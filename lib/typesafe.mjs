// Shared proxy logic: forwards one request to the TypeSafe API with the key
// the visitor supplied. Used by server.mjs (local) and api/*.js (Vercel).

export const TYPESAFE_BASE = process.env.TYPESAFE_API_BASE || "https://api.typesafe.ai";
// Optional host-provided fallback key; leave unset for a public deployment.
export const SERVER_KEY = process.env.TYPESAFE_API_KEY || "";
export const UPSTREAM_TIMEOUT_MS = 30_000;

export function keyFromAuthorization(header) {
  const match = /^Bearer\s+(.+)$/i.exec((header || "").trim());
  const key = match ? match[1].trim() : "";
  return key || SERVER_KEY;
}

// Returns { status, headers, body } ready to be written to the client.
export async function forwardToTypeSafe({ path, method, key, body }) {
  if (!key) {
    return jsonResult(401, {
      detail: { error_type: "authentication_error", message: "Enter your TypeSafe API key in the page first." },
    });
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
  try {
    const upstream = await fetch(`${TYPESAFE_BASE}${path}`, {
      method,
      headers: {
        Authorization: `Bearer ${key}`,
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: method === "POST" ? body : undefined,
      signal: controller.signal,
    });
    const text = await upstream.text();
    const headers = {
      "Content-Type": upstream.headers.get("content-type") || "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    };
    const retryAfter = upstream.headers.get("retry-after");
    if (retryAfter) headers["Retry-After"] = retryAfter;
    const requestId = upstream.headers.get("x-typesafe-request-id");
    if (requestId) headers["X-Typesafe-Request-Id"] = requestId;
    return { status: upstream.status, headers, body: text };
  } catch (err) {
    const timedOut = err?.name === "AbortError";
    return jsonResult(timedOut ? 504 : 502, {
      detail: {
        error_type: timedOut ? "timeout" : "upstream_error",
        message: timedOut ? "TypeSafe API did not answer in time." : `Could not reach TypeSafe API: ${err.message}`,
      },
    });
  } finally {
    clearTimeout(timer);
  }
}

export function jsonResult(status, obj) {
  return {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
    body: JSON.stringify(obj),
  };
}

export function configResult() {
  return jsonResult(200, { serverKeyConfigured: Boolean(SERVER_KEY), model: "jev-latest" });
}

// Writes a result object to a Node http.ServerResponse.
export function writeResult(res, result) {
  res.writeHead(result.status, { ...result.headers, "Content-Length": Buffer.byteLength(result.body) });
  res.end(result.body);
}

// Reads a request body as a string. Vercel pre-parses JSON into req.body;
// plain Node leaves the stream untouched. Handles both.
export async function readJsonBody(req, maxBytes = 1_000_000) {
  if (req.body !== undefined && req.body !== null) {
    return typeof req.body === "string" ? req.body : JSON.stringify(req.body);
  }
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > maxBytes) {
        reject(new Error("Request body too large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

// Validates that the body is JSON; returns an error result or null.
export function validateJson(body) {
  try {
    JSON.parse(body);
    return null;
  } catch {
    return jsonResult(400, { detail: { error_type: "bad_request", message: "Body must be JSON." } });
  }
}
