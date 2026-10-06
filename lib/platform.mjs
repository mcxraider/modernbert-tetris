// Passthrough proxy for the GovTech Platform AI gateway (OpenAI-compatible
// chat-completions). Unlike the other proxies, the key is NOT supplied by the
// visitor — it is read from the server environment (PLATFORM_API_KEY), so the
// page never sees or sends a key. The page sends an OpenAI chat-completions
// body (including `model`); this forwards it unchanged.

import { jsonResult } from "./typesafe.mjs";
import { PLATFORM_BASE_URL } from "./config.mjs";

export const PLATFORM_BASE = (process.env.PLATFORM_API_BASE_URL || PLATFORM_BASE_URL).replace(/\/+$/, "");
export const PLATFORM_KEY = process.env.PLATFORM_API_KEY || "";
const UPSTREAM_TIMEOUT_MS = 30_000;

export async function forwardToPlatform({ body }) {
  if (!PLATFORM_KEY) {
    return jsonResult(500, {
      error: { type: "server_misconfigured", message: "PLATFORM_API_KEY is not set on the server." },
    });
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
  try {
    const upstream = await fetch(`${PLATFORM_BASE}/v1/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${PLATFORM_KEY}`,
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body,
      signal: controller.signal,
    });
    const text = await upstream.text();
    const headers = {
      "Content-Type": upstream.headers.get("content-type") || "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    };
    const retryAfter = upstream.headers.get("retry-after");
    if (retryAfter) headers["Retry-After"] = retryAfter;
    return { status: upstream.status, headers, body: text };
  } catch (err) {
    const timedOut = err?.name === "AbortError";
    return jsonResult(timedOut ? 504 : 502, {
      error: {
        type: timedOut ? "timeout" : "upstream_error",
        message: timedOut ? "Platform API did not answer in time." : `Could not reach Platform API: ${err.message}`,
      },
    });
  } finally {
    clearTimeout(timer);
  }
}
