// Vercel serverless function: POST /api/platform -> GovTech Platform AI
// (OpenAI-compatible chat-completions). The key is read from the environment
// by forwardToPlatform; the client sends no key.
import { readJsonBody, validateJson, writeResult, jsonResult } from "../lib/typesafe.mjs";
import { forwardToPlatform } from "../lib/platform.mjs";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    writeResult(res, jsonResult(405, { error: { type: "method_not_allowed", message: "Use POST." } }));
    return;
  }
  const body = await readJsonBody(req);
  const invalid = validateJson(body);
  if (invalid) {
    writeResult(res, invalid);
    return;
  }
  writeResult(res, await forwardToPlatform({ body }));
}
