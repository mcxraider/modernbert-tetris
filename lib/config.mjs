// Non-secret runtime config, read from config.json at the repo root.
// Secrets (API keys) stay in .env; everything else lives here so it can be
// committed and shared with the Python worker (models/config.py reads the
// same file). Env vars still win when set, for one-off overrides.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const config = JSON.parse(readFileSync(join(REPO_ROOT, "config.json"), "utf8"));

export const PLATFORM_BASE_URL = config.platformApiBaseUrl;
