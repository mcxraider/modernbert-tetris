// Bridge to the long-lived ModernBERT inference worker (models/serve.py).
//
// ModernBERT is a local PyTorch model, not an HTTP LLM — reloading it per move
// would be unusable. So we spawn ONE python3 process on first use, load the
// model once, and talk to it over stdio: one JSON request line in, one reply
// line out. Requests are serialized through a single in-flight queue (the worker
// handles one line at a time); fine for one solo game or two battle sides at
// human speed. ponytail: single worker, add a pool if throughput ever matters.
//
// Local dev only (server.mjs). Vercel functions can't hold a torch process.

import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { existsSync } from "node:fs";
import { jsonResult } from "./typesafe.mjs";

const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
// Prefer the project venv (where torch lives) over bare system python3.
const VENV_PYTHON = join(REPO_ROOT, ".venv", "bin", "python");
const PYTHON = process.env.PYTHON || (existsSync(VENV_PYTHON) ? VENV_PYTHON : "python3");
const READY_TIMEOUT_MS = 120_000; // first load may download ModernBERT-base + compile
const REQUEST_TIMEOUT_MS = 30_000;

let worker = null; // { proc, ready: Promise<{source}>, queue: [], current }

function startWorker() {
  const proc = spawn(PYTHON, [join(REPO_ROOT, "models", "serve.py")], {
    cwd: REPO_ROOT,
    stdio: ["pipe", "pipe", "inherit"], // python stderr → server console
  });
  const state = { proc, queue: [], current: null, ready: null };
  const rl = createInterface({ input: proc.stdout });

  state.ready = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("ModernBERT worker did not become ready in time.")), READY_TIMEOUT_MS);
    rl.once("line", (line) => {
      clearTimeout(timer);
      let msg;
      try {
        msg = JSON.parse(line);
      } catch {
        reject(new Error(`ModernBERT worker sent non-JSON on startup: ${line.slice(0, 200)}`));
        return;
      }
      if (msg.ready) resolve({ source: msg.source });
      else reject(new Error(msg.error || "ModernBERT worker failed to load."));
    });
  });

  // After the ready line, every subsequent line answers the current request.
  rl.on("line", (line) => {
    if (!state.current) return; // the ready line, handled above
    const { resolve } = state.current;
    state.current = null;
    try {
      resolve(JSON.parse(line));
    } catch {
      resolve({ error: `Worker sent non-JSON: ${line.slice(0, 200)}` });
    }
    pump(state);
  });

  const fail = (err) => {
    const dead = new Error(`ModernBERT worker exited: ${err?.message || err}`);
    state.ready.catch(() => {}); // avoid unhandled rejection if it never readied
    for (const item of state.queue.splice(0)) item.reject(dead);
    if (state.current) {
      state.current.reject(dead);
      state.current = null;
    }
    if (worker === state) worker = null; // next request respawns
  };
  proc.on("exit", () => fail(new Error("process closed")));
  proc.on("error", fail);
  return state;
}

function pump(state) {
  if (state.current || state.queue.length === 0) return;
  const item = state.queue.shift();
  state.current = item;
  const timer = setTimeout(() => {
    if (state.current === item) {
      state.current = null;
      item.reject(new Error("ModernBERT worker timed out."));
    }
  }, REQUEST_TIMEOUT_MS);
  const done = (fn) => (v) => {
    clearTimeout(timer);
    fn(v);
  };
  item.resolve = done(item._resolve);
  item.reject = done(item._reject);
  state.proc.stdin.write(JSON.stringify(item.payload) + "\n");
}

// Spawn and load the worker at server startup so the first move isn't slow.
// Logs when the model is ready (or why it isn't). Safe to skip calling.
export async function warmModernBert() {
  if (!worker) worker = startWorker();
  try {
    const { source } = await worker.ready;
    console.log(`ModernBERT loaded (${source}).`);
  } catch (err) {
    worker = null; // let a real request respawn (e.g. after installing torch)
    console.log(`ModernBERT not available: ${err.message}`);
  }
}

// Run one inference request. Returns a server result object (via jsonResult).
export async function forwardToModernBert({ body }) {
  if (!worker) worker = startWorker();
  const state = worker;
  try {
    await state.ready;
  } catch (err) {
    worker = null; // let the next attempt respawn (e.g. after installing torch)
    return jsonResult(503, { error: { type: "model_unavailable", message: err.message } });
  }

  let req;
  try {
    req = JSON.parse(body);
  } catch {
    return jsonResult(400, { error: { type: "bad_request", message: "Body must be JSON." } });
  }
  if (!Array.isArray(req.prompts) || !Array.isArray(req.candidate_ids)) {
    return jsonResult(400, { error: { type: "bad_request", message: "Expected { prompts: [], candidate_ids: [] }." } });
  }

  try {
    const reply = await new Promise((resolve, reject) => {
      const item = { payload: { prompts: req.prompts, candidate_ids: req.candidate_ids }, resolve, reject, _resolve: resolve, _reject: reject };
      state.queue.push(item);
      pump(state);
    });
    if (reply.error) return jsonResult(422, { error: { type: "inference_error", message: reply.error } });
    return jsonResult(200, reply);
  } catch (err) {
    return jsonResult(502, { error: { type: "worker_error", message: err.message } });
  }
}
