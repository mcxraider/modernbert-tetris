// Plays N headless solo Tetris games with Jev in parallel, each to its own
// battle-log file (same format the browser solo page writes via /api/log).
//
//   TYPESAFE_API_KEY=... node tools/solo-batch.mjs [--runs 5] [--mode jev-better-prompt]
//                                                  [--max-plies 200] [--seeds 1-5] [--delay 0]
//
// Default mode is jev-better-prompt (shortlist + board_after per candidate),
// matching the solo page's "Jev (better prompt)" toggle. Each run explores a
// fresh random seed unless --seeds pins them, and writes:
//   data/battles/solo-<mode>-<ISO>-r<seed>.json
//
// Live controls (TTY only): space = pause/resume, + / - = slower/faster
// (±100 ms between moves, the browser slider's analogue), q = quit gracefully.

import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { saveStateRows } from "../lib/dataset.mjs";
import { newGame, stepSolo } from "../lib/solo.mjs";

const BATTLE_DIR = join(fileURLToPath(new URL("..", import.meta.url)), "data", "battles");
const ENDPOINT = process.env.TYPESAFE_BASE
  ? `${process.env.TYPESAFE_BASE.replace(/\/+$/, "")}/v1/systemone`
  : "https://api.typesafe.ai/v1/systemone";

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Shared live controls across all runs. `delayMs` throttles between moves;
// `paused` holds every loop at its next move; `quit` ends gracefully.
const control = { delayMs: 0, paused: false, quit: false };

// Blocks until unpaused, then waits delayMs. Returns false if we're quitting.
async function gate() {
  while (control.paused && !control.quit) await sleep(100);
  if (control.quit) return false;
  if (control.delayMs > 0) await sleep(control.delayMs);
  return !control.quit;
}

// Raw-mode key handling on the TTY: space/+/-/q. No deps; restores on exit.
function wireKeys() {
  const stdin = process.stdin;
  if (!stdin.isTTY) return () => {};
  stdin.setRawMode(true);
  stdin.resume();
  stdin.setEncoding("utf8");
  const onKey = (key) => {
    if (key === " ") {
      control.paused = !control.paused;
      console.log(control.paused ? "⏸  paused" : "▶  resumed");
    } else if (key === "+" || key === "=") {
      control.delayMs += 100;
      console.log(`slower: ${control.delayMs} ms between moves`);
    } else if (key === "-" || key === "_") {
      control.delayMs = Math.max(0, control.delayMs - 100);
      console.log(`faster: ${control.delayMs} ms between moves`);
    } else if (key === "q" || key === "") {
      control.quit = true;
      console.log("quitting after in-flight moves…");
    }
  };
  stdin.on("data", onKey);
  return () => {
    stdin.off("data", onKey);
    stdin.setRawMode(false);
    stdin.pause();
  };
}

function parseSeeds(spec) {
  const m = /^(\d+)-(\d+)$/.exec(spec);
  if (m) {
    const [a, b] = [Number(m[1]), Number(m[2])];
    return Array.from({ length: b - a + 1 }, (_, i) => a + i);
  }
  return spec.split(",").map(Number);
}

async function playRun({ seed, mode, maxPlies, apiKey }) {
  const betterPrompt = mode === "jev-better-prompt";
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const runId = `solo-${mode}-${stamp}-r${seed}`;
  const out = join(BATTLE_DIR, `${runId}.json`);
  const game = newGame(seed);
  let total = 0;
  for (let ply = 0; ply < maxPlies; ply++) {
    if (!(await gate())) break;
    const { rows, done } = await stepSolo(game, apiKey, { betterPrompt, endpoint: ENDPOINT });
    if (done) break;
    await saveStateRows(out, rows.map((r) => ({ ...r, run: runId }))); // per-ply, like the browser's /api/log
    total += rows.length;
    if ((ply + 1) % 10 === 0) console.log(`run r${seed}: ply ${ply + 1}, lines ${game.lines}, ${total} candidates`);
  }
  console.log(`run r${seed}: done, lines ${game.lines}, ${total} candidates → ${out}`);
  return { seed, lines: game.lines, candidates: total };
}

async function main() {
  const apiKey = process.env.TYPESAFE_API_KEY;
  if (!apiKey) throw new Error("Set TYPESAFE_API_KEY in the environment.");
  const runs = Number(arg("runs", 5));
  const mode = arg("mode", "jev-better-prompt");
  if (mode !== "jev-better-prompt" && mode !== "jev") throw new Error(`--mode must be jev or jev-better-prompt, got ${mode}`);
  const maxPlies = Number(arg("max-plies", 200));
  control.delayMs = Number(arg("delay", 0));
  const seedsArg = arg("seeds", null);
  // Default: one fresh random seed per run so each batch explores new states.
  const seeds = seedsArg ? parseSeeds(seedsArg) : Array.from({ length: runs }, () => Math.floor(Math.random() * 1e9));

  console.log(`Playing ${seeds.length} ${mode} game(s) in parallel (max ${maxPlies} plies each)…`);
  if (process.stdin.isTTY) console.log("controls: space=pause/resume  +/-=slower/faster  q=quit\n");
  const unwire = wireKeys();
  try {
    const results = await Promise.all(seeds.map((seed) => playRun({ seed, mode, maxPlies, apiKey })));
    const totalCandidates = results.reduce((a, r) => a + r.candidates, 0);
    console.log(`\nDone. ${results.length} runs, ${totalCandidates} candidates total.`);
  } finally {
    unwire();
  }
}

main().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
