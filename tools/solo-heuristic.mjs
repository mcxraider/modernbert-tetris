// Plays N headless solo Tetris games where the built-in heuristic picks each
// move (argmax of placement.heuristic), recording the SAME per-candidate
// battle-log format as the Jev model driver — only the decision source differs.
// Labels are a softmax over heuristic scores; the argmax is is_choice. No API.
//
//   node tools/solo-heuristic.mjs [--runs 5] [--seeds 1-5] [--max-plies 200] [--temp 1]
//
// Writes data/heuristic/solo-heuristic-<ISO>-r<seed>.json (kept separate from the
// model-generated data/battles/, so collapse_battles.py does not pick these up).

import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { mkdirSync } from "node:fs";
import { saveStateRows } from "../lib/dataset.mjs";
import { newGame, stepSolo, heuristicDecision } from "../lib/solo.mjs";

const OUT_DIR = join(fileURLToPath(new URL("..", import.meta.url)), "data", "heuristic");
const SHORTLIST_K = 10;

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

function parseSeeds(spec) {
  const m = /^(\d+)-(\d+)$/.exec(spec);
  if (m) {
    const [a, b] = [Number(m[1]), Number(m[2])];
    return Array.from({ length: b - a + 1 }, (_, i) => a + i);
  }
  return spec.split(",").map(Number);
}

async function playRun({ seed, maxPlies, temperature }) {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const runId = `solo-heuristic-${stamp}-r${seed}`;
  const out = join(OUT_DIR, `${runId}.json`);
  const game = newGame(seed);
  const decide = (offered) => heuristicDecision(offered, { temperature });
  let total = 0;
  for (let ply = 0; ply < maxPlies; ply++) {
    const { rows, done } = await stepSolo(game, null, { betterPrompt: true, decide, shortlistK: SHORTLIST_K });
    if (done) break;
    await saveStateRows(out, rows.map((r) => ({ ...r, run: runId })));
    total += rows.length;
    if ((ply + 1) % 20 === 0) console.log(`run r${seed}: ply ${ply + 1}, lines ${game.lines}, ${total} candidates`);
  }
  console.log(`run r${seed}: done, lines ${game.lines}, ${total} candidates → ${out}`);
  return { seed, lines: game.lines, candidates: total };
}

async function main() {
  const runs = Number(arg("runs", 5));
  const maxPlies = Number(arg("max-plies", 200));
  const temperature = Number(arg("temp", 1));
  const seedsArg = arg("seeds", null);
  const seeds = seedsArg ? parseSeeds(seedsArg) : Array.from({ length: runs }, () => Math.floor(Math.random() * 1e9));

  mkdirSync(OUT_DIR, { recursive: true });
  console.log(`Playing ${seeds.length} heuristic game(s) in parallel (max ${maxPlies} plies each, temp ${temperature})…`);
  console.log(`Top-${SHORTLIST_K} shortlist per state (line-clears always kept, so a state may offer a few more).`);
  const results = await Promise.all(seeds.map((seed) => playRun({ seed, maxPlies, temperature })));
  const totalCandidates = results.reduce((a, r) => a + r.candidates, 0);
  console.log(`\nDone. ${results.length} runs, ${totalCandidates} candidates total.`);
}

main().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
