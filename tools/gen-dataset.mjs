// Headless dataset generator: self-plays Tetris with Jev and records, for each
// (state S, candidate placement) pair, a rendered STATE/CANDIDATE prompt string
// and the probability Jev assigned to that candidate. Emits JSON grouped by state
// (4-space indent). No browser, no gravity, no timers — just the pure engine
// (tetris.js) plus Jev's request builder (jev.js).
//
//   TYPESAFE_API_KEY=sk-... node tools/gen-dataset.mjs --seeds 1-50 --out data/jev.json
//
// Format: { state_id: { candidate_id: { prompt, probability, is_choice } } }.
// Candidates sharing a state_id belong to one state; their
// probabilities sum to ~1. The prompt is the exact string you train on;
// candidate ids (p0,p1,...) are positional within a state and group elements.

import { saveStateRows } from "../lib/dataset.mjs";
import { newGame, stepSolo } from "../lib/solo.mjs";

const ENDPOINT = process.env.TYPESAFE_BASE
  ? `${process.env.TYPESAFE_BASE.replace(/\/+$/, "")}/v1/systemone`
  : "https://api.typesafe.ai/v1/systemone";

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

async function main() {
  const apiKey = process.env.TYPESAFE_API_KEY;
  if (!apiKey) throw new Error("Set TYPESAFE_API_KEY in the environment.");
  const seeds = parseSeeds(arg("seeds", "1-10"));
  const maxPlies = Number(arg("max-plies", 200));
  const out = arg("out", "data/jev.json");
  let total = 0;
  for (const seed of seeds) {
    const game = newGame(seed);
    const seedRows = [];
    for (let ply = 0; ply < maxPlies; ply++) {
      const { rows, done } = await stepSolo(game, apiKey, { betterPrompt: false, endpoint: ENDPOINT });
      if (done) break;
      seedRows.push(...rows);
    }
    await saveStateRows(out, seedRows); // atomically merge after each seed
    total += seedRows.length;
    console.log(`seed ${seed}: done, ${total} candidates so far`);
  }
  console.log(`Wrote ${total} candidates grouped by state to ${out}`);
}

main().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
