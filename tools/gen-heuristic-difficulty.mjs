// Generates a difficulty-balanced corpus of heuristic-played Tetris states —
// the same per-candidate battle-log format as the other heuristic driver, but
// bucketed by stack height so each difficulty band gets a fixed quota.
//
//   node tools/gen-heuristic-difficulty.mjs [--out-dir data/heuristic-difficulty]
//          [--max-plies 200] [--temp 1] [--limit N] [--seed 1000]
//          [--max-per-game 10] [--selftest]
//
// A "state" = one board position with its ~10 candidate rows; quotas count
// states. Difficulty is the engine's describeHeight band of the board BEFORE the
// move (very-easy ≤4, easy-medium 5-8, medium 9-12, hard 13-15, near-topout
// 16-19). The DT heuristic keeps the board low, so the harder bands never occur
// naturally — those buckets seed the board with random garbage first, then let
// the heuristic play (and dig out) from there. Writes one file per bucket:
//   data/heuristic-difficulty/<bucket>.json  (+ a `difficulty` field per row)

import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { mkdirSync } from "node:fs";
import {
  WIDTH,
  HEIGHT,
  GARBAGE,
  emptyBoard,
  boardStats,
  seededRandom,
} from "../public/tetris.js";
import { saveStateRows } from "../lib/dataset.mjs";
import { newGame, stepSolo, heuristicDecision } from "../lib/solo.mjs";

const OUT_DIR_DEFAULT = join(fileURLToPath(new URL("..", import.meta.url)), "data", "heuristic-difficulty");
const SHORTLIST_K = 10;
const MAX_SEEDS = 100000; // backstop so an unreachable quota can't loop forever

// Editable per-bucket config. `lo/hi` = inclusive maxHeight band (matches
// describeHeight). `seedHeight` = rows of garbage to pre-fill (0 = natural play
// from empty). `density` = garbage fill fraction (realism knob). near-topout
// caps seedHeight at 16 so the spawn row stays clear and the game doesn't
// instantly top out — the heuristic then climbs into 17-19 under pressure.
const BUCKETS = [
  { name: "very-easy", lo: 0, hi: 4, quota: 2500, seedHeight: 0, density: 0 },
  { name: "easy-medium", lo: 5, hi: 8, quota: 2500, seedHeight: 6, density: 0.5 },
  { name: "medium", lo: 9, hi: 12, quota: 3000, seedHeight: 10, density: 0.65 },
  { name: "hard", lo: 13, hi: 15, quota: 2500, seedHeight: 14, density: 0.8 },
  { name: "near-topout", lo: 16, hi: 19, quota: 2000, seedHeight: 16, density: 0.85 },
];

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}
const hasFlag = (name) => process.argv.includes(`--${name}`);

// Fills the bottom `height` rows of a fresh board with random garbage at the
// given density, guaranteeing no fully-complete row (so clearLines won't fire
// and the stack height is exactly `height`). Deterministic given `rng`.
function seedBoard(rng, height, density) {
  const board = emptyBoard();
  for (let y = HEIGHT - height; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) if (rng() < density) board[y][x] = GARBAGE;
    // Guarantee at least one filled cell (so the row contributes to height) and
    // at least one gap (so the row is never complete).
    if (board[y].every((c) => c)) board[y][(rng() * WIDTH) | 0] = null;
    if (board[y].every((c) => !c)) board[y][(rng() * WIDTH) | 0] = GARBAGE;
  }
  return board;
}

// Plays games with the heuristic and collects in-band candidate rows (no file
// I/O), until `quota` states or MAX_SEEDS games. Returns { rows, states }.
// ponytail: maxPerGame caps in-band states taken from one game, trading
// game-turnover for episode diversity — each game carries one episode_id, so a
// low cap means many episodes for a trustworthy per-game holdout. Raise it if a
// bucket can't meet quota within MAX_SEEDS games.
async function collectBucketRows(bucket, { maxPlies, temperature, limit, baseSeed, maxPerGame }) {
  const { name, lo, hi, seedHeight, density } = bucket;
  const quota = limit ? Math.min(bucket.quota, limit) : bucket.quota;
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const runId = `heuristic-difficulty-${name}-${stamp}`;
  const decide = (offered, gameInfo) => heuristicDecision(offered, gameInfo.nextPiece, { temperature });
  const seedRng = seededRandom(baseSeed + BUCKETS.indexOf(bucket) * 1_000_003);

  const rows = [];
  let states = 0;
  let seed = baseSeed;
  while (states < quota && seed - baseSeed < MAX_SEEDS) {
    const game = newGame(seed++);
    if (seedHeight > 0) game.board = seedBoard(seedRng, seedHeight, density);
    let inBand = 0; // in-band states taken from THIS game
    for (let ply = 0; ply < maxPlies && states < quota && inBand < maxPerGame; ply++) {
      const h = boardStats(game.board).maxHeight; // the board stepSolo is about to record
      const { rows: r, done } = await stepSolo(game, null, { betterPrompt: true, decide, shortlistK: SHORTLIST_K });
      if (done) break;
      if (h >= lo && h <= hi) {
        rows.push(...r.map((row) => ({ ...row, run: runId, difficulty: name })));
        states += 1;
        inBand += 1;
      }
    }
  }
  return { rows, states, quota };
}

async function fillBucket(bucket, opts) {
  const { name } = bucket;
  const { rows, states, quota } = await collectBucketRows(bucket, opts);
  await saveStateRows(join(opts.outDir, `${name}.json`), rows); // single write per bucket
  const shortfall = states < quota ? `  ⚠ quota not met (${states}/${quota}) within ${MAX_SEEDS} seeds` : "";
  console.log(`${name}: ${states} states, ${rows.length} candidate rows → ${name}.json${shortfall}`);
  return states;
}

function selftest() {
  // seedBoard produces exactly `height` stack with no complete row.
  const rng = seededRandom(7);
  for (const height of [6, 10, 14, 16]) {
    for (let t = 0; t < 20; t++) {
      const board = seedBoard(rng, height, 0.7);
      const stats = boardStats(board);
      if (stats.maxHeight !== height) throw new Error(`seedBoard height ${stats.maxHeight} !== ${height}`);
      if (board.some((row) => row.every((c) => c))) throw new Error("seedBoard left a complete row");
    }
  }
  // Bucket bands tile the describeHeight boundaries with no gap/overlap.
  const sorted = [...BUCKETS].sort((a, b) => a.lo - b.lo);
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].lo !== sorted[i - 1].hi + 1) throw new Error(`band gap/overlap at ${sorted[i].name}`);
  }
  if (sorted[0].lo !== 0 || sorted.at(-1).hi !== HEIGHT - 1) throw new Error("bands do not cover 0..19");
  return selftestEpisodeDiversity();
}

// The per-game cap must force game turnover: with cap C, N states need ≥ ⌈N/C⌉
// distinct episode_ids. very-easy keeps one game in-band for ~160 plies, so an
// uncapped run would collapse to a handful of episodes — this guards that.
async function selftestEpisodeDiversity() {
  const cap = 5;
  const { rows, states } = await collectBucketRows(
    BUCKETS[0], { maxPlies: 200, temperature: 1, limit: 20, baseSeed: 1000, maxPerGame: cap });
  const episodes = new Set(rows.map((r) => r.episode_id));
  const expected = Math.ceil(states / cap);
  if (episodes.size < expected) {
    throw new Error(`episode diversity: ${episodes.size} episodes for ${states} states (cap ${cap}, expected ≥ ${expected})`);
  }
  const perGame = {};
  for (const r of rows) perGame[r.episode_id] = (perGame[r.episode_id] || 0) + 1;
  // rows are per-candidate; count distinct states per episode via state_id.
  const statesPerGame = {};
  for (const r of rows) (statesPerGame[r.episode_id] ||= new Set()).add(r.state_id);
  if (Object.values(statesPerGame).some((s) => s.size > cap)) {
    throw new Error("episode diversity: a game exceeded the per-game cap");
  }
  console.log(`selftest ok (${episodes.size} episodes for ${states} states, cap ${cap})`);
}

async function main() {
  if (hasFlag("selftest")) return selftest();
  const outDir = arg("out-dir", OUT_DIR_DEFAULT);
  const maxPlies = Number(arg("max-plies", 200));
  const temperature = Number(arg("temp", 1));
  const limit = arg("limit", null) ? Number(arg("limit", null)) : null;
  const baseSeed = Number(arg("seed", 1000));
  const maxPerGame = Number(arg("max-per-game", 10));

  mkdirSync(outDir, { recursive: true });
  const total = BUCKETS.reduce((a, b) => a + (limit ? Math.min(b.quota, limit) : b.quota), 0);
  console.log(`Generating ${total} heuristic states across ${BUCKETS.length} difficulty buckets → ${outDir}`);

  let got = 0;
  for (const bucket of BUCKETS) got += await fillBucket(bucket, { outDir, maxPlies, temperature, limit, baseSeed, maxPerGame });
  console.log(`\nDone. ${got} states total.`);
}

main().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
