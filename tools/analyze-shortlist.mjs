// Measures how aggressively the jev-better-prompt candidate shortlist can be
// trimmed. Self-plays Jev headlessly (same rollout as gen-dataset.mjs), and for
// each state records where Jev's own pick — made while seeing EVERY placement —
// lands in the heuristic ranking, and whether it survives shortlistPlacements()
// at a sweep of K. Answers "is K=12 too big?" with a recall@K curve.
//
//   TYPESAFE_API_KEY=sk-... node tools/analyze-shortlist.mjs --seeds 1-20 --ks 3,5,8,12
//
// recall@K = fraction of states where Jev's full-candidate pick is still in the
// shortlist of size K. The smallest K with recall ~1.0 is the K you can ship.
//
// ponytail: duplicates the tiny self-play rollout from gen-dataset.mjs rather
// than refactoring it to export one — the shared surface is ~15 lines and the
// two tools emit different things. Merge into a shared rollout if a third
// consumer appears.

import { randomUUID } from "node:crypto";
import {
  emptyBoard,
  boardStats,
  enumeratePlacements,
  lockPiece,
  clearLines,
  collides,
  makeBag,
  seededRandom,
  shortlistPlacements,
  PIECES,
} from "../public/tetris.js";
import { buildRequest, askJev, pickPlacement } from "../public/jev.js";

const SPAWN_X = 3;
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

function newGame(seed) {
  const random = seededRandom(seed);
  let bag = [];
  const draw = () => {
    if (bag.length === 0) bag = makeBag(random);
    return bag.pop();
  };
  return { board: emptyBoard(), current: draw(), next: draw(), draw, lines: 0 };
}

// Rank of the chosen placement in the heuristic ordering (0 = heuristic's own
// top pick). Ties broken by id so the rank is deterministic.
function heuristicRank(placements, chosenId) {
  const sorted = placements
    .slice()
    .sort((a, b) => b.heuristic - a.heuristic || a.id.localeCompare(b.id));
  return sorted.findIndex((p) => p.id === chosenId);
}

async function playState(game, apiKey, ks, acc) {
  const { board, current, next, lines } = game;
  const spawn = PIECES[current][0];
  if (collides(board, spawn.cells, SPAWN_X, 0)) return true;

  const placements = enumeratePlacements(board, current);
  if (placements.length === 0) return true;

  // Jev sees every placement — this is the baseline pick we try to preserve.
  const gameInfo = { board, piece: current, nextPiece: next, stats: boardStats(board), linesCleared: lines };
  const request = buildRequest(gameInfo, placements);
  request.questions = { placement: request.questions.placement };
  const { response } = await askJev(request, apiKey, { endpoint: ENDPOINT, maxAttempts: 4 });
  const { chosen } = pickPlacement(response, placements);
  const probs = response?.answers?.placement?.probabilities || {};
  const choiceId = response?.answers?.placement?.choice ?? chosen.id;

  acc.states += 1;
  acc.candidates.push(placements.length);
  acc.ranks.push(heuristicRank(placements, choiceId));
  // How much of Jev's probability mass the top-K heuristic candidates capture.
  for (const k of ks) {
    const inShort = new Set(shortlistPlacements(placements, k).map((p) => p.id));
    if (inShort.has(choiceId)) acc.recall[k] += 1;
    acc.shortSizes[k].push(inShort.size);
    const mass = placements.filter((p) => inShort.has(p.id)).reduce((s, p) => s + (probs[p.id] ?? 0), 0);
    acc.mass[k] += mass;
  }

  const locked = lockPiece(board, chosen.piece, chosen.rotation, chosen.x, chosen.y);
  const { board: after, cleared } = clearLines(locked);
  game.board = after;
  game.lines += cleared;
  game.current = game.next;
  game.next = game.draw();
  return false;
}

function histogram(ranks) {
  const max = Math.max(...ranks);
  const counts = Array(max + 1).fill(0);
  for (const r of ranks) counts[r] += 1;
  return counts;
}

function pct(n, d) {
  return d ? `${((100 * n) / d).toFixed(1)}%` : "–";
}

async function main() {
  const apiKey = process.env.TYPESAFE_API_KEY;
  if (!apiKey) throw new Error("Set TYPESAFE_API_KEY in the environment.");
  const seeds = parseSeeds(arg("seeds", "1-10"));
  const maxPlies = Number(arg("max-plies", 200));
  const ks = arg("ks", "3,5,8,12").split(",").map(Number).sort((a, b) => a - b);

  const acc = {
    states: 0,
    candidates: [],
    ranks: [],
    recall: Object.fromEntries(ks.map((k) => [k, 0])),
    mass: Object.fromEntries(ks.map((k) => [k, 0])),
    shortSizes: Object.fromEntries(ks.map((k) => [k, []])),
  };

  for (const seed of seeds) {
    const game = newGame(seed);
    for (let ply = 0; ply < maxPlies; ply++) {
      if (await playState(game, apiKey, ks, acc)) break;
    }
    process.stderr.write(`seed ${seed}: ${acc.states} states so far\n`);
  }

  const avg = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
  const counts = histogram(acc.ranks);

  console.log(`\nStates measured: ${acc.states}`);
  console.log(`Avg candidates per state: ${avg(acc.candidates).toFixed(1)} (max ${Math.max(...acc.candidates)})`);

  console.log(`\nHeuristic rank of Jev's pick (0 = heuristic's own top choice):`);
  for (let r = 0; r < counts.length; r++) {
    if (!counts[r]) continue;
    const bar = "█".repeat(Math.round((40 * counts[r]) / acc.states));
    console.log(`  rank ${String(r).padStart(2)}: ${String(counts[r]).padStart(4)} ${pct(counts[r], acc.states).padStart(6)} ${bar}`);
  }

  console.log(`\nShortlist recall@K (Jev's pick survives) and avg shortlist size:`);
  console.log(`   K   recall   avg size   prob-mass kept`);
  for (const k of ks) {
    console.log(
      `  ${String(k).padStart(2)}   ${pct(acc.recall[k], acc.states).padStart(6)}   ${avg(acc.shortSizes[k]).toFixed(1).padStart(8)}   ${(acc.mass[k] / acc.states).toFixed(3)}`,
    );
  }
  console.log(`\nSmallest K with ~100% recall is the safe shortlist size. If recall is\nalready 100% at a small K, 12 is too generous; drop it.`);
}

main().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
