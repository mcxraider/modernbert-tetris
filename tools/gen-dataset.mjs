// Headless dataset generator: self-plays Tetris with Jev and records, for each
// (state S, candidate placement) pair, a rendered STATE/CANDIDATE prompt string
// and the probability Jev assigned to that candidate. Emits one JSON array
// (4-space indent). No browser, no gravity, no timers — just the pure engine
// (tetris.js) plus Jev's request builder (jev.js).
//
//   TYPESAFE_API_KEY=sk-... node tools/gen-dataset.mjs --seeds 1-50 --out data/jev.json
//
// Each element is { state_id, candidate_id, prompt, probability, is_choice }.
// Elements sharing a state_id are the candidates of one state; their
// probabilities sum to ~1. The prompt is the exact string you train on;
// candidate ids (p0,p1,...) are positional within a state and group elements.

import { writeFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
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
  PIECES,
} from "../public/tetris.js";
import { buildRequest, askJev, pickPlacement } from "../public/jev.js";
import { stateRows } from "../public/prompt.js";

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

// One deterministic game: a 7-bag queue the same way the arena draws it.
function newGame(seed) {
  const random = seededRandom(seed);
  let bag = [];
  const draw = () => {
    if (bag.length === 0) bag = makeBag(random);
    return bag.pop();
  };
  return { board: emptyBoard(), current: draw(), next: draw(), draw, lines: 0 };
}

async function recordState(game, apiKey) {
  const { board, current, next, lines } = game;
  const spawn = PIECES[current][0];
  if (collides(board, spawn.cells, SPAWN_X, 0)) return { done: true };

  const placements = enumeratePlacements(board, current);
  if (placements.length === 0) return { done: true };

  const gameInfo = { board, piece: current, nextPiece: next, stats: boardStats(board), linesCleared: lines };
  const request = buildRequest(gameInfo, placements);
  request.questions = { placement: request.questions.placement }; // drop extra questions, as the battle does

  const { response } = await askJev(request, apiKey, { endpoint: ENDPOINT, maxAttempts: 4 });
  const { chosen } = pickPlacement(response, placements);
  const probs = response?.answers?.placement?.probabilities || {};
  const choiceId = response?.answers?.placement?.choice ?? chosen.id;

  const rows = stateRows({
    stateId: `${new Date().toISOString()}-${randomUUID()}`,
    board,
    piece: current,
    nextPiece: next,
    lines,
    placements,
    probs,
    choiceId,
  });

  // Advance the board by the chosen placement (argmax rollout).
  const locked = lockPiece(board, chosen.piece, chosen.rotation, chosen.x, chosen.y);
  const { board: after, cleared } = clearLines(locked);
  game.board = after;
  game.lines += cleared;
  game.current = game.next;
  game.next = game.draw();
  return { rows, done: false };
}

async function main() {
  const apiKey = process.env.TYPESAFE_API_KEY;
  if (!apiKey) throw new Error("Set TYPESAFE_API_KEY in the environment.");
  const seeds = parseSeeds(arg("seeds", "1-10"));
  const maxPlies = Number(arg("max-plies", 200));
  const out = arg("out", "data/jev.json");
  await mkdir(dirname(out), { recursive: true });

  // A JSON array can't be appended incrementally, so accumulate then write once.
  const all = [];
  for (const seed of seeds) {
    const game = newGame(seed);
    for (let ply = 0; ply < maxPlies; ply++) {
      const { rows, done } = await recordState(game, apiKey);
      if (done) break;
      all.push(...rows);
    }
    await writeFile(out, JSON.stringify(all, null, 4)); // checkpoint after each seed
    console.log(`seed ${seed}: done, ${all.length} rows so far`);
  }
  console.log(`Wrote ${all.length} rows to ${out}`);
}

main().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
