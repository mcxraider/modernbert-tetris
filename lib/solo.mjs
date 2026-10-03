// Shared headless solo-rollout step: self-plays one Tetris game with Jev, no
// browser/gravity/timers. Used by tools/gen-dataset.mjs (plain Jev, merged
// dataset) and tools/solo-batch.mjs (parallel per-run battle logs).
//
// `betterPrompt` mirrors the browser's "Jev (better prompt)" solo mode: shortlist
// candidates and send `board_after` per candidate (see public/app.js jevDecision).

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
import { stateRows } from "../public/prompt.js";

const SPAWN_X = 3;

// One deterministic game: a 7-bag queue the same way the arena draws it.
export function newGame(seed) {
  const random = seededRandom(seed);
  let bag = [];
  const draw = () => {
    if (bag.length === 0) bag = makeBag(random);
    return bag.pop();
  };
  return { board: emptyBoard(), current: draw(), next: draw(), draw, lines: 0 };
}

// Default decider: ask Jev for one move. Returns { probs, choiceId } over the
// offered placements, same shape as any pluggable decider.
async function jevDecide(offered, gameInfo, apiKey, { betterPrompt, endpoint }) {
  const request = buildRequest(gameInfo, offered, { withBoardAfter: betterPrompt });
  request.questions = { placement: request.questions.placement }; // drop extra questions, as the battle does
  const { response } = await askJev(request, apiKey, { endpoint, maxAttempts: 4 });
  const { chosen } = pickPlacement(response, offered);
  return {
    probs: response?.answers?.placement?.probabilities || {},
    choiceId: response?.answers?.placement?.choice ?? chosen.id,
  };
}

// Softmax over each placement's precomputed .heuristic; argmax is the choice.
// A pure, API-free decider — same { probs, choiceId } contract as jevDecide.
// ponytail: temperature is the calibration knob — raw scores are near one-hot.
export function heuristicDecision(offered, { temperature = 1 } = {}) {
  const scores = offered.map((p) => p.heuristic / temperature);
  const max = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = {};
  offered.forEach((p, i) => {
    probs[p.id] = exps[i] / sum;
  });
  return { probs, choiceId: offered[scores.indexOf(max)].id };
}

// Picks one move, records (state, candidate) rows, advances the board by the
// argmax pick. The decision source is pluggable via `decide(offered, gameInfo)`
// → { probs, choiceId }; defaults to Jev. Returns { rows, done, cleared }
// ({ done: true } at top-out).
export async function stepSolo(game, apiKey, { betterPrompt = false, endpoint, decide, shortlistK = 12 } = {}) {
  const { board, current, next, lines } = game;
  const spawn = PIECES[current][0];
  if (collides(board, spawn.cells, SPAWN_X, 0)) return { done: true };

  const placements = enumeratePlacements(board, current);
  if (placements.length === 0) return { done: true };
  const offered = betterPrompt ? shortlistPlacements(placements, shortlistK) : placements;

  const gameInfo = { board, piece: current, nextPiece: next, stats: boardStats(board), linesCleared: lines };
  const { probs, choiceId } = decide
    ? decide(offered, gameInfo)
    : await jevDecide(offered, gameInfo, apiKey, { betterPrompt, endpoint });
  const chosen = offered.find((p) => p.id === choiceId) || offered[0];

  const rows = stateRows({
    stateId: `${new Date().toISOString()}-${randomUUID()}`,
    board,
    piece: current,
    nextPiece: next,
    lines,
    placements: offered,
    probs,
    choiceId,
  });

  const locked = lockPiece(board, chosen.piece, chosen.rotation, chosen.x, chosen.y);
  const { board: after, cleared } = clearLines(locked);
  game.board = after;
  game.lines += cleared;
  game.current = game.next;
  game.next = game.draw();
  return { rows, done: false, cleared };
}
