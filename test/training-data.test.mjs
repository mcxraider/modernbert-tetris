import { test } from "node:test";
import assert from "node:assert/strict";
import { stateRows } from "../public/prompt.js";
import { emptyBoard, enumeratePlacements } from "../public/tetris.js";
import { newGame, stepSolo, heuristicDecision } from "../lib/solo.mjs";

test("teacher metadata separates unoffered, evaluated-zero, and missing probabilities", () => {
  const board = emptyBoard();
  const placements = enumeratePlacements(board, "O").slice(0, 3);
  const rows = stateRows({ stateId: "s", board, piece: "O", nextPiece: "I", lines: 0,
    placements, offeredIds: placements.slice(0, 2).map(p => p.id),
    probs: { [placements[0].id]: 0 } });
  assert.equal(rows[0].offered_to_jev, true);
  assert.equal(rows[0].probability_observed, true);
  assert.equal(rows[0].probability, 0);
  assert.equal(rows[1].offered_to_jev, true);
  assert.equal(rows[1].probability_observed, false);
  assert.equal(rows[2].offered_to_jev, false);
});

test("separate games have separate episode IDs, retained across decisions", async () => {
  const a = newGame(42), b = newGame(42);
  assert.notEqual(a.episodeId, b.episodeId);
  const decide = (offered, info) => heuristicDecision(offered, info.nextPiece);
  for (let i = 0; i < 2; i++) {
    const { rows } = await stepSolo(a, null, { decide, betterPrompt: true });
    assert.ok(rows.length);
    assert.ok(rows.every(row => row.episode_id === a.episodeId));
  }
});
