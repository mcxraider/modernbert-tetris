// Renders a Jev game state and one candidate placement into the STATE/CANDIDATE
// training-prompt string. Pure (only the engine's describe* helpers), so both
// the browser battle logger and the headless dataset generator share it and
// cannot drift apart.

import { boardStats, describePlacement, describeHeight, describeHoles, describeSurface } from "./tetris.js";

// The STATE block is identical for every candidate of a state.
export function renderState(board, piece, nextPiece, lines) {
  const stats = boardStats(board);
  const grid = board.map((row) => row.map((c) => (c ? 1 : 0)).join("")).join("\n");
  return [
    "STATE",
    "board_top_to_bottom:",
    grid,
    `column_heights: ${stats.heights.join(" ")}`,
    `stack_height: ${describeHeight(stats.maxHeight)}`,
    `holes: ${describeHoles(stats.holes)}`,
    `surface: ${describeSurface(stats.bumpiness)}`,
    `current_piece: ${piece}`,
    `next_piece: ${nextPiece}`,
    `lines_cleared_so_far: ${lines}`,
  ].join("\n");
}

export function renderCandidate(p) {
  const f = describePlacement(p);
  return [
    "CANDIDATE",
    `where: ${f.where}`,
    `lines_cleared: ${f.lines_cleared}`,
    `holes_created: ${f.holes_created}`,
    `holes_uncovered: ${f.holes_uncovered}`,
    `stack_height_after: ${f.stack_height_after}`,
    `height_change: ${f.height_change}`,
    `surface_after: ${f.surface_after}`,
    `wells_after: ${f.wells_after}`,
    "board_after:",
    p.afterBoard.map((row) => row.map((c) => (c ? 1 : 0)).join("")).join("\n"),
  ].join("\n");
}

// Full STATE + CANDIDATE prompt for one (state, candidate) pair.
export function renderPrompt(board, piece, nextPiece, lines, candidate) {
  return `${renderState(board, piece, nextPiece, lines)}\n\n${renderCandidate(candidate)}`;
}

// One dataset row per candidate of a state: same STATE block, differing
// CANDIDATE block, that candidate's probability as the label. `probs` is Jev's
// probabilities map ({ p0: 0.42, ... }); `choiceId` is its argmax pick.
export function stateRows({ stateId, board, piece, nextPiece, lines, placements, probs, choiceId, offeredIds }) {
  const stateBlock = renderState(board, piece, nextPiece, lines);
  return placements.map((p) => ({
    state_id: stateId,
    candidate_id: p.id,
    prompt: `${stateBlock}\n\n${renderCandidate(p)}`,
    probability: probs[p.id] ?? 0,
    is_choice: p.id === choiceId,
    ...(offeredIds ? { offered_to_jev: offeredIds.includes(p.id),
      probability_observed: Object.hasOwn(probs, p.id) } : {}),
  }));
}
