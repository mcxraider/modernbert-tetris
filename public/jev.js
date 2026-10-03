// Builds the TypeSafe request for one Tetris move and reads the answer.
//
// One request per piece, following the speculative fan-out pattern: the
// placement Choice is the question that drives the game; the others are
// cheap extra judgments over the same state that the UI displays.

import { boardToText, describePlacement, describeHeight, describeSurface, describeHoles } from "./tetris.js";

export const MODEL = "jev-latest";

export const STRATEGY_OPTIONS = {
  build_clean: "The stack is low and tidy. Keep building a flat surface and wait for a chance to clear several lines at once.",
  clear_lines: "Lines can be cleared soon. Take clears as they come and keep the stack from growing.",
  repair_surface: "The surface is jagged or has holes. Prefer placements that smooth it out or uncover holes, even without clearing lines.",
  survive: "The stack is close to the top. Take any placement that lowers or does not raise the stack, even if it is ugly.",
};

export const HEALTH_LEVELS = [
  "Clean: low, flat stack with no holes",
  "Fine: some unevenness or a hole or two, plenty of room",
  "Rough: several holes or a jagged surface, room is shrinking",
  "Critical: stack near the top, the game may be lost within a few pieces",
];

export function buildState({ board, piece, nextPiece, stats, linesCleared }) {
  return {
    game: {
      rules: "Standard Tetris. Board is 10 columns wide and 20 rows tall. Rows fill left to right; a full row disappears. The game is lost when the stack reaches the top.",
      board_rows_top_to_bottom: boardToText(board),
      legend: "# is a filled cell, . is an empty cell. The first row is the top of the board.",
      column_heights_left_to_right: stats.heights,
      stack_height: describeHeight(stats.maxHeight),
      holes_in_stack: describeHoles(stats.holes),
      surface: describeSurface(stats.bumpiness),
      current_piece: piece,
      next_piece: nextPiece,
      lines_cleared_so_far: linesCleared,
    },
  };
}

// Renders a board (truthy-string/null cells) as "001..." rows, top first.
export function boardAfterRows(board) {
  return board.map((row) => row.map((c) => (c ? 1 : 0)).join(""));
}

export function buildQuestions(placements, { withBoardAfter = false } = {}) {
  const criteria = {};
  for (const p of placements) {
    criteria[p.id] = describePlacement(p);
    if (withBoardAfter) criteria[p.id].board_after = boardAfterRows(p.afterBoard);
  }
  return {
    placement: {
      type: "choice",
      instructions: {
        question: withBoardAfter
          ? "Which placement of `game.current_piece` should the player choose? Each option describes the board after that placement AND gives `board_after`: the exact resulting board as grid rows, top row first, 1 = filled and 0 = empty. Use board_after to reconstruct and verify the outcome before choosing."
          : "Which placement of `game.current_piece` should the player choose? Each option describes the board after that placement.",
        priorities: [
          "Clearing lines is good. Clearing more lines at once is better.",
          "Never seal an empty cell to keep the stack lower. A placement with holes_created of none beats one that creates holes even if the hole-maker leaves a lower stack_height_after. Only accept holes_created when that same placement clears far more lines right now, or the stack is already dangerously high and there is no clean option.",
          "A single empty column that is still open to the top is not a hole, it is a slot you can fill later. Prefer a placement that leaves such a slot open over one that drops a piece on top of it and seals it. Check game.next_piece and the piece after: an I, L, or J can usually fill a one-wide slot cleanly, so keep it open and wait. Only fill or cover a one-wide slot when no upcoming piece can reach it and the stack is getting high.",
          "Keep the stack low, but not by creating holes. Among placements that create no holes, prefer a lower stack_height_after and a height_change that does not grow the stack.",
          "Keep the surface flat. Prefer surface_after of flat over slightly uneven, bumpy, or very jagged.",
          "Think about the shape you leave behind, not just its height. A one-wide notch or a cell tucked under an overhang can only be filled by one specific piece, and a gap sealed against the left or right wall or in a corner may be impossible to fill at all. When two placements of the current piece are otherwise similar (for example the two orientations of an S or Z), prefer the one whose leftover empty cells stay open to a reachable piece, such as a flat ledge or an edge an L, J, or I can still slide into, over the one that walls a cell against the board edge. Use board_after to see the exact shape before choosing.",
          "One deep well is acceptable because the next I piece can fill it. Several deep wells are bad.",
          "When the stack is dangerously high, survival matters more than a clean surface.",
        ],
      },
      criteria,
    },
    strategy: {
      type: "choice",
      instructions: "Looking at `game`, which strategy fits the current situation best for the next few pieces?",
      criteria: STRATEGY_OPTIONS,
    },
    board_health: {
      type: "score",
      instructions: "How healthy is the stack in `game` for a Tetris player who wants to keep playing for a long time?",
      criteria: HEALTH_LEVELS,
    },
    next_piece_fits: {
      type: "noul",
      instructions: "Given `game.column_heights_left_to_right` and `game.surface`, is there an obvious clean spot for `game.next_piece` after this move, without creating holes?",
      criteria: {
        true: "A clean spot is easy to see.",
        false: "The next piece will be awkward to place.",
      },
    },
  };
}

export function buildRequest(gameInfo, placements, opts = {}) {
  return {
    state: buildState(gameInfo),
    model: MODEL,
    questions: buildQuestions(placements, opts),
  };
}

const RETRY_STATUSES = new Set([429, 529]);

export class JevError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

// Calls the local proxy (server.mjs), which forwards to api.typesafe.ai with
// the key in the Authorization header. Retries 429/529 with backoff.
export async function askJev(request, apiKey, { signal, maxAttempts = 4, endpoint = "api/systemone" } = {}) {
  let delay = 500;
  for (let attempt = 1; ; attempt++) {
    const started = performance.now();
    const headers = { "Content-Type": "application/json" };
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
    const res = await fetch(endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify(request),
      signal,
    });
    const latencyMs = performance.now() - started;
    let body = null;
    const text = await res.text();
    try {
      body = text ? JSON.parse(text) : null;
    } catch {
      body = { raw: text };
    }
    if (res.ok) return { response: body, latencyMs };
    if (RETRY_STATUSES.has(res.status) && attempt < maxAttempts) {
      const retryAfter = Number(res.headers.get("retry-after"));
      const wait = Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter * 1000 : delay;
      await new Promise((r) => setTimeout(r, wait));
      delay *= 2;
      continue;
    }
    const message = body?.detail?.message || body?.error || body?.message || body?.raw || `HTTP ${res.status}`;
    throw new JevError(message, res.status, body);
  }
}

// Reads the Choice answer and maps it back onto the placements. Falls back
// to the first placement if the answer somehow names an unknown option.
export function pickPlacement(response, placements) {
  const answer = response?.answers?.placement;
  if (!answer || answer.type !== "choice") throw new JevError("Response has no placement choice", 500, response);
  const byId = new Map(placements.map((p) => [p.id, p]));
  const ranked = Object.entries(answer.probabilities || {})
    .filter(([id]) => byId.has(id))
    .sort((a, b) => b[1] - a[1])
    .map(([id, probability]) => ({ placement: byId.get(id), probability }));
  const chosen = byId.get(answer.choice) || ranked[0]?.placement || placements[0];
  return { chosen, ranked, confidence: answer.confidence ?? 0 };
}
