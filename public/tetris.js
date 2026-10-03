// Pure Tetris engine. No DOM, no network. Used by the browser and by node tests.
//
// Code owns everything deterministic here: piece geometry, collision, line
// clears, enumerating every legal placement and describing each outcome.
// Jev (see jev.js) only picks between the placements this module produces.

export const WIDTH = 10;
export const HEIGHT = 20;

// Each piece: list of rotation states; each state is a list of [x, y] cells
// with y growing downwards. States are listed in clockwise order.
const RAW_PIECES = {
  I: [
    [[0, 1], [1, 1], [2, 1], [3, 1]],
    [[2, 0], [2, 1], [2, 2], [2, 3]],
  ],
  O: [[[0, 0], [1, 0], [0, 1], [1, 1]]],
  T: [
    [[1, 0], [0, 1], [1, 1], [2, 1]],
    [[1, 0], [1, 1], [2, 1], [1, 2]],
    [[0, 1], [1, 1], [2, 1], [1, 2]],
    [[1, 0], [0, 1], [1, 1], [1, 2]],
  ],
  S: [
    [[1, 0], [2, 0], [0, 1], [1, 1]],
    [[1, 0], [1, 1], [2, 1], [2, 2]],
  ],
  Z: [
    [[0, 0], [1, 0], [1, 1], [2, 1]],
    [[2, 0], [1, 1], [2, 1], [1, 2]],
  ],
  J: [
    [[0, 0], [0, 1], [1, 1], [2, 1]],
    [[1, 0], [2, 0], [1, 1], [1, 2]],
    [[0, 1], [1, 1], [2, 1], [2, 2]],
    [[1, 0], [1, 1], [0, 2], [1, 2]],
  ],
  L: [
    [[2, 0], [0, 1], [1, 1], [2, 1]],
    [[1, 0], [1, 1], [1, 2], [2, 2]],
    [[0, 1], [1, 1], [2, 1], [0, 2]],
    [[0, 0], [1, 0], [1, 1], [1, 2]],
  ],
};

export const PIECE_NAMES = Object.keys(RAW_PIECES);

export const PIECE_COLORS = {
  I: "#3fd6f0",
  O: "#f5d547",
  T: "#b565f2",
  S: "#5fe38a",
  Z: "#f26b6b",
  J: "#5b8cff",
  L: "#f4a340",
};

// Normalise every rotation so its bounding box starts at (0,0) and record its size.
export const PIECES = Object.fromEntries(
  Object.entries(RAW_PIECES).map(([name, states]) => [
    name,
    states.map((cells) => {
      const minX = Math.min(...cells.map((c) => c[0]));
      const minY = Math.min(...cells.map((c) => c[1]));
      const norm = cells.map(([x, y]) => [x - minX, y - minY]);
      return {
        cells: norm,
        width: Math.max(...norm.map((c) => c[0])) + 1,
        height: Math.max(...norm.map((c) => c[1])) + 1,
      };
    }),
  ]),
);

export function emptyBoard() {
  return Array.from({ length: HEIGHT }, () => new Array(WIDTH).fill(null));
}

export function cloneBoard(board) {
  return board.map((row) => row.slice());
}

export function collides(board, cells, x, y) {
  for (const [cx, cy] of cells) {
    const bx = x + cx;
    const by = y + cy;
    if (bx < 0 || bx >= WIDTH || by >= HEIGHT) return true;
    if (by >= 0 && board[by][bx]) return true;
  }
  return false;
}

// Returns the y at which the piece rests when dropped from `y` straight down.
export function dropY(board, cells, x, y) {
  let cur = y;
  while (!collides(board, cells, x, cur + 1)) cur += 1;
  return cur;
}

export function lockPiece(board, piece, rotation, x, y) {
  const next = cloneBoard(board);
  for (const [cx, cy] of PIECES[piece][rotation].cells) {
    const by = y + cy;
    if (by >= 0) next[by][x + cx] = piece;
  }
  return next;
}

// Removes full rows and returns { board, cleared, rows } (rows = indexes removed).
export function clearLines(board) {
  const rows = [];
  const kept = [];
  board.forEach((row, i) => {
    if (row.every((c) => c)) rows.push(i);
    else kept.push(row.slice());
  });
  while (kept.length < HEIGHT) kept.unshift(new Array(WIDTH).fill(null));
  return { board: kept, cleared: rows.length, rows };
}

export function columnHeights(board) {
  const heights = new Array(WIDTH).fill(0);
  for (let x = 0; x < WIDTH; x++) {
    for (let y = 0; y < HEIGHT; y++) {
      if (board[y][x]) {
        heights[x] = HEIGHT - y;
        break;
      }
    }
  }
  return heights;
}

// A hole is an empty cell with at least one filled cell above it in its column.
export function countHoles(board) {
  let holes = 0;
  for (let x = 0; x < WIDTH; x++) {
    let covered = false;
    for (let y = 0; y < HEIGHT; y++) {
      if (board[y][x]) covered = true;
      else if (covered) holes += 1;
    }
  }
  return holes;
}

export function bumpiness(heights) {
  let total = 0;
  for (let x = 0; x < heights.length - 1; x++) total += Math.abs(heights[x] - heights[x + 1]);
  return total;
}

// Wells: columns at least `depth` lower than both neighbours (walls count as tall).
export function wells(heights, depth = 3) {
  const found = [];
  for (let x = 0; x < heights.length; x++) {
    const left = x === 0 ? Infinity : heights[x - 1];
    const right = x === heights.length - 1 ? Infinity : heights[x + 1];
    const d = Math.min(left, right) - heights[x];
    if (d >= depth) found.push({ column: x, depth: d });
  }
  return found;
}

export function boardStats(board) {
  const heights = columnHeights(board);
  return {
    heights,
    maxHeight: Math.max(...heights),
    aggregateHeight: heights.reduce((a, b) => a + b, 0),
    holes: countHoles(board),
    bumpiness: bumpiness(heights),
    wells: wells(heights),
  };
}

// ---- Dellacherie-Thiery afterstate evaluation -----------------------------
// Nine-feature linear evaluation of a placement outcome: the six classic
// Dellacherie features plus the three BCTS/Dellacherie-Thiery extensions. Used
// by the built-in heuristic player (1-ply) and, via lookaheadScore, as the leaf
// value of its 2-ply search; also orders shortlistPlacements for Jev's prompt.
//
// Weights are a published starting point (the six Dellacherie weights are the
// El-Tetris / YuhanXiaoJY set; the three DT extensions are seeded modestly).
// ponytail: these assume these exact feature definitions AND our hard-drop-only
// action space — retuning against this engine (CMA-ES/CE/GA) is the follow-up,
// which is why this is one editable block. Zero a weight to drop its feature.
export const FEATURE_WEIGHTS = {
  landingHeight: -4.5,
  erodedCells: 3.42,
  rowTransitions: -3.22,
  columnTransitions: -9.35,
  holes: -7.9,
  wellSum: -3.39,
  holeDepth: -0.5, // BCTS
  rowsWithHoles: -1.0, // BCTS
  patternDiversity: -0.3, // DT (fuzziest feature; zero its weight to disable)
};

// Filled↔empty flips across each row, with the left/right walls counted as
// filled (so an exposed cell at the edge is a transition). High when rows are
// perforated — a surface the next piece can't sit flush on.
function rowTransitions(board) {
  let t = 0;
  for (let y = 0; y < HEIGHT; y++) {
    let prevFilled = true; // left wall
    for (let x = 0; x < WIDTH; x++) {
      const filled = !!board[y][x];
      if (filled !== prevFilled) t += 1;
      prevFilled = filled;
    }
    if (!prevFilled) t += 1; // right wall
  }
  return t;
}

// Filled↔empty flips down each column, with the floor (below the board) counted
// as filled and the ceiling (above) as empty.
function columnTransitions(board) {
  let t = 0;
  for (let x = 0; x < WIDTH; x++) {
    let prevFilled = false; // ceiling is empty
    for (let y = 0; y < HEIGHT; y++) {
      const filled = !!board[y][x];
      if (filled !== prevFilled) t += 1;
      prevFilled = filled;
    }
    if (!prevFilled) t += 1; // floor is filled
  }
  return t;
}

// Σ over every well of its cumulative depth 1+2+…+d (Dellacherie's well sum),
// for wells of ANY depth — distinct from wells()/boardStats which only flag
// deep (≥3) wells for the text prompt. A well cell is empty with filled cells
// (or a wall) on both sides; the well's depth is how many stack downward.
function wellSum(board) {
  let sum = 0;
  for (let x = 0; x < WIDTH; x++) {
    for (let y = 0; y < HEIGHT; y++) {
      const empty = !board[y][x];
      const leftFilled = x === 0 || !!board[y][x - 1];
      const rightFilled = x === WIDTH - 1 || !!board[y][x + 1];
      if (empty && leftFilled && rightFilled) {
        // Count the contiguous empty run downward from here as the well depth.
        let d = 0;
        for (let yy = y; yy < HEIGHT && !board[yy][x]; yy++) d += 1;
        sum += (d * (d + 1)) / 2;
        break; // the run is this column's single well; stop so it isn't recounted
      }
    }
  }
  return sum;
}

// Per-column holes with depth: a hole is an empty cell under a filled one.
// Returns { holes, holeDepth, rowsWithHoles } in one pass over the grid.
// holeDepth = Σ filled cells stacked directly above each hole (buried holes
// cost more); rowsWithHoles = count of rows containing at least one hole.
function holeFeatures(board) {
  let holes = 0;
  let holeDepth = 0;
  const rowsWithHoles = new Set();
  for (let x = 0; x < WIDTH; x++) {
    let coverage = 0; // filled cells seen above in this column
    for (let y = 0; y < HEIGHT; y++) {
      if (board[y][x]) coverage += 1;
      else if (coverage > 0) {
        holes += 1;
        holeDepth += coverage;
        rowsWithHoles.add(y);
      }
    }
  }
  return { holes, holeDepth, rowsWithHoles: rowsWithHoles.size };
}

// Thiery's surface diversity: how many distinct adjacent column-height
// differences appear (each clamped to [-2, 2]). A varied surface accommodates
// more piece shapes than a monotone one.
function patternDiversity(heights) {
  const diffs = new Set();
  for (let x = 0; x < heights.length - 1; x++) {
    let d = heights[x + 1] - heights[x];
    d = Math.max(-2, Math.min(2, d));
    diffs.add(d);
  }
  return diffs.size;
}

// The nine Dellacherie-Thiery features of a placement outcome. Pure: reads only
// the data enumeratePlacements already attached (afterBoard, cells, clearedRows,
// linesCleared, after.heights).
export function dellacherieFeatures(outcome) {
  const board = outcome.afterBoard;
  const ys = outcome.cells.map(([, y]) => y);
  const landingHeight = HEIGHT - (Math.min(...ys) + Math.max(...ys)) / 2;
  // Eroded cells: the piece's own cells that sat in rows this move cleared.
  // outcome.cells and clearedRows are both pre-clear coordinates, so this is a
  // valid intersection.
  const cleared = new Set(outcome.clearedRows);
  const pieceCellsCleared = ys.filter((y) => cleared.has(y)).length;
  const { holes, holeDepth, rowsWithHoles } = holeFeatures(board);
  return {
    landingHeight,
    erodedCells: outcome.linesCleared * pieceCellsCleared,
    rowTransitions: rowTransitions(board),
    columnTransitions: columnTransitions(board),
    holes,
    wellSum: wellSum(board),
    holeDepth,
    rowsWithHoles,
    patternDiversity: patternDiversity(outcome.after.heights),
  };
}

export function heuristicScore(outcome) {
  const f = dellacherieFeatures(outcome);
  let score = 0;
  for (const k in FEATURE_WEIGHTS) score += FEATURE_WEIGHTS[k] * f[k];
  return score;
}

// Enumerates every reachable (rotation, x) for `piece` on `board`. A placement
// is reachable when the piece fits at the spawn row and can drop straight down.
export function enumeratePlacements(board, piece) {
  const before = boardStats(board);
  const seen = new Set();
  const placements = [];
  PIECES[piece].forEach((state, rotation) => {
    for (let x = 0; x <= WIDTH - state.width; x++) {
      if (collides(board, state.cells, x, 0)) continue;
      const y = dropY(board, state.cells, x, 0);
      const locked = lockPiece(board, piece, rotation, x, y);
      const key = locked.map((r) => r.map((c) => (c ? "#" : ".")).join("")).join("/");
      if (seen.has(key)) continue; // identical outcome from another rotation (O, I, S, Z)
      seen.add(key);
      const { board: after, cleared, rows } = clearLines(locked);
      const stats = boardStats(after);
      const outcome = {
        id: `p${placements.length}`,
        piece,
        rotation,
        x,
        y,
        cells: state.cells.map(([cx, cy]) => [x + cx, y + cy]),
        linesCleared: cleared,
        clearedRows: rows,
        holesCreated: Math.max(0, stats.holes - before.holes),
        holesRemoved: Math.max(0, before.holes - stats.holes),
        heightDelta: stats.maxHeight - before.maxHeight,
        before,
        after: stats,
        afterBoard: after,
      };
      outcome.heuristic = heuristicScore(outcome);
      placements.push(outcome);
    }
  });
  return placements;
}

// 2-ply afterstate value: the placement's own 1-ply score plus the best reply
// to `nextPiece` on the board it leaves — Q(s,a) = h(s') + γ·max_a' h(s'').
// This is where the next-piece lookahead lives (NOT in enumeratePlacements), so
// it only costs the extra enumeration when a decider actually searches. Falls
// back to the 1-ply score when nextPiece is unknown.
export const LOOKAHEAD_GAMMA = 1;
const TOPOUT_PENALTY = -1e6;

export function lookaheadScore(outcome, nextPiece) {
  if (!nextPiece) return outcome.heuristic;
  const replies = enumeratePlacements(outcome.afterBoard, nextPiece);
  if (replies.length === 0) return TOPOUT_PENALTY; // next piece has nowhere to go
  const best = Math.max(...replies.map((r) => r.heuristic));
  return outcome.heuristic + LOOKAHEAD_GAMMA * best;
}

// Trims the candidate list to the strongest `k` placements, so a large prompt
// (jev-better-prompt sends a full board per candidate) stays small. Two stages:
// (A) drop placements dominated on every axis at once — a safe prune that keeps
// any move that is best on some dimension; (B) if still over k, keep the top-k
// by the linear heuristic, always retaining every line-clear. Ids are left as
// enumerated so pickPlacement maps answers back unchanged.
export function shortlistPlacements(placements, k = 12) {
  if (placements.length <= k) return placements;
  // Stage A: Pareto-dominance prune.
  // ponytail: O(n²) dominance scan, n≤~34 so no index needed.
  const dominated = (a, b) => {
    // true when b is at least as good as a on every axis and strictly better on one
    const ge =
      b.linesCleared >= a.linesCleared &&
      b.after.maxHeight <= a.after.maxHeight &&
      b.after.holes <= a.after.holes &&
      b.after.bumpiness <= a.after.bumpiness;
    const gt =
      b.linesCleared > a.linesCleared ||
      b.after.maxHeight < a.after.maxHeight ||
      b.after.holes < a.after.holes ||
      b.after.bumpiness < a.after.bumpiness;
    return ge && gt;
  };
  const frontier = placements.filter((a) => !placements.some((b) => b !== a && dominated(a, b)));
  if (frontier.length <= k) return frontier;
  // Stage B: keep every line-clear, then fill the rest by heuristic score.
  const clears = frontier.filter((p) => p.linesCleared > 0);
  const rest = frontier
    .filter((p) => p.linesCleared === 0)
    .sort((a, b) => b.heuristic - a.heuristic);
  const kept = [...clears, ...rest].slice(0, Math.max(k, clears.length));
  // Preserve enumeration order among the survivors for a stable prompt.
  const keep = new Set(kept);
  return placements.filter((p) => keep.has(p));
}

// ---- Descriptions for Jev -------------------------------------------------
// Jev reads text, not numbers. Every feature becomes a short named bucket so
// the model compares situations rather than doing arithmetic.

export function describeLines(n) {
  return ["none", "one line", "two lines", "three lines", "four lines (a Tetris)"][n] ?? `${n} lines`;
}

export function describeHoles(n) {
  if (n === 0) return "none";
  if (n === 1) return "one hole";
  if (n === 2) return "two holes";
  return "three or more holes";
}

export function describeHeight(maxHeight) {
  if (maxHeight <= 4) return "very low";
  if (maxHeight <= 8) return "low";
  if (maxHeight <= 12) return "medium";
  if (maxHeight <= 15) return "high";
  return "dangerously high, close to the top";
}

export function describeSurface(bump) {
  if (bump <= 4) return "flat";
  if (bump <= 9) return "slightly uneven";
  if (bump <= 16) return "bumpy";
  return "very jagged";
}

export function describeWells(list) {
  if (list.length === 0) return "no deep wells";
  if (list.length === 1) return `one deep well at column ${list[0].column + 1}`;
  return `${list.length} deep wells`;
}

function describeWhere(placement) {
  const xs = placement.cells.map((c) => c[0]);
  const left = Math.min(...xs) + 1;
  const right = Math.max(...xs) + 1;
  return left === right ? `column ${left}` : `columns ${left}-${right}`;
}

function describeHeightChange(placement) {
  const d = placement.heightDelta;
  if (placement.linesCleared > 0 && d < 0) return "stack gets lower";
  if (d <= 0) return "stack does not get taller";
  if (d === 1) return "stack grows by one row";
  return "stack grows by several rows";
}

// The Choice criteria entry for one placement. Same field names on every
// option so Jev can compare them directly.
export function describePlacement(p) {
  const holes = p.holesCreated;
  return {
    where: describeWhere(p),
    lines_cleared: describeLines(p.linesCleared),
    holes_created: describeHoles(holes),
    holes_uncovered: p.holesRemoved > 0 ? describeHoles(p.holesRemoved) : "none",
    stack_height_after: describeHeight(p.after.maxHeight),
    height_change: describeHeightChange(p),
    surface_after: describeSurface(p.after.bumpiness),
    wells_after: describeWells(p.after.wells),
  };
}

export function boardToText(board) {
  return board.map((row) => row.map((c) => (c ? "#" : ".")).join(""));
}

// ---- Garbage (versus mode) --------------------------------------------------------
// Pushes `count` garbage rows in from the bottom, each full except one gap.
// Returns { board, overflow }: overflow is true when the shift pushed filled
// cells out of the top, which ends the receiving player's game.
export const GARBAGE = "G";

export function addGarbage(board, count, gapColumn) {
  if (count <= 0) return { board, overflow: false };
  const n = Math.min(count, HEIGHT);
  let overflow = false;
  for (let y = 0; y < n; y++) if (board[y].some((c) => c)) overflow = true;
  const rows = board.slice(n).map((r) => r.slice());
  for (let i = 0; i < n; i++) {
    const row = new Array(WIDTH).fill(GARBAGE);
    row[gapColumn] = null;
    rows.push(row);
  }
  return { board: rows, overflow };
}

// ---- Random bag ------------------------------------------------------------

// Small seeded PRNG (mulberry32) so two players can share one piece sequence.
export function seededRandom(seed) {
  let a = seed >>> 0;
  return function random() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeBag(random = Math.random) {
  const bag = PIECE_NAMES.slice();
  for (let i = bag.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [bag[i], bag[j]] = [bag[j], bag[i]];
  }
  return bag;
}

export function scoreForLines(lines, level) {
  return [0, 100, 300, 500, 800][lines] * level;
}
