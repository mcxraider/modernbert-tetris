import { test } from "node:test";
import assert from "node:assert/strict";
import {
  HEIGHT,
  WIDTH,
  PIECES,
  PIECE_NAMES,
  emptyBoard,
  enumeratePlacements,
  clearLines,
  countHoles,
  columnHeights,
  describePlacement,
  makeBag,
  shortlistPlacements,
  dellacherieFeatures,
  heuristicScore,
  lookaheadScore,
  FEATURE_WEIGHTS,
} from "../public/tetris.js";
import { buildRequest, pickPlacement, buildQuestions } from "../public/jev.js";

function boardFrom(rows) {
  // rows: bottom-most last, strings of '#' and '.', padded to HEIGHT at the top
  const board = emptyBoard();
  const offset = HEIGHT - rows.length;
  rows.forEach((row, i) => {
    for (let x = 0; x < WIDTH; x++) board[offset + i][x] = row[x] === "#" ? "X" : null;
  });
  return board;
}

test("every piece has 4 cells in every rotation", () => {
  for (const name of PIECE_NAMES) {
    for (const state of PIECES[name]) assert.equal(state.cells.length, 4, name);
  }
});

test("enumeratePlacements on an empty board covers every column", () => {
  const board = emptyBoard();
  const o = enumeratePlacements(board, "O");
  assert.equal(o.length, WIDTH - 1); // 9 distinct positions for a 2-wide block
  const i = enumeratePlacements(board, "I");
  assert.equal(i.length, 7 + 10); // 7 horizontal + 10 vertical
  const t = enumeratePlacements(board, "T");
  assert.equal(t.length, 2 * 8 + 2 * 9); // two 3-wide and two 2-wide rotations
  for (const p of [...o, ...i]) assert.equal(p.holesCreated, 0);
  // A T pointing down rests on its stem and covers two cells: real holes.
  const stemDown = t.find((p) => p.rotation === 2 && p.x === 0);
  assert.equal(stemDown.holesCreated, 2);
  for (const p of [...o, ...i, ...t]) {
    assert.equal(p.linesCleared, 0);
    for (const [x, y] of p.cells) {
      assert.ok(x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT);
    }
  }
});

test("an I piece dropped into a well clears four lines", () => {
  const rows = ["#########.", "#########.", "#########.", "#########."];
  const board = boardFrom(rows);
  const placements = enumeratePlacements(board, "I");
  const tetris = placements.find((p) => p.linesCleared === 4);
  assert.ok(tetris, "expected a four-line clear");
  assert.equal(tetris.x, 9);
  assert.equal(tetris.after.maxHeight, 0);
  assert.equal(describePlacement(tetris).lines_cleared, "four lines (a Tetris)");
  assert.equal(describePlacement(tetris).holes_created, "none");
});

test("holes are counted and reported when a placement covers a gap", () => {
  const board = boardFrom(["##..######"]);
  const placements = enumeratePlacements(board, "O");
  const covering = placements.find((p) => p.x === 1);
  assert.ok(covering);
  assert.equal(covering.holesCreated, 1); // O at x=1 rests on column 1, column 2 stays empty underneath
  assert.equal(describePlacement(covering).holes_created, "one hole");
  const clean = placements.find((p) => p.x === 2);
  assert.equal(clean.holesCreated, 0);
  assert.equal(clean.linesCleared, 1);
});

test("clearLines removes full rows and keeps the board height", () => {
  const board = boardFrom(["##########", "#....#####", "##########"]);
  const { board: after, cleared, rows } = clearLines(board);
  assert.equal(cleared, 2);
  assert.deepEqual(rows, [17, 19]);
  assert.equal(after.length, HEIGHT);
  assert.equal(countHoles(after), 0);
  assert.deepEqual(columnHeights(after), [1, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
});

test("no placements when the board is full to the top", () => {
  const rows = Array.from({ length: HEIGHT }, () => "#########.");
  const board = boardFrom(rows);
  // Only the vertical I in the last column fits; every other piece is stuck.
  assert.equal(enumeratePlacements(board, "O").length, 0);
  assert.equal(enumeratePlacements(board, "I").length, 1);
});

test("makeBag returns every piece once", () => {
  const bag = makeBag(() => 0.42);
  assert.deepEqual([...bag].sort(), [...PIECE_NAMES].sort());
});

test("buildRequest produces one criteria entry per placement with stable fields", () => {
  const board = emptyBoard();
  const placements = enumeratePlacements(board, "L");
  const request = buildRequest(
    { board, piece: "L", nextPiece: "I", stats: placements[0].before, linesCleared: 0 },
    placements,
  );
  assert.equal(request.model, "jev-latest");
  assert.equal(Object.keys(request.questions.placement.criteria).length, placements.length);
  const fields = Object.keys(request.questions.placement.criteria.p0);
  for (const id of Object.keys(request.questions.placement.criteria)) {
    assert.deepEqual(Object.keys(request.questions.placement.criteria[id]), fields);
  }
  assert.equal(request.state.game.board_rows_top_to_bottom.length, HEIGHT);
  assert.equal(request.questions.strategy.type, "choice");
  assert.equal(request.questions.board_health.type, "score");
  assert.equal(request.questions.next_piece_fits.type, "noul");
  // Request should stay far below the 32k token state budget.
  assert.ok(JSON.stringify(request).length < 20_000);
});

test("shortlistPlacements caps the count, keeps members and clears, drops dominated moves", () => {
  // A well in the last column with a near-full stack: an I piece there clears
  // lines, and there are far more than 12 legal placements overall.
  const rows = ["#########.", "#########.", "#########.", "#########."];
  const board = boardFrom(rows);
  const all = enumeratePlacements(board, "I");
  assert.ok(all.length > 12, "need more than k placements to exercise the cap");

  const short = shortlistPlacements(all, 12);
  assert.ok(short.length <= 12, "capped to k");
  // Every survivor is an original placement object (ids preserved for pickPlacement).
  for (const p of short) assert.ok(all.includes(p));
  // Order follows enumeration order among survivors.
  const idx = short.map((p) => all.indexOf(p));
  assert.deepEqual(idx, [...idx].sort((a, b) => a - b));
  // The four-line clear is retained even though it may not top the heuristic sort.
  assert.ok(short.some((p) => p.linesCleared === 4), "line-clear kept");

  // Below the cap: returns the input untouched.
  const few = all.slice(0, 5);
  assert.equal(shortlistPlacements(few, 12), few);

  // Dominance: a strictly-worse placement is dropped. Two placements where one
  // is taller, has more holes, is bumpier and clears no more lines than another.
  const a = { id: "a", linesCleared: 0, heuristic: 0, after: { maxHeight: 5, holes: 2, bumpiness: 8 } };
  const b = { id: "b", linesCleared: 0, heuristic: 1, after: { maxHeight: 3, holes: 0, bumpiness: 2 } };
  const filler = Array.from({ length: 12 }, (_, i) => ({
    id: `f${i}`,
    linesCleared: 0,
    heuristic: -i,
    after: { maxHeight: 10 + i, holes: 5 + i, bumpiness: 20 + i },
  }));
  const out = shortlistPlacements([a, b, ...filler], 12);
  assert.ok(!out.includes(a), "dominated placement dropped");
  assert.ok(out.includes(b), "dominating placement kept");
});

test("pickPlacement maps the answer back and ranks alternatives", () => {
  const placements = enumeratePlacements(emptyBoard(), "T");
  const response = {
    answers: {
      placement: {
        type: "choice",
        choice: "p3",
        confidence: 0.4,
        probabilities: Object.fromEntries(placements.map((p) => [p.id, p.id === "p3" ? 0.5 : p.id === "p1" ? 0.3 : 0.01])),
      },
    },
  };
  const { chosen, ranked, confidence } = pickPlacement(response, placements);
  assert.equal(chosen.id, "p3");
  assert.equal(ranked[0].placement.id, "p3");
  assert.equal(ranked[1].placement.id, "p1");
  assert.equal(confidence, 0.4);
  assert.equal(Object.keys(buildQuestions(placements)).length, 4);
});

test("seeded bags give two players the same piece sequence", async () => {
  const { seededRandom } = await import("../public/tetris.js");
  const a = seededRandom(42);
  const b = seededRandom(42);
  const seqA = [...makeBag(a), ...makeBag(a), ...makeBag(a)];
  const seqB = [...makeBag(b), ...makeBag(b), ...makeBag(b)];
  assert.deepEqual(seqA, seqB);
  const c = seededRandom(43);
  assert.notDeepEqual([...makeBag(c)], seqA.slice(0, 7));
});

test("Platform prompt lists every option with the current piece and legend", async () => {
  const { buildPlatformPrompt } = await import("../public/players.js");
  const board = emptyBoard();
  const placements = enumeratePlacements(board, "S");
  const prompt = JSON.parse(buildPlatformPrompt({ board, piece: "S", nextPiece: "O", stats: placements[0].before, linesCleared: 0 }, placements));
  assert.equal(Object.keys(prompt.options).length, placements.length);
  assert.equal(prompt.current_piece, "S");
  assert.equal(prompt.next_piece, "O");
});

test("garbage rows push the stack up and report overflow at the top", async () => {
  const { addGarbage, GARBAGE } = await import("../public/tetris.js");
  const board = emptyBoard();
  board[HEIGHT - 1][0] = "X";
  const { board: after, overflow } = addGarbage(board, 2, 4);
  assert.equal(overflow, false);
  assert.equal(after.length, HEIGHT);
  assert.equal(after[HEIGHT - 3][0], "X"); // old bottom row moved up two
  assert.deepEqual(after[HEIGHT - 1].map((c) => (c ? 1 : 0)), [1, 1, 1, 1, 0, 1, 1, 1, 1, 1]);
  assert.equal(after[HEIGHT - 1][0], GARBAGE);
  const tall = emptyBoard();
  tall[1][3] = "X";
  assert.equal(addGarbage(tall, 2, 0).overflow, true);
  assert.equal(addGarbage(tall, 1, 0).overflow, false);
});

// Builds the minimal outcome shape dellacherieFeatures/heuristicScore read.
function makeOutcome(afterRows, { cells = [], clearedRows = [], linesCleared = 0 } = {}) {
  const afterBoard = boardFrom(afterRows);
  return { afterBoard, cells, clearedRows, linesCleared, after: { heights: columnHeights(afterBoard) } };
}

test("holeDepth distinguishes a buried hole from a surface hole", () => {
  // Both boards have exactly one hole; the DT holeDepth is what tells them apart
  // (the weakness the 4-feature heuristic couldn't see).
  const surface = makeOutcome(["#.........", ".........."]); // 1 filled cell over 1 empty
  const buried = makeOutcome(["#.........", "#.........", "#.........", ".........."]);
  const fs = dellacherieFeatures(surface);
  const fb = dellacherieFeatures(buried);
  assert.equal(fs.holes, 1);
  assert.equal(fb.holes, 1);
  assert.equal(fs.holeDepth, 1);
  assert.equal(fb.holeDepth, 3); // three filled cells stacked above the hole
  assert.equal(fs.rowsWithHoles, 1);
  assert.equal(fb.rowsWithHoles, 1);
});

test("erodedCells = linesCleared x the placed piece's own cells in cleared rows", () => {
  // The O piece's two bottom cells (rows 19) sat in the single row this move
  // cleared; its two top cells (row 18) did not.
  const o = makeOutcome(["##########"], {
    cells: [[0, 18], [1, 18], [0, 19], [1, 19]],
    clearedRows: [19],
    linesCleared: 1,
  });
  assert.equal(dellacherieFeatures(o).erodedCells, 1 * 2);
});

test("wellSum is the triangular (cumulative) depth of a well", () => {
  // Column 0 empty for 4 rows against the left wall and a filled column 1:
  // a single well of depth 4 → 1+2+3+4 = 10.
  const o = makeOutcome([".#........", ".#........", ".#........", ".#........"]);
  assert.equal(dellacherieFeatures(o).wellSum, 10);
});

test("heuristicScore is the dot product of the weights and the features", () => {
  const o = makeOutcome(["#.........", "#.........", "..........", ".#........", ".#........"]);
  const f = dellacherieFeatures(o);
  const expected = Object.keys(FEATURE_WEIGHTS).reduce((s, k) => s + FEATURE_WEIGHTS[k] * f[k], 0);
  assert.equal(heuristicScore(o), expected);
});

test("lookaheadScore adds the best next-piece reply, falls back at 1-ply, penalises top-out", () => {
  const board = emptyBoard();
  const o = { afterBoard: board, heuristic: 5 };
  // No next piece → pure 1-ply.
  assert.equal(lookaheadScore(o, null), 5);
  // With a next piece → own score plus the best reply's heuristic on afterBoard.
  const replies = enumeratePlacements(board, "O");
  const best = Math.max(...replies.map((r) => r.heuristic));
  assert.equal(lookaheadScore(o, "O"), 5 + best);
  // A completely full board tops out any next piece → heavy penalty.
  const full = emptyBoard().map((row) => row.map(() => "X"));
  assert.ok(lookaheadScore({ afterBoard: full, heuristic: 5 }, "I") <= -1e6);
});

test("lookahead prefers the continuation the next piece can clear", () => {
  // Row 19 is one cell short of a line at column 9; an afterstate that keeps it
  // open lets an I (vertical) or O finish it next ply. Compare that against the
  // same board with column 9 already covered by a hole (no clear available).
  const rows = Array.from({ length: 10 }, (_, i) => (i < 9 ? "#" : ".")).join("");
  const openBoard = boardFrom([rows]); // "#########." — one gap at col 9
  const open = { afterBoard: openBoard, heuristic: 0 };
  const sealedBoard = boardFrom(["#########.", "##########"]); // gap now buried
  const sealed = { afterBoard: sealedBoard, heuristic: 0 };
  assert.ok(lookaheadScore(open, "I") > lookaheadScore(sealed, "I"));
});

test("Platform tool schema enumerates the options and the parser reads the OpenAI tool call", async () => {
  const { buildPlatformTool, parsePlatformChoice } = await import("../public/players.js");
  const placements = enumeratePlacements(emptyBoard(), "J");
  const fn = buildPlatformTool(placements).function;
  assert.equal(fn.name, "place_piece");
  assert.deepEqual(fn.parameters.properties.option_id.enum, placements.map((p) => p.id));
  const call = (id) => ({ choices: [{ message: { content: "", tool_calls: [{ function: { name: "place_piece", arguments: JSON.stringify({ option_id: id }) } }] } }] });
  const say = (text) => ({ choices: [{ message: { content: text } }] });
  assert.equal(parsePlatformChoice(call("p2"), placements).id, "p2");
  assert.equal(parsePlatformChoice(say("I'd go with p4."), placements).id, "p4");
  assert.equal(parsePlatformChoice(call("p999"), placements), null);
  assert.equal(parsePlatformChoice({}, placements), null);
});
