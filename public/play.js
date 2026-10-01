// Human vs Jev. The human side is keyboard-driven (move, rotate, soft and hard
// drop, with a short lock delay); Jev's side is the same real-time model loop
// the battle uses. Both share the piece sequence, the gravity ramp, and the
// versus garbage rules from arena.js.

import { PIECES, collides, dropY } from "./tetris.js";
import { createJevPlayer } from "./players.js";
import {
  SPEEDUPS,
  SPAWN_X,
  PRESENT,
  sleep,
  formatClock,
  fmtUsd,
  describeSpeedup,
  createRamp,
  makeSide,
  resetSide,
  previewSide,
  runModelSide,
  applyGarbage,
  settlePiece,
  markTopOut,
  drawSide,
  renderSideStats,
  comparisonTable,
  avgLatency,
} from "./arena.js";

const $ = (id) => document.getElementById(id);
const LOCK_DELAY_MS = 450;
const MAX_LOCK_RESETS = 8;
const KICKS = [0, -1, 1, -2, 2];

const ui = {
  gravity: $("gravity"),
  gravityLabel: $("gravityLabel"),
  speedup: $("speedup"),
  level: $("level"),
  seed: $("seed"),
  limit: $("limit"),
  garbage: $("garbage"),
  jevGravity: $("jevGravity"),
  start: $("start"),
  stop: $("stop"),
  clock: $("clock"),
  result: $("result"),
  error: $("error"),
  matchInfo: $("matchInfo"),
  touch: $("touch"),
};

function sideEls(id) {
  return {
    canvas: $(`board${id}`),
    overlay: $(`overlay${id}`),
    incomingEl: $(`incoming${id}`),
    statsEl: $(`stats${id}`),
    moveEl: id === "R" ? $("moveR") : null,
    section: $(id === "L" ? "left" : "right"),
  };
}

const HUMAN = { name: "You", short: "You" };

function humanStatsRows(side) {
  const s = side.stats;
  const minutes = side.startedAt ? Math.max(1 / 60, (performance.now() - side.startedAt) / 60_000) : 0;
  const ppm = minutes ? (side.pieces / minutes).toFixed(1) : "–";
  return PRESENT
    ? [
        ["Lines", side.lines],
        ["Pieces", side.pieces],
        ["Sent", s.sent],
        ["Received", s.received],
        ["Score", side.score],
        ["Pieces/min", ppm],
        ["Hard drops", s.hardDrops || 0],
        ["Tetrises", s.clears[4]],
      ]
    : [
        ["Lines", side.lines],
        ["Pieces", side.pieces],
        ["Score", side.score],
        ["Sent", s.sent],
        ["Received", s.received],
        ["Pieces/min", ppm],
      ];
}

const comparisonRows = [
  ["Lines", (s) => s.lines],
  ["Pieces", (s) => s.pieces],
  ["Lines / piece", (s) => (s.pieces ? (s.lines / s.pieces).toFixed(2) : "–")],
  ["Tetrises", (s) => s.stats.clears[4]],
  ["Garbage sent", (s) => s.stats.sent],
  ["Garbage received", (s) => s.stats.received],
  ["Avg latency", (s) => (s.player === HUMAN ? "–" : avgLatency(s))],
  ["Missed deadlines", (s) => (s.player === HUMAN ? "–" : s.stats.missed)],
  ["Tokens in", (s) => (s.player === HUMAN ? "–" : s.stats.inputTokens.toLocaleString())],
  ["Cost", (s) => (s.player === HUMAN ? "–" : fmtUsd(s.stats.cost))],
];

let sides = [];
let match = null; // { ramp, abort, timer, limitMs, garbage }
let human = null; // the human side while a match runs

function gravityNow() {
  return match ? match.ramp.gravityNow() : Number(ui.gravity.value);
}

function renderLevel() {
  const level = match ? match.ramp.level() : 1;
  ui.level.textContent = `Level ${level} · ${gravityNow()} ms per row`;
  ui.level.classList.toggle("hot", level >= 4);
}

function showError(message) {
  ui.error.textContent = message;
  ui.error.classList.remove("hidden");
}

function hideError() {
  ui.error.classList.add("hidden");
}

function topOut(side) {
  markTopOut(side, match ? match.ramp.elapsed() : 0);
  checkEnd();
}

// ---- Human side --------------------------------------------------------------------------------

// Controls for the piece currently in play; null between pieces.
let control = null;

function tryMove(side, dx, dy) {
  const a = side.active;
  const cells = PIECES[a.piece][a.rotation].cells;
  if (collides(side.board, cells, a.x + dx, a.y + dy)) return false;
  a.x += dx;
  a.y += dy;
  return true;
}

function tryRotate(side, dir) {
  const a = side.active;
  const states = PIECES[a.piece].length;
  const rotation = (a.rotation + dir + states) % states;
  const cells = PIECES[a.piece][rotation].cells;
  for (const kick of KICKS) {
    if (!collides(side.board, cells, a.x + kick, a.y)) {
      a.rotation = rotation;
      a.x += kick;
      return true;
    }
  }
  return false;
}

function resting(side) {
  const a = side.active;
  return collides(side.board, PIECES[a.piece][a.rotation].cells, a.x, a.y + 1);
}

function humanAction(act) {
  if (!control || !human?.active) return;
  const side = human;
  let moved = false;
  switch (act) {
    case "left":
      moved = tryMove(side, -1, 0);
      break;
    case "right":
      moved = tryMove(side, 1, 0);
      break;
    case "rotateCw":
      moved = tryRotate(side, 1);
      break;
    case "rotateCcw":
      moved = tryRotate(side, -1);
      break;
    case "soft":
      if (tryMove(side, 0, 1)) {
        side.score += 1;
        control.restartGravity();
      }
      break;
    case "hard": {
      const a = side.active;
      const y = dropY(side.board, PIECES[a.piece][a.rotation].cells, a.x, a.y);
      side.score += 2 * (y - a.y);
      side.stats.hardDrops = (side.stats.hardDrops || 0) + 1;
      a.y = y;
      control.lockNow();
      return;
    }
    default:
      return;
  }
  drawSide(side);
  if (moved && resting(side)) control.touchLock();
}

const KEYMAP = {
  ArrowLeft: "left",
  ArrowRight: "right",
  ArrowUp: "rotateCw",
  x: "rotateCw",
  X: "rotateCw",
  z: "rotateCcw",
  Z: "rotateCcw",
  ArrowDown: "soft",
  " ": "hard",
};

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  const act = KEYMAP[e.key];
  if (!act) return;
  e.preventDefault();
  if (act === "hard" && e.repeat) return;
  humanAction(act);
});

ui.touch.addEventListener("pointerdown", (e) => {
  const btn = e.target.closest("button[data-act]");
  if (!btn) return;
  e.preventDefault();
  humanAction(btn.dataset.act);
});

// One piece: gravity ticks, a lock delay when resting, and the controls above.
function playPiece(side, signal) {
  return new Promise((resolve) => {
    let gravityTimer = null;
    let lockTimer = null;
    let resets = 0;
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(gravityTimer);
      clearTimeout(lockTimer);
      control = null;
      const a = side.active;
      resolve({ rotation: a.rotation, x: a.x, y: a.y });
    };
    const armLock = () => {
      clearTimeout(lockTimer);
      lockTimer = setTimeout(finish, LOCK_DELAY_MS);
    };
    const tick = () => {
      if (done || signal.aborted) return;
      if (tryMove(side, 0, 1)) {
        drawSide(side);
        clearTimeout(lockTimer);
        lockTimer = null;
        gravityTimer = setTimeout(tick, gravityNow());
      } else {
        if (!lockTimer) armLock();
        gravityTimer = setTimeout(tick, gravityNow());
      }
    };
    control = {
      lockNow: finish,
      restartGravity: () => {
        clearTimeout(gravityTimer);
        gravityTimer = setTimeout(tick, gravityNow());
      },
      // A successful move or rotation while resting restarts the lock delay, a few times.
      touchLock: () => {
        if (resets < MAX_LOCK_RESETS) {
          resets += 1;
          armLock();
        }
      },
    };
    signal.addEventListener("abort", finish, { once: true });
    gravityTimer = setTimeout(tick, gravityNow());
  });
}

async function runHumanSide(side, ctx) {
  const { signal } = ctx;
  side.showGhost = true;
  side.startedAt = performance.now();
  while (!side.over && !signal.aborted) {
    if (applyGarbage(side)) {
      ctx.onTopOut(side);
      return;
    }
    const piece = side.current;
    if (collides(side.board, PIECES[piece][0].cells, SPAWN_X, 0)) {
      ctx.onTopOut(side);
      return;
    }
    side.active = { piece, rotation: 0, x: SPAWN_X, y: 0 };
    drawSide(side);
    const landed = await playPiece(side, signal);
    if (signal.aborted) return;
    await settlePiece(side, landed, ctx);
  }
}

// ---- Match control -------------------------------------------------------------------------------

function checkEnd(timeUp = false) {
  if (!match) return;
  const [L, R] = sides;
  const bothOver = L.over && R.over;
  const oneOver = L.over || R.over;
  if (!oneOver && !timeUp) return;
  if (oneOver && !bothOver && !timeUp) {
    const loser = L.over ? L : R;
    const survivor = loser === L ? R : L;
    if (match.garbage) {
      finish(survivor, loser, survivor === L ? "You win: Jev topped out first" : "Jev wins: you topped out first");
      return;
    }
    if (survivor.pieces > loser.pieces) {
      finish(survivor, loser, survivor === L ? `You win: survived past ${loser.pieces} pieces` : `Jev wins: survived past ${loser.pieces} pieces`);
    } else {
      loser.overlay.textContent = `Topped out at ${formatClock(loser.lostAt)} after ${loser.pieces} pieces · ${survivor.player.name} must pass ${loser.pieces}`;
    }
    return;
  }
  let winner = null;
  let loser = null;
  let reason;
  if (bothOver && L.pieces !== R.pieces) {
    winner = L.pieces > R.pieces ? L : R;
    loser = winner === L ? R : L;
    reason = winner === L ? "You win: lasted more pieces" : "Jev wins: lasted more pieces";
  } else if (L.lines !== R.lines) {
    winner = L.lines > R.lines ? L : R;
    loser = winner === L ? R : L;
    reason = `${winner === L ? "You win" : "Jev wins"} on lines${timeUp ? " at the time limit" : ""}`;
  } else {
    reason = timeUp ? "Draw at the time limit" : "Draw";
  }
  finish(winner, loser, reason);
}

function finish(winner, loser, reason) {
  const [L, R] = sides;
  match.abort.abort();
  clearInterval(match.timer);
  control = null;
  const elapsed = match.ramp.elapsed();
  if (winner) {
    winner.section.classList.add("winner");
    loser?.section.classList.add("loser");
  }
  const note = `gravity ${match.ramp.gravityMs} → ${gravityNow()} ms/row, level ${match.ramp.level()}${match.garbage ? ", versus" : ""}`;
  ui.result.innerHTML = `${reason}<small>${formatClock(elapsed)} elapsed · seed ${ui.seed.value} · ${note}</small>${comparisonTable(L, R, comparisonRows)}`;
  ui.result.classList.remove("hidden");
  ui.start.disabled = false;
  ui.stop.disabled = true;
  document.body.classList.remove("running");
  document.body.classList.add("played");
  match = null;
  human = null;
}

async function startMatch() {
  hideError();
  const seed = Number(ui.seed.value) || 42;
  const gravityMs = Number(ui.gravity.value);
  const speedup = SPEEDUPS[ui.speedup.value] || SPEEDUPS.gentle;
  const garbage = ui.garbage.checked;
  const jevGravity = ui.jevGravity.checked;
  const limitMs = Math.max(1, Number(ui.limit.value) || 5) * 60_000;
  sides = [makeSide("L", HUMAN, sideEls("L"), humanStatsRows), makeSide("R", createJevPlayer(), sideEls("R"))];
  sides[0].opponent = sides[1];
  sides[1].opponent = sides[0];
  for (const s of sides) resetSide(s, seed);
  human = sides[0];
  ui.result.classList.add("hidden");
  ui.start.disabled = true;
  ui.stop.disabled = false;
  ui.matchInfo.textContent = `seed ${seed} · ${gravityMs} ms per row, ${describeSpeedup(ui.speedup.value)}${garbage ? " · cleared lines attack" : ""}`;
  document.body.classList.add("running");
  for (const n of [3, 2, 1]) {
    ui.clock.textContent = String(n);
    await sleep(700);
  }
  const abort = new AbortController();
  match = { ramp: createRamp({ gravityMs, speedup }), abort, timer: null, limitMs, garbage };
  renderLevel();
  match.timer = setInterval(() => {
    if (!match) return;
    const elapsed = match.ramp.elapsed();
    ui.clock.textContent = formatClock(elapsed);
    renderLevel();
    renderSideStats(sides[0]);
    if (elapsed >= match.limitMs) checkEnd(true);
  }, 250);
  const ctx = { signal: abort.signal, gravityNow, garbage, onTopOut: topOut, onPiece: () => checkEnd(), showError };
  runHumanSide(sides[0], ctx);
  runModelSide(sides[1], { ...ctx, lockstep: !jevGravity });
  ui.start.blur();
}

function giveUp() {
  if (!match) return;
  topOut(sides[0]);
}

// ---- Wiring ----------------------------------------------------------------------------------------

ui.start.addEventListener("click", startMatch);
ui.stop.addEventListener("click", giveUp);
ui.gravity.addEventListener("input", () => {
  ui.gravityLabel.textContent = `${ui.gravity.value} ms per row`;
  if (!match) renderLevel();
});
ui.gravityLabel.textContent = `${ui.gravity.value} ms per row`;
renderLevel();

sides = [makeSide("L", HUMAN, sideEls("L"), humanStatsRows), makeSide("R", { name: "Jev", short: "Jev" }, sideEls("R"))];
for (const s of sides) previewSide(s, Number(ui.seed.value) || 42);
