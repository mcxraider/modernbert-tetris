// Shared arena machinery for the two-board pages (battle.js: model vs model,
// play.js: human vs model): per-side state, drawing, stats, garbage, the
// gravity ramp, and the real-time loop for a model-driven side.

import {
  WIDTH,
  HEIGHT,
  PIECES,
  PIECE_COLORS,
  emptyBoard,
  enumeratePlacements,
  lockPiece,
  clearLines,
  boardStats,
  collides,
  dropY,
  makeBag,
  seededRandom,
  addGarbage,
  GARBAGE,
  scoreForLines,
  describePlacement,
} from "./tetris.js";

export const SPEEDUPS = {
  none: { everyMs: Infinity, factor: 1 },
  gentle: { everyMs: 30_000, factor: 0.9 },
  normal: { everyMs: 20_000, factor: 0.85 },
  brutal: { everyMs: 10_000, factor: 0.8 },
};
export const MIN_GRAVITY_MS = 40;
export const SPAWN_X = 3;

// ?present strips a page down to the boards and the clock for recordings.
export const PRESENT = new URLSearchParams(location.search).has("present");

if (PRESENT) document.body.classList.add("present");

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function fmtMs(ms) {
  return `${Math.round(ms)} ms`;
}

export function fmtUsd(v, digits = 4) {
  return `$${v.toFixed(digits)}`;
}

export function formatClock(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export function describeSpeedup(name) {
  const cfg = SPEEDUPS[name] || SPEEDUPS.normal;
  if (!Number.isFinite(cfg.everyMs)) return "constant gravity";
  return `${Math.round((1 - cfg.factor) * 100)}% faster every ${cfg.everyMs / 1000} s`;
}

// ---- Gravity ramp: one shared clock, level up every `everyMs` --------------------------

export function createRamp({ gravityMs, speedup }) {
  const startedAt = performance.now();
  const level = () => (Number.isFinite(speedup.everyMs) ? Math.floor((performance.now() - startedAt) / speedup.everyMs) + 1 : 1);
  return {
    startedAt,
    gravityMs,
    speedup,
    level,
    elapsed: () => performance.now() - startedAt,
    gravityNow: () => Math.max(MIN_GRAVITY_MS, Math.round(gravityMs * Math.pow(speedup.factor, level() - 1))),
  };
}

// ---- Per-side state ---------------------------------------------------------------------

export function freshStats() {
  return {
    calls: 0,
    latency: 0,
    minLatency: Infinity,
    maxLatency: 0,
    missed: 0,
    invalid: 0,
    errors: 0,
    inputTokens: 0,
    outputTokens: 0,
    cost: 0,
    sent: 0,
    received: 0,
    clears: [0, 0, 0, 0, 0],
  };
}

export function recordDecision(side, decision) {
  const s = side.stats;
  s.calls += 1;
  s.latency += decision.latencyMs;
  s.minLatency = Math.min(s.minLatency, decision.latencyMs);
  s.maxLatency = Math.max(s.maxLatency, decision.latencyMs);
  s.inputTokens += decision.inputTokens;
  s.outputTokens += decision.outputTokens;
  s.cost += decision.cost;
}

// `els`: { canvas, overlay, incomingEl, statsEl, moveEl, section }.
// `statsRows(side)` returns [[label, value], ...] for the stats grid.
export function makeSide(id, player, els, statsRows = modelStatsRows) {
  return {
    id,
    player,
    ...els,
    ctx: els.canvas.getContext("2d"),
    cell: els.canvas.width / WIDTH,
    statsRows,
    board: emptyBoard(),
    random: null,
    garbageRandom: null,
    pendingGarbage: 0,
    opponent: null,
    bag: [],
    current: null,
    next: null,
    active: null, // { piece, rotation, x, y }
    target: null, // placement cells to outline once decided
    flash: [],
    lines: 0,
    pieces: 0,
    score: 0,
    over: false,
    lostAt: null,
    startedAt: null,
    stats: freshStats(),
  };
}

export function nextPiece(side) {
  if (side.bag.length === 0) side.bag = makeBag(side.random);
  return side.bag.pop();
}

export function resetSide(side, seed) {
  side.board = emptyBoard();
  side.random = seededRandom(seed);
  side.garbageRandom = seededRandom(seed * 7919 + (side.id === "L" ? 1 : 2));
  side.pendingGarbage = 0;
  side.bag = [];
  side.current = nextPiece(side);
  side.next = nextPiece(side);
  side.active = null;
  side.target = null;
  side.flash = [];
  side.lines = 0;
  side.pieces = 0;
  side.score = 0;
  side.over = false;
  side.lostAt = null;
  side.startedAt = null;
  side.stats = freshStats();
  side.overlay.classList.add("hidden");
  side.section.classList.remove("winner", "loser", "hit");
  side.incomingEl.textContent = "";
  if (side.moveEl) side.moveEl.textContent = "Ready.";
  drawSide(side);
  renderSideStats(side);
}

// Idle boards before a match: a seeded preview so the page is not blank.
export function previewSide(side, seed) {
  side.random = seededRandom(seed);
  side.current = nextPiece(side);
  side.next = nextPiece(side);
  drawSide(side);
  renderSideStats(side);
}

// ---- Rendering ------------------------------------------------------------------------------

export function drawSide(side) {
  const { ctx, cell, canvas } = side;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  ctx.lineWidth = 1;
  for (let x = 1; x < WIDTH; x++) {
    ctx.beginPath();
    ctx.moveTo(x * cell, 0);
    ctx.lineTo(x * cell, canvas.height);
    ctx.stroke();
  }
  for (let y = 1; y < HEIGHT; y++) {
    ctx.beginPath();
    ctx.moveTo(0, y * cell);
    ctx.lineTo(canvas.width, y * cell);
    ctx.stroke();
  }
  const paint = (x, y, color, alpha = 1, outline = false) => {
    ctx.globalAlpha = alpha;
    if (outline) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x * cell + 2, y * cell + 2, cell - 4, cell - 4);
    } else {
      ctx.fillStyle = color;
      ctx.fillRect(x * cell + 1, y * cell + 1, cell - 2, cell - 2);
      ctx.fillStyle = "rgba(255,255,255,0.18)";
      ctx.fillRect(x * cell + 1, y * cell + 1, cell - 2, 3);
    }
    ctx.globalAlpha = 1;
  };
  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      const c = side.board[y][x];
      if (c) paint(x, y, side.flash.includes(y) ? "#ffffff" : c === GARBAGE ? "#5b6472" : PIECE_COLORS[c]);
    }
  }
  if (side.target) for (const [x, y] of side.target) paint(x, y, "#ffffff", 0.7, true);
  if (side.active) {
    const { piece, rotation, x, y } = side.active;
    if (side.showGhost) {
      const cells = PIECES[piece][rotation].cells;
      const gy = dropY(side.board, cells, x, y);
      for (const [cx, cy] of cells) if (gy + cy >= 0) paint(x + cx, gy + cy, PIECE_COLORS[piece], 0.35, true);
    }
    for (const [cx, cy] of PIECES[piece][rotation].cells) if (y + cy >= 0) paint(x + cx, y + cy, PIECE_COLORS[piece]);
  }
  // next piece preview in the top-right corner
  if (side.next) {
    const state = PIECES[side.next][0];
    const s = cell * 0.5;
    const ox = canvas.width - state.width * s - 6;
    ctx.globalAlpha = 0.85;
    ctx.fillStyle = PIECE_COLORS[side.next];
    for (const [cx, cy] of state.cells) ctx.fillRect(ox + cx * s, 6 + cy * s, s - 1, s - 1);
    ctx.globalAlpha = 1;
  }
}

export function modelStatsRows(side) {
  const s = side.stats;
  const avg = s.calls ? fmtMs(s.latency / s.calls) : "–";
  return PRESENT
    ? [
        ["Lines", side.lines],
        ["Pieces", side.pieces],
        ["Sent", s.sent],
        ["Received", s.received],
        ["Latency", avg],
        ["Missed", s.missed],
        ["In tok", s.inputTokens.toLocaleString()],
        ["Out tok", s.outputTokens.toLocaleString()],
        ["Cost", fmtUsd(s.cost)],
        ["Per move", s.calls ? fmtUsd(s.cost / s.calls, 5) : "–"],
        ["Calls", s.calls],
        ["Invalid", s.invalid + s.errors],
      ]
    : [
        ["Lines", side.lines],
        ["Pieces", side.pieces],
        ["Score", side.score],
        ["Sent", s.sent],
        ["Received", s.received],
        ["Avg latency", avg],
        ["Missed", s.missed],
        ["Invalid", s.invalid + s.errors],
        ["Tokens in", s.inputTokens.toLocaleString()],
        ["Tokens out", s.outputTokens.toLocaleString()],
        ["Cost", fmtUsd(s.cost)],
        ["Per move", s.calls ? fmtUsd(s.cost / s.calls, 5) : "–"],
      ];
}

export function renderSideStats(side) {
  side.statsEl.innerHTML = side
    .statsRows(side)
    .map(([label, value]) => `<div class="stat"><span class="label">${label}</span><span>${value}</span></div>`)
    .join("");
}

// Side-by-side table for the result card. `rows` = [[label, fn(side)], ...].
export function comparisonTable(L, R, rows = modelComparisonRows) {
  const row = (label, f) => `<tr><th>${label}</th><td>${f(L)}</td><td>${f(R)}</td></tr>`;
  return `<table class="compare">
    <colgroup><col class="metric" /><col /><col /></colgroup>
    <thead><tr><th></th><th>${L.player.short || L.player.name}</th><th>${R.player.short || R.player.name}</th></tr></thead>
    <tbody>${rows.map(([label, f]) => row(label, f)).join("")}</tbody>
  </table>`;
}

export const avgLatency = (s) => (s.stats.calls ? fmtMs(s.stats.latency / s.stats.calls) : "–");
export const latencyRange = (s) => (s.stats.calls ? `${Math.round(s.stats.minLatency)}–${Math.round(s.stats.maxLatency)}` : "–");

export const modelComparisonRows = [
  ["Lines", (s) => s.lines],
  ["Pieces", (s) => s.pieces],
  ["Lines / piece", (s) => (s.pieces ? (s.lines / s.pieces).toFixed(2) : "–")],
  ["Garbage sent", (s) => s.stats.sent],
  ["Garbage received", (s) => s.stats.received],
  ["Avg latency", avgLatency],
  ["Min–max ms", latencyRange],
  ["Missed", (s) => s.stats.missed],
  ["Invalid", (s) => s.stats.invalid + s.stats.errors],
  ["Model calls", (s) => s.stats.calls],
  ["Tokens in", (s) => s.stats.inputTokens.toLocaleString()],
  ["Tokens out", (s) => s.stats.outputTokens.toLocaleString()],
  ["Cost", (s) => fmtUsd(s.stats.cost)],
  ["Cost / move", (s) => (s.stats.calls ? fmtUsd(s.stats.cost / s.stats.calls, 5) : "–")],
];

// ---- Garbage -----------------------------------------------------------------------------------

export function renderIncoming(side) {
  side.incomingEl.textContent = side.pendingGarbage > 0 ? `▼ ${side.pendingGarbage} incoming` : "";
}

export function sendGarbage(from, count) {
  const to = from.opponent;
  if (!to || to.over || count <= 0) return;
  to.pendingGarbage += count;
  from.stats.sent += count;
  renderIncoming(to);
}

// Incoming garbage lands when the receiver's piece has locked, before the next spawn.
// Returns true when the push overflowed the top (the receiver is out).
export function applyGarbage(side) {
  const n = side.pendingGarbage;
  if (n <= 0) return false;
  side.pendingGarbage = 0;
  const gap = Math.floor(side.garbageRandom() * WIDTH);
  const { board, overflow } = addGarbage(side.board, n, gap);
  side.board = board;
  side.stats.received += n;
  renderIncoming(side);
  side.section.classList.add("hit");
  setTimeout(() => side.section.classList.remove("hit"), 350);
  drawSide(side);
  renderSideStats(side);
  return overflow;
}

// Locks a piece, clears lines (with a flash), sends garbage, advances the queue.
export async function settlePiece(side, landed, ctx) {
  const piece = side.active.piece;
  side.active = null;
  side.target = null;
  const locked = lockPiece(side.board, piece, landed.rotation, landed.x, landed.y);
  const { board, cleared, rows } = clearLines(locked);
  if (cleared > 0) {
    side.board = locked;
    side.flash = rows;
    drawSide(side);
    await sleep(90);
    side.flash = [];
  }
  side.board = board;
  side.lines += cleared;
  side.stats.clears[cleared] += 1;
  if (ctx.garbage && cleared > 0) sendGarbage(side, cleared);
  side.score += scoreForLines(cleared, Math.floor(side.lines / 10) + 1);
  side.pieces += 1;
  side.current = side.next;
  side.next = nextPiece(side);
  drawSide(side);
  renderSideStats(side);
  ctx.onPiece?.(side);
}

// Decides whether a round is over. Returns null while play continues, else
// { winner, loser, reason } (both null on a draw). In versus mode the first to
// top out loses. With independent boards a fast player cycles through more
// pieces per minute, so wall-clock survival would reward slowness: a lone
// top-out only ends the round once the survivor has passed the loser's piece
// count, and the loser's overlay says what has to be beaten meanwhile.
export function judgeRound(L, R, { garbage, timeUp = false }) {
  const bothOver = L.over && R.over;
  const oneOver = L.over || R.over;
  if (!oneOver && !timeUp) return null;
  if (oneOver && !bothOver && !timeUp) {
    const loser = L.over ? L : R;
    const survivor = loser === L ? R : L;
    if (garbage) return { winner: survivor, loser, reason: `${survivor.player.name} wins: ${loser.player.name} topped out first` };
    if (survivor.pieces > loser.pieces) return { winner: survivor, loser, reason: `${survivor.player.name} wins: survived past ${loser.pieces} pieces` };
    loser.overlay.textContent = `Topped out at ${formatClock(loser.lostAt)} after ${loser.pieces} pieces · ${survivor.player.name} must pass ${loser.pieces}`;
    return null;
  }
  let winner = null;
  let loser = null;
  let reason;
  if (bothOver && L.pieces !== R.pieces) {
    winner = L.pieces > R.pieces ? L : R;
    reason = `${winner.player.name} wins: lasted more pieces`;
  } else if (L.lines !== R.lines) {
    winner = L.lines > R.lines ? L : R;
    reason = `${winner.player.name} wins on lines${timeUp ? " at the time limit" : ""}`;
  } else if (L.stats.missed !== R.stats.missed) {
    winner = L.stats.missed < R.stats.missed ? L : R;
    reason = `${winner.player.name} wins on fewer missed deadlines`;
  } else {
    reason = timeUp ? "Draw at the time limit" : "Draw";
  }
  if (winner) loser = winner === L ? R : L;
  return { winner, loser, reason };
}

export function markTopOut(side, elapsedMs) {
  side.over = true;
  side.lostAt = elapsedMs;
  side.active = null;
  side.target = null;
  drawSide(side);
  side.overlay.textContent = `Topped out at ${formatClock(elapsedMs)}`;
  side.overlay.classList.remove("hidden");
  side.section.classList.add("loser");
  renderSideStats(side);
}

// ---- Real-time loop for a model-driven side ------------------------------------------------------
// ctx: { signal, lockstep, gravityNow(), garbage, onTopOut(side), onPiece(side), showError(msg) }

export async function runModelSide(side, ctx) {
  const { signal, lockstep, gravityNow } = ctx;
  while (!side.over && !signal.aborted) {
    if (applyGarbage(side)) {
      ctx.onTopOut(side);
      return;
    }
    const piece = side.current;
    const spawn = PIECES[piece][0];
    if (collides(side.board, spawn.cells, SPAWN_X, 0)) {
      ctx.onTopOut(side);
      return;
    }
    const placements = enumeratePlacements(side.board, piece);
    if (placements.length === 0) {
      ctx.onTopOut(side);
      return;
    }
    side.active = { piece, rotation: 0, x: SPAWN_X, y: 0 };
    side.target = null;
    drawSide(side);

    // Ask the model right away; the piece falls while we wait.
    const gameInfo = { board: side.board, piece, nextPiece: side.next, stats: boardStats(side.board), linesCleared: side.lines };
    const askedAt = performance.now();
    let decision = null;
    let settled = false;
    let failed = false;
    const pending = side.player
      .decide(gameInfo, placements, signal)
      .then((d) => {
        decision = d;
      })
      .catch((err) => {
        if (!signal.aborted) {
          failed = true;
          side.stats.errors += 1;
          if (side.moveEl) side.moveEl.textContent = `Error: ${err.message}`;
          if (err.status === 401 || err.status === 403) ctx.showError?.(`${side.player.name}: ${err.message}`);
        }
      })
      .finally(() => {
        settled = true;
      });

    let outcome = null; // "decided" | "missed"
    if (lockstep) {
      await pending;
      outcome = decision ? "decided" : "missed";
    } else {
      // Gravity loop: one row per gravityNow() until the answer arrives or the piece lands.
      while (!signal.aborted) {
        if (settled) {
          outcome = decision ? "decided" : "missed";
          break;
        }
        await sleep(gravityNow());
        if (signal.aborted) return;
        if (settled) {
          outcome = decision ? "decided" : "missed";
          break;
        }
        const a = side.active;
        if (!collides(side.board, PIECES[a.piece][a.rotation].cells, a.x, a.y + 1)) {
          a.y += 1;
          drawSide(side);
        } else {
          outcome = "missed";
          break;
        }
      }
    }
    if (signal.aborted) return;

    const a = side.active;
    let landed;
    if (outcome === "decided" && decision.chosen) {
      recordDecision(side, decision);
      const t = decision.chosen;
      const cells = PIECES[piece][t.rotation].cells;
      if (!collides(side.board, cells, t.x, a.y)) {
        // Slide into place quickly, then drop.
        side.target = t.cells;
        a.rotation = t.rotation;
        while (a.x !== t.x && !signal.aborted) {
          a.x += Math.sign(t.x - a.x);
          drawSide(side);
          await sleep(18);
        }
        const restY = dropY(side.board, cells, t.x, a.y);
        while (a.y < restY && !signal.aborted) {
          a.y += 1;
          drawSide(side);
          await sleep(10);
        }
        landed = { rotation: t.rotation, x: t.x, y: restY };
        const d = describePlacement(t);
        if (side.moveEl) side.moveEl.textContent = `${piece} → ${d.where} in ${Math.round(decision.latencyMs)} ms (${decision.note})`;
      } else {
        // Answer came too late for the rotation to fit at this height: lock as is.
        side.stats.missed += 1;
        landed = { rotation: a.rotation, x: a.x, y: dropY(side.board, PIECES[piece][a.rotation].cells, a.x, a.y) };
        if (side.moveEl) side.moveEl.textContent = `${piece}: answer arrived too late to fit (${Math.round(decision.latencyMs)} ms)`;
      }
    } else {
      if (outcome === "decided") {
        // Model replied but named no valid option.
        recordDecision(side, decision);
        side.stats.invalid += 1;
        if (side.moveEl) side.moveEl.textContent = `${piece}: ${decision.note}; piece dropped where it was`;
      } else {
        side.stats.missed += 1;
        if (!failed && side.moveEl) side.moveEl.textContent = `${piece}: no answer before landing (${Math.round(performance.now() - askedAt)} ms); locked in place`;
      }
      landed = { rotation: a.rotation, x: a.x, y: dropY(side.board, PIECES[piece][a.rotation].cells, a.x, a.y) };
    }
    if (signal.aborted) return;
    await settlePiece(side, landed, ctx);
  }
}
