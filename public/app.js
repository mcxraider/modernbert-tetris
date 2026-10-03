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
  makeBag,
  scoreForLines,
  describePlacement,
  shortlistPlacements,
  lookaheadScore,
} from "./tetris.js";
import { buildRequest, askJev, pickPlacement, JevError, STRATEGY_OPTIONS, HEALTH_LEVELS } from "./jev.js";
import { stateRows } from "./prompt.js";
import { requestModernBert } from "./players.js";

const $ = (id) => document.getElementById(id);
const PRICE_PER_TOKEN = 0.042 / 1_000_000; // $0.042 per million input tokens (docs.typesafe.ai/models)

// ---- DOM ------------------------------------------------------------------
const canvas = $("board");
const ctx = canvas.getContext("2d");
const nextCanvas = $("next");
const nextCtx = nextCanvas.getContext("2d");
const CELL = canvas.width / WIDTH;

const ui = {
  status: $("status"),
  overlay: $("overlay"),
  start: $("start"),
  pause: $("pause"),
  reset: $("reset"),
  speed: $("speed"),
  score: $("score"),
  lines: $("lines"),
  level: $("level"),
  pieces: $("pieces"),
  calls: $("calls"),
  latency: $("latency"),
  tokens: $("tokens"),
  cost: $("cost"),
  agree: $("agree"),
  top8: $("top8"),
  top10: $("top10"),
  top12: $("top12"),
  chosen: $("chosen"),
  pageTitle: $("pageTitle"),
  decisionTitle: $("decisionTitle"),
  readsNote: $("readsNote"),
  alternatives: $("alternatives"),
  strategy: $("strategy"),
  healthFill: $("healthFill"),
  healthValue: $("healthValue"),
  healthLegend: $("healthLegend"),
  nextFitFill: $("nextFitFill"),
  nextFitValue: $("nextFitValue"),
  reqJson: $("reqJson"),
  resJson: $("resJson"),
  error: $("error"),
};

// ---- Game state -------------------------------------------------------------
const game = {
  board: emptyBoard(),
  bag: [],
  current: null,
  next: null,
  score: 0,
  lines: 0,
  pieces: 0,
  over: false,
  running: false,
  active: null, // { piece, rotation, x, y } while animating a drop
  ghosts: [], // [{ cells, probability, chosen }]
  flash: [], // row indexes being cleared
};

const stats = { calls: 0, latency: 0, tokens: 0, agreements: 0, decisions: 0, top8: 0, top10: 0, top12: 0 };

// One run id per solo game; ply counts Jev's logged moves. Reset in resetGame.
let runId = null;
let ply = 0;

let abort = null;
// False when the page is served without its proxy (static hosting, file://).
// Jev mode needs the proxy because api.typesafe.ai rejects browser origins.
let backendAvailable = true;
const NO_BACKEND_MESSAGE =
  "This copy of the page has no proxy server, so it cannot reach api.typesafe.ai (the API blocks browser origins). " +
  "Run `npm start` locally or deploy the repo to get Jev mode. The built-in heuristic works here.";

const MODEL_LABELS = {
  jev: "Jev",
  "jev-better-prompt": "Jev (better prompt)",
  modernbert: "ModernBERT",
  heuristic: "Heuristic",
};

function mode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function modelLabel(source = mode()) {
  return MODEL_LABELS[source] || "Jev";
}

function applyModelTitle() {
  const label = modelLabel();
  ui.pageTitle.textContent = `${label} plays Tetris`;
  document.title = `${label} plays Tetris (solo)`;
}

function nextPiece() {
  if (game.bag.length === 0) game.bag = makeBag();
  return game.bag.pop();
}

function resetGame() {
  game.board = emptyBoard();
  game.bag = [];
  game.current = nextPiece();
  game.next = nextPiece();
  game.score = 0;
  game.lines = 0;
  game.pieces = 0;
  game.over = false;
  game.active = null;
  game.ghosts = [];
  game.flash = [];
  Object.assign(stats, { calls: 0, latency: 0, tokens: 0, agreements: 0, decisions: 0, top8: 0, top10: 0, top12: 0 });
  runId = `solo-${mode()}-${new Date().toISOString().replace(/[:.]/g, "-")}`;
  ply = 0;
  ui.overlay.classList.add("hidden");
  ui.chosen.innerHTML = '<span class="muted">No move yet.</span>';
  ui.alternatives.innerHTML = "";
  ui.decisionTitle.textContent = `${modelLabel()}'s move`;
  ui.readsNote.classList.add("hidden");
  renderReads(null);
  ui.reqJson.textContent = "–";
  ui.resJson.textContent = "–";
  hideError();
  renderStats();
  draw();
  drawNext();
}

// ---- Rendering ----------------------------------------------------------------
function drawCell(x, y, color, alpha = 1, outline = false) {
  const px = x * CELL;
  const py = y * CELL;
  ctx.globalAlpha = alpha;
  if (outline) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(px + 2, py + 2, CELL - 4, CELL - 4);
  } else {
    ctx.fillStyle = color;
    ctx.fillRect(px + 1, py + 1, CELL - 2, CELL - 2);
    ctx.fillStyle = "rgba(255,255,255,0.18)";
    ctx.fillRect(px + 1, py + 1, CELL - 2, 4);
  }
  ctx.globalAlpha = 1;
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  ctx.lineWidth = 1;
  for (let x = 1; x < WIDTH; x++) {
    ctx.beginPath();
    ctx.moveTo(x * CELL, 0);
    ctx.lineTo(x * CELL, canvas.height);
    ctx.stroke();
  }
  for (let y = 1; y < HEIGHT; y++) {
    ctx.beginPath();
    ctx.moveTo(0, y * CELL);
    ctx.lineTo(canvas.width, y * CELL);
    ctx.stroke();
  }
  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      const c = game.board[y][x];
      if (c) drawCell(x, y, game.flash.includes(y) ? "#ffffff" : PIECE_COLORS[c]);
    }
  }
  for (const g of game.ghosts) {
    const color = g.chosen ? PIECE_COLORS[game.current] : "#ffffff";
    const alpha = g.chosen ? 0.9 : Math.max(0.15, Math.min(0.6, g.probability * 1.5));
    for (const [x, y] of g.cells) drawCell(x, y, color, alpha, true);
    if (!g.chosen && g.probability >= 0.05) {
      const [lx, ly] = g.cells.reduce((a, b) => (b[1] < a[1] || (b[1] === a[1] && b[0] < a[0]) ? b : a));
      ctx.globalAlpha = 0.8;
      ctx.fillStyle = "#fff";
      ctx.font = "10px system-ui, sans-serif";
      ctx.fillText(`${Math.round(g.probability * 100)}%`, lx * CELL + 4, ly * CELL + 12);
      ctx.globalAlpha = 1;
    }
  }
  if (game.active) {
    const { piece, rotation, x, y } = game.active;
    for (const [cx, cy] of PIECES[piece][rotation].cells) {
      if (y + cy >= 0) drawCell(x + cx, y + cy, PIECE_COLORS[piece]);
    }
  }
}

function drawNext() {
  nextCtx.clearRect(0, 0, nextCanvas.width, nextCanvas.height);
  if (!game.next) return;
  const state = PIECES[game.next][0];
  const size = 12;
  const ox = (nextCanvas.width - state.width * size) / 2;
  const oy = (nextCanvas.height - state.height * size) / 2;
  nextCtx.fillStyle = PIECE_COLORS[game.next];
  for (const [cx, cy] of state.cells) nextCtx.fillRect(ox + cx * size + 1, oy + cy * size + 1, size - 2, size - 2);
}

function renderStats() {
  ui.score.textContent = game.score;
  ui.lines.textContent = game.lines;
  ui.level.textContent = level();
  ui.pieces.textContent = game.pieces;
  ui.calls.textContent = stats.calls;
  ui.latency.textContent = stats.calls ? `${Math.round(stats.latency / stats.calls)} ms` : "–";
  ui.tokens.textContent = stats.tokens.toLocaleString();
  ui.cost.textContent = `$${(stats.tokens * PRICE_PER_TOKEN).toFixed(4)}`;
  ui.agree.textContent = stats.decisions ? `${Math.round((100 * stats.agreements) / stats.decisions)}%` : "–";
  const pct = (n) => (stats.decisions ? `${Math.round((100 * n) / stats.decisions)}%` : "–");
  ui.top8.textContent = pct(stats.top8);
  ui.top10.textContent = pct(stats.top10);
  ui.top12.textContent = pct(stats.top12);
}

function level() {
  return Math.floor(game.lines / 10) + 1;
}

function setStatus(text, cls = "") {
  ui.status.textContent = text;
  ui.status.className = `status ${cls}`;
}

function showError(message) {
  ui.error.textContent = message;
  ui.error.classList.remove("hidden");
}

function hideError() {
  ui.error.classList.add("hidden");
}

function barRow(name, probability, chosen = false, fillClass = "") {
  const pct = Math.round(probability * 100);
  return `<div class="bar-row${chosen ? " chosen" : ""}">
    <span class="name" title="${name}">${name}</span>
    <div class="bar"><div class="bar-fill ${fillClass}" style="width:${Math.max(1, pct)}%"></div></div>
    <span class="bar-value">${probability.toFixed(2)}</span>
  </div>`;
}

function describeForHumans(p) {
  const d = describePlacement(p);
  const lines = p.linesCleared ? `clears ${d.lines_cleared}` : "no clear";
  const holes = p.holesCreated ? d.holes_created : "no holes";
  return `${d.where} · ${lines} · ${holes}`;
}

function renderDecision(decision, source) {
  const { chosen, ranked, confidence } = decision;
  const d = describePlacement(chosen);
  const isJev = source.startsWith("jev");
  const isModel = isJev || source === "modernbert"; // makes a model call; heuristic doesn't
  const conf = isModel ? `confidence ${confidence.toFixed(2)}` : "code heuristic, no model call";
  ui.decisionTitle.textContent = `${modelLabel(source)}'s move`;
  // "Jev's read of the board" panel is Jev-specific; hide its note only for Jev.
  ui.readsNote.classList.toggle("hidden", isJev);
  ui.chosen.innerHTML = `<strong>${chosen.piece}</strong> → ${d.where}
    <span class="muted">(${conf})</span>
    <div class="desc">
      <span>lines <b>${d.lines_cleared}</b></span>
      <span>holes <b>${d.holes_created}</b></span>
      <span>height <b>${d.stack_height_after}</b></span>
      <span>surface <b>${d.surface_after}</b></span>
      <span>wells <b>${d.wells_after}</b></span>
    </div>`;
  ui.alternatives.innerHTML = ranked
    .slice(0, 6)
    .map((r) => barRow(describeForHumans(r.placement), r.probability, r.placement === chosen))
    .join("");
}

function renderReads(answers) {
  if (!answers) {
    ui.strategy.innerHTML = Object.keys(STRATEGY_OPTIONS).map((k) => barRow(k.replaceAll("_", " "), 0)).join("");
    ui.healthFill.style.left = "0%";
    ui.healthValue.textContent = "";
    ui.healthLegend.innerHTML = HEALTH_LEVELS.map((l) => `<span>${l.split(":")[0]}</span>`).join("");
    ui.nextFitFill.style.width = "0%";
    ui.nextFitValue.textContent = "";
    return;
  }
  const strategy = answers.strategy;
  if (strategy?.probabilities) {
    ui.strategy.innerHTML = Object.entries(strategy.probabilities)
      .sort((a, b) => b[1] - a[1])
      .map(([k, p]) => barRow(k.replaceAll("_", " "), p, k === strategy.choice))
      .join("");
  }
  const health = answers.board_health;
  if (typeof health?.score === "number") {
    const max = HEALTH_LEVELS.length - 1;
    ui.healthFill.style.left = `calc(${(100 * health.score) / max}% - 2px)`;
    const nearest = HEALTH_LEVELS[Math.round(health.score)] || "";
    ui.healthValue.textContent = `${health.score.toFixed(2)} · ${nearest.split(":")[0]}`;
  }
  const fit = answers.next_piece_fits;
  if (typeof fit?.noul === "number") {
    ui.nextFitFill.style.width = `${Math.round(fit.noul * 100)}%`;
    ui.nextFitValue.textContent = fit.noul.toFixed(2);
  }
}

// ---- Deciding -------------------------------------------------------------------
function heuristicDecision(placements, nextPiece) {
  // Score by 2-ply lookahead (next-piece aware), falling back to 1-ply when
  // nextPiece is unknown. Cache per placement so the sort and softmax agree.
  const scored = placements.map((p) => ({ p, score: lookaheadScore(p, nextPiece) }));
  scored.sort((a, b) => b.score - a.score);
  // Softmax over the scores so the panel shows a distribution too.
  const max = scored[0].score;
  const weights = scored.map((s) => Math.exp((s.score - max) / 0.5));
  const sum = weights.reduce((a, b) => a + b, 0);
  return {
    chosen: scored[0].p,
    ranked: scored.map((s, i) => ({ placement: s.p, probability: weights[i] / sum })),
    confidence: weights[0] / sum,
  };
}

async function jevDecision(placements, signal, { betterPrompt = false } = {}) {
  // jev-better-prompt sends a full board per candidate, so trim to the
  // strongest few first; plain Jev still sees every placement.
  const offered = betterPrompt ? shortlistPlacements(placements) : placements;
  const request = buildRequest(
    {
      board: game.board,
      piece: game.current,
      nextPiece: game.next,
      stats: boardStats(game.board),
      linesCleared: game.lines,
    },
    offered,
    { withBoardAfter: betterPrompt },
  );
  ui.reqJson.textContent = JSON.stringify(request, null, 2);
  const who = betterPrompt ? "Jev (better prompt)" : "Jev";
  setStatus(`Asking ${who} where to put the ${game.current} (${offered.length} options)…`, "thinking");
  const { response, latencyMs } = await askJev(request, "", { signal });
  ui.resJson.textContent = JSON.stringify(response, null, 2);
  stats.calls += 1;
  stats.latency += latencyMs;
  stats.tokens += response?.usage?.input_tokens || 0;
  renderReads(response.answers);
  // Log this decision (one row per offered candidate, with Jev's probability)
  // to data/battles/<run>.json (grouped by state) via /api/log. Fire-and-forget like battle.js.
  const probs = response?.answers?.placement?.probabilities || {};
  const rows = stateRows({
    stateId: `${runId}:${ply}`,
    board: game.board,
    piece: game.current,
    nextPiece: game.next,
    lines: game.lines,
    placements: offered,
    offeredIds: offered.map((p) => p.id),
    probs,
    choiceId: response?.answers?.placement?.choice,
  }).map((r) => ({ ...r, run: runId }));
  ply += 1;
  fetch("api/log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run: runId, rows }),
  }).catch(() => {});
  const decision = pickPlacement(response, offered);
  // Debug: Jev's softmax per candidate, mirroring serve.py's ModernBERT dump.
  // TypeSafe returns probabilities (already softmaxed), so logit shown is log(p).
  const tag = betterPrompt ? "jev-better-prompt" : "jev";
  console.log(`[${tag}] ${decision.ranked.length} candidates, chosen=${decision.chosen.id}`);
  for (const { placement, probability } of decision.ranked) {
    console.log(`  ${String(placement.id).padEnd(6)} logit=${Math.log(probability).toFixed(4)}  p=${probability.toFixed(4)}`);
  }
  return decision;
}

async function modernbertDecision(placements, signal) {
  setStatus(`Asking ModernBERT where to put the ${game.current} (${placements.length} options)…`, "thinking");
  const { probabilities, chosen, latencyMs } = await requestModernBert(
    { board: game.board, piece: game.current, nextPiece: game.next, lines: game.lines, placements },
    signal,
  );
  stats.calls += 1;
  stats.latency += latencyMs;
  renderReads(null); // ModernBERT has no strategy/health read to show
  const ranked = placements
    .map((p) => ({ placement: p, probability: probabilities[p.id] ?? 0 }))
    .sort((a, b) => b.probability - a.probability);
  const pick = chosen || ranked[0].placement;
  return { chosen: pick, ranked, confidence: probabilities[pick.id] ?? 0 };
}

// ---- Animating --------------------------------------------------------------------
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function stepDelay() {
  // slider 1..10 → 220ms..10ms per animation step
  const v = Number(ui.speed.value);
  return Math.round(240 - v * 23);
}

async function animateDrop(target, signal) {
  const { piece, rotation, x: tx, y: ty } = target;
  const spawnX = Math.min(3, WIDTH - PIECES[piece][rotation].width);
  const cells = PIECES[piece][rotation].cells;
  let x = collides(game.board, cells, spawnX, 0) ? tx : spawnX;
  const active = { piece, rotation, x, y: 0 };
  game.active = active;
  draw();
  while (active.x !== tx) {
    if (signal.aborted) return;
    await sleep(stepDelay());
    const nx = active.x + Math.sign(tx - active.x);
    active.x = collides(game.board, cells, nx, active.y) ? tx : nx;
    draw();
  }
  while (active.y < ty) {
    if (signal.aborted) return;
    await sleep(stepDelay() * 0.6);
    active.y += 1;
    draw();
  }
}

// ---- Main loop --------------------------------------------------------------------
async function playLoop(signal) {
  while (game.running && !game.over && !signal.aborted) {
    const placements = enumeratePlacements(game.board, game.current);
    if (placements.length === 0) {
      endGame();
      return;
    }
    const heuristic = heuristicDecision(placements, game.next);
    let decision;
    const source = mode();
    if (source === "jev" || source === "jev-better-prompt") {
      try {
        decision = await jevDecision(placements, signal, { betterPrompt: source === "jev-better-prompt" });
      } catch (err) {
        if (signal.aborted) return;
        handleJevError(err);
        pause();
        return;
      }
    } else if (source === "modernbert") {
      try {
        decision = await modernbertDecision(placements, signal);
      } catch (err) {
        if (signal.aborted) return;
        showError(`ModernBERT call failed${err.status ? ` (${err.status})` : ""}: ${err.message}`);
        setStatus("Paused after an error.", "error");
        pause();
        return;
      }
    } else {
      decision = heuristic;
      setStatus(`Heuristic picks a spot for the ${game.current}`);
    }
    if (signal.aborted) return;

    stats.decisions += 1;
    if (decision.chosen === heuristic.chosen) stats.agreements += 1;
    const rank = heuristic.ranked.findIndex((r) => r.placement.id === decision.chosen.id);
    if (rank > -1 && rank < 8) stats.top8 += 1;
    if (rank > -1 && rank < 10) stats.top10 += 1;
    if (rank > -1 && rank < 12) stats.top12 += 1;
    renderDecision(decision, source);
    renderStats();
    hideError();

    game.ghosts = decision.ranked
      .slice(0, 4)
      .map((r) => ({ cells: r.placement.cells, probability: r.probability, chosen: r.placement === decision.chosen }));
    setStatus(`Dropping ${game.current} at ${describePlacement(decision.chosen).where}`);
    await animateDrop(decision.chosen, signal);
    if (signal.aborted) return;

    game.active = null;
    game.ghosts = [];
    const locked = lockPiece(game.board, decision.chosen.piece, decision.chosen.rotation, decision.chosen.x, decision.chosen.y);
    const { board, cleared, rows } = clearLines(locked);
    if (cleared > 0) {
      game.board = locked;
      game.flash = rows;
      draw();
      await sleep(Math.max(80, stepDelay() * 1.5));
      game.flash = [];
    }
    game.board = board;
    game.score += scoreForLines(cleared, level());
    game.lines += cleared;
    game.pieces += 1;
    game.current = game.next;
    game.next = nextPiece();
    renderStats();
    draw();
    drawNext();
    if (collides(game.board, PIECES[game.current][0].cells, 3, 0)) {
      endGame();
      return;
    }
  }
}

function endGame() {
  game.over = true;
  game.running = false;
  ui.overlay.textContent = `Game over · ${game.lines} lines · ${game.score} points`;
  ui.overlay.classList.remove("hidden");
  setStatus("Game over. Press Reset to play again.");
  ui.start.disabled = true;
  ui.pause.disabled = true;
}

function handleJevError(err) {
  if (err instanceof JevError) {
    if (err.status === 401 || err.status === 403) {
      showError(`TypeSafe rejected the server key (${err.status}): ${err.message}. Check TYPESAFE_API_KEY in .env.`);
    } else if (err.status === 429) {
      showError("Rate limited by TypeSafe even after retries. Wait a moment, then press Start.");
    } else if (err.status === 422) {
      showError(`TypeSafe rejected the request (422): ${JSON.stringify(err.detail)}`);
    } else {
      showError(`Jev call failed (${err.status}): ${err.message}`);
    }
  } else {
    showError(`Jev call failed: ${err.message}`);
  }
  setStatus("Paused after an error.", "error");
}

// ---- Controls -----------------------------------------------------------------------
function start() {
  if (game.over) return;
  const needsProxy = mode().startsWith("jev") || mode() === "modernbert";
  if (needsProxy && !backendAvailable) {
    showError(NO_BACKEND_MESSAGE);
    return;
  }
  hideError();
  game.running = true;
  ui.start.disabled = true;
  ui.pause.disabled = false;
  abort = new AbortController();
  playLoop(abort.signal).finally(() => {
    if (!game.over) {
      ui.start.disabled = false;
      ui.pause.disabled = true;
    }
  });
}

function pause() {
  game.running = false;
  abort?.abort();
  abort = null;
  ui.start.disabled = game.over;
  ui.pause.disabled = true;
  if (!game.over) setStatus(ui.status.classList.contains("error") ? ui.status.textContent : "Paused.");
}

function reset() {
  pause();
  resetGame();
  ui.start.disabled = false;
  ui.start.textContent = "Start";
  setStatus(mode().startsWith("jev") ? "Ready. Press Start and Jev takes over." : "Ready. Press Start for the built-in heuristic.");
}

// ---- Wiring ---------------------------------------------------------------------------
ui.start.addEventListener("click", start);
ui.pause.addEventListener("click", pause);
ui.reset.addEventListener("click", reset);
document.querySelectorAll('input[name="mode"]').forEach((el) =>
  el.addEventListener("change", () => {
    applyModelTitle();
    if (!game.running && !game.over) {
      setStatus(mode().startsWith("jev") ? "Ready. Press Start and Jev takes over." : "Ready. Press Start for the built-in heuristic.");
    }
  }),
);
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === " ") {
    e.preventDefault();
    if (game.running) pause();
    else start();
  }
});

function markNoBackend() {
  backendAvailable = false;
  document.querySelector('input[name="mode"][value="heuristic"]').checked = true;
  applyModelTitle();
  setStatus("Preview without a proxy server: the heuristic plays. Press Start.");
}

fetch("api/config", { cache: "no-store" })
  .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
  .catch(markNoBackend);

applyModelTitle();
resetGame();
setStatus("Press Start and Jev takes over.");
