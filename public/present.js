// Presentation page: a best-of-N series between Jev and one opponent, laid
// out for screen recordings. The boards fill the window height, the centre
// column carries the series score, the round clock and the round result, and
// nothing else is on screen once the series starts. Rounds run back to back
// each on a fresh random seed shared by both sides (add ?seed=N to the URL for
// a reproducible series instead); the series ends when one side has a majority.

import { OPPONENTS, createJevPlayer } from "./players.js";
import {
  SPEEDUPS,
  sleep,
  fmtMs,
  fmtUsd,
  formatClock,
  createRamp,
  freshStats,
  makeSide,
  resetSide,
  previewSide,
  runModelSide,
  markTopOut,
  comparisonTable,
  judgeRound,
  avgLatency,
} from "./arena.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

const ui = {
  setup: $("setup"),
  setupError: $("setupError"),
  opponent: $("opponent"),
  rounds: $("roundsInput"),
  hold: $("hold"),
  limit: $("limit"),
  gravity: $("gravity"),
  gravityLabel: $("gravityLabel"),
  speedup: $("speedup"),
  garbage: $("garbage"),
  lockstep: $("lockstep"),
  start: $("start"),
  badgeR: $("badgeR"),
  modelR: $("modelR"),
  nameL: $("nameL"),
  nameR: $("nameR"),
  scoreL: $("scoreL"),
  scoreR: $("scoreR"),
  seriesTitle: $("seriesTitle"),
  roundLabel: $("roundLabel"),
  clock: $("clock"),
  level: $("level"),
  result: $("result"),
  seriesCard: $("seriesCard"),
  next: $("next"),
  roundsList: $("rounds"),
  error: $("error"),
};

function sideEls(id) {
  return {
    canvas: $(`board${id}`),
    overlay: $(`overlay${id}`),
    incomingEl: $(`incoming${id}`),
    statsEl: $(`stats${id}`),
    moveEl: null,
    section: $(id === "L" ? "left" : "right"),
  };
}

// Two rows of four under each board.
function presentRows(side) {
  const s = side.stats;
  return [
    ["Lines", side.lines],
    ["Pieces", side.pieces],
    ["Sent", s.sent],
    ["Received", s.received],
    ["Latency", s.calls ? fmtMs(s.latency / s.calls) : "–"],
    ["Missed", s.missed],
    ["Cost", fmtUsd(s.cost)],
    ["Calls", s.calls],
  ];
}

// Short provider names for the narrow side panels.

const ROUND_ROWS = [
  ["Lines", (s) => s.lines],
  ["Pieces", (s) => s.pieces],
  ["Garbage sent", (s) => s.stats.sent],
  ["Avg latency", avgLatency],
  ["Missed", (s) => s.stats.missed],
  ["Cost", (s) => fmtUsd(s.stats.cost)],
];

const SERIES_ROWS = [
  ["Rounds won", (t) => t.wins],
  ["Lines", (t) => t.lines],
  ["Pieces", (t) => t.pieces],
  ["Avg latency", avgLatency],
  ["Missed", (t) => t.stats.missed],
  ["Tokens in", (t) => t.stats.inputTokens.toLocaleString()],
  ["Cost", (t) => fmtUsd(t.stats.cost)],
];

let sides = [];
let series = null; // settings + { target, rounds, wins, draws, totals, stopped, done }
let round = null; // { n, seed, ramp, abort, timer }
let skipHold = null; // set while waiting between rounds; call to continue now

// ---- Helpers ------------------------------------------------------------------------------

function opponentChoice() {
  const key = params.get("opponent");
  if (key && OPPONENTS[key] && ui.opponent.value !== key) ui.opponent.value = key;
  return OPPONENTS[ui.opponent.value] || Object.values(OPPONENTS)[0];
}

function renderOpponent() {
  const opp = opponentChoice();
  ui.badgeR.textContent = opp.name;
  ui.badgeR.className = `badge ${opp.badge}`;
  ui.modelR.textContent = opp.label;
  ui.nameR.textContent = opp.short;
  document.title = `Jev vs ${opp.name} · presentation`;
}

function gravityNow() {
  return round ? round.ramp.gravityNow() : Number(ui.gravity.value);
}

function renderLevel() {
  const level = round ? round.ramp.level() : 1;
  ui.level.textContent = series?.lockstep ? "no gravity" : `Level ${level}`;
  ui.level.classList.toggle("hot", level >= 4);
}

function renderScore() {
  ui.scoreL.textContent = series ? series.wins.L : 0;
  ui.scoreR.textContent = series ? series.wins.R : 0;
}

function renderRounds() {
  const opp = opponentChoice();
  ui.roundsList.innerHTML = series.rounds
    .map((r) => {
      const cls = r.winner === "L" ? "jev" : r.winner === "R" ? opp.badge : "draw";
      const who = r.winner === "L" ? "Jev" : r.winner === "R" ? opp.short : "Draw";
      return `<li><span class="n">R${r.n}</span><span class="w ${cls}">${who}</span><span class="t">${formatClock(r.elapsed)}</span><span class="l">${r.lines.L}–${r.lines.R}</span></li>`;
    })
    .join("");
}

function showError(message) {
  const el = ui.setup.classList.contains("hidden") ? ui.error : ui.setupError;
  el.textContent = message;
  el.classList.remove("hidden");
}

function hideErrors() {
  ui.error.classList.add("hidden");
  ui.setupError.classList.add("hidden");
}

function freshTotals(player) {
  return { player, wins: 0, lines: 0, pieces: 0, stats: freshStats() };
}

function addTotals(t, side) {
  t.lines += side.lines;
  t.pieces += side.pieces;
  const a = t.stats;
  const b = side.stats;
  for (const k of ["calls", "latency", "missed", "invalid", "errors", "inputTokens", "outputTokens", "cost", "sent", "received"]) a[k] += b[k];
  a.minLatency = Math.min(a.minLatency, b.minLatency);
  a.maxLatency = Math.max(a.maxLatency, b.maxLatency);
}

// ---- Series --------------------------------------------------------------------------------

async function startSeries() {
  const opp = opponentChoice();
  hideErrors();
  ui.start.disabled = true;
  const total = Math.min(99, Math.max(1, Number(ui.rounds.value) || 1));
  series = {
    total,
    target: Math.floor(total / 2) + 1,
    fixedSeed: params.has("seed") ? Number(params.get("seed")) || 42 : null,
    gravityMs: Number(ui.gravity.value),
    speedup: ui.speedup.value,
    limitMs: Math.max(1, Number(ui.limit.value) || 5) * 60_000,
    holdMs: Math.max(0, Number(ui.hold.value) || 0) * 1000,
    garbage: ui.garbage.checked,
    lockstep: ui.lockstep.checked,
    rounds: [],
    wins: { L: 0, R: 0 },
    draws: 0,
    totals: null,
    stopped: false,
    done: false,
  };
  sides = [makeSide("L", createJevPlayer(), sideEls("L"), presentRows), makeSide("R", opp.create(), sideEls("R"), presentRows)];
  sides[0].opponent = sides[1];
  sides[1].opponent = sides[0];
  series.totals = { L: freshTotals(sides[0].player), R: freshTotals(sides[1].player) };
  ui.setup.classList.add("hidden");
  ui.start.disabled = false;
  ui.seriesCard.classList.add("hidden");
  ui.roundsList.innerHTML = "";
  ui.seriesTitle.textContent = `Real-time Tetris · best of ${total}`;
  document.body.classList.add("live");
  renderScore();
  startRound();
}

async function startRound() {
  const n = series.rounds.length + 1;
  const seed = series.fixedSeed !== null ? series.fixedSeed + n - 1 : Math.floor(Math.random() * 1_000_000);
  for (const s of sides) resetSide(s, seed);
  ui.result.classList.add("hidden");
  ui.next.classList.add("hidden");
  ui.roundLabel.textContent = `Round ${n} of ${series.total}`;
  renderLevel();
  for (const c of [3, 2, 1]) {
    ui.clock.textContent = String(c);
    await sleep(800);
    if (!series || series.stopped) return;
  }
  const abort = new AbortController();
  round = { n, seed, ramp: createRamp({ gravityMs: series.gravityMs, speedup: SPEEDUPS[series.speedup] || SPEEDUPS.normal }), abort, timer: null };
  renderLevel();
  round.timer = setInterval(() => {
    if (!round) return;
    const elapsed = round.ramp.elapsed();
    ui.clock.textContent = formatClock(elapsed);
    renderLevel();
    if (elapsed >= series.limitMs) checkEnd(true);
  }, 250);
  const ctx = { signal: abort.signal, lockstep: series.lockstep, gravityNow, garbage: series.garbage, onTopOut: topOut, onPiece: () => checkEnd(), showError };
  for (const s of sides) runModelSide(s, ctx);
}

function topOut(side) {
  markTopOut(side, round ? round.ramp.elapsed() : 0);
  checkEnd();
}

function checkEnd(timeUp = false) {
  if (!round) return;
  const verdict = judgeRound(sides[0], sides[1], { garbage: series.garbage, timeUp });
  if (verdict) finishRound(verdict.winner, verdict.loser, verdict.reason);
}

function finishRound(winner, loser, reason) {
  const [L, R] = sides;
  const r = round;
  round = null;
  r.abort.abort();
  clearInterval(r.timer);
  const elapsed = r.ramp.elapsed();
  ui.clock.textContent = formatClock(elapsed);
  if (winner) {
    winner.section.classList.add("winner");
    loser?.section.classList.add("loser");
  }
  const key = winner ? winner.id : null;
  if (key) series.wins[key] += 1;
  else series.draws += 1;
  for (const s of sides) addTotals(series.totals[s.id], s);
  series.rounds.push({ n: r.n, seed: r.seed, winner: key, reason, elapsed, lines: { L: L.lines, R: R.lines } });
  renderScore();
  renderRounds();
  ui.result.innerHTML = `${reason}<small>Round ${r.n} · ${formatClock(elapsed)} · seed ${r.seed}</small>${comparisonTable(L, R, ROUND_ROWS)}`;
  ui.result.classList.remove("hidden");
  const decided = series.wins.L >= series.target || series.wins.R >= series.target || series.rounds.length >= series.total;
  holdThen(decided || series.stopped ? finishSeries : startRound, decided ? "Series result" : `Round ${r.n + 1}`);
}

// Waits holdMs (or for Space when holdMs is 0), then continues.
async function holdThen(next, label) {
  const holdMs = series.holdMs;
  const until = performance.now() + holdMs;
  let skipped = false;
  skipHold = () => {
    skipped = true;
  };
  while (!skipped && series && (holdMs === 0 || performance.now() < until)) {
    ui.next.textContent = holdMs === 0 ? `${label}: press Space` : `${label} in ${Math.ceil((until - performance.now()) / 1000)} s · Space to skip`;
    ui.next.classList.remove("hidden");
    await sleep(200);
  }
  skipHold = null;
  ui.next.classList.add("hidden");
  if (!series) return;
  if (series.stopped && next === startRound) {
    finishSeries();
    return;
  }
  next();
}

function finishSeries() {
  const [L, R] = sides;
  const w = series.wins;
  series.done = true;
  series.totals.L.wins = w.L;
  series.totals.R.wins = w.R;
  const winner = w.L > w.R ? L : w.R > w.L ? R : null;
  const title = winner ? `${winner.player.name} wins the series ${w[winner.id]}–${w[winner === L ? "R" : "L"]}` : `Series drawn ${w.L}–${w.R}`;
  const played = series.rounds.length;
  const note = `${played} round${played === 1 ? "" : "s"} played${series.draws ? `, ${series.draws} drawn` : ""} · first to ${series.target}${series.stopped ? " · stopped early" : ""}`;
  ui.result.classList.add("hidden");
  ui.seriesCard.innerHTML = `${title}<small>${note}</small>${comparisonTable(series.totals.L, series.totals.R, SERIES_ROWS)}`;
  ui.seriesCard.classList.remove("hidden");
  ui.roundLabel.textContent = "Series over";
  ui.next.textContent = "N for a new series";
  ui.next.classList.remove("hidden");
  for (const s of sides) s.section.classList.remove("winner", "loser");
  if (winner) {
    winner.section.classList.add("winner");
    (winner === L ? R : L).section.classList.add("loser");
  }
  document.body.classList.remove("live");
}

function endRound() {
  if (round) {
    checkEnd(true);
    return;
  }
  if (series && !series.done) {
    // Between rounds: a second Esc ends the series.
    series.stopped = true;
    skipHold?.();
  }
}

function newSeries() {
  if (round) return;
  if (series && !series.done) {
    series.stopped = true;
    skipHold?.();
    return;
  }
  series = null;
  round = null;
  ui.setup.classList.remove("hidden");
  ui.start.disabled = false;
}

// ---- Wiring ----------------------------------------------------------------------

ui.start.addEventListener("click", startSeries);
ui.opponent.addEventListener("change", () => {
  params.delete("opponent");
  renderOpponent();
  if (!series) sides[1].player = { name: opponentChoice().name, short: opponentChoice().short };
});
ui.gravity.addEventListener("input", () => {
  ui.gravityLabel.textContent = `${ui.gravity.value} ms per row`;
});
document.addEventListener("keydown", (e) => {
  if (!ui.setup.classList.contains("hidden")) return; // typing in the form
  if (e.code === "Space") {
    e.preventDefault();
    skipHold?.();
  } else if (e.key === "Escape") {
    endRound();
  } else if (e.key === "n" || e.key === "N") {
    newSeries();
  }
});

for (const [id, el] of [
  ["rounds", ui.rounds],
  ["hold", ui.hold],
  ["limit", ui.limit],
  ["gravity", ui.gravity],
]) {
  if (params.has(id)) el.value = params.get(id);
}
if (params.has("speedup") && SPEEDUPS[params.get("speedup")]) ui.speedup.value = params.get("speedup");
ui.gravityLabel.textContent = `${ui.gravity.value} ms per row`;

renderOpponent();
ui.roundLabel.textContent = `Best of ${ui.rounds.value}`;
sides = [makeSide("L", { name: "Jev", short: "Jev" }, sideEls("L"), presentRows), makeSide("R", { name: opponentChoice().name, short: opponentChoice().short }, sideEls("R"), presentRows)];
for (const s of sides) previewSide(s, 42);
renderLevel();
