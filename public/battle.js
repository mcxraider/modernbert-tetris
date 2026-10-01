// Real-time Tetris battle: Jev vs Claude Haiku on one seeded piece sequence
// and one clock. Each side runs its own loop (see arena.js): the model is asked
// when a piece spawns, gravity keeps pulling the piece down meanwhile, and a
// piece that lands before the answer arrives locks where it is. Gravity
// strengthens on a shared schedule. In versus mode cleared lines become
// garbage for the opponent and the first to top out loses; with independent
// boards the survivor must outlast the loser's piece count.

import { OPPONENTS, createJevPlayer } from "./players.js";
import {
  SPEEDUPS,
  PRESENT,
  sleep,
  formatClock,
  describeSpeedup,
  createRamp,
  makeSide,
  resetSide,
  previewSide,
  runModelSide,
  markTopOut,
  comparisonTable,
  judgeRound,
} from "./arena.js";

const $ = (id) => document.getElementById(id);
const ui = {
  opponent: $("opponent"),
  badgeR: $("badgeR"),
  modelR: $("modelR"),
  gravity: $("gravity"),
  gravityLabel: $("gravityLabel"),
  speedup: $("speedup"),
  level: $("level"),
  seed: $("seed"),
  limit: $("limit"),
  lockstep: $("lockstep"),
  garbage: $("garbage"),
  start: $("start"),
  stop: $("stop"),
  clock: $("clock"),
  result: $("result"),
  error: $("error"),
  matchInfo: $("matchInfo"),
};

function sideEls(id) {
  return {
    canvas: $(`board${id}`),
    overlay: $(`overlay${id}`),
    incomingEl: $(`incoming${id}`),
    statsEl: $(`stats${id}`),
    moveEl: $(`move${id}`),
    section: $(id === "L" ? "left" : "right"),
  };
}

let sides = [];
let battle = null; // { ramp, abort, timer, limitMs, lockstep, garbage }

function gravityNow() {
  return battle ? battle.ramp.gravityNow() : Number(ui.gravity.value);
}

function renderLevel() {
  const level = battle ? battle.ramp.level() : 1;
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
  markTopOut(side, battle ? battle.ramp.elapsed() : 0);
  checkEnd();
}

// ---- Battle control -----------------------------------------------------------------------

function checkEnd(timeUp = false) {
  if (!battle) return;
  const verdict = judgeRound(sides[0], sides[1], { garbage: battle.garbage, timeUp });
  if (verdict) finish(verdict.winner, verdict.loser, verdict.reason);
}

function finish(winner, loser, reason) {
  const [L, R] = sides;
  battle.abort.abort();
  clearInterval(battle.timer);
  const elapsed = battle.ramp.elapsed();
  if (winner) {
    winner.section.classList.add("winner");
    loser?.section.classList.add("loser");
  }
  const gravityNote =
    (battle.lockstep ? "lockstep, no gravity" : `gravity ${battle.ramp.gravityMs} → ${gravityNow()} ms/row, level ${battle.ramp.level()}`) +
    (battle.garbage ? ", versus" : "");
  ui.result.innerHTML = `${reason}<small>${formatClock(elapsed)} elapsed · seed ${ui.seed.value} · ${gravityNote}</small>${comparisonTable(L, R)}`;
  ui.result.classList.remove("hidden");
  ui.start.disabled = false;
  ui.stop.disabled = true;
  document.body.classList.remove("running");
  document.body.classList.add("played");
  battle = null;
}

function opponentChoice() {
  const key = new URLSearchParams(location.search).get("opponent");
  if (key && OPPONENTS[key] && ui.opponent.value !== key) ui.opponent.value = key;
  return OPPONENTS[ui.opponent.value] || Object.values(OPPONENTS)[0];
}

function renderOpponent() {
  const opp = opponentChoice();
  ui.badgeR.textContent = opp.name;
  ui.badgeR.className = `badge ${opp.badge}`;
  ui.modelR.textContent = opp.label;
  document.title = `Jev vs ${opp.name}`;
}

async function startBattle() {
  const opp = opponentChoice();
  hideError();
  const seed = Number(ui.seed.value) || 42;
  const gravityMs = Number(ui.gravity.value);
  const speedup = SPEEDUPS[ui.speedup.value] || SPEEDUPS.normal;
  const lockstep = ui.lockstep.checked;
  const garbage = ui.garbage.checked;
  const limitMs = Math.max(1, Number(ui.limit.value) || 5) * 60_000;
  // Log every Jev decision (all candidates + its probability over each) to
  // data/battles/<run>.jsonl via /api/log. One run id per battle; ply counts
  // Jev's moves. Fire-and-forget so logging never stalls the real-time loop.
  const run = `jev-better-seed${seed}-${new Date().toISOString().replace(/[:.]/g, "-")}`;
  let ply = 0;
  const onDecision = (rows) => {
    const stamped = rows.map((r) => ({ ...r, run, state_id: `${run}:${ply}` }));
    ply += 1;
    fetch("api/log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run, rows: stamped }),
    }).catch(() => {});
  };
  // jev-better-prompt is always the left contestant and the only logged side.
  sides = [
    makeSide("L", createJevPlayer({ betterPrompt: true, onDecision }), sideEls("L")),
    makeSide("R", opp.create(), sideEls("R")),
  ];
  sides[0].opponent = sides[1];
  sides[1].opponent = sides[0];
  for (const s of sides) resetSide(s, seed);
  ui.result.classList.add("hidden");
  ui.start.disabled = true;
  ui.stop.disabled = false;
  ui.matchInfo.textContent =
    (lockstep ? `seed ${seed} · lockstep, no gravity` : `seed ${seed} · ${gravityMs} ms per row, ${describeSpeedup(ui.speedup.value)}`) +
    (garbage ? " · cleared lines attack" : "");
  if (PRESENT) {
    document.body.classList.add("running");
    for (const n of [3, 2, 1]) {
      ui.clock.textContent = String(n);
      await sleep(800);
    }
  }
  const abort = new AbortController();
  battle = { ramp: createRamp({ gravityMs, speedup }), abort, timer: null, limitMs, lockstep, garbage };
  renderLevel();
  battle.timer = setInterval(() => {
    if (!battle) return;
    const elapsed = battle.ramp.elapsed();
    ui.clock.textContent = formatClock(elapsed);
    renderLevel();
    if (elapsed >= battle.limitMs) checkEnd(true);
  }, 250);
  const ctx = { signal: abort.signal, lockstep, gravityNow, garbage, onTopOut: topOut, onPiece: () => checkEnd(), showError };
  for (const s of sides) runModelSide(s, ctx);
}

function stopBattle() {
  if (!battle) return;
  checkEnd(true);
}

// ---- Wiring -----------------------------------------------------------------------

ui.start.addEventListener("click", startBattle);
ui.stop.addEventListener("click", stopBattle);
ui.opponent.addEventListener("change", () => {
  renderOpponent();
  if (!battle) sides[1].player = { name: opponentChoice().name, short: opponentChoice().short };
});
ui.gravity.addEventListener("input", () => {
  ui.gravityLabel.textContent = `${ui.gravity.value} ms per row`;
  if (!battle) renderLevel();
});
ui.gravityLabel.textContent = `${ui.gravity.value} ms per row`;
renderLevel();

renderOpponent();
sides = [makeSide("L", { name: "jev-better-prompt", short: "Jev" }, sideEls("L")), makeSide("R", { name: opponentChoice().name, short: opponentChoice().short }, sideEls("R"))];
for (const s of sides) previewSide(s, Number(ui.seed.value) || 42);
