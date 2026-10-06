// Headless A/B: play full solo Tetris games with the local ModernBERT worker,
// one model at a time, on the SAME seeds, and compare stats. No browser, no
// gravity, no API keys — talks straight to models/serve.py over stdio with the
// per-request `model` field ("base" | "checkpoint").
//
//   .venv must exist (torch). Run:
//     node tools/solo-modernbert-eval.mjs [--seeds 1-10] [--max-plies 300]
//
// Prints per-seed lines/pieces for each model and a summary (mean/median/total
// lines, win count on matched seeds). Models run sequentially, never together.

import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import {
  emptyBoard, boardStats, enumeratePlacements, lockPiece, clearLines,
  collides, makeBag, seededRandom, shortlistPlacements, PIECES,
} from "../public/tetris.js";
import { renderPrompt } from "../public/prompt.js";

const REPO_ROOT = join(fileURLToPath(new URL("..", import.meta.url)));
const VENV_PYTHON = join(REPO_ROOT, ".venv", "bin", "python");
const PYTHON = process.env.PYTHON || (existsSync(VENV_PYTHON) ? VENV_PYTHON : "python3");
const SPAWN_X = 3;

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}
function parseSeeds(spec) {
  const m = /^(\d+)-(\d+)$/.exec(spec);
  if (m) return Array.from({ length: +m[2] - +m[1] + 1 }, (_, i) => +m[1] + i);
  return spec.split(",").map(Number);
}

// One long-lived worker; one in-flight request at a time (serve.py is serial).
function startWorker() {
  const proc = spawn(PYTHON, [join(REPO_ROOT, "models", "serve.py")], {
    cwd: REPO_ROOT,
    stdio: ["pipe", "pipe", "inherit"], // python stderr → our console
  });
  const rl = createInterface({ input: proc.stdout });
  const lines = [];
  let waiter = null;
  rl.on("line", (line) => (waiter ? (waiter(line), (waiter = null)) : lines.push(line)));
  const next = () => new Promise((res) => (lines.length ? res(lines.shift()) : (waiter = res)));
  const send = (obj) => proc.stdin.write(JSON.stringify(obj) + "\n");
  return { proc, next, send };
}

// Ask the worker to score one state's candidates with the given model.
async function score(worker, { board, piece, nextPiece, lines }, offered, model) {
  const prompts = offered.map((p) => renderPrompt(board, piece, nextPiece, lines, p));
  worker.send({ prompts, candidate_ids: offered.map((p) => p.id), model });
  const reply = JSON.parse(await worker.next());
  if (reply.error) throw new Error(reply.error);
  return reply.chosen_candidate_id;
}

// Same 7-bag draw as lib/solo.mjs, so a seed reproduces the browser's sequence.
function newGame(seed) {
  const random = seededRandom(seed);
  let bag = [];
  const draw = () => ((bag.length === 0 && (bag = makeBag(random))), bag.pop());
  return { board: emptyBoard(), current: draw(), next: draw(), draw, lines: 0, pieces: 0 };
}

async function playGame(worker, seed, model, maxPlies) {
  const game = newGame(seed);
  for (let ply = 0; ply < maxPlies; ply++) {
    const spawn = PIECES[game.current][0];
    if (collides(game.board, spawn.cells, SPAWN_X, 0)) break;
    const placements = enumeratePlacements(game.board, game.current);
    if (placements.length === 0) break;
    const offered = shortlistPlacements(placements); // mirrors the solo page's ModernBERT path
    const gameInfo = { board: game.board, piece: game.current, nextPiece: game.next, lines: game.lines };
    const choiceId = await score(worker, gameInfo, offered, model);
    const chosen = offered.find((p) => p.id === choiceId) || offered[0];
    const { board, cleared } = clearLines(lockPiece(game.board, chosen.piece, chosen.rotation, chosen.x, chosen.y));
    game.board = board;
    game.lines += cleared;
    game.pieces += 1;
    game.current = game.next;
    game.next = game.draw();
  }
  return { seed, lines: game.lines, pieces: game.pieces };
}

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const median = (a) => { const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

async function runModel(worker, model, seeds, maxPlies) {
  console.log(`\n=== ${model} (${seeds.length} games, ≤${maxPlies} plies) ===`);
  const results = [];
  for (const seed of seeds) {
    const r = await playGame(worker, seed, model, maxPlies);
    results.push(r);
    console.log(`  seed ${String(seed).padStart(3)}: ${String(r.lines).padStart(3)} lines, ${String(r.pieces).padStart(3)} pieces`);
  }
  return results;
}

async function main() {
  const seeds = parseSeeds(arg("seeds", "1-10"));
  const maxPlies = Number(arg("max-plies", 300));
  const worker = startWorker();
  // serve.py's first stdout line is the readiness signal; wait for it.
  const ready = JSON.parse(await worker.next());
  if (!ready.ready) throw new Error(`worker failed to load: ${ready.error}`);
  console.log(`worker ready (base source=${ready.source})`);

  try {
    const base = await runModel(worker, "base", seeds, maxPlies);
    const ckpt = await runModel(worker, "checkpoint", seeds, maxPlies);

    const bl = base.map((r) => r.lines), cl = ckpt.map((r) => r.lines);
    let ckptWins = 0, baseWins = 0, ties = 0;
    for (let i = 0; i < seeds.length; i++) {
      if (cl[i] > bl[i]) ckptWins++; else if (cl[i] < bl[i]) baseWins++; else ties++;
    }
    console.log(`\n=== summary (same seeds) ===`);
    console.log(`                      base        checkpoint`);
    console.log(`  mean lines     ${mean(bl).toFixed(2).padStart(10)}  ${mean(cl).toFixed(2).padStart(10)}`);
    console.log(`  median lines   ${String(median(bl)).padStart(10)}  ${String(median(cl)).padStart(10)}`);
    console.log(`  total lines    ${String(bl.reduce((a, b) => a + b, 0)).padStart(10)}  ${String(cl.reduce((a, b) => a + b, 0)).padStart(10)}`);
    console.log(`  mean pieces    ${mean(base.map((r) => r.pieces)).toFixed(2).padStart(10)}  ${mean(ckpt.map((r) => r.pieces)).toFixed(2).padStart(10)}`);
    console.log(`\n  per-seed wins: checkpoint ${ckptWins}, base ${baseWins}, tie ${ties}`);
    const verdict = ckptWins > baseWins ? "checkpoint is better" : baseWins > ckptWins ? "base is better" : "tied";
    console.log(`  verdict: ${verdict} (by lines on matched seeds)`);
  } finally {
    worker.proc.kill();
  }
}

main().catch((err) => { console.error(err.message); process.exit(1); });
