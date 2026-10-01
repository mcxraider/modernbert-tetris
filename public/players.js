// The two contestants of the battle page. Each player gets identical
// information (the board plus the same described placement options) and
// must return one option id. Both go through the local proxy in server.mjs.
//
// Jev answers a TypeSafe Choice; the opponent is a Claude/GPT/Gemini model on
// the GovTech Platform AI gateway (OpenAI chat-completions), chosen from the
// dropdown. Neither needs a key in the page — the server holds both keys.

import { describePlacement, boardToText, describeHeight, describeHoles, describeSurface } from "./tetris.js";
import { buildRequest, askJev, pickPlacement } from "./jev.js";
import { stateRows } from "./prompt.js";

export const JEV_PRICE = { input: 0.042 / 1e6, output: 0 };

// ---- Jev ----------------------------------------------------------------------

// `onDecision(rows)` (optional): called with the dataset rows for this move —
// one per candidate, with Jev's probability over each. Used by the battle page
// to log every decision; ignored when absent.
export function createJevPlayer({ onDecision, betterPrompt = false } = {}) {
  return {
    name: betterPrompt ? "jev-better-prompt" : "Jev",
    short: betterPrompt ? "jev-better-prompt" : "Jev",
    model: "jev-latest",
    async decide(gameInfo, placements, signal) {
      const request = buildRequest(gameInfo, placements, { withBoardAfter: betterPrompt });
      // The battle only needs the placement Choice; drop the extra questions.
      request.questions = { placement: request.questions.placement };
      // No key argument: the server fills in TYPESAFE_API_KEY from its env.
      const { response, latencyMs } = await askJev(request, "", { signal, maxAttempts: 2 });
      const { chosen, confidence } = pickPlacement(response, placements);
      if (onDecision) {
        const probs = response?.answers?.placement?.probabilities || {};
        onDecision(
          stateRows({
            board: gameInfo.board,
            piece: gameInfo.piece,
            nextPiece: gameInfo.nextPiece,
            lines: gameInfo.linesCleared,
            placements,
            probs,
            choiceId: response?.answers?.placement?.choice ?? chosen.id,
          }),
        );
      }
      const usage = response.usage || {};
      return {
        chosen,
        latencyMs,
        inputTokens: usage.input_tokens || 0,
        outputTokens: usage.output_tokens || 0,
        cost: (usage.input_tokens || 0) * JEV_PRICE.input,
        note: `confidence ${confidence.toFixed(2)}`,
      };
    },
  };
}

// ---- Platform opponent (GovTech Platform AI, OpenAI-compatible) ------------------
// Same board text and described option list for every opponent; the answer is a
// forced function call whose option_id is an enum of the offered placements.
// Routed through the server's /api/platform proxy which holds PLATFORM_API_KEY.

const PLATFORM_SYSTEM = [
  "You are playing Tetris in real time. Each turn you get the board and a list of every legal placement for the current piece, each described by its outcome.",
  "Pick the best placement. Good play: clear lines (more at once is better), never create holes unless every option does, keep the stack low and the surface flat, avoid several deep wells.",
  "The piece is falling while you think, so decide immediately by calling the place_piece tool.",
].join(" ");

export function buildPlatformPrompt(gameInfo, placements) {
  const { board, piece, nextPiece, stats, linesCleared } = gameInfo;
  const options = {};
  for (const p of placements) options[p.id] = describePlacement(p);
  return JSON.stringify(
    {
      board_rows_top_to_bottom: boardToText(board),
      legend: "# filled, . empty",
      column_heights_left_to_right: stats.heights,
      stack_height: describeHeight(stats.maxHeight),
      holes_in_stack: describeHoles(stats.holes),
      surface: describeSurface(stats.bumpiness),
      current_piece: piece,
      next_piece: nextPiece,
      lines_cleared_so_far: linesCleared,
      options,
    },
    null,
    0,
  );
}

export function buildPlatformTool(placements) {
  return {
    type: "function",
    function: {
      name: "place_piece",
      description: "Choose where to drop the current piece by naming one option id from the options list.",
      parameters: {
        type: "object",
        properties: { option_id: { type: "string", enum: placements.map((p) => p.id) } },
        required: ["option_id"],
      },
    },
  };
}

export function parsePlatformChoice(json, placements) {
  const byId = new Map(placements.map((p) => [p.id, p]));
  const message = json?.choices?.[0]?.message || {};
  const call = (message.tool_calls || []).find((c) => c.function?.name === "place_piece");
  if (call) {
    try {
      const id = JSON.parse(call.function.arguments || "{}").option_id;
      if (byId.has(id)) return byId.get(id);
    } catch {
      /* fall through to text scan */
    }
  }
  // Fallback for a plain-text reply naming an id.
  for (const m of String(message.content || "").match(/\bp\d+\b/g) || []) if (byId.has(m)) return byId.get(m);
  return null;
}

// Rough rate for the cost column; the gateway bills centrally so this is only a
// display estimate (Haiku 4.5 $1/$5 per MTok). ponytail: single flat rate for
// every model, split per-model if the cost readout ever needs to be exact.
const PLATFORM_PRICE = { input: 1 / 1e6, output: 5 / 1e6 };

export function createPlatformPlayer({ name, short, model, endpoint = "api/platform" }) {
  return {
    name,
    short,
    model,
    async decide(gameInfo, placements, signal) {
      const body = {
        model,
        max_tokens: 64,
        messages: [
          { role: "system", content: PLATFORM_SYSTEM },
          { role: "user", content: buildPlatformPrompt(gameInfo, placements) },
        ],
        tools: [buildPlatformTool(placements)],
        tool_choice: { type: "function", function: { name: "place_piece" } },
      };
      const started = performance.now();
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
      });
      const latencyMs = performance.now() - started;
      const text = await res.text();
      let json = null;
      try {
        json = JSON.parse(text);
      } catch {
        json = null;
      }
      if (!res.ok) {
        const message = json?.error?.message || text || `HTTP ${res.status}`;
        const err = new Error(message);
        err.status = res.status;
        throw err;
      }
      const chosen = parsePlatformChoice(json, placements);
      const answer = chosen ? chosen.id : (json?.choices?.[0]?.message?.content || "").slice(0, 60);
      const usage = json.usage || {};
      const inputTokens = usage.prompt_tokens || 0;
      const outputTokens = usage.completion_tokens || 0;
      return {
        chosen,
        latencyMs,
        inputTokens,
        outputTokens,
        cost: inputTokens * PLATFORM_PRICE.input + outputTokens * PLATFORM_PRICE.output,
        note: chosen ? `picked ${answer}` : `invalid reply ${answer}`,
      };
    },
  };
}

// ---- Opponent table for the battle and presentation pages -------------------------------
// Every opponent is a model on the GovTech Platform gateway. `create()` takes no
// key — the server holds PLATFORM_API_KEY. Keep this list in sync with the
// <select id="opponent"> options in index.html / present.html.
function platformOpponent(model, name, short, badge) {
  return { name, short, badge, label: `${model} · GovTech Platform`, create: () => createPlatformPlayer({ name, short, model }) };
}

export const OPPONENTS = {
  "jev-better-prompt": {
    name: "jev-better-prompt",
    short: "jev-better-prompt",
    badge: "jev",
    label: "Jev · better prompt (board_after)",
    create: () => createJevPlayer({ betterPrompt: true }),
  },
  "haiku-4-5": platformOpponent("bedrock.claude-haiku-4-5", "Claude Haiku 4.5", "Haiku 4.5", "haiku"),
  "gpt-6-luna": platformOpponent("bedrock.gpt-6-luna", "GPT-6 Luna", "GPT-6 Luna", "gpt"),
  "gpt-5-6-sol": platformOpponent("bedrock.gpt-5.6-sol", "GPT-5.6 Sol", "GPT-5.6 Sol", "gpt"),
  "sonnet-5-5": platformOpponent("bedrock.claude-sonnet-5-5", "Claude Sonnet 5.5", "Sonnet 5.5", "haiku"),
};
