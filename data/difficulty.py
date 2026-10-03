"""Per-state curriculum difficulty for the distillation dataset.

Reads data/battles.jsonl (one state per line, with its candidate prompts) and
writes data/difficulty.jsonl with two orthogonal axes per state:

  S  structural difficulty  -- how messy/dangerous the board is, plus how bad
                               the teacher's *best* available move is forced to be
  A  decision ambiguity     -- normalized teacher entropy over the offered set

All signals are parsed straight from the prompt text; the categorical bins are
produced by describeHeight/Holes/Surface/HeightChange in public/tetris.js and are
exactly the model's input vocabulary, so we score on them rather than
re-deriving floats from the board grid.

States are bucketed into 5 stages by the empirical quantiles of S (the dataset
is heavily skewed toward low stacks, so fixed cutpoints would collapse into one
stage). A is emitted as a separate column so a 2D curriculum (stage, then
ambiguity within stage) needs no recompute.
"""

import json, math, re, statistics, pathlib

# Ordinal maps for the fixed prompt vocabularies (public/tetris.js:273-313).
HEIGHT = {"very low": 0, "low": 1, "medium": 2, "high": 3,
          "dangerously high, close to the top": 4}
HOLES = {"none": 0, "one hole": 1, "two holes": 2, "three or more holes": 3}
SURF = {"flat": 0, "slightly uneven": 1, "bumpy": 2, "very jagged": 3}
GROW = {"stack gets lower": 0, "stack does not get taller": 0,
        "stack grows by one row": 1, "stack grows by several rows": 2}

# ponytail: calibration knob -- the "right" weighting is a tuning decision, not
# a fact. Bump these if a term should dominate the curriculum ordering.
W = {"height": 1.0, "holes": 1.0, "surface": 1.0, "forced": 1.0}

N_STAGES = 5


def _field(prompt, name):
    """First `name: value` line in the prompt (state fields precede CANDIDATE)."""
    m = re.search(r"^" + re.escape(name) + r": (.+)$", prompt, re.M)
    if m is None:
        raise ValueError(f"missing field {name!r} in prompt")
    return m.group(1).strip()


def _cand_fields(prompt):
    """The CANDIDATE block's fields for one candidate prompt."""
    body = prompt.split("CANDIDATE", 1)[1]
    return {k: _field(body, k) for k in
            ("holes_created", "height_change", "stack_height_after")}


def structural_difficulty(state):
    """S for one state record from battles.jsonl."""
    best = max(state["candidates"], key=lambda c: c["probability"])
    p = best["prompt"]
    f = _cand_fields(p)
    forced = HOLES[f["holes_created"]] + GROW[f["height_change"]] + HEIGHT[f["stack_height_after"]]
    return (W["height"] * HEIGHT[_field(p, "stack_height")]
            + W["holes"] * HOLES[_field(p, "holes")]
            + W["surface"] * SURF[_field(p, "surface")]
            + W["forced"] * forced)


def ambiguity(state):
    """Normalized teacher entropy A in [0,1]; 0 for a single-candidate state."""
    ps = [c["probability"] for c in state["candidates"]]
    n = len(ps)
    if n <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in ps if p > 0)
    return h / math.log(n)


def stage_of(s, cuts):
    """1..N_STAGES by quantile cut points (len == N_STAGES-1, ascending)."""
    for i, c in enumerate(cuts):
        if s <= c:
            return i + 1
    return N_STAGES


def score_all(states):
    """Return per-state rows with S, A, and quantile-bucketed stage."""
    rows = [{"state_id": st["state_id"], "n_candidates": len(st["candidates"]),
             "S": structural_difficulty(st), "A": ambiguity(st)} for st in states]
    cuts = statistics.quantiles([r["S"] for r in rows], n=N_STAGES)
    for r in rows:
        r["stage"] = stage_of(r["S"], cuts)
    return rows, cuts


def main():
    here = pathlib.Path(__file__).parent
    states = [json.loads(line) for line in (here / "battles.jsonl").open()]
    rows, cuts = score_all(states)

    out = here / "difficulty.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    hist = {s: 0 for s in range(1, N_STAGES + 1)}
    for r in rows:
        hist[r["stage"]] += 1
    singles = sum(1 for r in rows if r["n_candidates"] == 1)

    print(f"Written {len(rows)} states to {out}")
    print("S quantile cuts:", [round(c, 3) for c in cuts])
    print("stage histogram:", hist)
    print(f"single-candidate states (A=0, ~0 KD loss): {singles} "
          f"({100 * singles / len(rows):.1f}%) -- consider excluding from KD batches")


def _selftest():
    def state(cands):  # cands: list of (probability, state_overrides, cand_overrides)
        sdef = dict(stack_height="very low", holes="none", surface="flat")
        cdef = dict(holes_created="none", height_change="stack does not get taller",
                    stack_height_after="very low")
        out = []
        for prob, so, co in cands:
            s, c = {**sdef, **so}, {**cdef, **co}
            prompt = (f"STATE\nstack_height: {s['stack_height']}\nholes: {s['holes']}\n"
                      f"surface: {s['surface']}\nCANDIDATE\nholes_created: {c['holes_created']}\n"
                      f"height_change: {c['height_change']}\n"
                      f"stack_height_after: {c['stack_height_after']}\n")
            out.append({"prompt": prompt, "probability": prob})
        return {"state_id": "t", "candidates": out}

    # best = max-probability candidate; a messy board + forced-bad best move scores high.
    hard = state([
        (0.9, dict(stack_height="high", holes="three or more holes", surface="bumpy"),
         dict(holes_created="one hole", height_change="stack grows by one row",
              stack_height_after="high")),
        (0.1, {}, {}),  # ignored for S: not the best move
    ])
    assert structural_difficulty(hard) == 3 + 3 + 2 + (1 + 1 + 3), structural_difficulty(hard)
    easy = state([(1.0, {}, {})])
    assert structural_difficulty(easy) == 0

    # ambiguity: 0 for n==1, 1 for an even 2-way split, analytic for a lopsided one.
    assert ambiguity(state([(1.0, {}, {})])) == 0.0
    assert abs(ambiguity(state([(0.5, {}, {}), (0.5, {}, {})])) - 1.0) < 1e-9
    a = ambiguity(state([(0.94, {}, {}), (0.06, {}, {})]))
    assert 0.0 < a < 1.0
    expected = -(0.94 * math.log(0.94) + 0.06 * math.log(0.06)) / math.log(2)
    assert abs(a - expected) < 1e-9, (a, expected)

    # staging partitions every state into exactly one of 1..N_STAGES.
    many = [state([(1.0, dict(stack_height=h), {})]) for h in
            ["very low", "low", "medium", "high", "dangerously high, close to the top"]]
    rows, cuts = score_all(many)
    assert len(cuts) == N_STAGES - 1
    assert all(1 <= r["stage"] <= N_STAGES for r in rows)
    assert sum(1 for r in rows) == len(many)
    print("selftest ok")


if __name__ == "__main__":
    _selftest()
    main()
