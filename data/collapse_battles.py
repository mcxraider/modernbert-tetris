import json, math, re, statistics, pathlib

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
    m = re.search(r"^" + re.escape(name) + r": (.+)$", prompt, re.M)
    if m is None:
        raise ValueError(f"missing field {name!r} in prompt")
    return m.group(1).strip()


def _cand_fields(prompt):
    body = prompt.split("CANDIDATE", 1)[1]
    return {k: _field(body, k) for k in
            ("holes_created", "height_change", "stack_height_after")}


def structural_difficulty(state):
    best = max(state["candidates"], key=lambda c: c["probability"])
    p = best["prompt"]
    f = _cand_fields(p)
    forced = HOLES[f["holes_created"]] + GROW[f["height_change"]] + HEIGHT[f["stack_height_after"]]
    return (W["height"] * HEIGHT[_field(p, "stack_height")]
            + W["holes"] * HOLES[_field(p, "holes")]
            + W["surface"] * SURF[_field(p, "surface")]
            + W["forced"] * forced)


def ambiguity(state):
    ps = [c["probability"] for c in state["candidates"]]
    n = len(ps)
    if n <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in ps if p > 0)
    return h / math.log(n)


def stage_of(s, cuts):
    for i, c in enumerate(cuts):
        if s <= c:
            return i + 1
    return N_STAGES


def export_states(battles_dir):
    """Preserve eligibility and episode provenance; never guess from positive mass."""
    states = []
    for f in sorted(battles_dir.glob("*.json")):
        for state_id, candidates in json.loads(f.read_text()).items():
            rows = []
            for candidate_id, candidate in candidates.items():
                row = dict(candidate, candidate_id=candidate_id)
                # Both solo entry points log only the offered list. Battle logs do
                # not; their missing flags must remain missing and fail validation.
                run = row.get("run", "")
                if "offered_to_jev" not in row and run.startswith("solo-jev") and run == f.stem:
                    row["offered_to_jev"] = True
                    row["eligibility_source"] = "legacy_solo_offered_only"
                rows.append(row)
            episodes = {r.get("episode_id") or r.get("run") for r in rows}
            if len(episodes) != 1 or not next(iter(episodes)):
                raise ValueError(f"{state_id}: missing or inconsistent episode metadata")
            states.append(dict(state_id=state_id, episode_id=episodes.pop(), candidates=rows))
    return states


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export Jev states with episode and eligibility metadata.")
    parser.add_argument("--input", type=pathlib.Path, default=pathlib.Path(__file__).parent / "battles")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path(__file__).parent / "battles-training.jsonl")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}; choose a new output path")
    states = export_states(args.input)
    if len(states) < N_STAGES:
        parser.error("Need at least five states to estimate difficulty cuts")
    s_vals = [structural_difficulty(st) for st in states]
    cuts = statistics.quantiles(s_vals, n=N_STAGES)
    hist = {s: 0 for s in range(1, N_STAGES + 1)}
    # Exclusive creation prevents reruns from replacing an existing export.
    with args.output.open("x") as out:
        for st, score in zip(states, s_vals):
            stage = stage_of(score, cuts)
            hist[stage] += 1
            out.write(json.dumps({**st, "n_candidates": len(st["candidates"]),
                                  "S": score, "A": ambiguity(st), "stage": stage}) + "\n")
    print(f"Written {len(states)} states to {args.output}")
    print("S quantile cuts:", [round(c, 3) for c in cuts])
    print("stage histogram:", hist)


if __name__ == "__main__":
    main()
