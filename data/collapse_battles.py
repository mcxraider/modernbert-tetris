import json, pathlib

battles_dir = pathlib.Path(__file__).parent / "battles"
out_path = pathlib.Path(__file__).parent / "battles.jsonl"

with out_path.open("w") as out:
    for f in sorted(battles_dir.glob("*.json")):
        data = json.loads(f.read_text())
        for state_id, candidates in data.items():
            out.write(json.dumps({
                "state_id": state_id,
                "candidates": [
                    {"prompt": c["prompt"], "probability": c["probability"]}
                    for c in candidates.values()
                ],
            }) + "\n")

print(f"Written to {out_path}")
