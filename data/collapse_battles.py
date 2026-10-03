import json, uuid, pathlib

battles_dir = pathlib.Path(__file__).parent / "battles"
out_path = pathlib.Path(__file__).parent / "battles.jsonl"

with out_path.open("w") as out:
    for f in sorted(battles_dir.glob("*.json")):
        data = json.loads(f.read_text())
        for game in data.values():
            for player in game.values():
                out.write(json.dumps({
                    "id": str(uuid.uuid4()),
                    "prompt": player["prompt"],
                    "probability": player["probability"],
                }) + "\n")

print(f"Written to {out_path}")
