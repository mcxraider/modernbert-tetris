import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { groupStateRows, saveStateRows } from "../lib/dataset.mjs";

const row = (state, candidate, probability = 0.5) => ({
  state_id: state, candidate_id: candidate, prompt: "STATE\nboard\n\nCANDIDATE\nmove",
  probability, is_choice: false, run: "test",
});

test("grouped dataset preserves rows, merges concurrent decisions, and refuses data loss", async () => {
  const directory = await mkdtemp(join(tmpdir(), "jev-dataset-"));
  const path = join(directory, "run.json");
  try {
    const rows = [row("s1", "p1"), row("s1", "p2"), row("s2", "p1")];
    const grouped = groupStateRows(rows);
    assert.deepEqual(Object.keys(grouped), ["s1", "s2"]);
    assert.deepEqual(Object.keys(grouped.s1), ["p1", "p2"]);
    const restored = Object.entries(grouped).flatMap(([state_id, candidates]) =>
      Object.entries(candidates).map(([candidate_id, candidate]) => ({ ...candidate, state_id, candidate_id })));
    assert.deepEqual(restored, rows);
    assert.throws(() => groupStateRows([rows[0], rows[0]]), /Duplicate/);
    assert.throws(() => groupStateRows([{ candidate_id: "p1" }]), /state_id/);
    await writeFile(path, JSON.stringify([rows[0]])); // Legacy file upgrades on next write.
    await Promise.all([saveStateRows(path, [rows[1]]), saveStateRows(path, [rows[2]])]);
    const read = async () => JSON.parse(await readFile(path, "utf8"));
    assert.deepEqual(await read(), JSON.parse(JSON.stringify(grouped)));
    await saveStateRows(path, [rows[0]]); // Network retry is idempotent.
    const before = await readFile(path, "utf8");
    await assert.rejects(saveStateRows(path, [row("s1", "p1", 0.9)]), /Conflicting/);
    assert.equal(await readFile(path, "utf8"), before);
    await saveStateRows(path, [row("s3", "p1")]); // Queue recovers after failure.
    assert.equal(Object.keys(await read()).length, 3);
    await writeFile(path, "{broken JSON");
    await assert.rejects(saveStateRows(path, [rows[0]]));
    assert.equal(await readFile(path, "utf8"), "{broken JSON");
    // Arbitrary user keys cannot modify object prototypes.
    await rm(path);
    await saveStateRows(path, [row("__proto__", "constructor")]);
    assert.equal((await read()).__proto__.constructor.prompt, rows[0].prompt);
    assert.equal({}.prompt, undefined);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
