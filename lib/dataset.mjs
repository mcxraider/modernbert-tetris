// Disk format: { state_id: { candidate_id: { prompt, probability, ... } } }.
import { mkdir, readFile, writeFile, rename, rm } from "node:fs/promises";
import { dirname } from "node:path";
import { randomUUID } from "node:crypto";

export function groupStateRows(rows) {
  if (!Array.isArray(rows)) throw new TypeError("Dataset rows must be an array");
  const states = Object.create(null);
  for (const row of rows) {
    if (!row || typeof row !== "object" || Array.isArray(row)) throw new TypeError("Invalid candidate row");
    const { state_id, candidate_id, ...candidate } = row;
    if (typeof state_id !== "string" || !state_id || typeof candidate_id !== "string" || !candidate_id) {
      throw new TypeError("Every candidate requires a nonempty state_id and candidate_id");
    }
    if (!Object.hasOwn(states, state_id)) states[state_id] = Object.create(null);
    if (Object.hasOwn(states[state_id], candidate_id)) throw new TypeError("Duplicate state/candidate ID");
    states[state_id][candidate_id] = candidate;
  }
  return states;
}

// ponytail: one process-wide queue and whole-file rewrites; use SQLite when
// multiple writers/processes or large logs require concurrent incremental writes.
let pendingWrite = Promise.resolve();
export function saveStateRows(path, rows) {
  const incoming = groupStateRows(rows);
  const operation = pendingWrite.then(async () => {
    let states = {};
    try {
      states = JSON.parse(await readFile(path, "utf8"));
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
    if (Array.isArray(states)) states = groupStateRows(states); // Existing flat datasets.
    if (!states || typeof states !== "object") throw new TypeError("Invalid saved dataset");
    for (const [stateId, candidates] of Object.entries(incoming)) {
      if (!Object.hasOwn(states, stateId)) {
        Object.defineProperty(states, stateId, { value: {}, enumerable: true });
      }
      const saved = states[stateId];
      if (!saved || typeof saved !== "object" || Array.isArray(saved)) throw new TypeError("Invalid saved state");
      for (const [id, candidate] of Object.entries(candidates)) {
        if (Object.hasOwn(saved, id)) {
          if (JSON.stringify(saved[id]) !== JSON.stringify(candidate)) {
            throw new TypeError(`Conflicting candidate ${stateId}/${id}; refusing to overwrite`);
          }
        } else {
          Object.defineProperty(saved, id, { value: candidate, enumerable: true });
        }
      }
    }
    await mkdir(dirname(path), { recursive: true });
    const temporary = `${path}.${randomUUID()}.tmp`;
    try {
      await writeFile(temporary, JSON.stringify(states, null, 4) + "\n", { flag: "wx" });
      await rename(temporary, path);
    } finally {
      await rm(temporary, { force: true });
    }
  });
  pendingWrite = operation.catch(() => {}); // A failed request must not block later writes.
  return operation;
}
