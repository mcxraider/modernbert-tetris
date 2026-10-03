#!/usr/bin/env bash
# 3 batches of a 5-run jev-better-prompt solo, each capped at 120s wall-clock.
# No `timeout` on macOS, so: run in background, sleep, kill the tree.
# SEED_BASE gives each worktree a disjoint block of 15 seeds; each batch takes
# the next 5, so all 60 seeds across 4 worktrees are unique — no repeats.
set -u
base=${SEED_BASE:?set SEED_BASE}
for i in 1 2 3; do
  start=$(( base + (i - 1) * 5 ))
  end=$(( start + 4 ))
  echo "=== batch $i (pwd: $(basename "$PWD")) seeds ${start}-${end} ==="
  node --env-file=.env tools/solo-batch.mjs \
    --runs 5 --mode jev-better-prompt --delay 250 --seeds "${start}-${end}" &
  pid=$!
  sleep 120
  kill -INT "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  echo "=== batch $i done ==="
done
