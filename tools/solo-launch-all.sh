#!/usr/bin/env bash
# Launch all 4 worktree drivers durably with disjoint seed bases.
root=/Users/Jerry_YANG_from.TP/govtech/jev-tetris
driver="$root/tools/solo-driver.sh"

wt="$root/.claude/worktrees/solo-1"; ( cd "$wt" && SEED_BASE=2000 nohup bash "$driver" > "$wt/driver.log" 2>&1 & ); echo "launched solo-1 base 2000"
wt="$root/.claude/worktrees/solo-2"; ( cd "$wt" && SEED_BASE=2100 nohup bash "$driver" > "$wt/driver.log" 2>&1 & ); echo "launched solo-2 base 2100"
wt="$root/.claude/worktrees/solo-3"; ( cd "$wt" && SEED_BASE=2200 nohup bash "$driver" > "$wt/driver.log" 2>&1 & ); echo "launched solo-3 base 2200"
wt="$root/.claude/worktrees/solo-4"; ( cd "$wt" && SEED_BASE=2300 nohup bash "$driver" > "$wt/driver.log" 2>&1 & ); echo "launched solo-4 base 2300"
