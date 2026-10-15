#!/usr/bin/env bash
# Runs the main experiment grid, 4 single-threaded jobs at a time (one per core).
cd "$(dirname "$0")/.."
mkdir -p results/logs
{
  for c in grown curriculum_only adult_scratch growth_only; do echo "0 $c"; done
  echo "0 shuffled_order"; echo "0 reversed_immaturity"
  for c in grown curriculum_only adult_scratch growth_only; do echo "1 $c"; done
  for c in grown curriculum_only adult_scratch growth_only; do echo "2 $c"; done
} | xargs -P 4 -L 1 bash -c 'python3 scripts/run_experiments.py --seed $0 --threads 1 --conditions $1 --out results/$1 > results/logs/seed$0_$1.log 2>&1'
echo QUEUE_DONE
