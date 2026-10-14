#!/usr/bin/env bash
# Runs after the seed-0 batch: more seeds + order controls, 4 jobs at a time.
cd "$(dirname "$0")/.."
while pgrep -f "run_experiments.py --seed 0 --threads 1 --conditions (grown|adult_scratch|curriculum_only|growth_only) " >/dev/null; do sleep 30; done
{
  echo "0 shuffled_order"; echo "0 reversed_immaturity"
  for s in 1 2; do for c in grown curriculum_only adult_scratch growth_only; do echo "$s $c"; done; done
} | xargs -P 4 -L 1 bash -c 'python3 scripts/run_experiments.py --seed $0 --threads 1 --conditions $1 --out results/$1 > results/logs/seed$0_$1.log 2>&1'
echo ROUND2_DONE
