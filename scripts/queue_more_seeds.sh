#!/usr/bin/env bash
# Seeds 3 and 4 for the four main regimes (n = 5), started after the control seeds finish.
cd "$(dirname "$0")/.."
mkdir -p results/logs
until grep -q CONTROLS_DONE results/logs/queue_controls.log 2>/dev/null; do sleep 60; done
{ for s in 3 4; do for c in curriculum_only adult_scratch grown growth_only; do echo "$s $c"; done; done; } \
  | xargs -P 4 -L 1 bash -c 'nice -n 10 python3 scripts/run_experiments.py --seed $0 --threads 1 --conditions $1 --out results/$1 > results/logs/seed$0_$1.log 2>&1'
echo MORE_SEEDS_DONE
