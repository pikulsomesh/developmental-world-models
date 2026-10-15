#!/usr/bin/env bash
# Extra seeds for the order controls (shuffled input quality, reversed immaturity).
cd "$(dirname "$0")/.."
mkdir -p results/logs
{ for s in 1 2; do for c in shuffled_order reversed_immaturity; do echo "$s $c"; done; done; } \
  | xargs -P 4 -L 1 bash -c 'nice -n 10 python3 scripts/run_experiments.py --seed $0 --threads 1 --conditions $1 --out results/$1 > results/logs/seed$0_$1.log 2>&1'
echo CONTROLS_DONE
