#!/usr/bin/env bash
# Bring every regime to five seeds: order controls (all senses) and front-end ablations (vision). Resumable.
cd "$(dirname "$0")/.."
mkdir -p results/logs
{
  for s in 3 4; do for c in shuffled_order reversed_immaturity; do echo "$s $c all"; done; done
  for s in 2 3 4; do for c in blur_only contrast_only; do echo "$s $c vision"; done; done
} | xargs -P 4 -L 1 bash -c '[ -f results/$1/seed$0/metrics.json ] || nice -n 10 python3 scripts/run_experiments.py --seed $0 --threads 1 $( [ "$2" = vision ] && echo "--senses vision" ) --conditions $1 --out results/$1 > results/logs/seed$0_$1.log 2>&1'
echo "$(date +%H:%M:%S) FIVE_SEEDS_DONE" >> results/logs/pipeline.log
