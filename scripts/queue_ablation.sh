#!/usr/bin/env bash
# Vision front-end ablations: only one component of immaturity follows age. Resumable.
cd "$(dirname "$0")/.."
mkdir -p results/logs
for s in 0 1 2; do for c in blur_only contrast_only; do echo "$s $c"; done; done \
  | xargs -P 6 -L 1 bash -c '[ -f results/$1/seed$0/metrics.json ] || nice -n 10 python3 scripts/run_experiments.py --seed $0 --threads 1 --senses vision --conditions $1 --out results/$1 > results/logs/seed$0_$1.log 2>&1'
echo "$(date +%H:%M:%S) ABLATIONS_DONE" >> results/logs/pipeline.log
