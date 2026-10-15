#!/usr/bin/env bash
# Resumable overnight pipeline: every step skips work whose output already exists, so it can be
# restarted after an interruption. Progress markers go to results/logs/pipeline.log.
cd "$(dirname "$0")/.."
mkdir -p results/logs
log() { echo "$(date +%H:%M:%S) $*" >> results/logs/pipeline.log; }
run_grid() {  # args: seeds conditions
  for s in $1; do for c in $2; do echo "$s $c"; done; done \
    | xargs -P 4 -L 1 bash -c '[ -f results/$1/seed$0/metrics.json ] || nice -n 10 python3 scripts/run_experiments.py --seed $0 --threads 1 --conditions $1 --out results/$1 > results/logs/seed$0_$1.log 2>&1'
}
log start
THREADS=1 nice -n 5 python3 scripts/analyze_extra.py --parts own_age > results/logs/extra_ownage.log 2>&1 &
run_grid "1 2" "shuffled_order reversed_immaturity"; log CONTROLS_DONE
wait; THREADS=2 python3 scripts/analyze_extra.py --parts own_age,stats >> results/logs/extra_ownage.log 2>&1; log N3_ANALYSES_DONE
run_grid "3 4" "curriculum_only adult_scratch grown growth_only"; log MORE_SEEDS_DONE
THREADS=2 python3 scripts/analyze_extra.py --parts own_age,stats >> results/logs/extra_ownage.log 2>&1
THREADS=4 python3 scripts/analyze_fusion_voe.py --seeds 0,1,2,3,4 --results results > results/logs/fusion_voe.log 2>&1; log ALL_DONE
