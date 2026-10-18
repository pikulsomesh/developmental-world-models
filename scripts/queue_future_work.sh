#!/bin/bash
# Runs the follow-up experiments of both papers, four processes at a time (one CPU thread each):
# paper 2's 24-month and latent runs for seeds 3-4, the main-study vision checkpoints for seeds 1-4
# (for the spectral analysis), and paper 1's three future-work experiments for seeds 0-4.
cd "$(dirname "$0")/.."
mkdir -p results/future_work/logs
P2=papers/baby-senses/experiments/age24_jepa
cmds=()
for s in 3 4; do cmds+=("python3 $P2/run.py --job vision_vep --seed $s" "python3 $P2/run.py --job vision_look --seed $s"); done
for s in 1 2 3 4; do for r in curriculum_only adult_scratch; do
  cmds+=("python3 scripts/run_experiments.py --senses vision --conditions $r --seed $s --out runs/$r --threads 1"); done; done
for s in 0 1 2 3 4; do for j in voe_gauss voe_jepa_vr matched_vision; do cmds+=("python3 scripts/run_future_work.py --job $j --seed $s"); done; done
for s in 3 4; do cmds+=("python3 $P2/run.py --job vision_jepa --seed $s" "python3 $P2/run.py --job audio --seed $s"); done
for s in 0 1 2 3 4; do cmds+=("python3 scripts/run_future_work.py --job matched_audio --seed $s"); done
i=0
for c in "${cmds[@]}"; do
  while [ $(jobs -r | wc -l) -ge 4 ]; do sleep 20; done
  log=results/future_work/logs/$(printf %02d $i)_$(echo "$c" | sed 's/.*--job \([a-z_]*\).*--seed \([0-9]\).*/\1_\2/;s/.*--conditions \([a-z_]*\) --seed \([0-9]\).*/main_\1_\2/').log
  echo "$(date +%H:%M:%S) start $c" >> results/future_work/logs/queue.log
  $c > $log 2>&1 &
  i=$((i+1))
done
wait
python3 scripts/run_future_work.py --job spectral --threads 4 > results/future_work/logs/spectral.log 2>&1
echo "$(date +%H:%M:%S) all done" >> results/future_work/logs/queue.log
