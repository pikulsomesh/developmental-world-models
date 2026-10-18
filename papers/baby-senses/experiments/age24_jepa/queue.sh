#!/bin/bash
# Runs every job of run.py, three at a time (the fourth core runs vision_vep seed 0 and the ceilings).
cd "$(dirname "$0")"
mkdir -p logs
jobs_list="vision_jepa_vr:0 vision_jepa_vr:1 vision_jepa_vr:2 vision_look:0 vision_look:1 vision_look:2 vision_vep:1 vision_vep:2 vision_jepa:0 vision_jepa:1 vision_jepa:2 audio:0 audio:1 audio:2"
for js in $jobs_list; do
  j=${js%:*}; s=${js#*:}
  while [ $(jobs -r | wc -l) -ge 3 ]; do sleep 20; done
  python3 run.py --job $j --seed $s > logs/${j}_$s.log 2>&1 &
done
wait
