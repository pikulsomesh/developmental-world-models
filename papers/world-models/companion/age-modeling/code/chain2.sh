cd "$(dirname "$0")"
while [ ! -f results_E2.json ]; do sleep 20; done
E3B=1 python3 run_av.py > e3b.log 2>&1
