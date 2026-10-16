cd "$(dirname "$0")"
python3 run_vision.py E1 > e1.log 2>&1
python3 run_av.py > e3.log 2>&1
python3 run_vision.py E2 > e2.log 2>&1
