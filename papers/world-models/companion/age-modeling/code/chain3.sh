cd "$(dirname "$0")"
python3 run_vision.py E2b > e2b.log 2>&1
python3 run_vision.py E1x > e1x.log 2>&1
