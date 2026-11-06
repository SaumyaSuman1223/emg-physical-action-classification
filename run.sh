#!/usr/bin/env bash
# runs the full emg classification pipeline (tasks 1-4)
set -e
cd "$(dirname "$0")"

DATA_DIR="data/EMG Physical Action Data Set"
if [ ! -d "$DATA_DIR" ]; then
    echo "dataset not found at '$DATA_DIR'"
    echo "download it from https://archive.ics.uci.edu/dataset/213 and extract it there"
    exit 1
fi

pip install -q -r requirements.txt
cd code
python3 run_all.py
echo "done: figures in figures/, results in results/"
