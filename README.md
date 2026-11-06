# EMG Physical Action Classification

Classification of 10 normal and 10 aggressive physical actions from 8-channel surface EMG
(UCI EMG Physical Action dataset) using segment mean/variance features and a kNN classifier.

## Setup
1. Download the dataset from https://archive.ics.uci.edu/dataset/213 and extract it so the
   folders are at `data/EMG Physical Action Data Set/sub1 ... sub4`.
2. Run:
```bash
./run.sh
```

## Files
- `code/emg_common.py` - loading, trial splitting, feature extraction
- `code/task1_read_plot.py` - task 1: list files, array sizes, signal plots
- `code/task2_features.py` - task 2: features and feature plots
- `code/task3_labels.py` - task 3: class labels
- `code/task4_classification.py` - task 4: binary and multi-class kNN analysis
- `code/run_all.py` - runs all tasks

Figures are written to `figures/` and numeric results to `results/`.
