import task1_read_plot
import task2_features
import task3_labels
import task4_classification
from emg_common import load_all_signals

if __name__ == "__main__":
    print("=== Task 1: read and plot ===")
    task1_read_plot.main()
    records = load_all_signals()
    print("\n=== Task 2: feature extraction ===")
    task2_features.main(records)
    print("\n=== Task 3: labels ===")
    task3_labels.main()
    print("\n=== Task 4: classification analysis ===")
    task4_classification.main(records)
