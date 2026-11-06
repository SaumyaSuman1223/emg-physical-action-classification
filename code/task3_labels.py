import numpy as np
from scipy.io import savemat

from emg_common import ACTIVITY_TYPES, N_TRIALS, RES_DIR, action_files, subject_dirs


def main():
    y_all = []
    for s in subject_dirs():
        y_normal, y_aggr = [], []
        offset = 0
        for act, target in zip(ACTIVITY_TYPES, (y_normal, y_aggr)):
            n_files = len(action_files(s, act))
            # one label per trial
            for k in range(1, n_files + 1):
                target += [offset + k] * N_TRIALS
            offset += n_files
        y_sub = np.concatenate([y_normal, y_aggr])
        print(f"{s.name}: {len(y_normal)} normal + {len(y_aggr)} aggressive labels, classes {y_sub.min()}..{y_sub.max()}")
        y_all.append(y_sub)
    y = np.concatenate(y_all).astype(int)
    print(f"Target vector: {y.shape}, counts per class: {np.bincount(y)[1:]}")
    np.save(RES_DIR / "labels.npy", y)
    savemat(RES_DIR / "labels.mat", {"y": y})
    return y


if __name__ == "__main__":
    main()
