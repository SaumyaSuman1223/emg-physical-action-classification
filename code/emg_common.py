from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "EMG Physical Action Data Set"
FIG_DIR = PROJECT_ROOT / "figures"
RES_DIR = PROJECT_ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

ACTIVITY_TYPES = ["Normal", "Aggressive"]
CHANNEL_NAMES = ["R-Bic", "R-Tri", "L-Bic", "L-Tri", "R-Thi", "R-Ham", "L-Thi", "L-Ham"]
N_CHANNELS = 8
N_TRIALS = 15
N_SEG = 2
K_NEIGHBOURS = 1
RANDOM_SEED = 42


def subject_dirs():
    return sorted([p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("sub")],
                  key=lambda p: int(p.name[3:]))


def action_files(subject_dir, activity):
    return sorted((subject_dir / activity / "txt").glob("*.txt"))


def load_signal(path):
    sig = np.loadtxt(path)
    assert sig.ndim == 2 and sig.shape[1] == N_CHANNELS, f"unexpected shape {sig.shape} in {path}"
    return sig


def split_trials(signal, n_trials=N_TRIALS):
    # equal-length trials, leftover samples dropped
    trial_len = signal.shape[0] // n_trials
    return signal[: trial_len * n_trials].reshape(n_trials, trial_len, signal.shape[1])


def trial_features(trial, n_seg=N_SEG):
    # per channel: n_seg means followed by n_seg variances
    seg_len = trial.shape[0] // n_seg
    segs = trial[: seg_len * n_seg].reshape(n_seg, seg_len, trial.shape[1])
    means = segs.mean(axis=1)
    variances = segs.var(axis=1)
    per_channel = np.concatenate([means.T, variances.T], axis=1)
    return per_channel.reshape(-1)


def feature_names(n_seg=N_SEG):
    names = []
    for ch in CHANNEL_NAMES:
        names += [f"{ch}_mean_s{s + 1}" for s in range(n_seg)]
        names += [f"{ch}_var_s{s + 1}" for s in range(n_seg)]
    return names


def load_all_signals():
    # labels 1..20 from the file count: normal 1..10, aggressive 11..20
    records = []
    for sdir in subject_dirs():
        label = 0
        for activity in ACTIVITY_TYPES:
            for f in action_files(sdir, activity):
                label += 1
                records.append(dict(subject=sdir.name, activity=activity, action=f.stem,
                                    label=label, signal=load_signal(f)))
    return records


def build_dataset(records, n_seg=N_SEG):
    X, y, g = [], [], []
    for r in records:
        for trial in split_trials(r["signal"]):
            X.append(trial_features(trial, n_seg))
            y.append(r["label"])
            g.append(int(r["subject"][3:]))
    return np.asarray(X), np.asarray(y), np.asarray(g)


def class_names(records):
    names = {}
    for r in records:
        names.setdefault(r["label"], r["action"])
    return [names[k] for k in sorted(names)]
