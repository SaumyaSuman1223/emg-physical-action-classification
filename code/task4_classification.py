import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold, cross_val_predict,
                                     cross_val_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emg_common import (FIG_DIR, K_NEIGHBOURS, N_SEG, RANDOM_SEED, RES_DIR, build_dataset,
                        class_names, load_all_signals)

N_FOLDS = 10
N_MC = 10
M_RANGE = range(2, 20)


def knn(k=K_NEIGHBOURS):
    # scaler is fitted inside each cv fold
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))


def cv_accuracy(X, y, k=K_NEIGHBOURS, seed=RANDOM_SEED):
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    return cross_val_score(knn(k), X, y, cv=cv)


def binary_analysis(X, y, names, results):
    # one example pair
    a, b = names.index("Waving") + 1, names.index("Punching") + 1
    mask = np.isin(y, [a, b])
    Xp, yp = X[mask], y[mask]
    scores = cv_accuracy(Xp, yp)
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cm = confusion_matrix(yp, cross_val_predict(knn(), Xp, yp, cv=cv), labels=[a, b])
    print(f"Binary {names[a-1]} vs {names[b-1]}: data {Xp.shape}, "
          f"10-fold CV acc = {scores.mean():.3f} +/- {scores.std():.3f}")
    results["binary_example"] = dict(classes=[names[a - 1], names[b - 1]], shape=list(Xp.shape),
                                     fold_acc=scores.tolist(), mean=scores.mean(), std=scores.std(),
                                     confusion=cm.tolist())

    # all 190 pairs
    P = np.full((20, 20), np.nan)
    for i, j in itertools.combinations(range(1, 21), 2):
        m = np.isin(y, [i, j])
        P[i - 1, j - 1] = P[j - 1, i - 1] = cv_accuracy(X[m], y[m]).mean()
    pair_acc = P[np.triu_indices(20, 1)]
    worst = np.unravel_index(np.nanargmin(P), P.shape)
    print(f"All 190 pairs: mean {pair_acc.mean():.3f}, median {np.median(pair_acc):.3f}, "
          f"min {pair_acc.min():.3f} ({names[worst[0]]} vs {names[worst[1]]}), max {pair_acc.max():.3f}")
    results["binary_all_pairs"] = dict(mean=pair_acc.mean(), median=float(np.median(pair_acc)),
                                       std=pair_acc.std(), min=pair_acc.min(), max=pair_acc.max(),
                                       worst_pair=[names[worst[0]], names[worst[1]]],
                                       frac_above_90=float((pair_acc >= 0.9).mean()),
                                       frac_below_75=float((pair_acc < 0.75).mean()))
    np.save(RES_DIR / "pairwise_accuracy.npy", P)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    im = axes[0].imshow(P, cmap="RdYlGn", vmin=0.5, vmax=1.0)
    axes[0].set_xticks(range(20))
    axes[0].set_yticks(range(20))
    axes[0].set_xticklabels(names, rotation=90, fontsize=7)
    axes[0].set_yticklabels(names, fontsize=7)
    axes[0].axhline(9.5, color="k", lw=0.8)
    axes[0].axvline(9.5, color="k", lw=0.8)
    axes[0].set_title("Pairwise 10-fold CV accuracy (kNN)", fontsize=10)
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].hist(pair_acc, bins=np.linspace(0.5, 1.0, 21), color="#3b6ea5", edgecolor="white")
    axes[1].axvline(pair_acc.mean(), color="#b8432f", ls="--", label=f"mean = {pair_acc.mean():.3f}")
    axes[1].set_xlabel("CV accuracy")
    axes[1].set_ylabel("Number of class pairs")
    axes[1].set_title("Distribution over all 190 pairs", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task4_binary_pairs.png", dpi=200)
    plt.close(fig)


def multiclass_analysis(X, y, results):
    rng = np.random.default_rng(RANDOM_SEED)
    labels = np.arange(1, 21)
    acc = np.zeros((len(M_RANGE), N_MC))
    chosen = {}
    # monte-carlo runs for each number of classes
    for r, M in enumerate(M_RANGE):
        combos = list(itertools.combinations(labels, M))
        picks = rng.integers(0, len(combos), size=N_MC)
        chosen[M] = [list(map(int, combos[p])) for p in picks]
        for mc in range(N_MC):
            sel = np.array(combos[picks[mc]])
            mask = np.isin(y, sel)
            acc[r, mc] = cv_accuracy(X[mask], y[mask], seed=RANDOM_SEED + mc).mean()
        print(f"M = {M:2d}: {len(combos):6d} combos, mean acc {acc[r].mean():.3f} +/- {acc[r].std():.3f}")
    full = cv_accuracy(X, y).mean()
    print(f"M = 20 (all classes): acc {full:.3f}")
    np.save(RES_DIR / "mc_accuracy.npy", acc)
    (RES_DIR / "mc_combos.json").write_text(json.dumps(chosen, indent=1))

    mean, std = acc.mean(1), acc.std(1)
    Ms = np.array(list(M_RANGE))
    drop = -np.diff(mean)
    results["multiclass"] = dict(M=Ms.tolist(), mean=mean.tolist(), std=std.tolist(),
                                 min=acc.min(1).tolist(), max=acc.max(1).tolist(), acc20=full,
                                 largest_drop_step=[int(Ms[np.argmax(drop)]), int(Ms[np.argmax(drop)] + 1)],
                                 drop_per_class=drop.tolist())

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(acc, cmap="viridis", aspect="auto", vmin=acc.min(), vmax=1)
    ax.set_yticks(range(len(Ms)))
    ax.set_yticklabels(Ms)
    ax.set_xticks(range(N_MC))
    ax.set_xticklabels(range(1, N_MC + 1))
    ax.set_xlabel("Monte-Carlo run (random class combination)")
    ax.set_ylabel("Number of classes M")
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            ax.text(j, i, f"{acc[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if acc[i, j] < 0.75 else "black")
    fig.colorbar(im, ax=ax, label="10-fold CV accuracy")
    ax.set_title("kNN accuracy vs number of classes and MC run", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task4_mc_heatmap.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.fill_between(Ms, acc.min(1), acc.max(1), color="#3b6ea5", alpha=0.15, label="min-max over MC runs")
    ax.errorbar(Ms, mean, yerr=std, fmt="o-", color="#3b6ea5", capsize=3, label="mean ± std")
    ax.plot(20, full, "s", color="#b8432f", label=f"all 20 classes ({full:.2f})")
    ax.plot(Ms, 1 / Ms, ":", color="grey", label="chance level 1/M")
    ax.set_xticks(list(Ms) + [20])
    ax.set_xlabel("Number of classes M")
    ax.set_ylabel("Mean 10-fold CV accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Mean kNN accuracy vs number of classes", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task4_accuracy_vs_classes.png", dpi=200)
    plt.close(fig)


def parameter_study(records, results):
    segs, ks = [1, 2, 3, 5, 8, 10, 15, 20], [1, 3, 5, 7, 9]
    grid = np.zeros((len(segs), len(ks)))
    for i, ns in enumerate(segs):
        X, y, _ = build_dataset(records, ns)
        for j, k in enumerate(ks):
            grid[i, j] = cv_accuracy(X, y, k=k).mean()
    print("N_seg x k grid (20-class acc):\n", np.round(grid, 3))
    results["param_study"] = dict(n_seg=segs, k=ks, acc=grid.tolist())
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for j, k in enumerate(ks):
        ax.plot(segs, grid[:, j], "o-", label=f"k = {k}")
    ax.set_xlabel(r"Number of segments per trial $N_{seg}$")
    ax.set_ylabel("20-class 10-fold CV accuracy")
    ax.set_xticks(segs)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(r"Effect of $N_{seg}$ and $k$ on 20-class accuracy", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task4_param_study.png", dpi=200)
    plt.close(fig)


def full_problem_diagnostics(X, y, g, names, results):
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    pred = cross_val_predict(knn(), X, y, cv=cv)
    cm = confusion_matrix(y, pred, labels=range(1, 21))
    per_class = cm.diagonal() / cm.sum(1)
    group_acc = np.mean((pred > 10) == (y > 10))
    results["per_class_acc"] = dict(zip(names, per_class.tolist()))
    results["group_acc_from_20class"] = group_acc
    print(f"20-class: Normal/Aggressive confusion-level accuracy {group_acc:.3f}")

    fig, ax = plt.subplots(figsize=(6.8, 6))
    cmn = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(20))
    ax.set_yticks(range(20))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.axhline(9.5, color="k", lw=0.6)
    ax.axvline(9.5, color="k", lw=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised rate")
    ax.set_title(f"20-class confusion matrix (accuracy {np.mean(pred == y):.2f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task4_confusion_20.png", dpi=200)
    plt.close(fig)

    # leave-one-subject-out
    loso = cross_val_score(knn(), X, y, groups=g, cv=LeaveOneGroupOut())
    print(f"Leave-one-subject-out 20-class accuracy per held-out subject: {np.round(loso, 3)}")
    results["loso"] = loso.tolist()
    results["per_subject"] = [cv_accuracy(X[g == s], y[g == s]).mean() for s in range(1, 5)]
    print(f"Per-subject 20-class CV accuracy: {np.round(results['per_subject'], 3)}")


def main(records=None):
    records = records or load_all_signals()
    X, y, g = build_dataset(records, N_SEG)
    y_saved = RES_DIR / "labels.npy"
    if y_saved.exists():
        assert np.array_equal(np.load(y_saved), y), "Task 3 labels and feature order disagree"
    names = class_names(records)
    print(f"Data: X {X.shape}, y {y.shape}  (Nf = 2 features x {N_SEG} segments x 8 channels)")
    results = dict(n_seg=N_SEG, k=K_NEIGHBOURS, n_folds=N_FOLDS, n_mc=N_MC, X_shape=list(X.shape))
    binary_analysis(X, y, names, results)
    multiclass_analysis(X, y, results)
    parameter_study(records, results)
    full_problem_diagnostics(X, y, g, names, results)
    (RES_DIR / "task4_results.json").write_text(json.dumps(results, indent=2, default=float))
    return results


if __name__ == "__main__":
    main()
