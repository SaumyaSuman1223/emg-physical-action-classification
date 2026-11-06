import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from emg_common import (CHANNEL_NAMES, FIG_DIR, N_SEG, RES_DIR, build_dataset, class_names,
                        feature_names, load_all_signals)


def main(records=None):
    records = records or load_all_signals()
    X, _, _ = build_dataset(records, N_SEG)
    names = class_names(records)
    print(f"Feature matrix: {X.shape}  (Nf = 8 channels x {N_SEG} segments x 2 features = {X.shape[1]})")

    np.save(RES_DIR / "features.npy", X)
    savemat(RES_DIR / "features.mat", {"X": X, "feature_names": np.array(feature_names(N_SEG), dtype=object)})

    # feature vectors of two classes
    _, y, _ = build_dataset(records, N_SEG)
    show = [(names.index(nm) + 1, nm) for nm in ("Jumping", "Punching")]
    nf = X.shape[1]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, (k, nm), col in zip(axes, show, ["#1f5fa8", "#b8432f"]):
        Z = np.log10(np.abs(X[y == k]) + 1)
        for row in Z:
            ax.plot(row, color=col, alpha=0.25, lw=0.7)
        ax.plot(Z.mean(0), color="k", lw=1.4, label="class mean")
        for c in range(1, 8):
            ax.axvline(c * 2 * N_SEG - 0.5, color="grey", lw=0.6, ls="--")
        ax.set_title(f"Class {k}: {nm} (60 trials)", fontsize=10)
        ax.set_ylabel(r"$\log_{10}(|f|+1)$")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    centers = [c * 2 * N_SEG + N_SEG - 0.5 for c in range(8)]
    axes[-1].set_xticks(centers)
    axes[-1].set_xticklabels(CHANNEL_NAMES)
    axes[-1].set_xlim(-0.5, nf - 0.5)
    axes[-1].set_xlabel(f"Feature index grouped by channel (each block: {N_SEG} means then {N_SEG} variances)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task2_feature_vectors.png", dpi=200)
    plt.close(fig)

    var_idx = np.array([[c * 2 * N_SEG + N_SEG + s for s in range(N_SEG)] for c in range(8)])
    M = np.array([[np.log10(X[y == k][:, var_idx[c]].mean()) for c in range(8)] for k in range(1, 21)])
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(8))
    ax.set_xticklabels(CHANNEL_NAMES, rotation=45, fontsize=8)
    ax.set_yticks(range(20))
    ax.set_yticklabels([f"{k + 1}. {n}" for k, n in enumerate(names)], fontsize=8)
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ mean segment variance")
    ax.set_title("Mean EMG power per class and channel", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task2_class_channel_power.png", dpi=200)
    plt.close(fig)
    return X


if __name__ == "__main__":
    main()
