import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emg_common import (ACTIVITY_TYPES, CHANNEL_NAMES, DATA_ROOT, FIG_DIR, RES_DIR,
                        action_files, load_signal, subject_dirs)


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    # task 1.1: number of folders
    subs = subject_dirs()
    out(f"Dataset location : {DATA_ROOT}")
    out(f"Number of subject folders: {len(subs)} -> {[s.name for s in subs]}")

    # task 1.2: list files
    for s in subs:
        for act in ACTIVITY_TYPES:
            names = [f.name for f in action_files(s, act)]
            out(f"{s.name}/{act} ({len(names)} files): {', '.join(names)}")

    # task 1.3: array sizes
    out()
    example = action_files(subs[0], "Normal")[0]
    out(f"Size of a single array ({subs[0].name}/Normal/{example.name}): {load_signal(example).shape}")
    out()
    out(f"{'subject':8s} {'activity':11s} {'action':14s} {'rows':>6s} {'cols':>4s}")
    for s in subs:
        for act in ACTIVITY_TYPES:
            for f in action_files(s, act):
                r, c = load_signal(f).shape
                out(f"{s.name:8s} {act:11s} {f.stem:14s} {r:6d} {c:4d}")

    (RES_DIR / "task1_listing.txt").write_text("\n".join(lines) + "\n")

    # task 1.4: plot all channels
    picks = [("Normal", "Waving"), ("Aggressive", "Punching")]
    fig, axes = plt.subplots(8, 2, figsize=(10, 11), sharex="col")
    for col, (act, name) in enumerate(picks):
        sig = load_signal(subs[0] / act / "txt" / f"{name}.txt")
        for ch in range(8):
            ax = axes[ch, col]
            ax.plot(sig[:, ch], lw=0.4, color="#1f5fa8" if col == 0 else "#b8432f")
            ax.set_ylabel(CHANNEL_NAMES[ch], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
        axes[0, col].set_title(f"{subs[0].name} - {act}: {name}", fontsize=10)
        axes[-1, col].set_xlabel("Sample index", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task1_signals.png", dpi=200)
    plt.close(fig)
    print(f"saved {FIG_DIR / 'task1_signals.png'}")


if __name__ == "__main__":
    main()
