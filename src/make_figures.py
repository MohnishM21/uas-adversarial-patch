"""Slide figures from finished runs (CPU only). Writes PNGs to figures/.

    python src/make_figures.py

1. size_vs_damage.png    - mAP50 vs patch size: clean reference, random noise, adversarial (val | test)
2. per_class.png         - per-class AP50, clean / random noise / adversarial at the main patch size
2b. transfer.png        - main patch against the model it was trained on and two unseen detectors
2c. robustness.png      - patch under controlled conditions (rotation, light, resolution, JPEG, perspective)
3. patch_evolution.png   - the patch at selected training epochs
4. before_after.png      - detections on clean vs patched test images
"""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUNS = Path("runs/attack")
OUT = Path("figures")
MAIN = "patch_s30"
SWEEP = [(0.1, "sweep_s01"), (0.2, "sweep_s02"), (0.3, MAIN), (0.4, "sweep_s04")]

# Validated reference palette (light): blue = adversarial, orange = random noise; clean is a neutral reference.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ADV, NOISE, CLEAN = "#2a78d6", "#eb6834", "#a8a69f"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid.axis": "y",
})


def metrics(run):
    return json.loads((RUNS / run / "attack_metrics.json").read_text())


def size_vs_damage():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    cover = [100 * s ** 2 for s, _ in SWEEP]
    for ax, split in zip(axes, ("val", "test")):
        res = [metrics(r) for _, r in SWEEP]
        clean = res[0]["clean"][split]["mAP50"]
        noise = [r["random_noise"][split]["mAP50"] for r in res]
        adv = [r["adversarial"][split]["mAP50"] for r in res]
        ax.axhline(clean, color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
        ax.text(cover[0] - 0.6, clean + 0.018, f"clean, no patch ({clean:.2f})", color=INK2, va="bottom",
                fontsize=11)
        for ys, col, label in ((noise, NOISE, "random noise"), (adv, ADV, "adversarial")):
            ax.plot(cover, ys, color=col, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3, label=label)
            ax.text(cover[-1] + 0.4, ys[-1], f"{label} {ys[-1]:.2f}", color=INK2, va="center", fontsize=11)
        ax.set_title(f"{split.capitalize()} set ({16} images)", loc="left")
        ax.set_xticks(cover, [f"{c:.0f}%" for c in cover])
        ax.set_xlim(0, cover[-1] + 7)
        ax.set_ylim(0, 0.75)
        ax.set_xlabel("Share of each object covered by the patch")
    axes[0].set_ylabel("mAP50")
    axes[0].legend(loc="lower left", frameon=False)
    fig.suptitle("Adversarial patch vs. random patch of the same size (YOLOv8s)", x=0.01, ha="left",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "size_vs_damage.png", dpi=200)
    plt.close(fig)


def per_class():
    res = metrics(MAIN)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    conds = (("clean", CLEAN, "clean"), ("random_noise", NOISE, "random noise"), ("adversarial", ADV, "adversarial"))
    for ax, split in zip(axes, ("val", "test")):
        names = list(res["clean"][split]["per_class"])
        x = np.arange(len(names))
        w = 0.26
        for i, (key, col, label) in enumerate(conds):
            vals = [res[key][split]["per_class"][n]["AP50"] for n in names]
            bars = ax.bar(x + (i - 1) * w, vals, w - 0.02, color=col, label=label, zorder=2)
            if key == "adversarial":
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center",
                            fontsize=10, color=INK2)
        ax.set_xticks(x, [n.replace("_", " ") for n in names])
        ax.set_title(f"{split.capitalize()} set", loc="left")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("AP50")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Per-class AP50 with the patch covering ~9% of each object", x=0.01, ha="left",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "per_class.png", dpi=200)
    plt.close(fig)


def transfer(targets=((MAIN, "YOLOv8s\ntrained on"), ("transfer_yolov8n", "YOLOv8n\nunseen"),
                      ("transfer_yolo11s", "YOLO11s\nunseen, new arch."))):
    targets = [(r, n) for r, n in targets if (RUNS / r / "attack_metrics.json").exists()]
    if len(targets) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    conds = (("clean", CLEAN, "clean"), ("random_noise", NOISE, "random noise"), ("adversarial", ADV, "adversarial"))
    for ax, split in zip(axes, ("val", "test")):
        x = np.arange(len(targets))
        w = 0.26
        for i, (key, col, lab) in enumerate(conds):
            vals = [metrics(r)[key][split]["mAP50"] for r, _ in targets]
            bars = ax.bar(x + (i - 1) * w, vals, w - 0.02, color=col, label=lab, zorder=2)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=10, color=INK2)
        ax.set_xticks(x, [n for _, n in targets], fontsize=11)
        ax.set_title(f"{split.capitalize()} set", loc="left")
        ax.set_ylim(0, 0.85)
    axes[0].set_ylabel("mAP50")
    fig.suptitle("Transferability: one patch (trained on YOLOv8s) against three detectors", x=0.01, ha="left",
                 fontsize=15, fontweight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=3, frameon=False,
               bbox_to_anchor=(0.99, 0.95))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "transfer.png", dpi=200)
    plt.close(fig)


ENS = "#184f95"  # ensemble patch: a darker step of the adversarial blue, same hue family


def transfer_ensemble():
    """Single-model patch vs ensemble patch on the three detectors (val | test)."""
    rows = (("patch_s30", "patch_ens", "YOLOv8s\nseen by both"),
            ("transfer_yolo11s", "ens_on_yolo11s", "YOLO11s\nseen by ensemble"),
            ("transfer_yolov8n", "ens_on_yolov8n", "YOLOv8n\nunseen by both"))
    if not all((RUNS / r / "attack_metrics.json").exists() for row in rows for r in row[:2]):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, split in zip(axes, ("val", "test")):
        x = np.arange(len(rows))
        w = 0.2
        series = (
            ("clean", CLEAN, [metrics(s)["clean"][split]["mAP50"] for s, _, _ in rows]),
            ("random noise", NOISE, [metrics(s)["random_noise"][split]["mAP50"] for s, _, _ in rows]),
            ("single patch (YOLOv8s)", ADV, [metrics(s)["adversarial"][split]["mAP50"] for s, _, _ in rows]),
            ("ensemble patch (v8s + 11s)", ENS, [metrics(e)["adversarial"][split]["mAP50"] for _, e, _ in rows]),
        )
        for i, (lab, col, vals) in enumerate(series):
            bars = ax.bar(x + (i - 1.5) * w, vals, w - 0.02, color=col, label=lab, zorder=2)
            if i >= 2:
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, color=INK2)
        ax.set_xticks(x, [n for _, _, n in rows], fontsize=11)
        ax.set_title(f"{split.capitalize()} set", loc="left")
        ax.set_ylim(0, 0.85)
    axes[0].set_ylabel("mAP50")
    fig.suptitle("Transfer: single-model patch vs. ensemble patch", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=4, frameon=False,
               bbox_to_anchor=(0.995, 0.95), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "transfer_ensemble.png", dpi=200)
    plt.close(fig)


ROBUST_PANELS = (
    ("rotation_deg", "Patch rotation", lambda v: f"{v}°"),
    ("brightness", "Scene brightness", lambda v: f"×{v}"),
    ("resolution", "Resolution (altitude)", lambda v: f"{round(float(v) * 100)}%"),
    ("jpeg_quality", "JPEG quality", str),
    ("perspective", "Perspective tilt", str),
)


def robustness(name="yolov8s"):
    path = Path("runs/robustness") / name / "robustness_metrics.json"
    if not path.exists():
        return
    res = json.loads(path.read_text())
    panels = [p for p in ROBUST_PANELS if p[0] in res]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 4.2), sharey=True)
    conds = (("clean", CLEAN, "clean"), ("random_noise", NOISE, "random noise"), ("adversarial", ADV, "adversarial"))
    for ax, (factor, title, fmt) in zip(np.atleast_1d(axes), panels):
        values = list(res[factor])
        x = np.arange(len(values))
        for key, col, lab in conds:
            ys = [res[factor][v][key]["mAP50"] for v in values]
            ls = (0, (4, 3)) if key == "clean" else "-"
            ax.plot(x, ys, color=col, lw=2, ls=ls, marker="o", ms=7, mec=SURFACE, mew=1.5, label=lab, zorder=3)
        ax.set_xticks(x, [fmt(v) for v in values], fontsize=10)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylim(0, 0.8)
    np.atleast_1d(axes)[0].set_ylabel("mAP50 (val + test, 32 images)")
    fig.suptitle("Robustness: one condition varied at a time (patch covers ~9% of each object, YOLOv8s)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="upper right", ncol=3, frameon=False,
               bbox_to_anchor=(0.995, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "robustness.png", dpi=200)
    plt.close(fig)


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (251, 252, 252), -1)
    cv2.putText(img, text, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (11, 11, 11), 2, cv2.LINE_AA)
    return img


def patch_evolution(epochs=(1, 3, 7, 14, 30)):
    tiles = []
    for e in epochs:
        p = RUNS / MAIN / "patch_history" / f"epoch_{e:03d}.png"
        if p.exists():
            tiles.append(label(cv2.resize(cv2.imread(str(p)), (360, 360)), f"epoch {e}"))
    gap = np.full((360, 12, 3), 252, np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.hstack([row, gap, t])
    cv2.imwrite(str(OUT / "patch_evolution.png"), row)


def before_after(stems=("aircraft-26", "building-24")):
    rows = []
    for stem in stems:
        clean = next((RUNS / MAIN / "examples_clean").glob(f"{stem}-*"), None)
        adv = next((RUNS / MAIN / "examples_adversarial").glob(f"{stem}-*"), None)
        if not (clean and adv):
            continue
        a = label(cv2.resize(cv2.imread(str(clean)), (560, 560)), "clean")
        b = label(cv2.resize(cv2.imread(str(adv)), (560, 560)), "with adversarial patch")
        rows.append(np.hstack([a, np.full((560, 12, 3), 252, np.uint8), b]))
    gap = np.full((12, rows[0].shape[1], 3), 252, np.uint8)
    grid = rows[0]
    for r in rows[1:]:
        grid = np.vstack([grid, gap, r])
    cv2.imwrite(str(OUT / "before_after.png"), grid)


def main():
    OUT.mkdir(exist_ok=True)
    size_vs_damage()
    per_class()
    transfer()
    transfer_ensemble()
    robustness()
    patch_evolution()
    before_after()
    print("Wrote:", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
