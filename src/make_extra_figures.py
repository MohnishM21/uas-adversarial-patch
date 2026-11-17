"""Figures and tables for the placement study and the total-variation ablation.

Run from the project root after src/placement_study.py and the tv_* runs:
    python src/make_extra_figures.py
Writes figures/placement.png, figures/tv_ablation.png and prints LaTeX-ready numbers.
"""
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path("figures")
GRID = [-0.3, -0.15, 0.0, 0.15, 0.3]
TVS = ["0", "1", "2.5", "5", "10"]
ADV, NOISE, CLEAN = "#1C5CAB", "#B4471B", "#7a7a7a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def placement():
    path = Path("runs/placement/yolov8s/placement_metrics.json")
    if not path.exists():
        print("no placement results yet")
        return
    r = json.loads(path.read_text())
    clean = r["clean"]["mAP50"]
    adv = np.array([[r[f"grid_{dx:+.2f}_{dy:+.2f}"]["adversarial"]["mAP50"] for dx in GRID] for dy in GRID])
    noi = np.array([[r[f"grid_{dx:+.2f}_{dy:+.2f}"]["random_noise"]["mAP50"] for dx in GRID] for dy in GRID])
    gap = noi - adv  # adversarial effect beyond occlusion

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, data, title, cmap, dark_high in [(axes[0], adv, "Adversarial mAP@0.5", "Blues_r", False),
                                             (axes[1], gap, "Noise $-$ adversarial (pattern effect)", "Oranges", True)]:
        im = ax.imshow(data, cmap=cmap, origin="upper")
        ax.set_xticks(range(5), [f"{v:+.2f}" for v in GRID])
        ax.set_yticks(range(5), [f"{v:+.2f}" for v in GRID])
        ax.set_xlabel(r"$\delta_x$ (box widths)")
        ax.set_ylabel(r"$\delta_y$ (box heights; $-$ = up)")
        ax.set_title(title)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if ((data[i, j] - data.min()) / (np.ptp(data) + 1e-9) > 0.55) == dark_high
                        else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    ring = lambda d: [k for k in r if k.startswith("grid_") and
                      round(max(abs(r[k]["dx"]), abs(r[k]["dy"])), 2) == d]
    groups = [("Centre", ["grid_+0.00_+0.00"]), ("Ring\n0.15", ring(0.15)), ("Ring\n0.30", ring(0.3)),
              ("Random\non-object", ["random"]), ("Detector\npeak", ["peak"]),
              ("Beside\nobject", [k for k in r if k.startswith("beside_")])]
    x = np.arange(len(groups))
    a = [np.mean([r[k]["adversarial"]["mAP50"] for k in ks]) for _, ks in groups]
    n = [np.mean([r[k]["random_noise"]["mAP50"] for k in ks]) for _, ks in groups]
    ax.bar(x - 0.2, n, 0.4, color=NOISE, label="Random noise")
    ax.bar(x + 0.2, a, 0.4, color=ADV, label="Adversarial")
    for xi, v in zip(x, a):
        ax.text(xi + 0.2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhline(clean, ls="--", color=CLEAN, lw=1)
    ax.text(len(groups) - 0.5, clean + 0.01, f"clean {clean:.2f}", ha="right", fontsize=7, color=CLEAN)
    ax.set_xticks(x, [g for g, _ in groups], fontsize=7)
    ax.set_ylabel("mAP@0.5 (val+test, 32 images)")
    ax.set_title("Placement strategies")
    ax.set_ylim(0, 0.68)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "placement.png", dpi=200)
    plt.close(fig)

    print(f"\n=== Placement (clean {clean:.3f}) : group  noise / adversarial / retained ===")
    for g, ks in groups:
        an = np.mean([r[k]["adversarial"]["mAP50"] for k in ks])
        nn = np.mean([r[k]["random_noise"]["mAP50"] for k in ks])
        lo = min(r[k]["adversarial"]["mAP50"] for k in ks)
        hi = max(r[k]["adversarial"]["mAP50"] for k in ks)
        print(f"  {g.replace(chr(10), ' '):18s} {nn:.3f} / {an:.3f} (range {lo:.3f}-{hi:.3f}) / {an / clean:.2f}")
    print("grid adversarial (rows dy=-0.3..0.3, cols dx=-0.3..0.3):\n", np.round(adv, 3))
    print("grid noise:\n", np.round(noi, 3))
    print("peak offsets:", r.get("_peak_offsets"))
    for k in sorted(k for k in r if k.startswith("beside_")):
        print(f"  {k}: noise {r[k]['random_noise']['mAP50']:.3f} adv {r[k]['adversarial']['mAP50']:.3f}")


def tv_runs():
    rows = []
    for tv in TVS:
        name = "patch_s30" if tv == "2.5" else f"tv_{tv}"
        d = Path("runs/attack") / name
        rob = Path("runs/robustness") / ("yolov8s" if tv == "2.5" else name) / "robustness_metrics.json"
        if not (d / "attack_metrics.json").exists():
            continue
        m = json.loads((d / "attack_metrics.json").read_text())
        log = json.loads((d / "train_log.json").read_text())
        row = {"tv": float(tv), "label": tv, "patch": d / "patch.png",
               "val": m["adversarial"]["val"]["mAP50"], "test": m["adversarial"]["test"]["mAP50"],
               "tv_final": log[-1]["tv_loss"], "det_final": log[-1]["det_loss"]}
        if rob.exists():
            rr = json.loads(rob.read_text())
            for f, v in [("resolution", "0.25"), ("resolution", "0.5"), ("jpeg_quality", "10"), ("jpeg_quality", "25")]:
                c = rr[f][v]
                row[f"{f}_{v}"] = c["adversarial"]["mAP50"] / c["clean"]["mAP50"]
        rows.append(row)
    if not rows:
        print("no TV results yet")
        return rows

    fig = plt.figure(figsize=(10.5, 2.6))
    gs = fig.add_gridspec(1, len(rows) + 1, width_ratios=[1] * len(rows) + [2.6])
    for i, row in enumerate(rows):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(cv2.cvtColor(cv2.imread(str(row["patch"])), cv2.COLOR_BGR2RGB))
        ax.set_title(rf"$\lambda_{{TV}}={row['label']}$" + f"\nTV={row['tv_final']:.3f}", fontsize=8)
        ax.axis("off")
    ax = fig.add_subplot(gs[0, len(rows)])
    x = np.arange(len(rows))
    ax.plot(x, [r["val"] for r in rows], "o-", color=ADV, label="val mAP (full res.)")
    ax.plot(x, [r["test"] for r in rows], "s-", color="#4c8bd6", label="test mAP (full res.)")
    if all("resolution_0.25" in r for r in rows):
        ax.plot(x, [r["resolution_0.25"] for r in rows], "^--", color=NOISE, label="retained @ 25% res.")
        ax.plot(x, [r["jpeg_quality_10"] for r in rows], "v--", color="#d98a2b", label="retained @ JPEG 10")
    ax.set_xticks(x, [r["label"] for r in rows])
    ax.set_xlabel(r"$\lambda_{TV}$")
    ax.set_ylabel("lower = stronger attack")
    ax.set_ylim(0.1, 0.56)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIG / "tv_ablation.png", dpi=200)
    plt.close(fig)

    print("\n=== TV ablation ===")
    for r in rows:
        extra = " ".join(f"{k}={r[k]:.2f}" for k in r if k.startswith(("resolution_", "jpeg_")))
        print(f"  lambda={r['label']:>4s} TV={r['tv_final']:.4f} det={r['det_final']:.3f} "
              f"val={r['val']:.3f} test={r['test']:.3f} {extra}")
    return rows


if __name__ == "__main__":
    placement()
    tv_runs()
