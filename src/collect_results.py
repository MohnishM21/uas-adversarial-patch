"""Copy the small result files (metrics, logs, learned patches) from runs/ into results/, which is
published with the code (runs/ itself holds large weights and patched copies of the dataset).

    python src/collect_results.py
"""
import shutil
from pathlib import Path

RUNS, OUT = Path("runs"), Path("results")
KEEP = ("attack_metrics.json", "args.json", "train_log.json", "patch.png", "baseline_metrics.json",
        "robustness_metrics.json", "results.csv")

if OUT.exists():
    shutil.rmtree(OUT)
n = 0
for f in RUNS.rglob("*"):
    if f.is_file() and f.name in KEEP and "eval_" not in str(f.parent) and "work" not in f.parts:
        dst = OUT / f.relative_to(RUNS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        n += 1
for f in (RUNS / "attack" / "summary.csv",):
    if f.exists():
        shutil.copy2(f, OUT / "attack" / f.name)
        n += 1
for f in sorted((RUNS / "attack" / "patch_s30" / "patch_history").glob("epoch_*.png")):
    if f.stem in ("epoch_001", "epoch_003", "epoch_007", "epoch_014", "epoch_030"):
        dst = OUT / "attack" / "patch_s30" / "patch_history" / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        n += 1
print(f"copied {n} files into {OUT}/")
