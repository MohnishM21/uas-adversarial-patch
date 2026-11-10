"""Tabulate every attack evaluation under runs/attack/ (size sweep, transfer runs, ...).

Run from the project root:
    python src/summarize.py            # prints the table, writes runs/attack/summary.csv
"""
import csv
import json
from pathlib import Path

CONDS = ("clean", "random_noise", "adversarial")


def main():
    rows = []
    for metrics in sorted(Path("runs/attack").glob("*/attack_metrics.json")):
        run = metrics.parent
        args = json.loads((run / "args.json").read_text())
        res = json.loads(metrics.read_text())
        row = {
            "run": run.name,
            "target_model": Path(w[0] if isinstance(w := args["weights"], list) else w).parent.parent.name,
            "patch": Path(args["patch"]).parent.name if args.get("patch") else run.name,
            "patch_scale": args["patch_scale"],
        }
        for split in ("val", "test"):
            for c in CONDS:
                row[f"{split}_{c}"] = round(res[c][split]["mAP50"], 3)
            clean, adv, noise = (res[c][split]["mAP50"] for c in ("clean", "adversarial", "random_noise"))
            row[f"{split}_adv_drop_%"] = round(100 * (1 - adv / clean), 1) if clean else None
            row[f"{split}_noise_drop_%"] = round(100 * (1 - noise / clean), 1) if clean else None
        rows.append(row)

    if not rows:
        print("No runs/attack/*/attack_metrics.json found.")
        return
    out = Path("runs/attack/summary.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'run':16s} {'target':10s} {'patch':12s} {'scale':>5s} | "
          f"{'val clean/noise/adv':>21s} {'drop%':>11s} | {'test clean/noise/adv':>21s} {'drop%':>11s}")
    for r in rows:
        print(f"{r['run']:16s} {r['target_model']:10s} {r['patch']:12s} {r['patch_scale']:5.2f} | "
              + " | ".join(f"{r[f'{s}_clean']:.3f}/{r[f'{s}_random_noise']:.3f}/{r[f'{s}_adversarial']:.3f}"
                           f"  {r[f'{s}_noise_drop_%']:4.0f}/{r[f'{s}_adv_drop_%']:4.0f}" for s in ("val", "test")))
    print(f"\ndrop% = noise/adversarial mAP50 drop relative to clean.  Saved {out}")


if __name__ == "__main__":
    main()
