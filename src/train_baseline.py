"""Fine-tune a COCO-pretrained YOLOv8 on the 4-class aerial dataset and report
precision, recall, per-class AP50 / AP50-95 and mAP on the val and test splits.

Run from the project root:
    python src/train_baseline.py --model-size s --epochs 100
    python src/train_baseline.py --arch yolo11 --model-size s
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def report(model, data, split, imgsz, out_dir, device=None, plots=True):
    m = model.val(data=data, split=split, imgsz=imgsz, plots=plots, device=device,
                  project=str(out_dir), name=f"eval_{split}", exist_ok=True)
    res = {
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
        "per_class": {
            m.names[c]: {"AP50": float(m.box.ap50[i]), "AP50-95": float(m.box.ap[i])}
            for i, c in enumerate(m.box.ap_class_index)
        },
    }
    print(f"\n[{split}] P={res['precision']:.3f} R={res['recall']:.3f} "
          f"mAP50={res['mAP50']:.3f} mAP50-95={res['mAP50-95']:.3f}")
    for name, ap in res["per_class"].items():
        print(f"  {name:15s} AP50={ap['AP50']:.3f} AP50-95={ap['AP50-95']:.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/dataset.yaml")
    ap.add_argument("--arch", default="yolov8", choices=["yolov8", "yolo11"],
                    help="detector family (a second family is used to test patch transferability)")
    ap.add_argument("--model-size", default="s", choices=list("nsmlx"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from its last.pt (original settings are kept)")
    args = ap.parse_args()

    out_dir = Path("runs/baseline").resolve()
    name = f"{args.arch}{args.model_size}"
    if args.resume:
        YOLO(out_dir / name / "weights" / "last.pt").train(resume=True)
    else:
        model = YOLO(f"{name}.pt")
        model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    workers=args.workers, patience=25, seed=0, deterministic=True,
                    project=str(out_dir), name=name, exist_ok=True)

    best = YOLO(out_dir / name / "weights" / "best.pt")
    results = {s: report(best, args.data, s, args.imgsz, out_dir / name) for s in ("val", "test")}
    (out_dir / name / "baseline_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved metrics to {out_dir / name / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
