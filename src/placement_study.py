"""Where on the object should the patch go? Placement sweep for the trained patch.

Placement is quantified as the offset of the patch centre from the box centre, normalised by the
box size:  dx = (x_patch - x_box) / w,  dy = (y_patch - y_box) / h.  (0, 0) is the box centre
(the roof of a vehicle or the fuselage centre in near-nadir images); dy < 0 is towards the top of
the image. Everything else is the standard evaluation setting (side 0.3*sqrt(wh), fixed random
rotation per object, no lighting change), so only the position changes between conditions.

Placements evaluated (patch and random-noise control, val + test pooled = 32 images):
  grid      dx, dy in {-0.3, -0.15, 0, 0.15, 0.3}: 25 positions on the object
  beside    dx or dy = +-0.75: the patch lies next to the object, not on it (off-object)
  random    a uniformly random on-object offset per object (|dx|, |dy| <= 0.3): imprecise placement
  peak      detector-guided: at the anchor point of the strongest clean prediction inside each box
            (where the detector "looks"), clipped so the patch centre stays within +-0.3

Run from the project root:
    python src/placement_study.py
"""
import argparse
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from attack_patch import letterbox, load_patch, paste, read_labels
from train_baseline import report

GRID = [-0.3, -0.15, 0.0, 0.15, 0.3]
BESIDE = [(0.75, 0.0), (-0.75, 0.0), (0.0, 0.75), (0.0, -0.75)]
KINDS = ("random_noise", "adversarial")


def placements():
    out = {f"grid_{dx:+.2f}_{dy:+.2f}": ("offset", dx, dy) for dy in GRID for dx in GRID}
    out.update({f"beside_{dx:+.2f}_{dy:+.2f}": ("offset", dx, dy) for dx, dy in BESIDE})
    out["random"] = ("random", 0, 0)
    out["peak"] = ("peak", 0, 0)
    return out


def anchor_points(imgsz, strides=(8, 16, 32)):
    """Centres of YOLOv8/11 grid cells in the order of the 8,400 predictions."""
    pts = []
    for s in strides:
        n = imgsz // s
        ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        pts.append(np.stack([(xs.ravel() + 0.5) * s, (ys.ravel() + 0.5) * s], 1))
    return np.concatenate(pts)


@torch.no_grad()
def peak_offsets(net, images, args, device):
    """Per object: offset of the anchor point with the highest clean class score inside the box."""
    anchors = anchor_points(args.imgsz)
    out = []
    for path, labels in images:
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        lb, r, left, top = letterbox(img, args.imgsz)
        x = torch.from_numpy(lb).permute(2, 0, 1)[None].float().div(255).to(device)
        scores = net(x)[0][0, 4:].amax(0).cpu().numpy()  # (8400,) best class score per prediction
        offs = []
        for _, cx, cy, w, h in labels:
            bx, by, bw, bh = cx * W * r + left, cy * H * r + top, w * W * r, h * H * r
            inside = (np.abs(anchors[:, 0] - bx) <= bw / 2) & (np.abs(anchors[:, 1] - by) <= bh / 2)
            if not inside.any():
                offs.append((0.0, 0.0))
                continue
            k = np.flatnonzero(inside)[scores[inside].argmax()]
            dx = float(np.clip((anchors[k, 0] - bx) / bw, -0.3, 0.3))
            dy = float(np.clip((anchors[k, 1] - by) / bh, -0.3, 0.3))
            offs.append((dx, dy))
        out.append(offs)
    return out


def build(images, patch, mode, dx, dy, peaks, root, args):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for i, (path, labels) in enumerate(images):
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        rng = random.Random(i)          # same rotations as robustness.py, for every placement
        pos_rng = random.Random(1000 + i)  # random-placement offsets, independent of rotations
        t = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(patch.device)
        for j, (_, cx, cy, w, h) in enumerate(labels):
            cx, cy, w, h = cx * W, cy * H, w * W, h * H
            angle = rng.uniform(0, 2 * math.pi)
            if min(w, h) < args.min_box * max(H, W) / args.imgsz:
                continue
            if mode == "random":
                ox, oy = pos_rng.uniform(-0.3, 0.3), pos_rng.uniform(-0.3, 0.3)
            elif mode == "peak":
                ox, oy = peaks[i][j]
            else:
                ox, oy = dx, dy
            t = paste(t, patch, cx + ox * w, cy + oy * h, args.patch_scale * math.sqrt(w * h), angle)
        out = (t.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        cv2.imwrite(str(root / "images" / (path.stem + ".png")), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        (root / "labels" / (path.stem + ".txt")).write_text(
            "".join(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/baseline/yolov8s/weights/best.pt")
    ap.add_argument("--patch", default="runs/attack/patch_s30/patch.png")
    ap.add_argument("--data-root", default="data_yolo")
    ap.add_argument("--name", default="yolov8s")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--patch-scale", type=float, default=0.3)
    ap.add_argument("--min-box", type=float, default=12)
    args = ap.parse_args()

    out_dir = (Path("runs/placement") / args.name).resolve()
    work = out_dir / "work"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    device = torch.device(args.device)
    adv = load_patch(args.patch, device)
    torch.manual_seed(0)
    patches = {"random_noise": torch.rand_like(adv), "adversarial": adv}  # same control as elsewhere

    images = []
    for split in ("val", "test"):
        for p in sorted((Path(args.data_root) / split / "images").iterdir()):
            images.append((p, read_labels(Path(args.data_root) / split / "labels" / (p.stem + ".txt")).tolist()))

    model = YOLO(args.weights)
    names = "".join(f"  {k}: {v}\n" for k, v in model.names.items())
    net = model.model.to(device).eval()
    peaks = peak_offsets(net, images, args, device)
    flat = np.array([o for offs in peaks for o in offs])
    print(f"peak offsets: mean dx={flat[:, 0].mean():+.3f} dy={flat[:, 1].mean():+.3f}, "
          f"mean |d|={np.hypot(flat[:, 0], flat[:, 1]).mean():.3f}")

    results_path = out_dir / "placement_metrics.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results["_peak_offsets"] = {"mean_dx": float(flat[:, 0].mean()), "mean_dy": float(flat[:, 1].mean()),
                                "mean_abs_offset": float(np.hypot(flat[:, 0], flat[:, 1]).mean())}

    def run(key, build_fn):
        root = work / key
        if root.exists():
            shutil.rmtree(root)
        build_fn(root)
        yaml = root / "data.yaml"
        yaml.write_text(f"path: {root.as_posix()}\ntrain: images\nval: images\nnames:\n{names}")
        res = report(model, str(yaml), "val", args.imgsz, root / "eval", args.device, plots=False)
        shutil.rmtree(root)
        return res

    if "clean" not in results:
        def clean(root):
            (root / "images").mkdir(parents=True)
            (root / "labels").mkdir()
            for p, labels in images:
                shutil.copy(p, root / "images" / p.name)
                shutil.copy(p.parent.parent / "labels" / (p.stem + ".txt"), root / "labels")
        results["clean"] = run("clean", clean)
        results_path.write_text(json.dumps(results, indent=2))

    for key, (mode, dx, dy) in placements().items():
        for kind in KINDS:
            if kind in results.get(key, {}):
                continue
            print(f"\n### {key} [{kind}]")
            res = run(f"{key}_{kind}", lambda root: build(images, patches[kind], mode, dx, dy, peaks, root, args))
            results.setdefault(key, {"mode": mode, "dx": dx, "dy": dy})[kind] = res
            results_path.write_text(json.dumps(results, indent=2))

    shutil.rmtree(work, ignore_errors=True)
    c = results["clean"]["mAP50"]
    print(f"\n=== mAP50, val+test pooled (clean {c:.3f}): noise / adversarial ===")
    for key in placements():
        r = results[key]
        print(f"  {key:22s} {r['random_noise']['mAP50']:.3f} / {r['adversarial']['mAP50']:.3f}")
    print(f"\nSaved {results_path}")


if __name__ == "__main__":
    main()
