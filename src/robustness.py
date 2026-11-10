"""Robustness of the trained patch under controlled real-world conditions.

One factor is varied at a time; everything else stays at the standard setting
(patch side 0.3*sqrt(box area), random rotation, no lighting change, full resolution).
The patch is pasted first (it is a physical object on the target), then the condition
is applied to the whole image, as a camera would see it. Clean and random-noise images
go through the same conditions, so each comparison is like-for-like.
Val and test are pooled (32 images) for more stable numbers.

Run from the project root:
    python src/robustness.py
    python src/robustness.py --weights runs/baseline/yolo11s/weights/best.pt --name yolo11s
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

from attack_patch import load_patch, paste, read_labels
from train_baseline import report

FACTORS = {
    "rotation_deg": [0, 45, 90, 180],             # patch orientation relative to the object
    "brightness": [0.5, 0.75, 1.0, 1.25, 1.5],    # scene illumination
    "resolution": [1.0, 0.5, 0.35, 0.25],         # pixels on target ~ altitude / distance
    "jpeg_quality": [95, 50, 25, 10],             # video-link compression
    "perspective": [0.0, 0.2, 0.4, 0.6],          # oblique view: top edge shrunk by this fraction
}
DEFAULTS = {"rotation_deg": None, "brightness": 1.0, "resolution": 1.0, "jpeg_quality": None, "perspective": 0.0}
KINDS = ("clean", "random_noise", "adversarial")
EXAMPLE_STEM = "aircraft-26"  # image saved as a thumbnail for every condition


def perspective_warp(img, labels, t):
    """Keystone warp: the top edge shrinks by fraction t, like a camera pitched away from nadir."""
    H, W = img.shape[:2]
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([[t * W / 2, 0], [W - t * W / 2, 0], [W, H], [0, H]])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (W, H), borderValue=(114, 114, 114))
    new = []
    for c, x, y, w, h in labels:
        pts = np.float32([[(x - w / 2) * W, (y - h / 2) * H], [(x + w / 2) * W, (y - h / 2) * H],
                          [(x + w / 2) * W, (y + h / 2) * H], [(x - w / 2) * W, (y + h / 2) * H]])
        p = cv2.perspectiveTransform(pts[None], M)[0]
        x0, y0 = np.clip(p.min(0), 0, [W, H])
        x1, y1 = np.clip(p.max(0), 0, [W, H])
        if x1 - x0 > 1 and y1 - y0 > 1:
            new.append([c, (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H])
    return out, new


def render(img, labels, patch, cond, seed, args):
    """Return (image, labels, file extension) for one image under one condition."""
    labels = [list(l) for l in labels]
    if patch is not None:
        rng = random.Random(seed)  # same placement for every condition
        H, W = img.shape[:2]
        t = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(patch.device)
        for _, cx, cy, w, h in labels:
            cx, cy, w, h = cx * W, cy * H, w * W, h * H
            angle = rng.uniform(0, 2 * math.pi)
            if min(w, h) < args.min_box * max(H, W) / args.imgsz:
                continue
            if cond["rotation_deg"] is not None:
                angle = math.radians(cond["rotation_deg"])
            t = paste(t, patch, cx, cy, args.patch_scale * math.sqrt(w * h), angle)
        img = (t.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)

    if cond["brightness"] != 1.0:
        img = np.clip(img.astype(np.float32) * cond["brightness"], 0, 255).astype(np.uint8)
    if cond["perspective"]:
        img, labels = perspective_warp(img, labels, cond["perspective"])
    if cond["resolution"] != 1.0:
        H, W = img.shape[:2]
        r = args.imgsz * cond["resolution"] / max(H, W)
        img = cv2.resize(img, (max(1, round(W * r)), max(1, round(H * r))), interpolation=cv2.INTER_AREA)
    if cond["jpeg_quality"] is not None:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, cond["jpeg_quality"]])
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), labels, ".jpg", buf
    return img, labels, ".png", None


def build(images, patch, cond, root, args):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for i, (path, labels) in enumerate(images):
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        out, lab, ext, buf = render(img, labels, patch, cond, seed=i, args=args)
        dst = root / "images" / (path.stem + ext)
        if buf is not None:
            dst.write_bytes(buf.tobytes())  # keep the exact compressed bytes
        else:
            cv2.imwrite(str(dst), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        (root / "labels" / (path.stem + ".txt")).write_text(
            "".join(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in lab))


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
    ap.add_argument("--factors", nargs="*", default=list(FACTORS), help="subset of factors to run")
    args = ap.parse_args()

    out_dir = (Path("runs/robustness") / args.name).resolve()  # absolute: see attack_patch.py
    work, thumbs = out_dir / "work", out_dir / "examples"
    thumbs.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device(args.device)
    adv = load_patch(args.patch, device)
    torch.manual_seed(0)
    noise = torch.rand_like(adv)  # same control patch as attack_patch.py
    patches = {"clean": None, "random_noise": noise, "adversarial": adv}

    images = []
    for split in ("val", "test"):
        for p in sorted((Path(args.data_root) / split / "images").iterdir()):
            images.append((p, read_labels(Path(args.data_root) / split / "labels" / (p.stem + ".txt")).tolist()))

    model = YOLO(args.weights)
    names = "".join(f"  {k}: {v}\n" for k, v in model.names.items())
    results_path = out_dir / "robustness_metrics.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for factor in args.factors:
        for value in FACTORS[factor]:
            cond = dict(DEFAULTS, **{factor: value})
            for kind in KINDS:
                key = f"{factor}={value}"
                if kind in results.get(factor, {}).get(str(value), {}):
                    continue  # already done (lets an interrupted run continue)
                if kind == "clean" and factor == "rotation_deg":
                    cond_eval = dict(DEFAULTS)  # rotation only changes the patch; clean is the default image
                else:
                    cond_eval = cond
                root = work / f"{factor}_{value}_{kind}"
                if root.exists():
                    shutil.rmtree(root)
                build(images, patches[kind], cond_eval, root, args)
                yaml = root / "data.yaml"
                yaml.write_text(f"path: {root.as_posix()}\ntrain: images\nval: images\nnames:\n{names}")
                print(f"\n### {key} [{kind}]")
                res = report(model, str(yaml), "val", args.imgsz, root / "eval", args.device, plots=False)
                ex = next((root / "images").glob(f"{EXAMPLE_STEM}*"), None)
                if ex is not None:
                    im = cv2.imread(str(ex))
                    s = 360 / max(im.shape[:2])
                    cv2.imwrite(str(thumbs / f"{factor}_{value}_{kind}.jpg"),
                                cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s))))
                shutil.rmtree(root)  # patched full-res images are large; keep only metrics + thumbnail
                results.setdefault(factor, {}).setdefault(str(value), {})[kind] = res
                results_path.write_text(json.dumps(results, indent=2))

    shutil.rmtree(work, ignore_errors=True)
    print("\n=== mAP50 on val+test (32 images): clean / random noise / adversarial ===")
    for factor in args.factors:
        print(f"{factor}:")
        for value in FACTORS[factor]:
            r = results[factor][str(value)]
            print(f"  {str(value):>5s}   " + " / ".join(f"{r[k]['mAP50']:.3f}" for k in KINDS))
    print(f"\nSaved {results_path}")


if __name__ == "__main__":
    main()
