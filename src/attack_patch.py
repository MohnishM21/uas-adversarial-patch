"""Object-disappearance adversarial patch against the fine-tuned YOLOv8 baseline.

Training (on the training split only):
  - One universal patch is pasted onto every target object in each image.
  - Expectation over Transformation (EOT): each paste gets a random 0-360 deg rotation
    (aerial views have no fixed heading), scale jitter, position jitter inside the box,
    and brightness / contrast / noise changes, so the patch has to work across all of them.
  - Loss = detection score of the strongest predictions whose centres fall inside a
    patched box (pushed down) + total-variation penalty (keeps the patch smooth/printable).

Evaluation (val + test): the converged patch, a random-noise patch of the same size
(control: separates "adversarial" from "simply covering part of the object"), and the
clean images are scored with the same metrics as the baseline.

Run from the project root:
    python src/attack_patch.py                      # train patch, then evaluate
    python src/attack_patch.py --resume                 # continue after an interruption
    python src/attack_patch.py --weights A.pt B.pt --name patch_ens   # one patch against an ensemble
    python src/attack_patch.py --eval-only --patch runs/attack/patch_s30/patch.png
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

from train_baseline import report

PAD_VALUE = 114 / 255


# --------------------------------------------------------------------------- data

def letterbox(img, size):
    """Resize keeping aspect ratio and pad to size x size, like YOLO's val pipeline."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, left, top


def read_labels(path):
    rows = [l.split() for l in Path(path).read_text().splitlines() if l.strip()]
    return np.array([[float(v) for v in r] for r in rows], np.float32).reshape(-1, 5)


class TrainImages(Dataset):
    """Letterboxed training images + their boxes in pixel coords (cls, cx, cy, w, h)."""

    def __init__(self, root, imgsz, limit=None):
        self.imgs = sorted((Path(root) / "images").iterdir())[:limit]
        self.imgsz = imgsz

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img, r, left, top = letterbox(img, self.imgsz)
        lab = read_labels(p.parent.parent / "labels" / (p.stem + ".txt"))
        boxes = lab.copy()
        boxes[:, 1] = lab[:, 1] * w * r + left
        boxes[:, 2] = lab[:, 2] * h * r + top
        boxes[:, 3] = lab[:, 3] * w * r
        boxes[:, 4] = lab[:, 4] * h * r
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(boxes)


def collate(batch):
    imgs, boxes = zip(*batch)
    return torch.stack(imgs).float() / 255, list(boxes)


# ------------------------------------------------------------------ patch applier

def paste(img, patch, cx, cy, side, angle):
    """Paste `patch` (3,P,P) rotated by `angle` with side length `side` px centred at (cx, cy).

    Only the window around the patch is resampled, so this stays cheap on large images.
    Differentiable w.r.t. `patch`.
    """
    _, H, W = img.shape
    half = side * 0.7072  # half-diagonal: bounds the rotated square
    x0, x1 = max(0, int(cx - half)), min(W, int(math.ceil(cx + half)))
    y0, y1 = max(0, int(cy - half)), min(H, int(math.ceil(cy + half)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return img
    ww, hh = x1 - x0, y1 - y0

    # Map window pixels (normalised to [-1, 1]) into patch coordinates (also [-1, 1]).
    c, s = math.cos(angle), math.sin(angle)
    rt = torch.tensor([[c, s], [-s, c]], dtype=img.dtype, device=img.device)
    a = rt @ torch.diag(torch.tensor([ww / side, hh / side], dtype=img.dtype, device=img.device))
    t = rt @ torch.tensor([(x0 + ww / 2 - cx) * 2 / side, (y0 + hh / 2 - cy) * 2 / side],
                          dtype=img.dtype, device=img.device)
    theta = torch.cat([a, t[:, None]], 1)[None]
    grid = F.affine_grid(theta, (1, 4, hh, ww), align_corners=False)

    rgba = torch.cat([patch, torch.ones_like(patch[:1])], 0)[None]
    warped = F.grid_sample(rgba, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    rgb, alpha = warped[:3], warped[3:]

    region = img[:, y0:y1, x0:x1] * (1 - alpha) + rgb * alpha  # read before the in-place write
    img = img.clone()
    img[:, y0:y1, x0:x1] = region
    return img


def eot_patch(patch, args):
    """Random photometric change: lighting/contrast differ between print and camera."""
    contrast = random.uniform(1 - args.eot_contrast, 1 + args.eot_contrast)
    bright = random.uniform(-args.eot_brightness, args.eot_brightness)
    noise = (torch.rand_like(patch) * 2 - 1) * args.eot_noise
    return (patch * contrast + bright + noise).clamp(0, 1)


def apply_eot(img, boxes, patch, args):
    """Paste the patch on up to `max_boxes` objects with random EOT; return image + patched boxes."""
    keep = [b for b in boxes.tolist() if min(b[3], b[4]) >= args.min_box]
    random.shuffle(keep)
    keep = keep[:args.max_boxes]
    for _, cx, cy, w, h in keep:
        side = args.patch_scale * math.sqrt(w * h) * random.uniform(1 - args.eot_scale, 1 + args.eot_scale)
        cx += random.uniform(-args.eot_shift, args.eot_shift) * w
        cy += random.uniform(-args.eot_shift, args.eot_shift) * h
        img = paste(img, eot_patch(patch, args), cx, cy, side, random.uniform(0, 2 * math.pi))
    return img, keep


# -------------------------------------------------------------------------- losses

def detection_loss(preds, patched):
    """Mean of the top scores among predictions centred inside patched boxes, plus the max.

    preds: (B, 4+nc, N) from the YOLOv8 head in eval mode, boxes as (cx, cy, w, h) pixels.
    Using the max over all classes means relabelling an object as another class doesn't help:
    the object must disappear.
    """
    centers, scores = preds[:, :2], preds[:, 4:].amax(1)  # (B,2,N), (B,N)
    losses = []
    for i, boxes in enumerate(patched):
        if not boxes:
            continue
        b = torch.tensor(boxes, device=preds.device)
        inside = ((centers[i, 0][None] - b[:, 1:2]).abs() <= b[:, 3:4] / 2) & \
                 ((centers[i, 1][None] - b[:, 2:3]).abs() <= b[:, 4:5] / 2)
        s = scores[i][inside.any(0)]
        if s.numel() == 0:
            continue
        k = min(s.numel(), 3 * len(boxes))
        losses.append(s.topk(k).values.mean() + s.max())
    return torch.stack(losses).mean() if losses else None


def tv_loss(p):
    return (p[:, 1:, :] - p[:, :-1, :]).abs().mean() + (p[:, :, 1:] - p[:, :, :-1]).abs().mean()


# ------------------------------------------------------------------------ training

def train_patch(dets, args, out_dir, device):
    """Optimise one patch against every detector in `dets` (an ensemble when there are several)."""
    loader = DataLoader(TrainImages(args.train_dir, args.imgsz, args.limit), batch_size=args.batch,
                        shuffle=True, num_workers=args.workers, collate_fn=collate,
                        persistent_workers=args.workers > 0)
    log, step, start = [], 0, 1
    if args.resume and (out_dir / "patch.png").exists():
        # Continue from the last saved epoch (optimizer state is not saved; Adam re-warms quickly).
        patch = load_patch(out_dir / "patch.png", device)
        log = json.loads((out_dir / "train_log.json").read_text())
        start = len(log) + 1
        print(f"Resuming from epoch {start} with {out_dir / 'patch.png'}")
    else:
        patch = torch.full((3, args.patch_size, args.patch_size), 0.5, device=device)
        patch = (patch + 0.1 * torch.randn_like(patch)).clamp(0, 1)
    patch.requires_grad_(True)
    opt = torch.optim.Adam([patch], lr=log[-1]["lr"] if log else args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    for epoch in range(start, args.epochs + 1):
        tot_det = tot_tv = n = 0
        for imgs, boxes in loader:
            imgs = imgs.to(device)
            patched_imgs, patched_boxes = [], []
            for img, b in zip(imgs, boxes):
                im, kept = apply_eot(img, b, patch, args)
                patched_imgs.append(im)
                patched_boxes.append(kept)
            batch = torch.stack(patched_imgs)
            opt.zero_grad()
            # Backward one detector at a time (gradients add up on the patch), so peak GPU memory
            # stays at one model's activations; the small pasting graph is kept for the next model.
            losses = []
            for det in dets:
                l = detection_loss(det(batch)[0], patched_boxes)
                if l is None:  # no patched objects in this batch (same for every model)
                    break
                (l / len(dets)).backward(retain_graph=True)
                losses.append(l.item())
                del l
            if len(losses) < len(dets):
                continue
            l_tv = tv_loss(patch)
            (args.tv_weight * l_tv).backward()
            opt.step()
            l_det = sum(losses) / len(losses)
            with torch.no_grad():
                patch.clamp_(0, 1)
            tot_det += l_det
            tot_tv += l_tv.item()
            n += 1
            step += 1
            if step % 25 == 0:
                per_model = " ".join(f"{v:.3f}" for v in losses) if len(dets) > 1 else ""
                print(f"  epoch {epoch} step {step}: det={l_det:.4f} {per_model} tv={l_tv.item():.4f}")
            if args.max_steps and step >= args.max_steps:
                break
        n = max(n, 1)
        sched.step(tot_det / n)
        log.append({"epoch": epoch, "det_loss": tot_det / n, "tv_loss": tot_tv / n,
                    "lr": opt.param_groups[0]["lr"]})
        print(f"epoch {epoch}/{args.epochs}: det_loss={tot_det / n:.4f} tv={tot_tv / n:.4f} "
              f"lr={opt.param_groups[0]['lr']:.4g}")
        save_patch(patch, out_dir / "patch.png")
        save_patch(patch, out_dir / "patch_history" / f"epoch_{epoch:03d}.png")
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
        if args.max_steps and step >= args.max_steps:
            break
    return patch.detach()


def save_patch(patch, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (patch.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def load_patch(path, device):
    arr = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(arr).permute(2, 0, 1).float().div(255).to(device)


# ---------------------------------------------------------------------- evaluation

def build_patched_split(src_split, dst_split, patch, args, seed):
    """Paste `patch` on every object (>= min_box px) at fixed scale, centred, random rotation."""
    rng = random.Random(seed)
    (dst_split / "images").mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_split / "labels", dst_split / "labels", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.cache"))
    for p in sorted((src_split / "images").iterdir()):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        t = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(patch.device)
        for _, cx, cy, w, h in read_labels(src_split / "labels" / (p.stem + ".txt")).tolist():
            cx, cy, w, h = cx * W, cy * H, w * W, h * H
            if min(w, h) < args.min_box * max(H, W) / args.imgsz:  # same min size as at 640 px
                continue
            t = paste(t, patch, cx, cy, args.patch_scale * math.sqrt(w * h), rng.uniform(0, 2 * math.pi))
        arr = (t.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        cv2.imwrite(str(dst_split / "images" / (p.stem + ".png")), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def evaluate(weights, patch, args, out_dir, device):
    data_root = Path(args.data_root)
    torch.manual_seed(0)
    conditions = {"adversarial": patch, "random_noise": torch.rand_like(patch)}
    model = YOLO(weights)
    results = {"clean": {}}
    for split in ("val", "test"):
        results["clean"][split] = report(model, args.data, split, args.imgsz, out_dir / "eval_clean", args.device)

    for cond, p in conditions.items():
        root = out_dir / "eval_data" / cond
        if root.exists():
            shutil.rmtree(root)
        for split in ("val", "test"):
            build_patched_split(data_root / split, root / split, p, args, seed=0)
        yaml = root / "dataset.yaml"
        yaml.write_text(f"path: {root.resolve().as_posix()}\ntrain: val/images\nval: val/images\n"
                        f"test: test/images\nnames:\n" +
                        "".join(f"  {k}: {v}\n" for k, v in model.names.items()))
        results[cond] = {s: report(model, str(yaml), s, args.imgsz, out_dir / f"eval_{cond}", args.device)
                         for s in ("val", "test")}

    (out_dir / "attack_metrics.json").write_text(json.dumps(results, indent=2))
    print_summary(results)

    # Before/after detections on a few test images, for slides.
    samples = sorted((out_dir / "eval_data" / "adversarial" / "test" / "images").iterdir())[:6]
    clean = [data_root / "test" / "images" / (s.stem + ".jpg") for s in samples]
    model.predict([str(c) for c in clean if c.exists()], save=True, project=str(out_dir),
                  name="examples_clean", exist_ok=True, verbose=False, device=args.device)
    model.predict([str(s) for s in samples], save=True, project=str(out_dir),
                  name="examples_adversarial", exist_ok=True, verbose=False, device=args.device)


def print_summary(results):
    for split in ("val", "test"):
        print(f"\n=== {split}: AP50 (recall)  clean -> random noise -> adversarial ===")
        names = list(results["clean"][split]["per_class"])
        for name in names + ["ALL"]:
            row = []
            for cond in ("clean", "random_noise", "adversarial"):
                r = results[cond][split]
                ap = r["mAP50"] if name == "ALL" else r["per_class"].get(name, {}).get("AP50", float("nan"))
                row.append(f"{ap:.3f}")
            rec = "" if name != "ALL" else "  (recall " + " -> ".join(
                f"{results[c][split]['recall']:.3f}" for c in ("clean", "random_noise", "adversarial")) + ")"
            print(f"  {name:15s} " + " -> ".join(row) + rec)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", default=["runs/baseline/yolov8s/weights/best.pt"],
                    help="detector(s) to attack; several = ensemble patch. Evaluation uses the first.")
    ap.add_argument("--data", default="configs/dataset.yaml")
    ap.add_argument("--data-root", default="data_yolo")
    ap.add_argument("--train-dir", default="data_yolo/train")
    ap.add_argument("--name", default=None, help="output folder under runs/attack (default from settings)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--patch-size", type=int, default=300, help="patch resolution in pixels")
    ap.add_argument("--patch-scale", type=float, default=0.3,
                    help="patch side as a fraction of sqrt(box area); 0.3 covers ~9%% of the box")
    ap.add_argument("--tv-weight", type=float, default=2.5)
    ap.add_argument("--max-boxes", type=int, default=24, help="max objects patched per training image")
    ap.add_argument("--min-box", type=float, default=12, help="skip objects smaller than this (px at imgsz)")
    ap.add_argument("--eot-scale", type=float, default=0.2)
    ap.add_argument("--eot-shift", type=float, default=0.15)
    ap.add_argument("--eot-brightness", type=float, default=0.1)
    ap.add_argument("--eot-contrast", type=float, default=0.2)
    ap.add_argument("--eot-noise", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=None, help="use only N training images (smoke tests)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop training after N steps (smoke tests)")
    ap.add_argument("--resume", action="store_true", help="continue training from the saved patch.png")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--patch", default=None, help="patch image to evaluate with --eval-only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # Absolute: ultralytics re-roots relative output paths under runs/detect/.
    out_dir = (Path("runs/attack") / (args.name or f"patch_s{round(args.patch_scale * 100)}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    if args.eval_only:
        patch = load_patch(args.patch or out_dir / "patch.png", device)
    else:
        dets = [YOLO(w).model.to(device).eval() for w in args.weights]
        for det in dets:
            for prm in det.parameters():
                prm.requires_grad_(False)
        patch = train_patch(dets, args, out_dir, device)
        del dets
        if device.type == "cuda":
            torch.cuda.empty_cache()

    evaluate(args.weights[0], patch, args, out_dir, device)


if __name__ == "__main__":
    main()
