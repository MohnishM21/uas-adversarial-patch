"""Convert the raw course dataset into a clean YOLO layout.

data/{Training,Validation,Testing} Data/{Images,Labels}
  -> data_yolo/{train,val,test}/{images,labels}

YOLO pairs images with labels by swapping '/images/' for '/labels/' in the path,
which is case-sensitive, so the raw 'Images'/'Labels' folders can't be used directly.
Polygon label lines (class + 4 corner points) are converted to bounding boxes.
Safe to re-run: it rebuilds data_yolo/ from scratch.
"""
import argparse
import shutil
from pathlib import Path

SPLITS = {"Training Data": "train", "Validation Data": "val", "Testing Data": "test"}


def to_box(parts):
    """Return a YOLO box line clipped to the image, or None if nothing is left.

    The rotation-augmented training images have boxes that spill past the frame;
    YOLO rejects the whole image for any coordinate outside [0, 1], so clip instead.
    """
    cls, coords = parts[0], [float(v) for v in parts[1:]]
    if len(coords) == 4:
        x, y, w, h = coords
        xs, ys = [x - w / 2, x + w / 2], [y - h / 2, y + h / 2]
    else:
        xs, ys = coords[0::2], coords[1::2]
    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
    if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
        return None
    return f"{cls} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data")
    ap.add_argument("--dst", default="data_yolo")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    # Empty dst rather than deleting it: Windows often holds a handle on the folder itself.
    if dst.exists():
        for child in dst.iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()

    for raw, split in SPLITS.items():
        img_out, lbl_out = dst / split / "images", dst / split / "labels"
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)
        n_img = n_box = n_poly = n_drop = 0
        for img in sorted((src / raw / "Images").iterdir()):
            lbl = src / raw / "Labels" / (img.stem + ".txt")
            if not lbl.exists():
                print(f"  [skip] no label for {img.name}")
                continue
            lines = []
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                n_poly += len(parts) != 5
                box = to_box(parts)
                if box is None:
                    n_drop += 1
                else:
                    lines.append(box)
            shutil.copy2(img, img_out / img.name)
            (lbl_out / lbl.name).write_text("\n".join(lines) + "\n")
            n_img += 1
            n_box += len(lines)
        print(f"{split:5s}: {n_img} images, {n_box} boxes "
              f"({n_poly} polygons converted, {n_drop} fully off-image boxes dropped)")


if __name__ == "__main__":
    main()
