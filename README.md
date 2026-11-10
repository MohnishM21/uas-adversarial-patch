# Physical Adversarial Patch Attack on UAS Object Detection

Course project for AE5510 (Security of Safety-Critical Systems), IIT Madras, by Mohnish M (AE22B021).

Drones increasingly rely on deep-learning object detectors. This project asks whether a single printed
pattern placed on a target can make it disappear from a UAS detector. A YOLOv8s detector is fine-tuned on
a 4-class aerial dataset (Aircraft, Building, Military Truck, Tank). One universal patch is then optimised
with Expectation over Transformation (random rotation, scale, position and lighting) and a total-variation
smoothness penalty, and evaluated against clean images and against a random-noise patch of the same size
and placement, which separates the effect of the learned pattern from plain occlusion.

## Results

mAP@0.5 with the patch covering about 9% of each object (validation / test):

| Detector | Clean | Random-noise patch | Adversarial patch |
|---|---|---|---|
| YOLOv8s (attacked model) | 0.622 / 0.490 | 0.376 / 0.393 | **0.129 / 0.226** |
| YOLOv8n (never seen by the patch) | 0.689 / 0.465 | 0.517 / 0.400 | **0.409 / 0.308** |
| YOLO11s (never seen by the patch) | 0.743 / 0.469 | 0.501 / 0.408 | **0.350 / 0.309** |

- **The damage comes from the pattern, not occlusion:** on validation the adversarial drop (−79%) is twice the drop of a same-size noise patch (−40%).
- **Size:** covering only 4% of each object already halves validation mAP (0.622 → 0.299).
- **Transfer:** the patch also degrades detectors it never saw; a patch trained against YOLOv8s + YOLO11s together gave a modest, inconclusive gain on the unseen YOLOv8n.
- **Robustness (simulated):** under patch rotation, brightness ×0.5–1.5, 4× lower resolution, JPEG quality 10 and perspective tilt, patched objects keep only 30–48% of their clean mAP (noise patch: 62–84%). The weakest case is a dark scene.
- **Limits:** obliquely photographed tanks resist (their sides stay visible around a roof patch); evaluation sets are small (16 + 16 images); all tests are digital, not printed.

All numbers are in [`results/`](results/) (`results/attack/summary.csv` collects every attack run). The full write-up is the IEEE-format report in [`report/`](report/).

## Repository layout

| Path | Contents |
|---|---|
| `src/` | All code (see below) |
| `configs/dataset.yaml` | YOLO dataset config with the verified class names |
| `results/` | Metrics, training logs and the learned patches for every run |
| `figures/` | Result charts and the patch's evolution during training |
| `report/` | LaTeX source of the report |

**Not included:** the course dataset (it is distributed through the course and contains third-party
watermarked imagery), trained weights, and figures that show dataset photos. Because of that, the report's
qualitative figures are not in `report/figures/`; the report compiles once they are regenerated with
`python src/make_figures.py` from your own runs.

## Setup

```
pip install -r requirements.txt     # install CUDA PyTorch first, see requirements.txt
```

Tested on Windows 11, Python 3.14, an RTX 4060 Laptop GPU (8 GB). Run every command from the repository root.

## Data

Place the course "Aerial Image Dataset" in `data/` so it contains `Training Data/`, `Validation Data/` and
`Testing Data/`, each with `Images/` and `Labels/`.

Class IDs in the label files are **0 = Aircraft, 1 = Building, 2 = Military Truck, 3 = Tank** (verified by
drawing each ID's boxes; this differs from the order in the dataset's README). The preparation step also
clips boxes that rotation augmentation pushed outside the frame (105 images had them; YOLO silently drops 94
of those images otherwise) and converts four polygon labels to boxes.

## Pipeline (run in this order)

| Step | Command | Time (RTX 4060) | Output |
|---|---|---|---|
| 1. Prepare data | `python src/prepare_data.py` | < 1 min | `data_yolo/` |
| 2. Baseline detector | `python src/train_baseline.py --model-size s` | ~1.8 h | `runs/baseline/yolov8s/` |
| 3. Patch attack | `python src/attack_patch.py` | ~1.1 h | `runs/attack/patch_s30/` |
| 4. Patch-size sweep | `python src/attack_patch.py --eval-only --patch runs/attack/patch_s30/patch.png --patch-scale 0.2 --name sweep_s02` (also 0.1, 0.4) | ~3 min each | `runs/attack/sweep_*/` |
| 5. Transfer models | `python src/train_baseline.py --model-size n` and `python src/train_baseline.py --arch yolo11 --model-size s` | ~0.6 h, ~1.5 h | `runs/baseline/yolov8n/`, `runs/baseline/yolo11s/` |
| 6. Transfer tests | `python src/attack_patch.py --eval-only --weights runs/baseline/yolov8n/weights/best.pt --patch runs/attack/patch_s30/patch.png --name transfer_yolov8n` (same for `yolo11s`) | ~3 min each | `runs/attack/transfer_*/` |
| 7. Ensemble patch | `python src/attack_patch.py --weights runs/baseline/yolov8s/weights/best.pt runs/baseline/yolo11s/weights/best.pt --epochs 20 --name patch_ens` | ~2 h | `runs/attack/patch_ens/` |
| 8. Ensemble transfer | `python src/attack_patch.py --eval-only --weights runs/baseline/yolov8n/weights/best.pt --patch runs/attack/patch_ens/patch.png --name ens_on_yolov8n` (same for `yolo11s`) | ~3 min each | `runs/attack/ens_on_*/` |
| 9. Robustness | `python src/robustness.py` | ~10 min | `runs/robustness/yolov8s/` |
| 10. Summary, figures, results | `python src/summarize.py`, `python src/make_figures.py`, `python src/collect_results.py` | seconds | `runs/attack/summary.csv`, `figures/`, `results/` |

Long runs can be continued after an interruption by repeating the command with `--resume` (steps 2, 3, 5
and 7) or by re-running it (step 9 skips finished conditions).

## Code

| File | Purpose |
|---|---|
| `src/prepare_data.py` | Raw dataset → YOLO layout; clips out-of-frame boxes, converts polygon labels |
| `src/train_baseline.py` | Fine-tunes YOLOv8 / YOLO11; reports precision, recall, per-class AP, mAP |
| `src/attack_patch.py` | EOT patch training (single model or ensemble); clean / random-noise / adversarial evaluation |
| `src/robustness.py` | One-factor-at-a-time robustness study (rotation, brightness, resolution, JPEG, perspective) |
| `src/summarize.py` | Collects all attack runs into one table |
| `src/make_figures.py` | Result figures |
| `src/collect_results.py` | Copies the small result files from `runs/` into `results/` |
