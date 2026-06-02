# Object Detection - YOLOv8 & Faster R-CNN on Custom Tool Dataset

Two-part object detection project: (1) zero-shot inference on a video using COCO-pretrained models, and (2) fine-tuning YOLOv8n and Detectron2 Faster R-CNN on a custom 6-class hand-tool dataset. Compares single-stage vs. two-stage detectors across speed and accuracy.

---

## Dataset

- **Source:** aiHub.v1i (Kaggle / Roboflow)
- **Classes (6):** Hammer, Pliers, Rope, Screw Driver, Tool Box, Wrench
- **Train:** 885 images | **Validation:** 51 images (82 instances) | **Test:** 26 images (45 instances)
- **Formats:** COCO JSON (Detectron2) and YOLOv8 YAML

**Part 1 video:** 251-frame clip (`video.mov`) used for zero-shot COCO inference without fine-tuning.

---

## Models

### Part 1 — Zero-Shot Video Inference (COCO pretrained)
- **Faster R-CNN ResNet50 FPN V2** (torchvision, confidence threshold 0.9)
- **YOLOv8n** (Ultralytics, pretrained)

### Part 2 — Fine-tuned on Tool Dataset
- **YOLOv8n** — 100 epochs, GPU, 2.69M params, 6.8 GFLOPs
- **Detectron2 Faster R-CNN R-50 FPN 3x** — 1,000 iterations, LR=0.00025, batch=2, 416×416 input

---

## Results

### YOLOv8n (100 epochs) — Validation Set

| Metric | Score |
|--------|-------|
| mAP@50 | **0.730** |
| mAP@[50:95] | 0.562 |
| Precision | 0.756 |
| Recall | 0.730 |

Per-class mAP@50: Tool Box 0.924 · Hammer 0.922 · Rope 0.840 · Pliers 0.754 · Wrench 0.624 · Screw Driver 0.313

### Detectron2 Faster R-CNN R-50 FPN 3x

| Metric | Validation | Test |
|--------|-----------|------|
| AP@[50:95] | 45.45 | **63.71** |
| AP@50 | 65.36 | **83.87** |
| AP@75 | 54.01 | 72.75 |
| AR@100 | 0.659 | 0.730 |

Per-class AP (test): Rope 88.46 · Pliers 82.57 · Hammer 71.80 · Tool Box 66.02 · Wrench 37.13 · Screw Driver 36.29

---

## Key Findings

- YOLOv8n achieved higher mAP@50 (0.730 vs. 0.654), while Detectron2 showed stronger localization precision at IoU=0.75 (0.540)
- Screw Driver was consistently the hardest class across both models due to visual similarity with other tools and mislabeled validation annotations
- Rope and Tool Box were the easiest classes — their unique shapes clearly differentiate them
- Detectron2 test AP (63.71) substantially exceeded validation AP (45.45), indicating the test set was easier

---

## Setup

```bash
pip install torch torchvision ultralytics
# For Detectron2:
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Place the dataset in `object-detection-using-cnn-architecture/` following the COCO/YOLOv8 folder structure, then open `object_detetction.ipynb`.
