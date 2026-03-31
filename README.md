# YOLOv8 Nanoparticle Detection (Oriented Bounding Boxes)

This repository implements a full training and benchmarking pipeline for **nanoparticle detection in microscopy images** using YOLOv8.

The project focuses on **detecting and measuring nanoparticles**, not only locating them.
It provides tools to train custom models, evaluate them rigorously, and compare multiple trained checkpoints.

Main scripts:

• `yolo_experiment.py` — training pipeline
• `test_generated_models.py` — inference and benchmarking

---

# Project Goal

Detect and localize **nanoparticles in microscopy imagery** using deep learning, providing a robust and efficient workflow suitable for experimentation and reproducible research.

The repository includes:

• Scripts for dataset handling and experimentation
• Fine-tuned YOLO models for nanoparticle detection
• Automated inference and benchmarking tools

---

# Environment Setup

Python 3.10+ recommended.

Install the main dependency:

```bash
pip install ultralytics opencv-python numpy
```

Clone the repository:

```bash
git clone https://github.com/izaias-saturnino/yolo-nanoparticle.git
cd yolo-nanoparticle
```

After cloning, **configure the dataset location inside the project directory**.

---

# Dataset Requirements

The scripts expect a YOLO-style dataset configured via a `.yaml` file.

Example structure:

```
data-obb/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

The dataset configuration file must be referenced by the training script:

```
data.yaml
```

When oriented bounding boxes are enabled, the scripts automatically switch to:

```
data-obb.yaml
```

---

# Training

Run:

```bash
python yolo_experiment.py
```

Before running, **configure the following variables inside the script**:

| Variable           | Description                     |
| ------------------ | ------------------------------- |
| `data_file`        | Path to dataset YAML            |
| `local_model_name` | Name of model to train/save     |
| `base_model_name`  | Base YOLO model to download/use |
| `oriented_bb`      | Enable oriented bounding boxes  |

The base model is automatically downloaded if missing, enabling retraining without manual model management.

---

# Multi-Phase Training Strategy

The training pipeline uses staged learning to improve stability and generalization.

Phase 1 — Baseline training
• No augmentation
• Learns stable features

Phase 2 — Augmented training
Adds:
• Rotation (up to 180° in OBB mode)
• Copy-paste
• MixUp
• Vertical flip

Phase 3 — Shear training (prepared but disabled)

Training artifacts and runs are automatically archived to prevent overwriting.

---

# Custom Geometry-Based Metric

This project introduces a metric focused on **particle diameter accuracy**.

Process:

1. Match predictions to ground truth using IoU
2. Extract particle diameter from the largest bounding-box edge
3. Normalize predicted diameter by true diameter
4. Compute **99% confidence interval of diameter error**

Lower values indicate more reliable particle measurement.

Training metrics are saved to:

```
training_phase_1.txt
training_phase_2.txt
```

---

# Model Evaluation / Benchmarking

Run:

```bash
python test_generated_models.py
```

Before running, configure:

| Variable       | Description                         |
| -------------- | ----------------------------------- |
| `data_file`    | Dataset YAML                        |
| `model_paths`  | List of trained models              |
| `confidence`   | Detection confidence threshold      |
| `metric_sets`  | Splits to evaluate (train/val/test) |
| `save_results` | Save prediction outputs             |
| `dir_path`     | Dataset directory                   |
| `oriented_bb`  | Whether models use OBB              |

---

# Metrics Produced

For each model and dataset split:

• Mean diameter error
• 99% CI of diameter error
• Precision
• Recall
• F1 score
• True positives
• False positives
• False negatives
• Instance counts

YOLO run folders are automatically archived for reproducibility.

---

# Typical Workflow

Train models:

```bash
python yolo_experiment.py
```

Benchmark models:

```bash
python test_generated_models.py
```

Compare metrics and select the best checkpoint.

---

# References

YOLOv8 documentation:
[https://docs.ultralytics.com/models/yolov8/](https://docs.ultralytics.com/models/yolov8/)

---

If you want, I can now produce a short GitHub-portfolio version or an academic paper style version.
