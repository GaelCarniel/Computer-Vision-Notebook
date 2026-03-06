# UC Merced Land Use Classification — Transfer Learning

Satellite image classification across 21 land-use categories using transfer learning on the [UC Merced Land Use Dataset](https://www.kaggle.com/datasets/ashikahmmed/uc-merce).

---

## Overview

This project applies transfer learning (pretrained on ImageNet) to classify aerial imagery into 21 land-use classes. The convolutional base is frozen and used as a fixed feature extractor; only a lightweight fully connected classifier is trained on top.

- Dataset size: 2100 images (21 classes × 100 images each)
- Image size: 256 × 256 pixels

**Classes:** agricultural, airplane, baseball diamond, beach, buildings, chaparral, dense residential, forest, freeway, golf course, harbor, intersection, medium residential, mobile home park, overpass, parking lot, river, runway, sparse residential, storage tanks, tennis court.

---

## Notebooks

### `train_vgg16_uc_merced.ipynb` — Baseline VGG16 classifier
The main notebook. Trains a VGG16-based classifier using bottleneck feature extraction.

1. **Data split** — 100 images/class → 80 train / 10 validation / 10 test
2. **Feature extraction** — Images passed once through frozen VGG16 (no top layer) to produce bottleneck feature vectors
3. **Classifier training** — Lightweight FC network trained on extracted features:
   - `Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(21, Softmax)`
   - Optimizer: Adam (`lr=2e-5`), Loss: categorical crossentropy, 20 epochs
4. **Evaluation** — Accuracy, classification report, and confusion matrix heatmap
5. **Grad-CAM** — Heatmap visualization of the regions driving each prediction
6. **External inference** — Model tested on arbitrary images (e.g. Google images)

### `comparison_architecture.ipynb` — Architecture comparison
Benchmarks three architectures side by side using the same training pipeline and dataset split.

| Architecture   | Test Accuracy | Notes               |
|----------------|---------------|---------------------|
| VGG16          | ~80%          | Baseline            |
| EfficientNetB0 | ~95%          | Fastest training    |
| ResNet50       | ~95%          | Good accuracy       |

Each architecture is evaluated with:
- **Confusion matrix** — side-by-side normalized heatmaps across all 21 classes
- **Grad-CAM** — per-architecture visualization of discriminative regions, with correct preprocessing applied per model
- **Training time** — wall-clock time reported for each architecture as a practical reference

> **Note on preprocessing:** Each architecture requires its own preprocessing function (`vgg16_preprocess`, `resnet_preprocess`, `efficientnet_preprocess`). Using the wrong one causes the model to produce random predictions — a subtle bug documented and fixed during development.

### `few_shot_efficientnet.ipynb` — Few-shot learning experiment
Investigates how classification performance degrades when training data is severely limited, using EfficientNetB0 as the backbone.

| Training shots | Total train images | Notes                        |
|----------------|--------------------|------------------------------|
| 10 shots       | 210 (10/class)     | Extreme low-data regime      |
| 80 shots       | 1680 (80/class)    | Full training set (baseline) |

The validation and test sets are kept **identical** across both experiments to ensure a fair comparison. Results are presented as:
- **Accuracy vs. shots curve** — quantifies the performance drop from full to few-shot training
- **Grad-CAM per shot size** — visualizes how the model's attention shifts with less training data

> EfficientNetB0 was chosen for this experiment as it is specifically designed to perform well in low-data regimes, making it the most meaningful architecture to stress-test with limited samples.

---

## Installation & Setup

**Requirements:** Python 3.8+

```bash
pip install tensorflow keras scikit-learn scikit-image opencv-python matplotlib kagglehub seaborn
```

**Download the dataset** (handled automatically in the notebooks via `kagglehub`):

```python
import kagglehub
path = kagglehub.dataset_download("ashikahmmed/uc-merce", output_dir="./data")
```

> A Kaggle account and API token (`~/.kaggle/kaggle.json`) are required.

---

## How to Run

```bash
# Baseline VGG16
jupyter notebook train_vgg16_uc_merced.ipynb

# Architecture comparison
jupyter notebook comparison_architecture.ipynb

# Few-shot experiment
jupyter notebook few_shot_efficientnet.ipynb
```

Run cells sequentially in all notebooks — the dataset is downloaded and split automatically.

**To test on your own images**, add `.jpg` files to `external_data/` and update the paths in the last section of `train_vgg16_uc_merced.ipynb`.

---

## Note
Model weights and dataset are not included in this repo — they are reproducible by running the notebooks from the top.

---

## Project Structure

```
.
├── train_vgg16_uc_merced.ipynb       # Baseline VGG16 notebook
├── comparison_architecture.ipynb     # Architecture comparison notebook
├── few_shot_efficientnet.ipynb       # Few-shot learning notebook
├── data/                             # Generated on first run (not tracked)
│   └── converted_uc_merced_data/
│       ├── train/
│       ├── validate/
│       └── test/
├── external_data/                    # Place custom test images here
└── figures/
    ├── confusion_matrix.png
    ├── gradcam_results.png
    ├── model_plot.png
    ├── comparison_confusion_matrices.png
    ├── gradcam_comparison.png
    ├── fewshot_accuracy_curve.png
    └── gradcam_fewshot.png
```

---

## References

- [UC Merced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) — Yang & Newsam, ACM SIGSPATIAL 2010
- [VGG16](https://arxiv.org/abs/1409.1556) — Simonyan & Zisserman, 2014
- [ResNet50](https://arxiv.org/abs/1512.03385) — He et al., 2015
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
- [Grad-CAM](https://arxiv.org/abs/1610.02391) — Selvaraju et al., 2017
- [Keras Applications](https://keras.io/api/applications/)