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

| Architecture   | Test Accuracy | Notes                          |
|----------------|---------------|--------------------------------|
| VGG16          | ~90%          | Baseline                       |
| EfficientNetB0 | ~90%          | Fastest training               |
| ResNet50       | ~96%          | Best accuracy                  |

Each architecture is evaluated with:
- **Confusion matrix** — side-by-side normalized heatmaps across all 21 classes
- **Grad-CAM** — per-architecture visualization of discriminative regions, with correct preprocessing applied per model
- **Training time** — wall-clock time reported for each architecture as a practical reference

> **Note on preprocessing:** Each architecture requires its own preprocessing function (`vgg16_preprocess`, `resnet_preprocess`, `efficientnet_preprocess`). Using the wrong one causes the model to produce random predictions — a subtle bug documented and fixed during development.

---

## Installation & Setup

**Requirements:** Python 3.8+

```bash
pip install tensorflow keras scikit-learn scikit-image opencv-python matplotlib kagglehub seaborn
```

**Download the dataset** (handled automatically in the notebook via `kagglehub`):

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
```

Run cells sequentially in both notebooks — the dataset is downloaded and split automatically.

**To test on your own images**, add `.jpg` files to `data/external_data/` and update the paths in the last section of `train_vgg16_uc_merced.ipynb`.

---

## Note
Model weights and dataset are not included in this repo — they are reproducible by running the notebooks from the top.

---

## Project Structure

```
.
├── train_vgg16_uc_merced.ipynb     # Baseline VGG16 notebook
├── comparison_architecture.ipynb   # Architecture comparison notebook
├── data/                           # Generated on first run (not tracked)
│   ├── converted_uc_merced_data/
│   │   ├── train/
│   │   ├── validate/
│   │   └── test/
├── external_data/                  # Place custom test images here
├── figure/
   ├── confusion_matrix.png
   ├── gradcam_results.png
   ├── model_plot.png
   ├── comparison_confusion_matrices.png
   └── gradcam_comparison.png
```

---

## References

- [UC Merced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) — Yang & Newsam, ACM SIGSPATIAL 2010
- [VGG16](https://arxiv.org/abs/1409.1556) — Simonyan & Zisserman, 2014
- [ResNet50](https://arxiv.org/abs/1512.03385) — He et al., 2015
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
- [Grad-CAM](https://arxiv.org/abs/1610.02391) — Selvaraju et al., 2017
- [Keras Applications](https://keras.io/api/applications/)