# UC Merced Land Use Classification — VGG16 Transfer Learning

Satellite image classification across 21 land-use categories using transfer learning on the [UC Merced Land Use Dataset](https://www.kaggle.com/datasets/ashikahmmed/uc-merce).

---

## Overview

This project applies **VGG16 transfer learning** (pretrained on ImageNet) to classify aerial imagery into 21 land-use classes. The convolutional base is frozen and used as a fixed feature extractor; only a lightweight fully connected classifier is trained on top.
Dataset size: 2100 images (21 classes × 100 images each)
Image size: 256 × 256 pixels

**Classes:** agricultural, airplane, baseball diamond, beach, buildings, chaparral, dense residential, forest, freeway, golf course, harbor, intersection, medium residential, mobile home park, overpass, parking lot, river, runway, sparse residential, storage tanks, tennis court.

---

## Approach

1. **Data split** — 100 images/class → 80 train / 10 validation / 10 test 
2. **Feature extraction** — Images passed once through frozen VGG16 (no top layer) to produce bottleneck feature vectors
3. **Classifier training** — Lightweight FC network trained on extracted features:
   - `Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(21, Softmax)`
   - Optimizer: Adam (`lr=2e-5`), Loss: categorical crossentropy, 20 epochs
4. **Evaluation** — Accuracy + classification report on held-out test set
5. **External inference** — Model tested on arbitrary images (e.g. Google images)

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
jupyter notebook train_vgg16_uc_merced.ipynb
```

Run cells sequentially. The notebook will:
- Download and split the dataset
- Extract bottleneck features from VGG16
- Train and save the classifier (`VGG16_model.h5`)
- Plot training curves
- Evaluate on the test set
- Run inference on external images placed in `data/external_data/`

**To test on your own images**, add `.jpg` files to `data/external_data/` and update the paths in the last notebook section.

---

## Results

| Split      | Accuracy |
|------------|----------|
| Training   | ~95%     |
| Validation | ~90%     |
| Test       | ~90%     |

Per-class performance is printed via `sklearn.metrics.classification_report` after evaluation.

---

## Project Structure

```
.
├── UC_merced2-08082018.ipynb   # Main notebook
├── VGG16_model.h5              # Saved model (generated after training)
├── data/
│   ├── converted_uc_merced_data/
│   │   ├── train/
│   │   ├── validate/
│   │   └── test/
│   └── external_data/          # Place custom test images here
└── model_plot.png              # Model architecture diagram
```

---

## References

- [UC Merced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) — Yang & Newsam, ACM SIGSPATIAL 2010
- [VGG16](https://arxiv.org/abs/1409.1556) — Simonyan & Zisserman, 2014
- [Keras Applications](https://keras.io/api/applications/vgg/)
