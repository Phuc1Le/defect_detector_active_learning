# Active Learning for Synthetic Bottle Defect Detection

A computer vision project that explores whether **active learning** can reduce manual labeling effort for industrial defect detection. Starting from a model trained only on **synthetically generated defects**, the system iteratively selects additional real images for labeling using either **uncertainty-based active learning** or **random sampling**, and compares their performance as more labeled data becomes available.

## Overview

Collecting labeled defect images is expensive because defects are relatively rare and require human inspection. This project investigates whether an active learning strategy can make better use of a limited labeling budget by prioritizing the most informative unlabeled images.

The workflow consists of:

1. Training an initial classifier using only synthetic defect images.
2. Evaluating the bootstrap model on a held-out real test set.
3. Iteratively revealing labels from an unlabeled pool.
4. Comparing:
   - **Active Learning** (uncertainty sampling)
   - **Random Sampling**
5. Measuring accuracy, precision, recall, and F1 score throughout the labeling process.

---

## Results

### Bootstrap Model (0 Real Labels)

The initial model was trained **only on synthetic defects**, without using any real labeled training images.

| Metric | Score |
|--------|------:|
| Accuracy | **0.729** |
| Precision | **0.429** |
| Recall | **0.692** |
| F1 Score | **0.529** |

Although the synthetic data provides a reasonable starting point, performance is limited due to the domain gap between synthetic and real defects.

---

### Final Performance

After progressively incorporating real labeled images:

| Strategy | Accuracy | F1 Score |
|----------|----------:|----------:|
| Active Learning | **0.932** | **0.850** |
| Random Sampling | **0.932** | **0.830** |

Both approaches substantially improved over the bootstrap model, while **active learning consistently achieved better performance than random sampling at every labeling checkpoint**, indicating that selecting uncertain samples provides a more efficient use of labeling effort.

More importantly, by combining active learning with synthetic defect generation, the model reached nearly 90% accuracy using only 50% of the real labeled data, highlighting the effectiveness of reducing annotation requirements while maintaining strong performance.
---

### Accuracy

![Accuracy](results/accuracy_chart.png)

---

### F1 Score

![F1](results/f1_chart.png)

---


## Source Files

### `make_splits.py`

Scans the dataset under `data/raw/<category>`, assigns binary labels (clean or defective), partitions the data into bootstrap, pool, and test sets, and generates `manifest.csv`.

### `synth_defects.py`

Generates synthetic defects from clean bottle images. Implemented defect generators include scratches, cracks, scuffs, blobs, and randomly combined defects to increase training diversity.

### `dataset.py`

Defines the `BottleDataset` class responsible for loading images, applying preprocessing transforms, and supplying samples to PyTorch's `DataLoader`.

### `train.py`

Builds a pretrained ResNet-18 classifier and trains it using weighted random sampling to address class imbalance.

### `evaluate.py`

Evaluates the trained model by computing:

- Accuracy
- Precision
- Recall
- F1 Score

It also returns prediction probabilities for active learning.

### `active_learning.py`

Implements the iterative labeling process.

Two sampling strategies are supported:

- **Active Learning:** selects images with prediction probabilities closest to 0.5 (highest uncertainty)
- **Random Sampling:** selects images uniformly at random

### `experiment.py`

Runs the complete experiment across multiple random seeds, averages the results, exports evaluation metrics, and generates comparison plots.

---

## Methodology

1. Create train/pool/test splits.
2. Generate synthetic defects from clean bottle images.
3. Train a bootstrap model using only synthetic defects.
4. Evaluate on real test images.
5. Repeat:
   - Predict probabilities on the unlabeled pool.
   - Select a batch of images:
     - uncertainty sampling (active learning), or
     - random sampling.
   - Reveal their labels.
   - Continue training with the expanded labeled dataset.
6. Compare both strategies throughout the labeling process.

---

## Model

- **Architecture:** ResNet-18
- **Transfer Learning:** ImageNet pretrained weights
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam
- **Input Resolution:** 224 × 224

---

## Dataset

The project uses the **Bottle** category from the MVTec Anomaly Detection dataset.

Training begins using synthetic defects generated from clean bottle images, while evaluation is performed exclusively on real defect images.

---

## Running the Project

The easiest way to reproduce the project is with the notebook:

```
notebooks/defect_detector.ipynb
```

The notebook walks through:

1. Dataset preparation
2. Synthetic defect generation
3. Bootstrap training
4. Active learning experiment
5. Evaluation
6. Result visualization

---

## Future Improvements

- More realistic synthetic defect generation
- Additional uncertainty sampling strategies (entropy, margin sampling)
- Fine-tuning larger pretrained vision models
- Multi-class defect classification
- Pixel-level defect localization using segmentation models
- Evaluation on additional MVTec object categories

---

## Technologies

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn