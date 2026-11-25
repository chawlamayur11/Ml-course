# Homework 8: Neural Networks and Deep Learning

Course: [Machine Learning Zoomcamp (2026 Cohort)](https://courses.datatalks.club/ml-zoomcamp-2026)  
Module: **08 - Neural Networks & Deep Learning**

---

## Overview

In this homework, we build, train, and evaluate a Convolutional Neural Network (CNN) binary classifier using PyTorch to predict whether an image contains curly hair (`curly`) or straight hair (`straight`).

We evaluate both:
1. **Baseline Model**: Trained for 10 epochs on normalized images without data augmentation.
2. **Augmented Model**: Trained for 10 epochs with random horizontal flipping and random rotation (up to 50 degrees).

---

## Model Architecture

The convolutional model is defined as follows:
- **Input shape**: `(3, 200, 200)`
- **Convolutional Layer**: `nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)`
  - Output shape: `(32, 198, 198)`
  - Trainable parameters: `(3 * 3 * 3 + 1) * 32 = 28 * 32 = 896`
- **Max Pooling Layer**: `nn.MaxPool2d(kernel_size=2, stride=2)`
  - Output shape: `(32, 99, 99)`
  - Trainable parameters: `0`
- **Flattening**: `32 * 99 * 99 = 313,632`
- **Dense Layer (Hidden)**: `nn.Linear(in_features=313632, out_features=64)`
  - Trainable parameters: `313,632 * 64 + 64 = 20,072,512`
- **Activation**: `nn.ReLU()`
- **Output Layer**: `nn.Linear(in_features=64, out_features=1)`
  - Trainable parameters: `64 * 1 + 1 = 65`

### Total Parameters
$$\text{Total Parameters} = 896 + 20,072,512 + 65 = 20,073,473 \approx 20\,\text{M}$$

---

## Homework Questions & Verified Answers

### Question 1: Loss Function
Since we are using PyTorch and our model does not have a Sigmoid layer at the output (it outputs raw logits), what is the appropriate loss function for binary classification?

- **Answer**: `nn.BCEWithLogitsLoss()`
- **Explanation**: `nn.BCEWithLogitsLoss` combines a `Sigmoid` activation layer and `BCELoss` into one single layer. This is mathematically and numerically more stable than using a separate `Sigmoid` followed by `BCELoss`, as it leverages the log-sum-exp trick for numerical stability.

---

### Question 2: Total Number of Parameters
What is the total number of parameters in our model?

- **Answer**: `20,073,473` (~20M)
- **Calculation**:
  - Conv2d weights: $3 \times 3 \times 3 \times 32 = 864$
  - Conv2d biases: $32$
  - Linear (hidden) weights: $313,632 \times 64 = 20,072,448$
  - Linear (hidden) biases: $64$
  - Linear (output) weights: $64 \times 1 = 64$
  - Linear (output) biases: $1$
  - Total: $896 + 20,072,512 + 65 = 20,073,473$.

---

### Question 3: Baseline Median Training Accuracy
What is the median of training accuracy across all 10 epochs for the baseline model?

- **Answer**: `0.81`
- **Metrics by Epoch**:
  - Epoch 1: 0.6450
  - Epoch 2: 0.7175
  - Epoch 3: 0.7550
  - Epoch 4: 0.7912
  - Epoch 5: 0.8088
  - Epoch 6: 0.8075
  - Epoch 7: 0.8362
  - Epoch 8: 0.8850
  - Epoch 9: 0.9100
  - Epoch 10: 0.9388
- **Median Calculation**:
  - Sorted: `[0.6450, 0.7175, 0.7550, 0.7912, 0.8075, 0.8088, 0.8362, 0.8850, 0.9100, 0.9388]`
  - Median: $(0.8075 + 0.8088) / 2 = 0.8081 \approx 0.81$.

---

### Question 4: Baseline Training Loss Standard Deviation
What is the standard deviation of training loss across all 10 epochs for the baseline model?

- **Answer**: `0.139` (population std) / `0.147` (sample std)
- **Loss values**: `[0.6385, 0.5441, 0.4955, 0.4330, 0.4048, 0.3953, 0.3437, 0.2806, 0.2236, 0.1584]`
- **Mean Loss**: `0.3918`
- **Population Standard Deviation (`ddof=0`)**: `0.1391`
- **Sample Standard Deviation (`ddof=1`)**: `0.1466`

---

### Question 5: Augmented Mean Evaluation Loss
What is the mean of evaluation loss across all 10 epochs for the model trained with data augmentations?

- **Answer**: `0.575` (closest to `0.6` / `0.57`)
- **Evaluation Loss per Epoch**:
  - `[0.6552, 0.6608, 0.5787, 0.5679, 0.5680, 0.5510, 0.5288, 0.5401, 0.5632, 0.5404]`
- **Mean**: $5.7541 / 10 = 0.5754 \approx 0.575$.

---

### Question 6: Augmented Average Evaluation Accuracy (Last 5 Epochs)
What is the average of test/evaluation accuracy for the last 5 epochs (epochs 6 to 10) for the model trained with augmentations?

- **Answer**: `0.72` (or `0.73`)
- **Evaluation Accuracy (Epochs 6-10)**:
  - Epoch 6: 0.7015 (70.1%)
  - Epoch 7: 0.7413 (74.1%)
  - Epoch 8: 0.7114 (71.1%)
  - Epoch 9: 0.7363 (73.6%)
  - Epoch 10: 0.7313 (73.1%)
- **Average**: $(0.7015 + 0.7413 + 0.7114 + 0.7363 + 0.7313) / 5 = 3.6218 / 5 = 0.7244 \approx 0.72$.

---

## Files in this Directory

- `README.md`: This summary report and solutions document.
- `train.py`: Self-contained PyTorch script to train both baseline and augmented models, calculate statistics, and export training metrics.
- `history_baseline.json`: Epoch-by-epoch loss and accuracy metrics for the baseline model.
- `history_augmented.json`: Epoch-by-epoch loss and accuracy metrics for the augmented model.
