# Burmese Handwritten Digit Recognition (BHDR) 🇲🇲

[![PyPI version](https://badge.fury.io/py/bhdr.svg)](https://badge.fury.io/py/bhdr)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, production-ready Deep Learning library that classifies handwritten Burmese digits (၀-၉) using a custom Convolutional Neural Network (CNN) built with PyTorch.

This package bridges the domain gap between clean academic datasets and real-world, unconstrained handwriting by utilizing custom aspect-aware scaling and anti-squish mathematical padding.

## 📦 Installation

You can install `bhdr` directly from PyPI using pip:

```bash
pip install bhdr
```

*Note: This package requires Python >= 3.10.*

## 🚀 Quick Start Usage

`bhdr` is designed to be incredibly simple to use inside your own Python projects. Just import the `predict_digit` function and pass it an image path!

```python
from bhdr import predict_digit

# Path to an image of a handwritten Burmese digit
image_path = "test_images/my_digit.jpg"

# Run the inference engine
predicted_class = predict_digit(image_path)

print(f"The model predicted the digit is: {predicted_class}")
```

## 🧠 Model Architecture (V2)

The core model is a highly optimized 2-layer Convolutional Neural Network.
* **Input:** `1x28x28` Grayscale Tensor
* **Layer 1:** Conv2D (16 filters) -> ReLU -> MaxPool2D
* **Layer 2:** Conv2D (32 filters) -> ReLU -> MaxPool2D
* **Classifier:** Fully Connected Linear Layer (10 output classes for digits ၀-၉)

## 🛠️ Intelligent Preprocessing Pipeline

Standard computer vision models often fail on real-world Burmese handwriting due to extreme variance in stroke thickness and aspect ratio distortion (especially for tall/thin digits like ၇ and ၂). 

This library features a custom inference pipeline that:
1. Inverts colors to a standard black background / white stroke format.
2. Crops the image exactly to the digit's bounding box to remove dead space.
3. Dynamically pads the shortest side to create a perfect square *without* stretching or distorting the original digit's aspect ratio.
4. Resizes to the 28x28 tensor expected by the CNN.

---
**Author:** Shin Thant Tun