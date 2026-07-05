# Burmese Handwritten Digit Recognition (CNN) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RichardHtunn/BHDR/blob/main/startingAnOCR.ipynb)

An end-to-end Computer Vision project focused on classifying handwritten Myanmar (Burmese) digits (၀-၉) using a custom PyTorch Convolutional Neural Network. 

This project explores the domain gap between academic datasets and real-world, unconstrained handwriting, utilizing data augmentation and custom preprocessing pipelines to improve model robustness.

## 📖 Overview
While standard models achieve high accuracy (+99%) on clean validation sets like the BHDD (Burmese Handwritten Digit Dataset), they frequently fail in real-world applications due to variance in stroke thickness, aspect ratio distortion, and background noise. 

Inspired by the recent `myMNIST` benchmark research, this project implements a 2-layer CNN architecture and a robust image preprocessing pipeline to bridge that domain gap, allowing the model to accurately read thin, edge-to-edge, unpadded handwriting.

## Model Architecture (V2)
The network is a custom PyTorch CNN designed to extract both simple edge features and complex geometric loops (crucial for Burmese digits like ၅, ၆, and ၉).
* **Layer 1:** `Conv2d` (1 -> 16 channels) + ReLU + Max Pooling
* **Layer 2:** `Conv2d` (16 -> 32 channels) + ReLU + Max Pooling
* **Classifier:** Fully Connected Linear Layers (1568 -> 128 -> 10)
* **Optimizer:** Adam (`lr=0.001`)
* **Loss:** Cross-Entropy

## 🛠️ Engineering Challenges & Solutions

During real-world inference testing, the initial model failed due to three primary data pipeline issues. Here is how they were resolved:

### 1. The Aspect Ratio Distortion
* **Problem:** Raw images drawn on rectangular canvases were being forcibly squished into `28x28` squares via standard `transforms.Resize`, mathematically destroying the shape of tall digits like ၇ and ၂.
* **Solution:** Implemented a custom `fit_and_pad` function using PIL. It scales the longest edge to 20 pixels while maintaining the exact aspect ratio, and then pastes it onto a pure black `28x28` canvas, perfectly mimicking the BHDD dataset padding.

### 2. The Box Artifact (Thresholding)
* **Problem:** Dark gray backgrounds from drawing apps were being pasted onto pure black tensors, creating a sharp square artifact. The CNN locked onto the square edge rather than the digit, resulting in total prediction failure.
* **Solution:** Injected a thresholding binarization step `img.point(lambda p: 255 if p > 128 else 0)` before padding to crush gray backgrounds to pure `0` and boost strokes to `255`.

### 3. Stroke Thickness & The Domain Gap
* **Problem:** The model failed on incredibly thin handwriting because the `F.max_pool2d` layers compressed the thin strokes into disconnected dots. 
* **Solution:** Rebuilt the `DataLoader` to include a Data Augmentation pipeline (`RandomRotation`, `RandomAffine` for scaling and translation) applied only to the training set. This forced the CNN to learn true geometric topology rather than memorizing thick-stroke pixel layouts.

## How to Run
This project was developed entirely in Google Colab. 
1. Click the `Open in Colab` badge at the top of the `.ipynb` file.
2. Run the notebook sequentially to instantiate the network and the `fit_and_pad` pipeline.
3. Use the final Inference cell to upload your own custom `.png` or `.jpg` drawings of Burmese digits to test the model.

## Author
**Shin Thant Tun** Computer Engineering Undergraduate  
King Mongkut's Institute of Technology Ladkrabang (KMITL)
