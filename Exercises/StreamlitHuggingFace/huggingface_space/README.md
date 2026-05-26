---
title: CIFAR-10 Image Classifier
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.50.0
app_file: app.py
pinned: false
license: mit
---

# 🤖 CIFAR-10 Image Classifier

A small Streamlit web app that classifies uploaded images into one of 10
CIFAR-10 categories using a SimpleCNN trained on the CIFAR-10 dataset.

Built as the deployment exercise for **SE 489: Machine Learning Engineering
for Production (MLOps)** at DePaul University.

## How to use

1. Open the app.
2. Upload a PNG or JPEG image.
3. The app resizes it to 32×32, runs it through the CNN, and shows per-class
   confidence scores.

## What the model is

- **Architecture:** 3-layer CNN with two fully-connected layers
- **Training data:** CIFAR-10 (50,000 images, 10 classes)
- **Input size:** 32 × 32 pixels (resized from your upload)
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Limitations

- The model is tiny and only briefly trained. Don't expect accuracy on
  high-resolution or out-of-distribution images.
- Predictions are deterministic per image — no on-device fine-tuning.

## Tech

- **UI:** Streamlit
- **Model:** PyTorch
- **Charts:** Plotly
- **Deployment:** Hugging Face Spaces, auto-deployed from GitHub Actions

## License

MIT.
