---
title: CIFAR-10 Image Classifier (Gradio)
emoji: 🤖
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.0.0
app_file: gradio_app.py
pinned: false
license: mit
---

# 🤖 CIFAR-10 Image Classifier (Gradio)

Gradio variant of the SE 489 MLOps bootstrap app. Identical model, identical
predictions — just a different UI framework.

This is the README you push to root if you choose the **Gradio** path. The
Streamlit equivalent lives in `../huggingface_space/README.md`.

## How to use

1. Open the app.
2. Either upload a single image on the first tab or upload many at once on
   the Batch tab.
3. The model resizes inputs to 32 x 32 and returns per-class confidences.

## What the model is

- **Architecture:** 3-layer CNN with two fully-connected layers
- **Training data:** CIFAR-10 (50,000 images, 10 classes)
- **Input size:** 32 x 32 pixels (resized from your upload)
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Tech

- **UI:** Gradio 5 (Blocks API)
- **Model:** PyTorch
- **Deployment:** Hugging Face Spaces, auto-deployed from GitHub Actions

## License

MIT.
