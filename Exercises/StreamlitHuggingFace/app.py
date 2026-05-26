"""Streamlit UI for the CIFAR-10 image classifier — complete reference app.

This file is a finished, deploy-ready bootstrap. Copy it into your own MLOps
project and adapt the model-specific bits (the SimpleCNN class, the class
labels, the preprocessing transforms). The framework scaffolding around it —
tabs, sidebar, batch upload, CSV download, session analytics — is intended
to be reused as-is.

Sections marked `# >>> REPLACE FOR YOUR MODEL` are the parts that change when
you swap in a different model. Everything else stays the same.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# --- Page config (MUST be the first Streamlit call) -------------------------
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# >>> REPLACE FOR YOUR MODEL ------------------------------------------------
# Class labels in the same order the model outputs them.
CIFAR10_CLASSES: list[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class SimpleCNN(nn.Module):
    """Tiny 3-conv CNN. Matches the architecture in train_model.py.

    If you swap in a different model, keep the class name `SimpleCNN` or
    rename it consistently across this file and `train_model.py`.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


@st.cache_resource
def load_model(weights_path: str = "model.pth") -> SimpleCNN:
    """Load model weights. `@st.cache_resource` ensures one load per session."""
    model = SimpleCNN(num_classes=len(CIFAR10_CLASSES))
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        st.warning(
            f"No `{weights_path}` found. Predictions will be random. "
            "Run `python train_model.py` (locally) or commit `model.pth` to "
            "your HF Space."
        )
    model.eval()
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Resize + normalize a PIL image. Adapt the transform to your model."""
    image = image.convert("RGB").resize((32, 32))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform(image).unsqueeze(0)


def predict(model: SimpleCNN, image_tensor: torch.Tensor) -> np.ndarray:
    """Run inference and return per-class probabilities."""
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.nn.functional.softmax(logits[0], dim=0)
    return probabilities.numpy()
# <<< REPLACE FOR YOUR MODEL ------------------------------------------------


# --- Framework helpers (reusable, model-agnostic) --------------------------
def top_k(probabilities: np.ndarray, classes: list[str], k: int = 3
          ) -> list[tuple[str, float]]:
    """Return the top-k (class, probability) pairs."""
    idx = np.argsort(probabilities)[::-1][:k]
    return [(classes[i], float(probabilities[i])) for i in idx]


def confidence_bar_chart(probabilities: np.ndarray, classes: list[str]):
    """Horizontal bar chart of class confidences (Plotly)."""
    df = (
        pd.DataFrame({"Class": classes, "Confidence": probabilities * 100})
        .sort_values("Confidence", ascending=True)
    )
    fig = px.bar(
        df, x="Confidence", y="Class", orientation="h",
        color="Confidence", color_continuous_scale="viridis",
        title="Prediction Confidence (%)",
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def log_prediction(image_name: str, predicted_class: str, confidence: float) -> None:
    """Append a prediction to the in-session analytics log."""
    st.session_state.setdefault("predictions_log", []).append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image_name": image_name,
        "predicted_class": predicted_class,
        "confidence": float(confidence),
    })


# --- Tabs ------------------------------------------------------------------
def render_single_tab(model: SimpleCNN, threshold: float, show_raw: bool) -> None:
    """Single-image upload + prediction view."""
    uploaded = st.file_uploader(
        "Choose an image", type=["png", "jpg", "jpeg"],
        help="Native input is 32x32; the app resizes any size you upload.",
        key="single_upload",
    )
    if uploaded is None:
        st.info("Upload an image to see a prediction.")
        return

    image = Image.open(io.BytesIO(uploaded.getvalue()))

    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption=uploaded.name, width="stretch")

    with right:
        with st.spinner("Running inference..."):
            probabilities = predict(model, preprocess_image(image))

        top_3 = top_k(probabilities, CIFAR10_CLASSES, k=3)
        top_class, top_confidence = top_3[0]

        if top_confidence < threshold:
            st.warning(
                f"Top prediction **{top_class}** is below your confidence "
                f"threshold of {threshold:.0%}."
            )
        else:
            st.success(f"**Prediction:** {top_class.capitalize()}")
            st.metric("Confidence", f"{top_confidence:.2%}")

        st.subheader("Top 3")
        for cls, prob in top_3:
            st.write(f"**{cls.capitalize()}** — {prob:.2%}")
            st.progress(prob)

        log_prediction(uploaded.name, top_class, top_confidence)

    st.plotly_chart(
        confidence_bar_chart(probabilities, CIFAR10_CLASSES), width="stretch"
    )

    if show_raw:
        st.subheader("Raw probabilities")
        st.dataframe(
            pd.DataFrame({
                "Class": CIFAR10_CLASSES, "Probability": probabilities,
            }).sort_values("Probability", ascending=False),
            width="stretch",
        )


def render_batch_tab(model: SimpleCNN) -> None:
    """Batch upload + CSV download."""
    files = st.file_uploader(
        "Upload multiple images", type=["png", "jpg", "jpeg"],
        accept_multiple_files=True, key="batch_upload",
    )
    if not files:
        st.info("Upload one or more images, then click **Process batch**.")
        return

    if not st.button(f"Process batch ({len(files)} images)"):
        return

    rows = []
    progress = st.progress(0.0)
    for i, f in enumerate(files, start=1):
        image = Image.open(io.BytesIO(f.getvalue()))
        probabilities = predict(model, preprocess_image(image))
        top_class, top_confidence = top_k(probabilities, CIFAR10_CLASSES, k=1)[0]
        rows.append({
            "filename": f.name,
            "prediction": top_class,
            "confidence": round(top_confidence, 4),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        log_prediction(f.name, top_class, top_confidence)
        progress.progress(i / len(files))

    results = pd.DataFrame(rows)
    st.subheader(f"Batch results ({len(results)} predictions)")
    st.dataframe(results, width="stretch")

    st.download_button(
        label="Download results as CSV",
        data=results.to_csv(index=False),
        file_name="batch_predictions.csv",
        mime="text/csv",
    )


def render_about_tab() -> None:
    """Static About panel — swap text for your own model."""
    st.markdown(
        """
        ### About this app

        - **Model:** SimpleCNN (3 conv layers, 2 FC layers)
        - **Dataset:** CIFAR-10 (50,000 training images, 10 classes)
        - **Input size:** 32 x 32 RGB
        - **Framework:** PyTorch + torchvision
        - **UI:** Streamlit
        - **Charts:** Plotly
        - **Deployment:** Hugging Face Spaces, via GitHub Actions

        This is a reference template from
        [SE 489 - MLOps](https://github.com/Alizadeh-DePaul/MLOps-Course-Codes).
        Fork it, swap in your own model class + preprocessing, and ship.

        ### Limitations

        - SimpleCNN is intentionally tiny — accuracy is OK, not great.
        - Predictions on out-of-distribution images (anything not resembling
          CIFAR-10) will be unreliable.
        - No on-device fine-tuning; the served model is the trained snapshot.

        ### License

        MIT.
        """
    )


# --- Sidebar ---------------------------------------------------------------
def render_sidebar() -> tuple[float, bool]:
    """Sidebar with model info + UI controls. Returns (threshold, show_raw)."""
    with st.sidebar:
        st.header("Model")
        st.metric("Architecture", "SimpleCNN")
        st.metric("Input", "32 x 32 RGB")
        st.metric("Classes", str(len(CIFAR10_CLASSES)))

        st.markdown("---")
        st.header("Settings")
        threshold = st.slider("Min confidence to display", 0.0, 1.0, 0.0, 0.05)
        show_raw = st.checkbox("Show raw probability table", value=False)

        st.markdown("---")
        st.caption("Classes:")
        st.write(", ".join(CIFAR10_CLASSES))

        log = st.session_state.get("predictions_log", [])
        if log:
            st.markdown("---")
            st.header("This session")
            st.metric("Predictions", len(log))
            st.metric("Avg confidence",
                      f"{np.mean([p['confidence'] for p in log]):.1%}")

    return threshold, show_raw


# --- Main ------------------------------------------------------------------
def main() -> None:
    st.title("🤖 CIFAR-10 Image Classifier")
    st.caption(
        "A complete Streamlit + Hugging Face Spaces reference app. "
        "Fork, swap in your model, ship."
    )

    threshold, show_raw = render_sidebar()
    model = load_model()

    single_tab, batch_tab, about_tab = st.tabs([
        "Single image", "Batch upload", "About",
    ])
    with single_tab:
        render_single_tab(model, threshold, show_raw)
    with batch_tab:
        render_batch_tab(model)
    with about_tab:
        render_about_tab()


if __name__ == "__main__":
    main()
