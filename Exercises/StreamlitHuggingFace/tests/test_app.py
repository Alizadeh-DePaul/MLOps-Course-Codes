"""Smoke tests run by CI before deploying to Hugging Face Spaces.

These tests deliberately avoid spinning up a Streamlit or Gradio server.
They check:
1. Both app modules import cleanly (catches syntax errors and stale imports).
2. The shared model architecture is wired up and produces sensibly-shaped output.
3. Pure helper functions return what the UIs expect.
4. Required packaging + HF Space front-matter files are present.

If a test here fails, the deploy job is blocked — that's the point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import app
import gradio_app


# --- Import + structure ----------------------------------------------------
def test_streamlit_app_module_imports() -> None:
    assert hasattr(app, "main")
    assert hasattr(app, "SimpleCNN")
    assert hasattr(app, "predict")
    assert hasattr(app, "preprocess_image")
    assert hasattr(app, "top_k")


def test_gradio_app_module_imports() -> None:
    assert hasattr(gradio_app, "build_interface")
    assert hasattr(gradio_app, "predict_single")
    assert hasattr(gradio_app, "predict_batch")
    assert hasattr(gradio_app, "SimpleCNN")


def test_cifar10_classes_count() -> None:
    assert len(app.CIFAR10_CLASSES) == 10
    assert app.CIFAR10_CLASSES == gradio_app.CIFAR10_CLASSES, (
        "Class lists must match between Streamlit and Gradio apps"
    )


# --- Model -----------------------------------------------------------------
def test_model_forward_shape() -> None:
    """Forward pass on a single 3x32x32 input returns 10 logits."""
    model = app.SimpleCNN()
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 10)


# --- Streamlit-side helpers ------------------------------------------------
def test_preprocess_image_resizes_to_32x32() -> None:
    big_image = Image.new("RGB", (256, 256), color=(128, 64, 200))
    tensor = app.preprocess_image(big_image)
    assert tensor.shape == (1, 3, 32, 32)


def test_predict_returns_probability_distribution() -> None:
    model = app.SimpleCNN()
    model.eval()
    img = Image.new("RGB", (32, 32), color=(0, 255, 0))
    tensor = app.preprocess_image(img)
    probs = app.predict(model, tensor)
    assert probs.shape == (10,)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)
    assert (probs >= 0).all()


def test_top_k_returns_sorted_pairs() -> None:
    probs = np.array([0.05, 0.02, 0.50, 0.10, 0.08, 0.07, 0.03, 0.05, 0.05, 0.05])
    top_3 = app.top_k(probs, app.CIFAR10_CLASSES, k=3)
    assert len(top_3) == 3
    assert top_3[0] == ("bird", 0.50)
    # Strictly descending
    assert top_3[0][1] >= top_3[1][1] >= top_3[2][1]


# --- Gradio-side helpers ---------------------------------------------------
def test_predict_single_returns_label_dict() -> None:
    img = Image.new("RGB", (32, 32), color=(10, 200, 10))
    result = gradio_app.predict_single(img)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(gradio_app.CIFAR10_CLASSES)
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_predict_single_handles_none() -> None:
    """Gradio passes None when the input slot is empty - don't crash."""
    assert gradio_app.predict_single(None) == {}


def test_gradio_build_interface_returns_blocks() -> None:
    demo = gradio_app.build_interface()
    # Smoke check: it's a Gradio Blocks instance with at least one tab
    import gradio as gr
    assert isinstance(demo, gr.Blocks)


# --- Packaging + HF Space config ------------------------------------------
def test_huggingface_streamlit_readme_has_frontmatter() -> None:
    readme = Path(__file__).resolve().parents[1] / "huggingface_space" / "README.md"
    assert readme.exists(), f"Missing {readme}"
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "sdk: streamlit" in text
    assert "app_file: app.py" in text


def test_huggingface_gradio_readme_has_frontmatter() -> None:
    readme = Path(__file__).resolve().parents[1] / "huggingface_space_gradio" / "README.md"
    assert readme.exists(), f"Missing {readme}"
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "sdk: gradio" in text
    assert "app_file: gradio_app.py" in text


def test_pyproject_pins_python_311() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11,<3.12"' in text


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "model.pth").exists(),
    reason="model.pth not present - run `python train_model.py` to generate it",
)
def test_load_model_with_real_weights() -> None:
    weights = Path(__file__).resolve().parents[1] / "model.pth"
    model = app.SimpleCNN()
    state_dict = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10)
