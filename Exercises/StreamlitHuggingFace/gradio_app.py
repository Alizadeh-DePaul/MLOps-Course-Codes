"""Gradio UI for the CIFAR-10 image classifier — complete reference app.

This file is a finished, deploy-ready bootstrap. Same idea as `app.py` but
using Gradio instead of Streamlit. Pick whichever framework you prefer for
your own MLOps project and adapt the model-specific bits.

Sections marked `# >>> REPLACE FOR YOUR MODEL` are the parts that change when
you swap in a different model. Everything else stays the same.

Run locally:
    python gradio_app.py

Deploy to Hugging Face Spaces using the YAML in
`huggingface_space_gradio/README.md`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# >>> REPLACE FOR YOUR MODEL ------------------------------------------------
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class SimpleCNN(nn.Module):
    """Same architecture as app.py / train_model.py. Keep them in sync."""

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


def _load_model(weights_path: str = "model.pth") -> SimpleCNN:
    model = SimpleCNN(num_classes=len(CIFAR10_CLASSES))
    try:
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
    except FileNotFoundError:
        print(f"WARNING: {weights_path} not found - predictions will be random.")
    model.eval()
    return model


MODEL = _load_model()
TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
# <<< REPLACE FOR YOUR MODEL ------------------------------------------------


# --- Inference helpers (model-agnostic) ------------------------------------
def predict_single(image: Image.Image | None) -> dict[str, float]:
    """Single-image inference. Returns a label-score dict for `gr.Label`."""
    if image is None:
        return {}
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.nn.functional.softmax(MODEL(tensor)[0], dim=0)
    return {cls: float(probabilities[i]) for i, cls in enumerate(CIFAR10_CLASSES)}


def predict_batch(files: list | None) -> tuple[pd.DataFrame, str | None]:
    """Batch inference. Returns a DataFrame + path to a downloadable CSV."""
    if not files:
        return pd.DataFrame(), None

    rows = []
    for f in files:
        # Gradio passes either a NamedString path or a tempfile object;
        # both expose `.name` as the path on disk.
        path = Path(f.name if hasattr(f, "name") else f)
        image = Image.open(path).convert("RGB")
        tensor = TRANSFORM(image).unsqueeze(0)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(MODEL(tensor)[0], dim=0)
        top_idx = int(torch.argmax(probs))
        rows.append({
            "filename": path.name,
            "prediction": CIFAR10_CLASSES[top_idx],
            "confidence": round(float(probs[top_idx]), 4),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    df = pd.DataFrame(rows)
    csv_path = Path("/tmp") / "batch_predictions.csv"
    df.to_csv(csv_path, index=False)
    return df, str(csv_path)


# --- Interface (Blocks API) ------------------------------------------------
ABOUT_MD = """\
### About this app

- **Model:** SimpleCNN (3 conv layers, 2 FC layers)
- **Dataset:** CIFAR-10 (10 classes, 32 x 32 RGB)
- **Framework:** PyTorch + torchvision
- **UI:** Gradio (Blocks)
- **Deployment:** Hugging Face Spaces, via GitHub Actions

This is a reference template from
[SE 489 - MLOps](https://github.com/Alizadeh-DePaul/MLOps-Course-Codes).
Fork it, swap in your own model class + preprocessing, and ship.

### Tips for adapting this

1. Replace `SimpleCNN` and the `TRANSFORM` pipeline.
2. Update `CIFAR10_CLASSES` with your label list (order matters - it must
   match your model's output order).
3. Commit your trained `model.pth` either to the HF Space directly or pull
   it from the Hub at startup with `huggingface_hub.hf_hub_download`.
4. If your inputs aren't images, swap `gr.Image` for `gr.Audio`, `gr.Text`,
   `gr.Video`, or whatever the user uploads.

### License

MIT.
"""


def build_interface() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), title="CIFAR-10 Classifier") as demo:
        gr.Markdown(
            "# 🤖 CIFAR-10 Image Classifier\n\n"
            "A complete Gradio + Hugging Face Spaces reference app. "
            "Fork, swap in your model, ship."
        )

        with gr.Tab("Single image"):
            with gr.Row():
                with gr.Column():
                    single_input = gr.Image(type="pil", label="Upload image")
                    single_btn = gr.Button("Classify", variant="primary")
                with gr.Column():
                    single_output = gr.Label(
                        num_top_classes=5,
                        label="Top 5 predictions",
                    )
            # Auto-classify on upload AND on button click.
            single_input.change(predict_single, single_input, single_output)
            single_btn.click(predict_single, single_input, single_output)

        with gr.Tab("Batch upload"):
            with gr.Row():
                batch_input = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="Upload one or more images",
                )
            batch_btn = gr.Button("Process batch", variant="primary")
            batch_table = gr.Dataframe(
                headers=["filename", "prediction", "confidence", "timestamp"],
                label="Results",
                wrap=True,
            )
            batch_csv = gr.File(label="Download CSV")
            batch_btn.click(
                predict_batch,
                inputs=batch_input,
                outputs=[batch_table, batch_csv],
            )

        with gr.Tab("About"):
            gr.Markdown(ABOUT_MD)

    return demo


if __name__ == "__main__":
    build_interface().launch()
