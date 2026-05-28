"""Optional: a FastAPI app that captions images with a vision-language model.

This is the (optional) machine-learning step of the exercise. It is a
*separate* FastAPI app from app/main.py so the core exercise stays light:
the model dependencies (torch + transformers) are heavy and only installed
via the optional `[ml]` extra.

Install the extra, then run this app on its own:

    uv pip install -e ".[ml]"          # alt: pip install -e ".[ml]"
    fastapi dev app/ml_caption.py      # alt: uvicorn app.ml_caption:app --reload

Then POST an image to /caption/ and (optionally) a max_length query param:

    curl -X POST "http://127.0.0.1:8000/caption/?max_length=24" \
         -F "data=@your_image.jpg"

Model: nlpconnect/vit-gpt2-image-captioning (VisionEncoderDecoder). The first
request downloads the weights (~1 GB) and is slow; later requests are fast.

Note: we use `ViTImageProcessor`, not the old `ViTFeatureExtractor`. The
feature-extractor classes are deprecated in transformers and are removed in
transformers v5 - image processors are the supported replacement.
"""

from __future__ import annotations

from http import HTTPStatus
from io import BytesIO

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

app = FastAPI(title="SE 489 Image Captioning API")

MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"

# Loaded once at import time and reused across requests.
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict_caption(image: Image.Image, max_length: int = 16, num_beams: int = 8) -> str:
    """Generate a single caption for one PIL image."""
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = image_processor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "num_return_sequences": 1}
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return preds[0].strip()


@app.post("/caption/")
async def caption(data: UploadFile = File(...), max_length: int = 16) -> dict:
    """Accept an uploaded image and return a generated caption string."""
    content = await data.read()
    image = Image.open(BytesIO(content))
    text = predict_caption(image, max_length=max_length)
    return {
        "caption": text,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.ml_caption:app", host="127.0.0.1", port=8000, reload=True)
