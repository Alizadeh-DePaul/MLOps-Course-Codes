"""Step 3 - Downloading binary data (a PNG image).

Goal: when the response body is not text, .json() and .text are useless.
You have to write response.content (raw bytes) straight to a file in
"write binary" mode ("wb").

This is the exact pattern you'll use to download model weights, datasets,
or any other artifact from a URL inside an MLOps pipeline.
"""

from __future__ import annotations

from pathlib import Path

import requests

PYTORCH_LOGO_URL = (
    "https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/"
    "pytorch-logo-dark.png"
)
OUTPUT_PATH = Path(__file__).parent / "img.png"


def download_image(url: str, destination: Path) -> Path:
    """Download a binary file and write it to `destination`.

    Returns the destination path so callers can chain on it.
    """
    response = requests.get(url)

    # TODO 1: Raise if the request failed. requests gives you
    #         response.raise_for_status() - one line, raises HTTPError on 4xx/5xx.

    # TODO 2: Open `destination` in "wb" mode (write binary) using a `with`
    #         block, and call f.write(response.content) inside it.
    #         The `with` block guarantees the file is closed even if write fails.

    return destination


def main() -> None:
    # NOTE: you'll sometimes see this written as r'img.png' (raw string).
    # The `r` prefix is only needed when the string contains backslashes you
    # don't want Python to interpret (e.g. r'C:\Users\me'). For a plain
    # filename, the `r` is noise - drop it.

    print(f"Downloading {PYTORCH_LOGO_URL}")
    out = download_image(PYTORCH_LOGO_URL, OUTPUT_PATH)

    # TODO 3: Print the size of the saved file so we can verify it landed.
    #         Use out.stat().st_size. A successful download should be ~5-15 KB.

    # TODO 4 (optional): Confirm the file really is a PNG by reading the
    #         first 8 bytes and comparing to the PNG magic number:
    #             b"\x89PNG\r\n\x1a\n"
    #         This is how MIME-sniffing libraries (and `file(1)`) work.


if __name__ == "__main__":
    main()
