"""Complete FastAPI reference application for SE 489 - MLOps.

This is a *complete, runnable example* - there is nothing to fill in. Every
endpoint from the exercise is implemented and correct, so you can read the
file top to bottom, run it, and see each concept work. Use it as the
reference when you wrap your own model in an API.

Run it (development, auto-reload):

    fastapi dev app/main.py            # modern FastAPI CLI (recommended)
    # alternative, shows the ASGI server explicitly:
    uvicorn app.main:app --reload --port 8888

Then open the interactive docs at http://127.0.0.1:8000/docs (or
http://127.0.0.1:8888/docs if you used the uvicorn command with --port 8888).

Endpoints, in the order the exercise introduces them:
    GET  /                     root + HTTP status code
    GET  /items/{item_id}      path parameter with int type validation
    GET  /restric_items/{id}   path parameter restricted to an Enum
    GET  /query_items          query parameter
    POST /login/               POST with form-style query params (toy DB)
    GET  /text_model/          single string input + regex email check
    POST /text_model/          JSON body (Pydantic model) + domain match
    POST /cv_model/            file upload, OpenCV resize, file response
"""

from __future__ import annotations

import re
from enum import Enum
from http import HTTPStatus

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(
    title="SE 489 FastAPI Application",
    description="Complete reference API for the MLOps FastAPI exercise.",
    version="1.0.0",
)

# Uploaded / generated files are written relative to the current working
# directory (the folder you launch the server from). When you run the app
# from this exercise folder, image.jpg / image_resize.jpg / database.csv land
# here and are ignored by git (see the repo .gitignore).


# --- 1. Root endpoint + HTTP status code -----------------------------------
@app.get("/")
def read_root() -> dict:
    """Root endpoint. Returns a human-readable message plus the status code.

    `HTTPStatus.OK` is the built-in enum for 200. `.phrase` is the text
    ("OK"); the member itself serializes to the integer 200 in the response.
    """
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }


# --- 2. Path parameter with type validation --------------------------------
@app.get("/items/{item_id}")
def read_item(item_id: int) -> dict:
    """Return the id passed in the path.

    `item_id: int` makes FastAPI validate (via Pydantic) that the path
    segment is an integer. Visiting /items/test returns a clean 422 error
    instead of crashing - try it.
    """
    return {"item_id": item_id}


# --- 3. Path parameter restricted to an Enum -------------------------------
class ItemEnum(str, Enum):
    """Allowed values for the restricted-items path parameter.

    Subclassing `str` makes the members JSON-serializable and renders them
    as a dropdown in the /docs UI.
    """

    item1 = "item1"
    item2 = "item2"
    item3 = "item3"


@app.get("/restric_items/{item_id}")
def read_restricted_item(item_id: ItemEnum) -> dict:
    """Only item1/item2/item3 are accepted; anything else returns a 422."""
    return {"item_id": item_id}


# --- 4. Query parameter ----------------------------------------------------
@app.get("/query_items")
def read_query_item(item_id: int) -> dict:
    """`item_id` is NOT in the path here, so FastAPI treats it as a query
    parameter. Call it as /query_items?item_id=42."""
    return {"item_id": item_id}


# --- 5. POST with form-style params (toy login store) ----------------------
database: dict[str, list[str]] = {"username": [], "password": []}


@app.post("/login/")
def login(username: str, password: str) -> str:
    """Save a username/password pair to an in-memory store and a CSV file.

    This is a deliberately naive example to show the POST verb - never store
    plaintext credentials in a real application.
    """
    username_db = database["username"]
    password_db = database["password"]
    if username not in username_db and password not in password_db:
        with open("database.csv", "a") as file:
            file.write(f"{username}, {password} \n")
        username_db.append(username)
        password_db.append(password)
    return "login saved"


# --- 6. Single string input + regex email check ----------------------------
# Word-boundary-anchored email pattern. `[A-Za-z]{2,}` matches the TLD.
EMAIL_REGEX = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"


@app.get("/text_model/")
def contains_email(data: str) -> dict:
    """Check whether `data` is a valid email address using a regex."""
    return {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(EMAIL_REGEX, data) is not None,
    }


# --- 7. JSON body (Pydantic model) + domain match --------------------------
class DomainEnum(str, Enum):
    """Email domains this endpoint knows how to validate against."""

    gmail = "gmail"
    depaul = "depaul"


class EmailItem(BaseModel):
    """Request body for the POST /text_model/ endpoint.

    Sending a JSON body (rather than query params) is the pattern you will
    use for model inference: the client POSTs a structured payload and
    FastAPI validates it against this schema before your code runs.
    """

    email: str
    domain: DomainEnum


@app.post("/text_model/")
def contains_email_domain(data: EmailItem) -> dict:
    """Check that `data.email` is valid AND matches the requested domain."""
    # Build a domain-specific pattern, e.g. ...@gmail\.<tld>
    domain_regex = rf"\b[A-Za-z0-9._%+-]+@{data.domain.value}\.[A-Za-z]{{2,}}\b"
    return {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(domain_regex, data.email) is not None,
    }


# --- 8. File upload -> OpenCV resize -> file response -----------------------
@app.post("/cv_model/")
async def cv_model(
    data: UploadFile = File(...),
    h: int = 28,
    w: int = 28,
) -> FileResponse:
    """Accept an uploaded image, resize it with OpenCV, and send it back.

    `data: UploadFile = File(...)` declares a multipart file upload (the
    `...` marks it required). `async`/`await` let the server read the upload
    without blocking other requests. `h` and `w` are optional query params
    (default 28x28); call without them to use the defaults.

    Note OpenCV's `resize` takes the target size as (width, height).

    The endpoint returns the resized image as the response body, so a client
    can save it directly:  curl ... --output resized.jpg
    """
    input_path = "image.jpg"
    output_path = "image_resize.jpg"

    with open(input_path, "wb") as image:
        content = await data.read()
        image.write(content)

    img = cv2.imread(input_path)
    resized = cv2.resize(img, (w, h))
    cv2.imwrite(output_path, resized)

    return FileResponse(output_path, media_type="image/jpeg")


if __name__ == "__main__":
    # Lets you run `python app/main.py` directly. In class we use the
    # `fastapi dev` CLI (or uvicorn) instead - see the module docstring.
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8888, reload=True)
