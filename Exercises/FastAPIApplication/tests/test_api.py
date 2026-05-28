"""Tests for the FastAPI Application exercise (the optional "testing" step).

FastAPI ships a `TestClient` (built on httpx) that drives the app in-process,
so you do NOT need a running uvicorn server to test it - the client calls the
ASGI app directly. Run with:

    pytest                       # from inside Exercises/FastAPIApplication/

Reference: https://fastapi.tiangolo.com/tutorial/testing/
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


# --- Root + status ---------------------------------------------------------
def test_root_returns_ok_status() -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "OK"
    assert body["status-code"] == 200


# --- Path parameter + type validation --------------------------------------
def test_items_accepts_int() -> None:
    response = client.get("/items/42")
    assert response.status_code == 200
    assert response.json() == {"item_id": 42}


def test_items_rejects_non_int() -> None:
    """A non-integer path segment is a 422, not a crash."""
    response = client.get("/items/not-a-number")
    assert response.status_code == 422


# --- Restricted enum path parameter ----------------------------------------
def test_restricted_item_accepts_allowed_value() -> None:
    response = client.get("/restric_items/item1")
    assert response.status_code == 200
    assert response.json() == {"item_id": "item1"}


def test_restricted_item_rejects_unknown_value() -> None:
    response = client.get("/restric_items/item9")
    assert response.status_code == 422


# --- Query parameter -------------------------------------------------------
def test_query_items() -> None:
    response = client.get("/query_items", params={"item_id": 7})
    assert response.status_code == 200
    assert response.json() == {"item_id": 7}


# --- POST login ------------------------------------------------------------
def test_login_saves() -> None:
    response = client.post("/login/", params={"username": "se489", "password": "mlops"})
    assert response.status_code == 200
    assert response.json() == "login saved"


# --- GET text_model (regex email check) ------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("se489@depaul.edu", True),
        ("not-an-email", False),
        ("a@b.co", True),
    ],
)
def test_contains_email(value: str, expected: bool) -> None:
    response = client.get("/text_model/", params={"data": value})
    assert response.status_code == 200
    assert response.json()["is_email"] is expected


# --- POST text_model (JSON body + domain match) ----------------------------
def test_domain_match_true() -> None:
    payload = {"email": "se489@depaul.edu", "domain": "depaul"}
    response = client.post("/text_model/", json=payload)
    assert response.status_code == 200
    assert response.json()["is_email"] is True


def test_domain_match_false_when_domain_differs() -> None:
    payload = {"email": "se489@depaul.edu", "domain": "gmail"}
    response = client.post("/text_model/", json=payload)
    assert response.status_code == 200
    assert response.json()["is_email"] is False


def test_domain_rejects_unknown_enum() -> None:
    payload = {"email": "x@yahoo.com", "domain": "yahoo"}
    response = client.post("/text_model/", json=payload)
    assert response.status_code == 422


# --- POST cv_model (file upload -> resize -> file response) -----------------
def _png_bytes(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(120, 80, 200)).save(buf, format="PNG")
    return buf.getvalue()


def test_cv_model_resizes_with_defaults() -> None:
    files = {"data": ("sample.png", _png_bytes(256, 256), "image/png")}
    response = client.post("/cv_model/", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    returned = Image.open(io.BytesIO(response.content))
    assert returned.size == (28, 28)  # default h=w=28


def test_cv_model_resizes_with_custom_dims() -> None:
    files = {"data": ("sample.png", _png_bytes(256, 256), "image/png")}
    response = client.post("/cv_model/", params={"h": 64, "w": 64}, files=files)
    assert response.status_code == 200
    returned = Image.open(io.BytesIO(response.content))
    assert returned.size == (64, 64)
