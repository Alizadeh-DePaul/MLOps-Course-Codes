"""Step 4 - POST: data= (form-encoded) vs json= (JSON).

Goal: feel the difference between sending the SAME payload as a form vs as
JSON. This is the single most common bug when integrating with a real API:
the server is happy with one and rejects the other, and the error message
is usually unhelpful.

We use httpbin.org/post, which is a small echo server: it sends back what
it received. That makes the difference visible without needing a real API.

If httpbin.org is down (it has had outages), the same exercise works against
https://postman-echo.com/post - swap the URL and re-run.
"""

from __future__ import annotations

import requests

ECHO_URL = "https://httpbin.org/post"
PAYLOAD = {"username": "depaul", "password": "se489"}


def post_form_encoded(url: str, payload: dict) -> dict:
    """Send `payload` as application/x-www-form-urlencoded.

    Pass payload via the `data=` kwarg. requests will set
    Content-Type: application/x-www-form-urlencoded for you.

    The echo server reports this in its response under the key "form".
    """
    # TODO 1: Call requests.post(url, data=payload). Return response.json().
    return {}


def post_json_body(url: str, payload: dict) -> dict:
    """Send `payload` as application/json.

    Pass payload via the `json=` kwarg. requests will serialize the dict to
    a JSON string AND set Content-Type: application/json for you.

    The echo server reports this in its response under the key "json".
    """
    # TODO 2: Call requests.post(url, json=payload). Return response.json().
    return {}


def main() -> None:
    print("--- POST with data= (form-encoded) ---")
    resp_form = post_form_encoded(ECHO_URL, PAYLOAD)
    print(f"  Content-Type sent:  {resp_form.get('headers', {}).get('Content-Type')}")
    print(f"  Echoed under 'form':  {resp_form.get('form')}")
    print(f"  Echoed under 'json':  {resp_form.get('json')}    <- None, server saw a form")

    print("\n--- POST with json= ---")
    resp_json = post_json_body(ECHO_URL, PAYLOAD)
    print(f"  Content-Type sent:  {resp_json.get('headers', {}).get('Content-Type')}")
    print(f"  Echoed under 'form':  {resp_json.get('form')}    <- empty, server saw JSON")
    print(f"  Echoed under 'json':  {resp_json.get('json')}")

    # TODO 3 (discussion, not code): in a comment below, answer:
    #   - Which encoding would a FastAPI endpoint defined with
    #     `def predict(payload: MyPydanticModel)` expect?
    #   - Why is json= almost always the right choice for ML model APIs?


if __name__ == "__main__":
    main()
