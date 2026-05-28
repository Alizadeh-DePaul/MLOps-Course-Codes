"""Step 1 - HTTP status codes with the requests package.

Goal: send GET requests, inspect the status code, and branch on it.
This is the foundation for every health-check / readiness call you will
make against a model server later in the course.

Reference: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
"""

from __future__ import annotations

import requests


def fetch_and_print_status(url: str) -> int:
    """Send a GET request to `url` and print + return the status code."""
    # TODO 1: Use requests.get(url) to send the GET request, store the
    #         response object in a variable called `response`.
    response = ...  # noqa: F841 - placeholder

    # TODO 2: Print response.status_code so the caller can see what came back.
    #         Then return it so the function is testable.
    return -1


def classify_status(code: int) -> str:
    """Return a one-word category for a status code.

    Examples:
        200, 201 -> "success"
        301, 302 -> "redirect"
        404, 403 -> "client_error"
        500, 503 -> "server_error"
        anything else -> "other"
    """
    # TODO 3: Use if/elif/else on `code` (or integer division by 100) to map
    #         the code to one of the strings in the docstring above.
    return "other"


def main() -> None:
    # Step 1.1 - a URL that returns 404 (the path /wrong-api-link does not exist)
    fetch_and_print_status("https://api.github.com/wrong-api-link")

    # Step 1.2 - a URL that returns 200 (the GitHub REST API root)
    fetch_and_print_status("https://api.github.com")

    # Step 1.3 - show the if/elif/else branching pattern from the exercise page
    code = fetch_and_print_status("https://api.github.com")
    if code == 200:
        print("Success!")
    elif code == 404:
        print("Error")
    else:
        print(f"Unhandled status: {code}")

    # Step 1.4 - prove your classify_status() works
    for c in (200, 301, 404, 500, 999):
        print(f"{c} -> {classify_status(c)}")


if __name__ == "__main__":
    main()
