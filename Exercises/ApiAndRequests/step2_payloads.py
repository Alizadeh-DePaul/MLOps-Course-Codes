"""Step 2 - Payloads: .content, .json(), and query parameters.

Goal: understand what comes back from a response object. .content is raw
bytes, .text is decoded string, .json() is parsed JSON (and raises if the
body is not JSON). Then use params= to add query parameters to a GET.

Reference: https://docs.github.com/en/rest/search?apiVersion=2022-11-28
"""

from __future__ import annotations

import requests


def inspect_html_response() -> None:
    """The course-repo HTML page is text/html, NOT JSON.

    Calling .json() here would raise json.JSONDecodeError. We deliberately
    fetch a *web page* (not an API) to make that mismatch concrete.
    """
    response = requests.get("https://github.com/Alizadeh-DePaul/MLOps-Course-Codes")

    # TODO 1: Print the type of response.content. (Hint: type(response.content).)
    #         You should see <class 'bytes'>.

    # TODO 2: Print the first 200 characters of response.text so you can
    #         see this is HTML, not JSON.

    # NOTE: We intentionally do NOT call response.json() here - it would raise.
    #       In production code you would check response.headers["Content-Type"]
    #       before assuming the body is JSON.


def inspect_json_response() -> None:
    """The GitHub API root IS JSON. .json() parses the body for you."""
    response = requests.get("https://api.github.com")

    # TODO 3: Call response.json() and store it in a variable called `data`.
    data: dict = {}

    # TODO 4: Print data["current_user_url"] so you can see one nested field.
    #         (The full dict has ~30 keys - it's the GitHub API's link map.)
    print(data.get("current_user_url"))


def search_github_repos(query: str) -> list[str]:
    """Use the GitHub Search API with the params= kwarg.

    Returns the full_name of the top 5 results, e.g. "psf/requests".

    Heads up - unauthenticated search is rate-limited to ~10 requests per
    minute. If you re-run this a lot, you may see a 403 with the header
    X-RateLimit-Remaining: 0. That is normal; wait a minute and retry.
    """
    # TODO 5: Call requests.get on "https://api.github.com/search/repositories"
    #         with params={"q": query}. Notice that requests handles the URL
    #         encoding for you (spaces, +, etc.).
    response = ...  # placeholder

    # TODO 6: If status_code != 200, print a warning and return [].
    #         (This is where the rate-limit 403 will land.)

    # TODO 7: Parse response.json(). The shape is:
    #         {"total_count": int, "items": [{...repo dict...}, ...]}
    #         Return [repo["full_name"] for repo in items[:5]].
    return []


def main() -> None:
    print("--- HTML response (bytes, not JSON) ---")
    inspect_html_response()

    print("\n--- JSON response (parsed dict) ---")
    inspect_json_response()

    print("\n--- GitHub search for 'requests+language:python' ---")
    for name in search_github_repos("requests+language:python"):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
