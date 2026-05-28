"""Hello-world HTTP function for Cloud Run functions (formerly Cloud Functions).

The simplest possible deployable unit: an HTTP-triggered function that greets
the caller. It defaults to "Hello World!" (matching the GCP starter template)
and reads an optional ``name`` from either the query string or a JSON body.

Run locally::

    functions-framework --source=hello/main.py --target=hello_mlops --debug
    curl "http://localhost:8080/?name=MLOPS%20engineer"   # -> Hello MLOPS engineer!

Deploy (2nd gen / Cloud Run functions)::

    gcloud functions deploy hello-mlops \\
        --gen2 --runtime=python311 --region=us-central1 \\
        --source=hello --entry-point=hello_mlops \\
        --trigger-http --allow-unauthenticated
"""

import functions_framework


@functions_framework.http
def hello_mlops(request):
    """Return a greeting; ``name`` may come from the query string or JSON body."""
    name = "World"
    request_json = request.get_json(silent=True)
    request_args = request.args
    if request_json and "name" in request_json:
        name = request_json["name"]
    elif request_args and "name" in request_args:
        name = request_args["name"]
    return f"Hello {name}!"
