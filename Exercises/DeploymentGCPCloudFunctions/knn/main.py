"""KNN classifier HTTP function for Cloud Run functions (formerly Cloud Functions).

Loads a pickled scikit-learn model from a Cloud Storage bucket at cold start,
then serves predictions over HTTP. The bucket and blob names come from
environment variables so they are never hard-coded into the source.

Set these when deploying (or exporting locally):

    BUCKET_NAME   the GCS bucket holding the model, e.g. mlops489-models
    MODEL_FILE    the blob name inside the bucket, e.g. model.pkl

Run locally::

    export BUCKET_NAME=mlops489-models MODEL_FILE=model.pkl
    functions-framework --source=knn/main.py --target=knn_classifier --debug
    curl -X POST localhost:8080 -H "Content-Type: application/json" \\
        -d '{"input_data": "5.1,3.5,1.4,0.2"}'

Deploy (2nd gen / Cloud Run functions)::

    gcloud functions deploy knn-classifier \\
        --gen2 --runtime=python311 --region=us-central1 \\
        --source=knn --entry-point=knn_classifier \\
        --trigger-http --allow-unauthenticated \\
        --set-env-vars=BUCKET_NAME=mlops489-models,MODEL_FILE=model.pkl
"""

import os
import pickle

import functions_framework
from google.cloud import storage

BUCKET_NAME = os.environ["BUCKET_NAME"]
MODEL_FILE = os.environ["MODEL_FILE"]

# Load the model once at cold start, not on every request. Anything at module
# scope runs once per instance and is reused across warm invocations.
_client = storage.Client()
_bucket = _client.get_bucket(BUCKET_NAME)
_blob = _bucket.get_blob(MODEL_FILE)
_model = pickle.loads(_blob.download_as_bytes())


@functions_framework.http
def knn_classifier(request):
    """Predict the iris class for a comma-separated feature vector.

    Expects JSON like ``{"input_data": "5.1,3.5,1.4,0.2"}`` (four floats: sepal
    length, sepal width, petal length, petal width).
    """
    request_json = request.get_json(silent=True)
    if request_json and "input_data" in request_json:
        data = request_json["input_data"]
        input_data = list(map(float, data.split(",")))
        prediction = _model.predict([input_data])
        return f"Belongs to class: {prediction}"
    return "No input data received"
