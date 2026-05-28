"""Basic FastAPI app deployed to Cloud Run for the SE 489 deployment exercise.

Two routes so you can see path params working once it is live:

    GET /              -> {"Hello": "World"}
    GET /items/{id}    -> {"item_id": <id>}

Run locally without Docker::

    uv run uvicorn basic_fastapi:app --reload      # alt: uvicorn ... after pip install
    # open http://localhost:8000/items/1
"""

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    """Root endpoint."""
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    """Get an item by id."""
    return {"item_id": item_id}
