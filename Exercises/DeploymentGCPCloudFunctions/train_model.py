"""Train a tiny KNN classifier on the iris dataset and pickle it.

Produces ``model.pkl`` in the current directory. Upload that file to a Cloud
Storage bucket so the ``knn/`` Cloud Run function can load it at cold start::

    uv run python train_model.py                 # writes model.pkl
    gcloud storage cp model.pkl gs://<bucket>/model.pkl

The model is deliberately small and fast to fit; the point of the exercise is
the deployment flow, not the model.
"""

import pickle

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


def main() -> None:
    iris_x, iris_y = datasets.load_iris(return_X_y=True)

    # Deterministic split: shuffle once with a fixed seed, hold out the last
    # 10 rows as a quick sanity-check test set.
    np.random.seed(0)
    indices = np.random.permutation(len(iris_x))
    iris_x_train = iris_x[indices[:-10]]
    iris_y_train = iris_y[indices[:-10]]
    iris_x_test = iris_x[indices[-10:]]

    knn = KNeighborsClassifier()
    knn.fit(iris_x_train, iris_y_train)
    print("Sample predictions on held-out rows:", knn.predict(iris_x_test))

    with open("model.pkl", "wb") as file:
        pickle.dump(knn, file)
    print("Wrote model.pkl")


if __name__ == "__main__":
    main()
