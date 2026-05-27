"""Minimal scikit-learn classifier used as the payload for the Artifact
Registry exercise.

Trains an SVM on the built-in digits dataset, prints a classification report,
then exits. The point is not the model — it is to have something small, fast,
and dependency-light to wrap in a Docker image and push to Artifact Registry.
"""

from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


def main() -> None:
    digits = datasets.load_digits()

    # Flatten the (8, 8) images into 64-feature vectors.
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    clf = svm.SVC(gamma=0.001)

    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}"
    )


if __name__ == "__main__":
    main()
