import argparse
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Train a logistic regression model on Iris dataset")
    parser.add_argument("--solver", type=str, default="lbfgs", help="Solver for logistic regression")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Enable auto-logging
    mlflow.sklearn.autolog()

    # Start a run
    with mlflow.start_run():
        # Log parameters manually (in addition to autolog)
        mlflow.log_param("dataset", "iris")

        # Create and train the model
        model = LogisticRegression(
            solver=args.solver,
            C=args.C,
            max_iter=args.max_iter
        )
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics manually (in addition to autolog)
        mlflow.log_metric("test_accuracy", accuracy)

        print(f"Solver: {args.solver}")
        print(f"C: {args.C}")
        print(f"Max iterations: {args.max_iter}")
        print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
