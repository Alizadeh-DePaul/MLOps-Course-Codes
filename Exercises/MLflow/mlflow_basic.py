import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set the tracking URI (uses local directory by default)
mlflow.set_tracking_uri("file:./mlruns")

# Set the experiment name
mlflow.set_experiment("iris-classification-week7")

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start an MLflow run
with mlflow.start_run(run_name="basic-logistic-regression"):
    # Log parameters
    params = {"C": 1.0, "solver": "lbfgs", "max_iter": 100}
    mlflow.log_params(params)

    # Train the model
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    # Log feature names for reference
    mlflow.log_dict({"feature_names": iris.feature_names}, "features.json")
    
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

# The run is automatically ended when exiting the context
