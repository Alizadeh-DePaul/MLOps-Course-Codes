import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Log the model with MLflow
mlflow.sklearn.log_model(
    model,
    "iris_rf_model",
    registered_model_name="iris_rf_docker"
)

# Build a Docker image for serving
model_uri = "models:/iris_rf_docker/1"  # Update version as needed
mlflow.models.build_docker(model_uri, name="mlflow-iris-rf:latest")
