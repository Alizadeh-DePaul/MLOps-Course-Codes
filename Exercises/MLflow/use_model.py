import mlflow
import mlflow.pyfunc
import numpy as np
from sklearn.datasets import load_iris

# Set the tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load the model by name and version
model_name = "iris-classifier"
model_version = 1  # Update with your version
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

# Load some test data
iris = load_iris()
# Take a few examples
X = iris.data[:5]
feature_names = iris.feature_names
class_names = iris.target_names

# Make predictions
predictions = model.predict(X)

# Print results
print("Predictions:")
for i, pred in enumerate(predictions):
    features = {name: value for name, value in zip(feature_names, X[i])}
    print(f"Example {i+1}:")
    print(f"  Features: {features}")
    print(f"  Predicted class: {class_names[int(pred)]}")
    print()
