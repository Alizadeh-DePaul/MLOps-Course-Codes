import mlflow
import mlflow.pyfunc
import sys

# Set the tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Get run ID from command line
if len(sys.argv) != 2:
    print("Usage: python register_model.py RUN_ID")
    sys.exit(1)

run_id = sys.argv[1]

# Register the model
model_uri = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri, "iris-classifier")

print(f"Model registered with name: {mv.name}")
print(f"Model version: {mv.version}")
