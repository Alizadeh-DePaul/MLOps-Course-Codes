import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Set the experiment
mlflow.set_experiment("iris-classification-advanced")

# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters to try
hyperparams = [
    {"n_estimators": 10, "max_depth": 3},
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10}
]

for params in hyperparams:
    with mlflow.start_run(run_name=f"rf-{params['n_estimators']}-{params['max_depth']}"):
        # Log parameters
        mlflow.log_params(params)

        # Train model
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log basic metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log feature importance as a parameter
        for idx, importance in enumerate(model.feature_importances_):
            # Sanitize feature name for MLflow metric key
            safe_name = feature_names[idx].replace("(", "").replace(")", "").replace(" ", "_")
            
            mlflow.log_metric(f"importance_{safe_name}", importance)

        # Create and log confusion matrix as a figure
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title("Confusion Matrix")
        plt.colorbar(im)
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        # Add text annotations to confusion matrix
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # Save and log the figure
        confusion_matrix_path = "confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        mlflow.log_artifact(confusion_matrix_path)
        plt.close(fig)

        # Log the classification report as a text file
        report = classification_report(y_test, y_pred, target_names=class_names)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        # Log the model
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Log a sample of the dataset
        sample_df = pd.DataFrame(X_test[:5], columns=feature_names)
        sample_df['target'] = y_test[:5]
        sample_df.to_csv("sample_data.csv", index=False)
        mlflow.log_artifact("sample_data.csv")

        print(f"Run completed with accuracy: {accuracy:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
