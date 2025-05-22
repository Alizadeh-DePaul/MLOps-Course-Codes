import mlflow
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set the tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris-hyperopt-week7")

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function
def objective(params):
    with mlflow.start_run(nested=True):
        # Log parameters
        mlflow.log_params(params)

        # Train model
        model = RandomForestClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            min_samples_split=int(params['min_samples_split']),
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        # Return negative accuracy for minimization
        return {'loss': -accuracy, 'status': STATUS_OK}

# Define the search space
search_space = {
    'n_estimators': hp.quniform('n_estimators', 10, 100, 1),
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1)
}

# Run hyperparameter optimization
with mlflow.start_run(run_name="hyperopt-tuning"):
    trials = Trials()
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials
    )

    # Log the best parameters
    mlflow.log_params({
        "best_n_estimators": int(best['n_estimators']),
        "best_max_depth": int(best['max_depth']),
        "best_min_samples_split": int(best['min_samples_split'])
    })

    # Get the run ID with the best performance
    accuracies = [-trial['result']['loss'] for trial in trials.trials]
    best_run_idx = np.argmax(accuracies)
    best_run_id = trials.trials[best_run_idx]['misc']['tid']
    best_accuracy = accuracies[best_run_idx]

    mlflow.log_metric("best_accuracy", best_accuracy)

    print(f"Best parameters: {best}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Best run ID: {best_run_id}")
