"""Hyperparameter tuning over a Random Forest with Optuna + MLflow.

Same teaching arc as the old hyperopt version, but using Optuna — the
maintained, MLflow-native option since MLflow 3. Each Optuna trial becomes a
nested MLflow run under one parent "optuna-tuning" run, so the UI shows you
the parent timing and a clean tree of trial children.

Search space:
    n_estimators       int  [10, 100]
    max_depth          int  [3, 10]
    min_samples_split  int  [2, 10]

Run:
    python optuna_tuning.py
Then open `mlflow ui` and click into the "iris-optuna-week7" experiment.
"""

import mlflow
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Local tracking — the same ./mlruns/ directory the other Week 7 scripts use.
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("iris-optuna-week7")

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def objective(trial: optuna.Trial) -> float:
    """One Optuna trial = one nested MLflow run."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    }

    with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
        mlflow.log_params(params)

        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)
        # MLflow 3.x: artifact subpath via `name=`.
        mlflow.sklearn.log_model(model, name="model")

        # Optuna *maximizes* when direction="maximize" below — return raw acc.
        return accuracy


def main() -> None:
    # The parent run wraps the whole sweep so the trials hang under one node
    # in the UI. Open the parent run and look at its "Child Runs" tab.
    with mlflow.start_run(run_name="optuna-tuning"):
        # TPESampler is Optuna's default; spelled out here so students see the
        # equivalent of hyperopt's tpe.suggest. seed= for reproducibility.
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=20)

        best = study.best_trial
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metric("best_accuracy", best.value)

        print(f"Best params: {best.params}")
        print(f"Best accuracy: {best.value:.4f}")
        print(f"Best trial number: {best.number}")


if __name__ == "__main__":
    main()
