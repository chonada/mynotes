"""
MLOps end-to-end example: Wine Quality classification.

This version reads the dataset from a local CSV file. If the file is not
present, it tries to fetch it via sklearn's OpenML connector (which uses
a more reliable certificate chain than UCI's own server). If both fail,
it prints clear manual-download instructions.

Demonstrates:
  1. Local CSV loading + SHA-256 content hash for data versioning.
  2. Reproducibility via pinned random_state and logged environment info.
  3. Hyperparameter sweep across 3 algorithms (parent + child runs).
  4. Rich metric, parameter, artifact, signature, and input_example logging.
  5. Confusion-matrix PNG + per-class classification report (JSON).
  6. Model registration with the modern alias API ('@champion').
  7. Loading the registered model back from the registry for inference.

Run:
    python mlops_wine_classification.py

Then view in the UI (from the same directory):
    mlflow ui --port 5000
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend, safe in scripts and CI
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "wine_quality_classification"
REGISTERED_MODEL_NAME = "WineQualityClassifier"
LOCAL_DATA_DIR = Path(".")
LOCAL_DATA_PATH = LOCAL_DATA_DIR / "winequality-red.csv"
RANDOM_STATE = 42

# Reliable mirrors. If you want to download manually, any of these works.
# UCI's own server (archive.ics.uci.edu) is excluded because its TLS
# certificate frequently fails verification on corporate networks.
MIRRORS = [
    "https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv",
    "https://raw.githubusercontent.com/parulnith/Wine-Quality-Analysis/master/winequality-red.csv",
]


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    """SHA-256 content hash of a file. Used as a data-version fingerprint."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_separator(path: Path) -> str:
    """Wine-quality CSVs ship with either ';' (UCI original) or ',' (some mirrors)."""
    first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    return ";" if first_line.count(";") > first_line.count(",") else ","


def try_fetch_via_openml() -> bool:
    """Last-resort online fetch via sklearn's OpenML connector. Returns True on success."""
    try:
        from sklearn.datasets import fetch_openml

        print("Local CSV not found. Attempting fetch via OpenML...")
        wine = fetch_openml(name="wine-quality-red", version=1, as_frame=True)
        LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        wine.frame.to_csv(LOCAL_DATA_PATH, sep=";", index=False)
        print(f"  Saved to {LOCAL_DATA_PATH}")
        return True
    except Exception as e:
        print(f"  OpenML fetch failed: {e}")
        return False


def print_manual_instructions() -> None:
    """Tell the user how to grab the file by hand if all auto-fetch paths failed."""
    print("\n" + "=" * 70)
    print("Could not load the dataset automatically.")
    print("=" * 70)
    print("\nPlease download winequality-red.csv manually from any of these URLs:\n")
    for url in MIRRORS:
        print(f"  - {url}")
    print(
        "\n  - https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "wine-quality/winequality-red.csv"
    )
    print(
        "\n  - https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009"
    )
    print(f"\nSave it as:  {LOCAL_DATA_PATH.resolve()}")
    print("Then re-run this script.\n")


def load_data() -> tuple[pd.DataFrame, pd.Series, str]:
    """Load + binarize the target (quality >= 7 => 'good' wine).

    Tries: (1) local file, (2) OpenML. If both fail, exits with instructions.
    """
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not LOCAL_DATA_PATH.exists():
        if not try_fetch_via_openml():
            print_manual_instructions()
            raise SystemExit(1)

    sep = detect_separator(LOCAL_DATA_PATH)
    df = pd.read_csv(LOCAL_DATA_PATH, sep=sep)

    # Some mirrors quote everything as a single ';'-joined column when read
    # with the wrong separator - sanity-check column count.
    if df.shape[1] < 2:
        sep = ";" if sep == "," else ","
        df = pd.read_csv(LOCAL_DATA_PATH, sep=sep)

    if "quality" not in df.columns:
        raise ValueError(
            f"Expected a 'quality' column, found: {list(df.columns)}. "
            f"Check that {LOCAL_DATA_PATH} is the red-wine quality dataset."
        )

    data_hash = file_sha256(LOCAL_DATA_PATH)
    y = (df["quality"] >= 7).astype(int).rename("is_good_wine")
    X = df.drop(columns=["quality"])
    return X, y, data_hash


# ---------------------------------------------------------------------------
# 2. Artifact helpers
# ---------------------------------------------------------------------------
def log_confusion_matrix(y_true, y_pred, labels, artifact_name: str) -> None:
    """Render the confusion matrix as a PNG and log it as an MLflow artifact."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    # Write to a temp path inside the working directory (cross-platform; /tmp
    # does not exist on Windows).
    tmp_dir = Path("./_mlflow_tmp")
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / artifact_name
    fig.savefig(tmp_path, format="png", dpi=120)
    plt.close(fig)

    mlflow.log_artifact(str(tmp_path), artifact_path="plots")
    tmp_path.unlink(missing_ok=True)


def log_classification_report(y_true, y_pred) -> dict:
    """Log the sklearn classification report as a JSON artifact and return it."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    tmp_dir = Path("./_mlflow_tmp")
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / "classification_report.json"
    tmp_path.write_text(json.dumps(report, indent=2))
    mlflow.log_artifact(str(tmp_path), artifact_path="reports")
    tmp_path.unlink(missing_ok=True)
    return report


# ---------------------------------------------------------------------------
# 3. Single training run
# ---------------------------------------------------------------------------
def train_one(
    estimator_name: str,
    estimator,
    params: dict,
    X_train,
    X_test,
    y_train,
    y_test,
    data_hash: str,
) -> tuple[str, float]:
    """Train one configuration, log everything to MLflow, return (run_id, f1)."""
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )

    with mlflow.start_run(run_name=estimator_name, nested=True) as run:
        mlflow.set_tags(
            {
                "estimator_family": estimator_name,
                "data.sha256": data_hash,
                "data.local_path": str(LOCAL_DATA_PATH.resolve()),
                "python.version": platform.python_version(),
                "sklearn.scaled": "StandardScaler",
                "task": "binary_classification",
            }
        )

        mlflow.log_params({**params, "random_state": RANDOM_STATE})

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "log_loss": log_loss(y_test, y_proba),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
        }
        mlflow.log_metrics(metrics)

        log_confusion_matrix(
            y_test, y_pred, labels=["not_good", "good"], artifact_name="confusion.png"
        )
        log_classification_report(y_test, y_pred)

        # Model signature + input example - schemas are enforced at inference time
        signature = infer_signature(X_train, pipe.predict(X_train))
        input_example = X_train.iloc[:3]

        mlflow.sklearn.log_model(
            sk_model=pipe,
            name="model",
            signature=signature,
            input_example=input_example,
        )

        print(
            f"  [{estimator_name}] f1={metrics['f1']:.4f}  "
            f"roc_auc={metrics['roc_auc']:.4f}  run_id={run.info.run_id}"
        )
        return run.info.run_id, metrics["f1"]


# ---------------------------------------------------------------------------
# 4. Sweep across models + hyperparameters
# ---------------------------------------------------------------------------
def build_search_space() -> list[tuple[str, object, dict]]:
    """Define the (name, estimator, params) tuples for the sweep."""
    space: list[tuple[str, object, dict]] = []

    for C in [0.1, 1.0, 10.0]:
        space.append(
            (
                "logistic_regression",
                LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_STATE),
                {"model": "LogisticRegression", "C": C, "max_iter": 1000},
            )
        )

    for n in [100, 300]:
        for depth in [None, 8]:
            space.append(
                (
                    "random_forest",
                    RandomForestClassifier(
                        n_estimators=n,
                        max_depth=depth,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                    {
                        "model": "RandomForestClassifier",
                        "n_estimators": n,
                        "max_depth": str(depth),
                    },
                )
            )

    for lr in [0.05, 0.1]:
        space.append(
            (
                "gradient_boosting",
                GradientBoostingClassifier(
                    learning_rate=lr, n_estimators=200, random_state=RANDOM_STATE
                ),
                {
                    "model": "GradientBoostingClassifier",
                    "learning_rate": lr,
                    "n_estimators": 200,
                },
            )
        )

    return space


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    # Defensive: clean up any lingering active run from an earlier crash
    # (common in notebooks where state persists across cells).
    if mlflow.active_run() is not None:
        mlflow.end_run()

    print("=" * 70)
    print("MLOps Wine Quality classification pipeline")
    print("=" * 70)

    X, y, data_hash = load_data()
    print(f"Dataset: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Class balance: {y.mean():.2%} positive ('good' wines)")
    print(f"Data sha256: {data_hash[:16]}...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    search_space = build_search_space()
    print(f"\nLaunching sweep over {len(search_space)} configurations...\n")

    parent_run_name = f"sweep_{datetime.now():%Y%m%d_%H%M%S}"
    with mlflow.start_run(run_name=parent_run_name) as parent:
        mlflow.set_tags(
            {
                "run_type": "sweep_parent",
                "data.sha256": data_hash,
                "n_configurations": str(len(search_space)),
            }
        )
        mlflow.log_param("dataset", "UCI Wine Quality (red)")
        mlflow.log_param("n_configurations", len(search_space))

        results: list[tuple[str, float]] = []
        for name, est, params in search_space:
            run_id, f1 = train_one(
                name, est, params, X_train, X_test, y_train, y_test, data_hash
            )
            results.append((run_id, f1))

        best_run_id, best_f1 = max(results, key=lambda t: t[1])
        mlflow.log_metric("best_f1", best_f1)
        mlflow.set_tag("best_run_id", best_run_id)
        print(f"\nBest child run: {best_run_id}  (f1={best_f1:.4f})")
        print(f"Parent run: {parent.info.run_id}")

    register_best_model(best_run_id)
    demo_load_and_score(X_test.head(5))


# ---------------------------------------------------------------------------
# 6. Model registry
# ---------------------------------------------------------------------------
def register_best_model(run_id: str) -> None:
    """Register the model logged in `run_id`, attach metadata, and tag as champion."""
    print("\nRegistering best model in the MLflow Model Registry...")
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    client = MlflowClient()

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=mv.version,
        description=(
            f"Best model from sweep on {datetime.now():%Y-%m-%d}. "
            f"Auto-registered from run {run_id}."
        ),
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=mv.version,
        key="validation_status",
        value="auto_promoted",
    )

    try:
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="champion",
            version=mv.version,
        )
        print(f"  Alias 'champion' -> {REGISTERED_MODEL_NAME} v{mv.version}")
    except Exception as e:
        print(f"  Alias API unavailable ({e}); using stage transition instead.")
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )


# ---------------------------------------------------------------------------
# 7. Load + serve demo
# ---------------------------------------------------------------------------
def demo_load_and_score(sample: pd.DataFrame) -> None:
    """Load the champion model back from the registry and score a small sample."""
    print("\nLoading champion model from the registry for inference...")
    try:
        model = mlflow.pyfunc.load_model(
            f"models:/{REGISTERED_MODEL_NAME}@champion"
        )
        source = "alias 'champion'"
    except Exception:
        model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}/latest")
        source = "stage 'latest'"

    preds = model.predict(sample)
    print(f"Loaded via {source}.")
    print("\nSample predictions:")
    out = sample.copy()
    out["prediction"] = preds
    print(out.tail().to_string())


if __name__ == "__main__":
    main()
