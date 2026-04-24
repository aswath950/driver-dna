"""
model.py — Train, evaluate, and save the driver classifier.

Reads the feature parquet produced by pipeline.py,
trains an XGBoost multi-class classifier with stratified CV,
produces a plotly confusion matrix and SHAP importance plot,
and persists the model + label encoder to ./models/.

Expected columns from dataset.parquet:
  driver, lap_time_seconds, mean_speed, max_speed, min_speed,
  throttle_mean, throttle_std, brake_mean, brake_events,
  gear_changes, steer_std, speed_trace, throttle_trace, brake_trace
"""

import sys
from collections.abc import Callable
from pathlib import Path

# Guard: must be run inside the project venv to avoid numpy/pandas conflicts
_expected_venv = Path(__file__).parent.parent / ".venv"
if _expected_venv.exists() and not sys.prefix.startswith(str(_expected_venv)):
    print(
        f"\nERROR: Wrong Python interpreter.\n"
        f"  Running: {sys.executable}\n"
        f"  Expected: {_expected_venv}/bin/python\n\n"
        f"  Activate the venv first:\n"
        f"    source .venv/bin/activate\n"
        f"  Then re-run:\n"
        f"    python src/model.py\n",
        file=sys.stderr,
    )
    sys.exit(1)

import json  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import joblib  # noqa: E402
import shap  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
DATASET_PATH = DATA_DIR / "dataset.parquet"

FEATURE_COLS = [
    "lap_time_seconds",
    "mean_speed", "max_speed", "min_speed",
    "throttle_mean", "throttle_std",
    "brake_mean", "brake_events",
    "gear_changes",
    "steer_std",
]


def load_and_prepare(path: Path = DATASET_PATH):
    """
    Load parquet, drop rows with any nulls, label-encode 'driver'.

    Returns:
        X (np.ndarray), y (np.ndarray), label_encoder (LabelEncoder)
    """
    # print(f"Loading dataset from {path}...")
    df = pd.read_parquet(path)
    # steer_std is NaN when the Steer channel is absent from a session;
    # treat it as zero variance rather than dropping the entire row.
    df["steer_std"] = df["steer_std"].fillna(0.0)
    # Scope the null check to scalar feature columns only — the list-type
    # trace columns (speed_trace etc.) would cause dropna() to misbehave.
    df = df.dropna(subset=FEATURE_COLS)
    le = LabelEncoder()
    y = np.asarray(le.fit_transform(df["driver"]))
    X = df[FEATURE_COLS].values
    return X, y, le


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    progress_cb: Callable[[float, str], None] | None = None,
) -> tuple[XGBClassifier, float]:
    """
    Train an XGBoostClassifier with 5-fold stratified cross-validation.
    Prints per-fold accuracy and mean accuracy, then returns the model
    fit on the full dataset.

    Args:
        progress_cb: Optional callback(fraction, message) fired before and
                     after each fold and at the final fit step.
                     fraction is in [0, 1].
    """
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores: list[float] = []

    for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        if progress_cb:
            # Scale folds to 0.0–0.90 so the final-fit callback (0.97) is always higher
            progress_cb(i / 5 * 0.90, f"Training fold {i + 1}/5…")
        fold_clf = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
        )
        fold_clf.fit(X[train_idx], y[train_idx])
        score = float(fold_clf.score(X[val_idx], y[val_idx]))
        fold_scores.append(score)
        print(f"  Fold {i + 1}: {score:.4f}")
        if progress_cb:
            progress_cb((i + 1) / 5 * 0.90, f"Fold {i + 1}/5 — accuracy: {score:.3f}")

    fold_arr = np.array(fold_scores)
    print(f"  Mean CV accuracy: {fold_arr.mean():.4f} ± {fold_arr.std():.4f}")

    if progress_cb:
        progress_cb(0.97, "Fitting final model on full dataset…")
    clf.fit(X, y)
    return clf, float(fold_arr.mean())


def evaluate_model(
    model: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
) -> None:
    """
    Produces and saves:
      - ./models/accuracy.txt           (mean 5-fold CV accuracy as plain float)
      - ./models/confusion_matrix.html  (plotly heatmap)
      - ./models/shap_importance.png    (SHAP top-10 mean |shap| bar chart)
    Also prints a full sklearn classification report.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    class_names = list(label_encoder.classes_)
    y_pred = model.predict(X)

    # --- Full-data (train) accuracy ---
    train_accuracy = float(accuracy_score(y, y_pred))
    print(f"Train accuracy (full dataset): {train_accuracy:.4f}")

    # --- CV accuracy → accuracy.txt ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    (MODELS_DIR / "accuracy.txt").write_text(f"{cv_mean:.3f}")
    print(f"CV accuracy: {cv_mean:.4f} ± {cv_std:.4f} (saved to accuracy.txt)")

    # --- Per-driver metrics from classification report ---
    report_dict = classification_report(
        y, y_pred, target_names=class_names, output_dict=True
    )
    per_driver = {
        driver: {
            "precision": round(report_dict[driver]["precision"], 4),
            "recall":    round(report_dict[driver]["recall"], 4),
            "f1_score":  round(report_dict[driver]["f1-score"], 4),
            "support":   int(report_dict[driver]["support"]),
        }
        for driver in class_names
        if driver in report_dict
    }

    # --- Save metrics.json ---
    metrics = {
        "cv_mean":       round(cv_mean, 4),
        "cv_std":        round(cv_std, 4),
        "fold_scores":   [round(float(s), 4) for s in cv_scores],
        "train_accuracy": round(train_accuracy, 4),
        "n_samples":     int(len(y)),
        "n_drivers":     int(len(class_names)),
        "drivers":       class_names,
        "per_driver":    per_driver,
    }
    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as _f:
        json.dump(metrics, _f, indent=2)
    print(f"Saved detailed metrics → {metrics_path}")

    # --- Classification report (human-readable) ---
    print(classification_report(y, y_pred, target_names=class_names))

    # --- Confusion matrix (plotly) ---
    cm = confusion_matrix(y, y_pred)
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
    )
    cm_path = MODELS_DIR / "confusion_matrix.html"
    fig.write_html(str(cm_path))
    print(f"Saved confusion matrix → {cm_path}")

    # --- SHAP importance (top 10) ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # SHAP output shapes vary by version and model type:
    #   Old SHAP  — list of (n_samples, n_features) per class
    #               → np.array gives (n_classes, n_samples, n_features), feature axis = 2
    #   New SHAP  — single array (n_samples, n_features, n_classes), feature axis = 1
    #   Binary    — single array (n_samples, n_features),              feature axis = 1
    # Detect the feature axis by matching its length to len(FEATURE_COLS).
    shap_arr = np.array(shap_values)
    n_features = len(FEATURE_COLS)
    if shap_arr.ndim == 2:
        # binary classification or single-output
        feature_importance = np.abs(shap_arr).mean(axis=0)
    elif shap_arr.ndim == 3 and shap_arr.shape[2] == n_features:
        # old SHAP: (n_classes, n_samples, n_features)
        feature_importance = np.abs(shap_arr).mean(axis=(0, 1))
    elif shap_arr.ndim == 3 and shap_arr.shape[1] == n_features:
        # new SHAP: (n_samples, n_features, n_classes)
        feature_importance = np.abs(shap_arr).mean(axis=(0, 2))
    else:
        # fallback: flatten everything except the last axis
        feature_importance = np.abs(shap_arr).mean(axis=tuple(range(shap_arr.ndim - 1)))
    top_idx = np.argsort(feature_importance)[::-1][:10]
    top_features = [FEATURE_COLS[i] for i in top_idx]
    top_values = feature_importance[top_idx]

    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_features[::-1], top_values[::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 10 Feature Importances (SHAP)")
    plt.tight_layout()
    shap_path = MODELS_DIR / "shap_importance.png"
    fig2.savefig(str(shap_path), dpi=150)
    plt.close(fig2)
    print(f"Saved SHAP importance plot → {shap_path}")


def save_model(model: XGBClassifier, label_encoder: LabelEncoder) -> None:
    """Persist the model and label encoder to ./models/ using joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "driver_dna_clf.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")
    print(f"Saved model and label encoder → {MODELS_DIR}/")


def load_model() -> tuple[XGBClassifier, LabelEncoder]:
    """Load and return the model and label encoder from ./models/."""
    clf = joblib.load(MODELS_DIR / "driver_dna_clf.joblib")
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")
    return clf, le


if __name__ == "__main__":
    print("Loading dataset...")
    X, y, le = load_and_prepare()
    print(f"  {len(X)} laps, {len(le.classes_)} drivers: {list(le.classes_)}\n")

    print("Training with 5-fold CV...")
    model, cv_accuracy = train_model(X, y)

    print("\nEvaluating...")
    evaluate_model(model, X, y, le)

    print("\nSaving model...")
    save_model(model, le)
