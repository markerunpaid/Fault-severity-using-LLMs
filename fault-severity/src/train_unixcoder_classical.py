"""Train classical classifiers on frozen UnixCoder 778-d features.

This script loads the existing UnixCoder embedding artifacts, concatenates
the 768-d embeddings with the 10 code-metric features, and trains four
classical classifiers:
  - XGBoost
  - CatBoost
  - LightGBM
  - SVM

Outputs are written to results/ as JSON files that follow the repo's existing
metric naming convention, plus a summary CSV for easy comparison.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - runtime dependency check
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc

try:
    from catboost import CatBoostClassifier
except ImportError as exc:  # pragma: no cover - runtime dependency check
    CatBoostClassifier = None
    CATBOOST_IMPORT_ERROR = exc

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - runtime dependency check
    lgb = None
    LIGHTGBM_IMPORT_ERROR = exc


warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
EMBEDDINGS_DIR = Path("embeddings")
FEATURE_MODEL = "unixcoder"
NUM_CLASSES = 4
EXPECTED_EMBED_DIM = 768
EXPECTED_METRIC_DIM = 10
RANDOM_STATE = 42

CLASS_NAMES = ["Critical(0)", "Major(1)", "Medium(2)", "Low(3)"]

MODEL_CONFIGS = {
    "xgboost": {
        "family": "XGBoost",
        "file_slug": "unixcoder_xgboost",
        "builder": "build_xgboost",
    },
    "catboost": {
        "family": "CatBoost",
        "file_slug": "unixcoder_catboost",
        "builder": "build_catboost",
    },
    "lightgbm": {
        "family": "LightGBM",
        "file_slug": "unixcoder_lightgbm",
        "builder": "build_lightgbm",
    },
    "svm": {
        "family": "SVM",
        "file_slug": "unixcoder_svm",
        "builder": "build_svm",
    },
}


def geometric_mean(y_true, y_pred):
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recalls[recalls > 0]
    return float(np.prod(recalls) ** (1.0 / len(recalls))) if len(recalls) > 0 else 0.0


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Macro F1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Weighted F1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
        "G-Mean": round(geometric_mean(y_true, y_pred), 4),
        "Precision (Macro)": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Recall (Macro)": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }

    try:
        metrics["ROC-AUC (Macro)"] = round(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"), 4
        )
    except Exception:
        metrics["ROC-AUC (Macro)"] = 0.0

    return metrics


def stable_softmax(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = matrix - np.max(matrix, axis=1, keepdims=True)
    exp = np.exp(matrix)
    return exp / np.sum(exp, axis=1, keepdims=True)


def load_split(split_name):
    prefix = EMBEDDINGS_DIR / f"{FEATURE_MODEL}_{split_name}"

    embeddings = np.load(f"{prefix}_embeddings.npy")
    metrics = np.load(f"{prefix}_metrics.npy")
    labels = np.load(f"{prefix}_labels.npy")

    if embeddings.ndim != 2 or embeddings.shape[1] != EXPECTED_EMBED_DIM:
        raise ValueError(
            f"{split_name} embeddings must have shape (N, {EXPECTED_EMBED_DIM}), got {embeddings.shape}"
        )
    if metrics.ndim != 2 or metrics.shape[1] != EXPECTED_METRIC_DIM:
        raise ValueError(
            f"{split_name} metrics must have shape (N, {EXPECTED_METRIC_DIM}), got {metrics.shape}"
        )
    if len(embeddings) != len(metrics) or len(metrics) != len(labels):
        raise ValueError(
            f"{split_name} arrays are misaligned: embeddings={len(embeddings)}, metrics={len(metrics)}, labels={len(labels)}"
        )

    features = np.concatenate([embeddings, metrics], axis=1).astype(np.float32)
    if features.shape[1] != EXPECTED_EMBED_DIM + EXPECTED_METRIC_DIM:
        raise ValueError(f"Expected 778-d features, got {features.shape[1]} dims")

    return features, labels.astype(np.int64)


def load_baseline():
    path = RESULTS_DIR / "unixcoder_results.json"
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_xgboost():
    if XGBClassifier is None:
        raise ImportError(f"xgboost is required: {XGBOOST_IMPORT_ERROR}")

    return XGBClassifier(
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=1,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_catboost():
    if CatBoostClassifier is None:
        raise ImportError(f"catboost is required: {CATBOOST_IMPORT_ERROR}")

    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        iterations=800,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        auto_class_weights="Balanced",
        thread_count=-1,
    )


def build_lightgbm():
    if lgb is None:
        raise ImportError(f"lightgbm is required: {LIGHTGBM_IMPORT_ERROR}")

    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=NUM_CLASSES,
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )


def build_svm():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="linear",
                    C=1.0,
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=8000,
                ),
            ),
        ]
    )


def fit_model(model_name, model, X_train, y_train, X_val, y_val):
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    if model_name == "xgboost":
        try:
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=50,
            )
        except TypeError:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        return model

    if model_name == "catboost":
        try:
            model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        except TypeError:
            model.fit(X_train, y_train)
        return model

    if model_name == "lightgbm":
        try:
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                eval_metric="multi_logloss",
            )
        except TypeError:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        return model

    if model_name == "svm":
        model.fit(X_train, y_train, svm__sample_weight=sample_weights)
        return model

    raise ValueError(f"Unknown model: {model_name}")


def predict_with_probabilities(model, X):
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)
    else:
        y_prob = stable_softmax(model.decision_function(X))

    return np.asarray(y_pred).reshape(-1).astype(np.int64), np.asarray(y_prob)


def print_report(model_label, metrics, y_true, y_pred):
    print(f"\n{'=' * 72}")
    print(f"  {model_label}")
    print(f"{'=' * 72}")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )
    print(f"  Accuracy      : {metrics['Accuracy']:.4f}")
    print(f"  Macro F1      : {metrics['Macro F1']:.4f}")
    print(f"  Weighted F1   : {metrics['Weighted F1']:.4f}")
    print(f"  MCC           : {metrics['MCC']:.4f}")
    print(f"  G-Mean        : {metrics['G-Mean']:.4f}")
    print(f"  ROC-AUC Macro : {metrics['ROC-AUC (Macro)']:.4f}")


def save_result(file_slug, result):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{file_slug}_results.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return output_path


def run_single(model_name, X_train, X_val, X_test, y_train, y_val, y_test, baseline=None):
    config = MODEL_CONFIGS[model_name]
    builder = globals()[config["builder"]]

    print(f"\n{'=' * 72}")
    print(f"TRAINING {config['family']} ON UNIXCODER 778-D FEATURES")
    print(f"{'=' * 72}")
    print(f"  Train / Val / Test: {len(X_train)} / {len(X_val)} / {len(X_test)}")
    print(f"  Feature dim       : {X_train.shape[1]}")

    model = builder()
    model = fit_model(model_name, model, X_train, y_train, X_val, y_val)

    y_pred, y_prob = predict_with_probabilities(model, X_test)
    metrics = compute_metrics(y_test, y_pred, y_prob)

    print_report(config["family"], metrics, y_test, y_pred)

    result = {
        "model": config["file_slug"],
        "model_name": f"UnixCoder + {config['family']}",
        "feature_provider": FEATURE_MODEL,
        "feature_dim": int(X_test.shape[1]),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        **metrics,
    }

    result["config"] = {
        "random_state": RANDOM_STATE,
        "train_split": 0.15,
        "feature_source": "embeddings + metrics (.npy)",
        "classifier": config["family"],
    }

    if baseline is not None:
        result["baseline_unixcoder_macro_f1"] = baseline.get("Macro F1")
        result["delta_macro_f1_vs_neural_baseline"] = round(
            result["Macro F1"] - float(baseline.get("Macro F1", 0.0)),
            4,
        )

    save_result(config["file_slug"], result)
    np.save(RESULTS_DIR / f"{config['file_slug']}_predictions.npy", y_pred)
    np.save(RESULTS_DIR / f"{config['file_slug']}_probabilities.npy", y_prob)

    return result


def main(models):
    if not EMBEDDINGS_DIR.exists():
        raise FileNotFoundError("embeddings/ directory not found. Run src/extract_embeddings.py first.")

    X_train_full, y_train_full = load_split("train")
    X_test, y_test = load_split("test")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    print(f"Train full : {X_train_full.shape} | Test: {X_test.shape}")
    print(f"Train split: {X_train.shape} | Val: {X_val.shape}")
    print(f"Labels     : train={len(y_train_full)} | test={len(y_test)}")

    baseline = load_baseline()
    if baseline is not None:
        print(
            f"Baseline  : UnixCoder neural Macro F1 = {baseline.get('Macro F1', 0.0):.4f}"
        )

    results = []
    for model_name in models:
        results.append(
            run_single(
                model_name,
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                baseline=baseline,
            )
        )

    summary = pd.DataFrame(results).sort_values("Macro F1", ascending=False).reset_index(drop=True)
    summary["Rank"] = np.arange(1, len(summary) + 1)
    summary = summary[["Rank", "model_name", "Accuracy", "Macro F1", "Weighted F1", "MCC", "G-Mean", "ROC-AUC (Macro)"]]

    summary_path = RESULTS_DIR / "unixcoder_classical_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n{'=' * 80}")
    print("UNIXCODER CLASSICAL MODEL COMPARISON")
    print(f"{'=' * 80}")
    print(summary.to_string(index=False))
    print(f"\nSaved summary -> {summary_path}")

    if baseline is not None and not summary.empty:
        best_row = summary.iloc[0]
        delta = float(best_row["Macro F1"]) - float(baseline.get("Macro F1", 0.0))
        print(f"Best classical model vs neural UnixCoder Macro F1: {delta:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train classical classifiers on UnixCoder 778-d features"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        default=["all"],
        help="Which classifiers to train. Use 'all' for XGBoost, CatBoost, LightGBM, and SVM.",
    )
    args = parser.parse_args()

    selected = list(MODEL_CONFIGS.keys()) if "all" in args.models else args.models
    main(selected)