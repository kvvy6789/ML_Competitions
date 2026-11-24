"""Reproduce experiments and three Kaggle runs for title-interest regression."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
SEED = 410510


def text_features(*, word: bool, char: bool) -> FeatureUnion:
    parts = []
    if word:
        parts.append(("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True, strip_accents="unicode")))
    if char:
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=5000, sublinear_tf=True)))
    return FeatureUnion(parts)


def ridge_model(*, word: bool, char: bool, alpha: float) -> Pipeline:
    features = ColumnTransformer(
        [("text", text_features(word=word, char=char), "title"),
         ("domain", OneHotEncoder(handle_unknown="ignore"), ["domain"])],
        transformer_weights={"text": 1.0, "domain": 0.5},
    )
    return Pipeline([("features", features), ("regressor", Ridge(alpha=alpha))])


def domain_model(alpha: float) -> Pipeline:
    features = ColumnTransformer([("domain", OneHotEncoder(handle_unknown="ignore"), ["domain"])])
    return Pipeline([("features", features), ("regressor", Ridge(alpha=alpha))])


def mlp_model(hidden: tuple[int, ...]) -> Pipeline:
    """A trained neural regressor over a compact dense TF-IDF representation."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("svd", TruncatedSVD(n_components=20, random_state=SEED)),
        ("scale", StandardScaler()),
        ("regressor", MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=0.01, learning_rate_init=0.003, max_iter=1500, early_stopping=True, validation_fraction=0.15, n_iter_no_change=40, random_state=SEED)),
    ])


def evaluate(model: Pipeline, x, y: np.ndarray) -> dict[str, float]:
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    rmses, maes = [], []
    for train_idx, valid_idx in splitter.split(np.zeros(len(y)), y.astype(int)):
        if isinstance(x, pd.DataFrame):
            x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        else:
            x_train, x_valid = x[train_idx], x[valid_idx]
        fitted = clone(model).fit(x_train, y[train_idx])
        pred = np.clip(fitted.predict(x_valid), 1.0, 6.0)
        rmses.append(mean_squared_error(y[valid_idx], pred) ** 0.5)
        maes.append(mean_absolute_error(y[valid_idx], pred))
    return {"dev_rmse": float(np.mean(rmses)), "dev_rmse_std": float(np.std(rmses)), "dev_mae": float(np.mean(maes))}


def validate_submission(frame: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(frame.columns) != list(sample.columns):
        raise ValueError("Submission columns do not match sample_submission.csv")
    if not frame["title_id"].equals(sample["title_id"]):
        raise ValueError("Submission title IDs or order do not match the sample")
    if frame["interest"].isna().any():
        raise ValueError("Submission contains missing predictions")
    if not frame["interest"].between(1.0, 6.0).all():
        raise ValueError("Predictions must be between 1 and 6")


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    x_frame = train[["title", "domain"]].fillna("")
    test_frame = test[["title", "domain"]].fillna("")
    x_text = train["title"].fillna("").to_numpy()
    test_text = test["title"].fillna("").to_numpy()
    y = train["interest"].to_numpy(dtype=float)

    attempts = {
        "global_mean": (DummyRegressor(strategy="mean"), x_frame, test_frame),
        "domain_ridge_a3": (domain_model(3.0), x_frame, test_frame),
        "domain_ridge_a10": (domain_model(10.0), x_frame, test_frame),
        "word_ridge_a3": (ridge_model(word=True, char=False, alpha=3.0), x_frame, test_frame),
        "word_ridge_a10": (ridge_model(word=True, char=False, alpha=10.0), x_frame, test_frame),
        "char_ridge_a3": (ridge_model(word=False, char=True, alpha=3.0), x_frame, test_frame),
        "word_char_ridge_a10": (ridge_model(word=True, char=True, alpha=10.0), x_frame, test_frame),
        "mlp_16": (mlp_model((16,)), x_text, test_text),
        "mlp_32_16": (mlp_model((32, 16)), x_text, test_text),
    }

    results = {}
    for name, (model, x, _) in attempts.items():
        metrics = evaluate(model, x, y)
        results[name] = metrics
        print(f"{name:24s} RMSE={metrics['dev_rmse']:.4f} (+/- {metrics['dev_rmse_std']:.4f}) MAE={metrics['dev_mae']:.4f}")

    run_specs = {
        "run1": {"attempt": "domain_ridge_a10", "approach": "Domain-only one-hot Ridge, alpha=10"},
        "run2": {"attempt": "word_ridge_a10", "approach": "Word unigram/bigram TF-IDF + Ridge, alpha=10"},
        "run3": {"attempt": "mlp_16", "approach": "Word TF-IDF + 20D SVD + trained 16-unit MLP"},
    }

    ARTIFACTS.mkdir(exist_ok=True)
    manifest = {}
    for run_id, spec in run_specs.items():
        attempt = spec["attempt"]
        model, x, x_test = attempts[attempt]
        fitted = clone(model).fit(x, y)
        predictions = np.clip(fitted.predict(x_test), 1.0, 6.0)
        submission = pd.DataFrame({"title_id": test["title_id"], "interest": predictions})
        validate_submission(submission, sample)
        output = ARTIFACTS / f"{run_id}.csv"
        submission.to_csv(output, index=False)
        manifest[run_id] = {**spec, **results[attempt], "seed": SEED, "file": output.name}

    pd.DataFrame([{"attempt": name, **metrics} for name, metrics in results.items()]).sort_values("dev_rmse").to_csv(ARTIFACTS / "experiment_summary.csv", index=False)
    (ARTIFACTS / "experiment_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (ARTIFACTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote experiment records and three runs to {ARTIFACTS}")


if __name__ == "__main__":
    main()
