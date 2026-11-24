"""Reproducible experiments for the Premature Problem Solving competition.

The official validation set is used only for final per-approach evaluation.
Hyperparameters are selected by stratified cross-validation on train.csv.
After evaluation, each selected configuration is refit on train+validation to
produce a Kaggle-ready prediction file with IDs in the required order.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SEED = 410510


@dataclass(frozen=True)
class Candidate:
    approach: str
    config_id: str
    build: Callable[[], BaseEstimator]
    settings: dict[str, Any]
    text_view: str = "combined"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def combine_text(frame: pd.DataFrame) -> pd.Series:
    """Keep fields explicit so the model can learn context/response roles."""
    return (
        "CONTEXT: "
        + frame["context"].fillna("").astype(str)
        + "\nRESPONSE: "
        + frame["response"].fillna("").astype(str)
    )


def render_text(frame: pd.DataFrame, view: str) -> pd.Series:
    context = frame["context"].fillna("").astype(str)
    response = frame["response"].fillna("").astype(str)
    if view == "combined":
        return combine_text(frame)
    if view == "response":
        return "RESPONSE: " + response
    if view == "context":
        return "CONTEXT: " + context
    if view == "opening_emphasis":
        # The label specifically concerns whether acknowledgment precedes advice.
        # Duplicating the opening gives its n-grams more influence while retaining
        # the complete context and response.
        opening = response.str.slice(0, 500)
        return (
            "RESPONSE_OPENING: "
            + opening
            + "\nCONTEXT: "
            + context
            + "\nRESPONSE_FULL: "
            + response
        )
    raise ValueError(f"Unknown text view: {view}")


def word_vectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True):
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=1.0,
        sublinear_tf=sublinear_tf,
        max_features=30_000,
    )


def char_vectorizer(ngram_range=(3, 5), min_df=2):
    return TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        max_features=40_000,
    )


def candidates() -> list[Candidate]:
    items: list[Candidate] = [
        Candidate(
            "Majority Baseline",
            "majority",
            lambda: Pipeline(
                [("tfidf", word_vectorizer()), ("clf", DummyClassifier(strategy="most_frequent"))]
            ),
            {"strategy": "most_frequent"},
        )
    ]

    for c in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
        items.append(
            Candidate(
                "Word TF-IDF Logistic Regression",
                f"word_lr_c{c:g}",
                lambda c=c: Pipeline(
                    [
                        ("tfidf", word_vectorizer((1, 2), min_df=1)),
                        (
                            "clf",
                            LogisticRegression(
                                C=c,
                                class_weight="balanced",
                                max_iter=5_000,
                                random_state=SEED,
                            ),
                        ),
                    ]
                ),
                {"C": c, "ngrams": [1, 2], "class_weight": "balanced"},
            )
        )

    for view in ("response", "context", "opening_emphasis"):
        for c in (0.1, 0.5, 1.0, 2.0, 5.0):
            display = {
                "response": "Response-only TF-IDF Logistic Regression",
                "context": "Context-only Diagnostic Logistic Regression",
                "opening_emphasis": "Opening-emphasis TF-IDF Logistic Regression",
            }[view]
            tag = {"response": "resp", "context": "ctx", "opening_emphasis": "open"}[view]
            items.append(
                Candidate(
                    display,
                    f"{tag}_lr_c{c:g}",
                    lambda c=c: Pipeline(
                        [
                            ("tfidf", word_vectorizer((1, 2), min_df=1)),
                            (
                                "clf",
                                LogisticRegression(
                                    C=c,
                                    class_weight="balanced",
                                    max_iter=5_000,
                                    random_state=SEED,
                                ),
                            ),
                        ]
                    ),
                    {"C": c, "ngrams": [1, 2], "class_weight": "balanced", "text_view": view},
                    text_view=view,
                )
            )

    for c in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        items.append(
            Candidate(
                "Character TF-IDF Linear SVM",
                f"char_svm_c{c:g}",
                lambda c=c: Pipeline(
                    [
                        ("tfidf", char_vectorizer((3, 5), min_df=2)),
                        ("clf", LinearSVC(C=c, class_weight="balanced", random_state=SEED)),
                    ]
                ),
                {"C": c, "char_ngrams": [3, 5], "class_weight": "balanced"},
            )
        )

    for c in (0.1, 0.25, 0.5, 1.0):
        items.append(
            Candidate(
                "Word+Character TF-IDF Linear SVM",
                f"word_char_svm_c{c:g}",
                lambda c=c: Pipeline(
                    [
                        (
                            "features",
                            FeatureUnion(
                                [
                                    ("word", word_vectorizer((1, 2), min_df=1)),
                                    ("char", char_vectorizer((3, 5), min_df=2)),
                                ]
                            ),
                        ),
                        ("clf", LinearSVC(C=c, class_weight="balanced", random_state=SEED)),
                    ]
                ),
                {
                    "C": c,
                    "word_ngrams": [1, 2],
                    "char_ngrams": [3, 5],
                    "class_weight": "balanced",
                },
            )
        )

    # A neural network trained locally, satisfying the assignment requirement.
    # SVD turns sparse TF-IDF into a compact dense representation for the MLP.
    for hidden, alpha in (((32,), 0.0001), ((64,), 0.001), ((64, 32), 0.001)):
        hidden_tag = "x".join(map(str, hidden))
        items.append(
            Candidate(
                "TF-IDF SVD + Trained MLP",
                f"mlp_h{hidden_tag}_a{alpha:g}",
                lambda hidden=hidden, alpha=alpha: Pipeline(
                    [
                        ("tfidf", word_vectorizer((1, 2), min_df=1)),
                        ("svd", TruncatedSVD(n_components=64, random_state=SEED)),
                        ("scale", StandardScaler()),
                        (
                            "clf",
                            MLPClassifier(
                                hidden_layer_sizes=hidden,
                                activation="relu",
                                solver="adam",
                                alpha=alpha,
                                learning_rate_init=0.001,
                                max_iter=1_000,
                                early_stopping=True,
                                validation_fraction=0.2,
                                n_iter_no_change=30,
                                random_state=SEED,
                            ),
                        ),
                    ]
                ),
                {
                    "svd_components": 64,
                    "hidden_layers": list(hidden),
                    "alpha": alpha,
                    "learning_rate": 0.001,
                    "early_stopping": True,
                },
            )
        )
    return items


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_positive": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_positive": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_positive": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    project_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data/premature-problem-solving")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts")
    parser.add_argument("--selection-metric", default="f1_macro", choices=["f1_macro", "accuracy"])
    args = parser.parse_args()

    set_seed()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir = args.output_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    validation = pd.read_csv(args.data_dir / "validation.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")

    required_train = {"id", "context", "response", "label"}
    required_test = {"id", "context", "response"}
    if set(train.columns) != required_train or set(validation.columns) != required_train:
        raise ValueError("Unexpected train/validation schema")
    if set(test.columns) != required_test:
        raise ValueError("Unexpected test schema")
    if sample["id"].tolist() != test["id"].tolist():
        raise ValueError("Sample submission IDs do not match test IDs/order")

    y_train = train["label"].astype(int)
    y_val = validation["label"].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = "f1_macro" if args.selection_metric == "f1_macro" else "accuracy"

    all_results: list[dict[str, Any]] = []
    all_candidates = candidates()
    for candidate in all_candidates:
        x_train = render_text(train, candidate.text_view)
        x_val = render_text(validation, candidate.text_view)
        model = candidate.build()
        cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        result = {
            "approach": candidate.approach,
            "config_id": candidate.config_id,
            "settings": candidate.settings,
            "cv_metric": scoring,
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "validation": metrics(y_val, val_pred),
        }
        all_results.append(result)
        print(
            f"{candidate.config_id:28s} cv={result['cv_mean']:.3f} "
            f"val_f1m={result['validation']['f1_macro']:.3f} "
            f"val_acc={result['validation']['accuracy']:.3f}"
        )

    # Choose one configuration per approach by CV only; validation remains an
    # honest final comparison rather than a hyperparameter-tuning surface.
    selected: list[tuple[Candidate, dict[str, Any]]] = []
    for approach in dict.fromkeys(c.approach for c in all_candidates):
        approach_results = [r for r in all_results if r["approach"] == approach]
        best_result = max(approach_results, key=lambda r: (r["cv_mean"], -r["cv_std"], r["config_id"]))
        best_candidate = next(c for c in all_candidates if c.config_id == best_result["config_id"])
        selected.append((best_candidate, best_result))

    combined = pd.concat([train, validation], ignore_index=True)
    y_combined = combined["label"].astype(int)
    run_manifest: list[dict[str, Any]] = []
    ranked = sorted(
        selected,
        key=lambda pair: (
            pair[1]["validation"][args.selection_metric],
            pair[1]["cv_mean"],
        ),
        reverse=True,
    )
    for run_number, (candidate, result) in enumerate(ranked, start=1):
        x_combined = render_text(combined, candidate.text_view)
        x_test = render_text(test, candidate.text_view)
        model = candidate.build()
        model.fit(x_combined, y_combined)
        test_pred = model.predict(x_test).astype(int)
        submission = pd.DataFrame({"id": test["id"], "label": test_pred})
        if not submission["label"].isin([0, 1]).all() or len(submission) != len(test):
            raise ValueError("Invalid submission predictions")
        path = submissions_dir / f"run{run_number}_{candidate.config_id}.csv"
        submission.to_csv(path, index=False)
        run_manifest.append(
            {
                "run_id": f"run{run_number}",
                "approach": candidate.approach,
                "config_id": candidate.config_id,
                "settings": candidate.settings,
                "dev_scores": result["validation"],
                "train_cv_mean": result["cv_mean"],
                "submission": str(path.as_posix()),
                "test_prediction_counts": submission["label"].value_counts().sort_index().to_dict(),
            }
        )

    payload = {
        "seed": SEED,
        "selection_metric": args.selection_metric,
        "data": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "train_label_counts": train["label"].value_counts().sort_index().to_dict(),
            "validation_label_counts": validation["label"].value_counts().sort_index().to_dict(),
        },
        "all_experiments": all_results,
        "selected_runs": run_manifest,
    }
    (args.output_dir / "experiment_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "approach": r["approach"],
                "config_id": r["config_id"],
                "cv_mean": r["cv_mean"],
                "cv_std": r["cv_std"],
                **{f"val_{k}": v for k, v in r["validation"].items() if k != "confusion_matrix"},
            }
            for r in all_results
        ]
    ).sort_values(["val_f1_macro", "cv_mean"], ascending=False).to_csv(
        args.output_dir / "experiment_summary.csv", index=False
    )
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {len(run_manifest)} validated submission files to {submissions_dir}")


if __name__ == "__main__":
    main()

