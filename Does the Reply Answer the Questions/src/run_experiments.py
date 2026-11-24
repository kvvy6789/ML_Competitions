"""Experiments for ARI 510: Does the Reply Match the Question?"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

SEED = 5102026


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def split_text(text: str) -> tuple[str, str]:
    parts = re.split(r"\n\s*REPLY\s*\n", str(text), maxsplit=1, flags=re.I)
    return (parts[0], parts[1]) if len(parts) == 2 else (str(text), "")


def render(frame: pd.DataFrame, view: str) -> pd.Series:
    pairs = frame["text"].fillna("").astype(str).map(split_text)
    if view == "full":
        return frame["text"].fillna("").astype(str)
    if view == "question":
        return pairs.map(lambda x: x[0])
    if view == "reply":
        return pairs.map(lambda x: x[1])
    if view == "coverage":
        return pairs.map(lambda x: f"QUESTION: {x[0]}\nREPLY: {x[1]}\nREPLY_EMPHASIS: {x[1]}")
    raise ValueError(view)


@dataclass(frozen=True)
class Candidate:
    approach: str
    config: str
    view: str
    build: Callable[[], BaseEstimator]


class CoverageFeatures(BaseEstimator, TransformerMixin):
    """Numeric request/response coverage features derived without test labels."""

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        rows = []
        for text in x:
            question, reply = split_text(str(text))
            q_words = re.findall(r"[a-z]+", question.lower())
            r_words = re.findall(r"[a-z]+", reply.lower())
            q_set, r_set = set(q_words), set(r_words)
            overlap = len(q_set & r_set) / max(1, len(q_set))
            rows.append([
                question.count("?"),
                len(question), len(reply), len(reply) / max(1, len(question)),
                len(q_words), len(r_words),
                overlap,
                len(re.findall(r"\b(and|also|or|another|second|additionally)\b", question.lower())),
                len(re.findall(r"\b(yes|no|because|should|can|need|use|try)\b", reply.lower())),
            ])
        return np.asarray(rows, dtype=float)


def word_vec():
    return TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=1,
        sublinear_tf=True, max_features=12_000
    )


def char_vec():
    return TfidfVectorizer(
        analyzer="char_wb", lowercase=True, ngram_range=(3, 5), min_df=1,
        sublinear_tf=True, max_features=20_000
    )


def all_candidates() -> list[Candidate]:
    out = [Candidate("Majority Baseline", "majority", "full", lambda: Pipeline([
        ("tfidf", word_vec()), ("clf", DummyClassifier(strategy="most_frequent"))]))]
    for view in ("full", "reply", "question", "coverage"):
        for c in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
            out.append(Candidate(
                f"{view.title()} Word TF-IDF Logistic Regression", f"{view}_lr_c{c:g}", view,
                lambda c=c: Pipeline([
                    ("tfidf", word_vec()),
                    ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=SEED)),
                ])
            ))
    for c in (0.05, 0.1, 0.25, 0.5, 1.0):
        out.append(Candidate(
            "Word+Character TF-IDF Linear SVM", f"word_char_svm_c{c:g}", "coverage",
            lambda c=c: Pipeline([
                ("features", FeatureUnion([("word", word_vec()), ("char", char_vec())])),
                ("clf", LinearSVC(C=c, class_weight="balanced", random_state=SEED)),
            ])
        ))
    for hidden in ((8,), (16,), (16, 8)):
        tag = "x".join(map(str, hidden))
        out.append(Candidate(
            "TF-IDF SVD + Trained MLP", f"mlp_h{tag}", "coverage",
            lambda hidden=hidden: Pipeline([
                ("tfidf", word_vec()),
                ("svd", TruncatedSVD(n_components=6, random_state=SEED)),
                ("scale", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=hidden, alpha=0.001, learning_rate_init=0.001,
                    max_iter=1500, early_stopping=False, random_state=SEED)),
            ])
        ))
    for depth in (1, 2, 3):
        out.append(Candidate(
            "Engineered Coverage Decision Tree", f"coverage_tree_d{depth}", "full",
            lambda depth=depth: Pipeline([
                ("features", CoverageFeatures()),
                ("clf", DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=SEED)),
            ])
        ))
    for c in (0.1, 1.0, 10.0):
        out.append(Candidate(
            "Engineered Coverage Logistic Regression", f"coverage_features_lr_c{c:g}", "full",
            lambda c=c: Pipeline([
                ("features", CoverageFeatures()), ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=SEED)),
            ])
        ))
    out.append(Candidate(
        "Engineered Coverage Random Forest", "coverage_rf", "full",
        lambda: Pipeline([
            ("features", CoverageFeatures()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=2, min_samples_leaf=2,
                class_weight="balanced", random_state=SEED)),
        ])
    ))
    return out


def score(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("Does the Reply Answer the Questions/data"))
    ap.add_argument("--output-dir", type=Path, default=Path("Does the Reply Answer the Questions/artifacts"))
    args = ap.parse_args()
    seed_everything()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    val = pd.read_csv(args.data_dir / "validation.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    if sample.id.tolist() != test.id.tolist():
        raise ValueError("Sample/test ID mismatch")
    if set(train.label.unique()) - {0, 1}:
        raise ValueError("Unexpected labels")

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    results = []
    candidates = all_candidates()
    for cand in candidates:
        xtr, xv = render(train, cand.view), render(val, cand.view)
        model = cand.build()
        cv_scores = cross_val_score(model, xtr, train.label, cv=cv, scoring="accuracy")
        model.fit(xtr, train.label)
        pred = model.predict(xv)
        item = {
            "approach": cand.approach, "config": cand.config, "view": cand.view,
            "cv_accuracy_mean": float(cv_scores.mean()), "cv_accuracy_std": float(cv_scores.std()),
            "validation": score(val.label, pred),
        }
        results.append(item)
        print(f"{cand.config:24s} cv={item['cv_accuracy_mean']:.3f} val={item['validation']['accuracy']:.3f}")

    # Hyperparameters are selected by train-only CV within each named approach.
    selected = []
    for approach in dict.fromkeys(c.approach for c in candidates):
        group = [r for r in results if r["approach"] == approach]
        best = max(group, key=lambda r: (r["cv_accuracy_mean"], -r["cv_accuracy_std"], r["config"]))
        selected.append(best)

    pd.DataFrame([{
        "approach": r["approach"], "config": r["config"], "view": r["view"],
        "cv_accuracy_mean": r["cv_accuracy_mean"], "cv_accuracy_std": r["cv_accuracy_std"],
        "val_accuracy": r["validation"]["accuracy"], "val_f1_macro": r["validation"]["f1_macro"],
    } for r in results]).sort_values(["val_accuracy", "cv_accuracy_mean"], ascending=False).to_csv(
        args.output_dir / "experiment_summary.csv", index=False)
    payload = {"seed": SEED, "results": results, "selected_by_approach": selected}
    (args.output_dir / "experiment_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Produce stable candidate files for the best validation configurations.
    combined = pd.concat([train, val], ignore_index=True)
    ranked = sorted(results, key=lambda r: (r["validation"]["accuracy"], r["cv_accuracy_mean"]), reverse=True)
    used = set()
    manifest = []
    for result in ranked:
        if result["approach"] == "Majority Baseline" or result["approach"] in used:
            continue
        cand = next(c for c in candidates if c.config == result["config"])
        used.add(cand.approach)
        run_id = f"run{len(manifest)+1}"
        model = cand.build()
        model.fit(render(combined, cand.view), combined.label)
        pred = model.predict(render(test, cand.view)).astype(int)
        submission = pd.DataFrame({"id": test.id, "label": pred})
        if len(submission) != 8 or not submission.label.isin([0, 1]).all():
            raise ValueError("Invalid submission")
        path = args.output_dir / f"{run_id}.csv"
        submission.to_csv(path, index=False)
        manifest.append({
            "run_id": run_id, "approach": cand.approach, "config": cand.config,
            "view": cand.view, "dev": result["validation"],
            "train_cv_accuracy": result["cv_accuracy_mean"],
            "prediction_counts": {str(k): int(v) for k, v in submission.label.value_counts().items()},
            "file": path.as_posix(),
        })
        if len(manifest) == 3:
            break
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

