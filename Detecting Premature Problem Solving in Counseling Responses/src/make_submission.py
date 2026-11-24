"""Create one stable, named Kaggle run from all labeled course data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from run_experiments import SEED, render_text, set_seed, word_vectorizer


RUNS = {
    "run1": {
        "approach": "Response-only TF-IDF Logistic Regression",
        "config_id": "resp_lr_c2",
        "text_view": "response",
        "C": 2.0,
        "dev_f1_macro": 0.7222222222222222,
        "dev_accuracy": 0.7333333333333333,
    },
    "run2": {
        "approach": "Opening-emphasis TF-IDF Logistic Regression",
        "config_id": "open_lr_c5",
        "text_view": "opening_emphasis",
        "C": 5.0,
        "dev_f1_macro": 0.6827262044653349,
        "dev_accuracy": 0.7,
    },
    "run3": {
        "approach": "Combined-text TF-IDF Logistic Regression",
        "config_id": "word_lr_c10",
        "text_view": "combined",
        "C": 10.0,
        "dev_f1_macro": 0.6827262044653349,
        "dev_accuracy": 0.7,
    },
}


def build_model(c: float) -> Pipeline:
    return Pipeline(
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
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, choices=sorted(RUNS))
    project_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data/premature-problem-solving")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "artifacts/final_submissions")
    args = parser.parse_args()
    set_seed()

    spec = RUNS[args.run_id]
    train = pd.read_csv(args.data_dir / "train.csv")
    validation = pd.read_csv(args.data_dir / "validation.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    labeled = pd.concat([train, validation], ignore_index=True)

    model = build_model(spec["C"])
    model.fit(render_text(labeled, spec["text_view"]), labeled["label"].astype(int))
    predictions = model.predict(render_text(test, spec["text_view"])).astype(int)
    submission = pd.DataFrame({"id": test["id"], "label": predictions})

    if submission["id"].tolist() != sample["id"].tolist():
        raise ValueError("Submission IDs/order do not match sample_submission.csv")
    if len(submission) != 60 or not submission["label"].isin([0, 1]).all():
        raise ValueError("Submission must contain 60 binary predictions")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.run_id}.csv"
    submission.to_csv(output, index=False)
    metadata = {
        "run_id": args.run_id,
        **spec,
        "seed": SEED,
        "training_rows": len(labeled),
        "test_rows": len(test),
        "prediction_counts": {
            str(k): int(v) for k, v in submission["label"].value_counts().sort_index().items()
        },
        "output": output.as_posix(),
    }
    (args.output_dir / f"{args.run_id}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

