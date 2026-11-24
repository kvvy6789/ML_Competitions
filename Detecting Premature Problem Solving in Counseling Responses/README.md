# Premature Problem Solving Competition

Reproducible code for the University of Michigan-Flint ARI 410/510 Kaggle
competition **Detecting Premature Problem Solving in Counseling Responses**.

The task is binary text classification over a paired counseling `context` and
`response`. Label `1` means the response moves into solution-directed guidance
before adequately acknowledging the emotional concern; label `0` means it does
not.

## Data and permitted use

The competition text is restricted to noncommercial course use and must not be
redistributed. Download the four competition CSV files from Kaggle and place
them under:

```text
Detecting Premature Problem Solving in Counseling Responses/data/premature-problem-solving/
  train.csv
  validation.csv
  test.csv
  sample_submission.csv
```

The `data/` directory is intentionally ignored by Git.

## Reproduce experiments

```powershell
python -m pip install -r "Detecting Premature Problem Solving in Counseling Responses/requirements.txt"
python "Detecting Premature Problem Solving in Counseling Responses/src/run_experiments.py"
```

The script fixes all random seeds, selects hyperparameters with five-fold
stratified cross-validation on `train.csv`, evaluates selected approaches on
the provided `validation.csv`, refits on train plus validation, and writes
Kaggle-ready files under `artifacts/submissions/`.

Generated files:

- `artifacts/experiment_summary.csv`: every attempted configuration
- `artifacts/experiment_results.json`: complete scores and confusion matrices
- `artifacts/run_manifest.json`: exact settings and output path for every run
- `artifacts/submissions/*.csv`: validated Kaggle submission files

The run IDs in `run_manifest.json` determine which settings produce each run.

## Stable Kaggle run IDs

Create an exact final submission with one of these commands:

```powershell
python "Detecting Premature Problem Solving in Counseling Responses/src/make_submission.py" --run-id run1
python "Detecting Premature Problem Solving in Counseling Responses/src/make_submission.py" --run-id run2
python "Detecting Premature Problem Solving in Counseling Responses/src/make_submission.py" --run-id run3
```

- `run1`: response-only word/bigram TF-IDF + balanced logistic regression,
  `C=2.0`, seed `410510`
- `run2`: opening-emphasis word/bigram TF-IDF + balanced logistic regression,
  `C=5.0`, seed `410510`
- `run3`: combined context/response word/bigram TF-IDF + balanced logistic
  regression, `C=10.0`, seed `410510`

These commands refit the frozen configuration on all 150 labeled examples and
write the final file under `artifacts/final_submissions/`.

## Named approaches

- **Majority Baseline**
- **Word TF-IDF Logistic Regression**
- **Character TF-IDF Linear SVM**
- **Word+Character TF-IDF Linear SVM**
- **TF-IDF SVD + Trained MLP** (the required trained neural-network approach)

The trained MLP was evaluated as an experiment but was not selected for a
leaderboard run because its validation result was substantially worse than the
linear models.


