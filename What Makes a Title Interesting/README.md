# ARI 410/510: What Makes a Title Interesting?

Reproducible experiments for predicting a title's reader-interest score from 1 through 6.

The course dataset is restricted and should not be included in a public code repository. Place the supplied files under `What Makes a Title Interesting/data/`.

Run all experiments and create three candidate submissions:

```powershell
python "What Makes a Title Interesting/src/run_experiments.py"
```

The script uses seed `410510`, 5-fold repeated stratified cross-validation, reports RMSE and MAE, refits each frozen run on all 120 labeled examples, and creates `artifacts/run1.csv` through `artifacts/run3.csv`.

The runs are the best domain-only Ridge model, a word TF-IDF Ridge model, and a trained TF-IDF/SVD neural regressor. The global mean remains documented in the experiment table. Add Kaggle public scores to `reports/form_response_draft.md` after submission.
