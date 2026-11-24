# Phase 2b Form Draft - What Makes a Title Interesting?

## Task

This task predicts an interest score from 1 through 6 for each content title. The supplied records also identify the source domain: article, podcast, Reddit, or YouTube. This is an ordinal regression problem, so I evaluated predictions using error magnitude rather than treating all mistakes as equally different.

## Methodology

The dataset contained 120 labeled training titles and 30 hidden test titles. Each of the four domains had 30 training examples. The target distribution was not uniform: scores 1 through 6 occurred 30, 20, 15, 25, 19, and 11 times, respectively. I preserved the raw title text, filled any missing text with an empty string, and used the supplied domain as a categorical feature when the model supported it. All predictions were clipped to the valid 1-6 range, and all randomized procedures used seed `410510`.

Because no separate validation file was supplied, I created my own dev evaluation with repeated stratified cross-validation. I used five folds and five repeats, stratifying by the integer interest label so each fold had a similar target distribution. I recorded mean RMSE as the main selection metric and mean absolute error as a secondary measure. Repeating the split reduced the chance that a conclusion depended on one unusually easy or difficult fold. After fixing each run configuration, I refit it on all 120 labeled examples before predicting the test set.

**Global-Mean Regression Baseline.** This approach ignores the title and predicts the training-set mean interest score for every record. It established whether more complex models learned a signal that generalized. Its repeated dev RMSE was 1.6778 and MAE was 1.4778. I retained it in the experiment table but did not select it as a leaderboard run.

**Domain-Only Ridge Regression.** I one-hot encoded the four domains and fit regularized linear models with Ridge `alpha` values 3 and 10. This tested whether interest differed systematically by content source without using title words. The `alpha=10` model achieved the best mean dev RMSE, 1.6742, with MAE 1.4667, and became `run1`. Its advantage over the global mean was only 0.0036 RMSE, so it should be treated as a very small improvement.

**Word TF-IDF Ridge Regression.** I lowercased titles, normalized Unicode accents, applied sublinear term frequency, and extracted word unigrams and bigrams that occurred in at least two training titles. I added a one-hot domain feature at half the text-feature weight. I tested Ridge `alpha` values 3 and 10. Stronger regularization was important because there were far more lexical features than examples. The `alpha=10` configuration achieved dev RMSE 1.6783 and MAE 1.4650 and became `run2` as the main lexical model.

**Character TF-IDF Ridge Regression.** I also tried character 3-5-grams within word boundaries, both alone and combined with word TF-IDF. Character features can capture style, punctuation, capitalization patterns, and word fragments, but they did not clearly beat the baseline on this small dataset.

**TF-IDF/SVD Trained Neural Regressor.** To test a trained neural-network approach without creating an unmanageably sparse neural input, I transformed word unigram/bigram TF-IDF into 20 dense dimensions using Truncated SVD, standardized the dimensions, and trained MLP regressors. I compared one hidden layer with 16 units against hidden layers of 32 and 16 units. Both used ReLU, Adam, L2 regularization `alpha=0.01`, learning rate 0.003, early stopping, and a maximum of 1,500 iterations. The smaller network achieved dev RMSE 2.0056 and MAE 1.6370 and became `run3`. The deeper model was slightly worse at RMSE 2.0184. These results show that the small dataset did not support the additional neural flexibility.

I kept every attempted configuration in `experiment_summary.csv`, including models not selected for a leaderboard submission. The run manifest records the exact approach, seed, dev RMSE, dev MAE, and output filename for each selected run. The code validates column names, test-ID order, missing predictions, and the 1-6 range against the supplied sample submission.

## Runs

Replace `TBD` after Kaggle scores each submitted file. Include only files actually submitted.

```text
run1 | Domain-only one-hot Ridge, alpha=10 | dev RMSE 1.6742, MAE 1.4667 | TBD
run2 | Word unigram/bigram TF-IDF + Ridge, alpha=10 | dev RMSE 1.6783, MAE 1.4650 | TBD
run3 | Word TF-IDF + 20D SVD + trained 16-unit MLP | dev RMSE 2.0056, MAE 1.6370 | TBD
```

Host baseline score(s): `TBD`.

## Reflection

The domain-only Ridge model worked best on the repeated dev folds, but its RMSE advantage over the global mean was only 0.0036. I therefore do not interpret it as decisive evidence that domain generalizes. The word TF-IDF model nearly matched the baseline and had slightly better MAE, suggesting that title words may help some cases while hurting others. With only 120 examples and highly diverse titles, lexical features are sparse and many phrases appear only once. The neural models performed much worse, which indicates that the 20-dimensional latent representation and extra nonlinear parameters introduced variance that the small dataset could not support. The main lesson is to use strong regularization, compare against a mean baseline, repeat validation splits, and avoid assuming that a more flexible text model will perform better. This reflection should be extended after leaderboard scores are available.

## Code

Public repository link: https://github.com/kvvy6789/ML_Competitions/tree/master/What%20Makes%20a%20Title%20Interesting

The code for this task is under `What Makes a Title Interesting/`. Run:

```powershell
python "What Makes a Title Interesting/src/run_experiments.py"
```

This reproduces the experiment table, run manifest, and `run1.csv` through `run3.csv` with seed `410510`.
