# Phase 2b Form Draft - Premature Problem Solving

## Task

The task was to classify a paired counseling context and response as either
premature problem-solving (`1`) or not premature (`0`). The important distinction
was not whether a response contained advice, but whether it introduced advice,
instructions, diagnosis, treatment, or other solution-directed guidance before
adequately acknowledging the emotional concern in the context.

## Methodology

I treated this as a small-data binary text-classification problem. The supplied
data contained 120 training examples, 30 labeled validation examples, and 60
unlabeled test examples. The training labels were somewhat imbalanced (74
positive and 46 negative), while the validation labels contained 18 positive
and 12 negative examples. I preserved the supplied split throughout model
development. I used the 120 training examples for five-fold stratified
cross-validation and kept the 30-example validation set separate for final
development-stage comparisons. All experiments used random seed `410510`.

Preprocessing was deliberately light because capitalization and punctuation
were not expected to carry much reliable signal in only 120 examples. I filled
missing strings defensively (although the downloaded data had no missing
cells), lowercased text, normalized Unicode accents, and extracted sublinear
TF-IDF features. Word models used unigrams and bigrams so that short phrases
related to acknowledgment or advice could be represented. Character models
used character n-grams of length 3-5 to capture lexical variants and partial
phrases. I used balanced class weights for discriminative linear models.

I named the approaches as follows:

**Majority Baseline.** This predicted the most frequent training class for every
example. It established a validation accuracy of 0.600 but only 0.375 macro-F1,
showing why accuracy alone could be misleading with the class imbalance.

**Word TF-IDF Logistic Regression.** I concatenated fields with explicit
`CONTEXT:` and `RESPONSE:` markers and trained balanced logistic regression. I
tested regularization values `C` in {0.1, 0.5, 1, 2, 5, 10}. The best validation
configurations reached 0.683 macro-F1 and 0.700 accuracy.

**Response-only TF-IDF Logistic Regression.** Because the target concerns the
ordering and content of the response, I modeled only the response text. I used
the same word/bigram TF-IDF representation and tested `C` in {0.1, 0.5, 1, 2,
5}. The `C=2` configuration performed best on the held-out validation set,
reaching 0.722 macro-F1 and 0.733 accuracy. This became `run1`.

**Opening-emphasis TF-IDF Logistic Regression.** I duplicated the first 500
characters of each response in a separate `RESPONSE_OPENING:` field, followed
by the complete context and response. The goal was to give early acknowledgment
or early solution language more weight while retaining the full pair. I tested
the same family of regularization values. The frozen `C=5` configuration reached
0.683 macro-F1 and 0.700 accuracy and became `run2`.

**Character TF-IDF Linear SVM.** I tried balanced linear SVMs over character
3-5-grams with `C` in {0.05, 0.1, 0.25, 0.5, 1, 2}. The strongest validation
result in this family was 0.641 macro-F1 and 0.667 accuracy. Character features
were robust to word variants but were less effective than the word-level
response-only model.

**Word+Character TF-IDF Linear SVM.** I combined word unigram/bigram features
and character 3-5-gram features, testing `C` in {0.1, 0.25, 0.5, 1}. The selected
configuration reached 0.612 macro-F1 and 0.633 accuracy, so I did not choose it
for submission.

**Context-only Diagnostic Logistic Regression.** I trained a model using only
the context as a diagnostic experiment. Its weaker validation performance
supported the interpretation that the response behavior, rather than topic
alone, was the more useful signal. Because contexts and labels can still be
correlated in a small sample, I did not use this diagnostic model as a final
system.

**TF-IDF SVD + Trained MLP.** To satisfy the requirement to train a neural
network, I converted word/bigram TF-IDF into 64 dense dimensions with truncated
SVD, standardized the dense representation, and trained MLP classifiers with
hidden structures `(32)`, `(64)`, and `(64, 32)`. I tested L2 penalties of
0.0001 and 0.001, used Adam with learning rate 0.001, early stopping, and the
fixed seed. The best cross-validation-selected MLP reached only 0.362 macro-F1
and 0.567 validation accuracy. With so few examples, the dense representation
and neural optimization were less stable than regularized sparse linear models,
so I documented the experiment but did not select it for a leaderboard run.

Hyperparameters were compared primarily using five-fold stratified
cross-validation on the training partition. The official validation set was
then used for final approach comparisons and run selection. After freezing a
run configuration, I refit it on all 150 labeled examples and generated exactly
one binary prediction for each test ID. The code checks that output IDs and
their order match `sample_submission.csv`.

## Runs

Replace each `TBD` after Kaggle reports the score. Include only runs actually
submitted.

```text
run1 | Response-only TF-IDF Logistic Regression (C=2) | macro-F1 0.722; accuracy 0.733 | TBD
run2 | Opening-emphasis TF-IDF Logistic Regression (C=5) | macro-F1 0.683; accuracy 0.700 | TBD
run3 | Combined-text TF-IDF Logistic Regression (C=10) | macro-F1 0.683; accuracy 0.700 | TBD
```

Host baseline score(s): `TBD - copy from the competition Overview/Leaderboard`.

## Reflection

The response-only logistic-regression approach worked best on my validation
set. This result makes sense because the label is defined by what the response
does and when it does it, while the context mainly supplies the emotional issue.
Including the context added many topic words that could be predictive by chance
in a small dataset but did not consistently help identify the target response
behavior. Word bigrams were useful because acknowledgment and directive language
often appears in short phrases. The experiment also showed that a more complex
model is not automatically better: the trained MLP was substantially less
stable than sparse linear classifiers with only 120 training examples. The most
important lessons were to preserve a genuinely held-out validation set, compare
against a majority baseline, use class-sensitive metrics, and design text views
that reflect the actual definition of the label.

## Code

Public repository/notebook link: `TBD - publish this repository before submitting the form`.

The code for this task is in `src/run_experiments.py` and
`src/make_submission.py`. Exact commands are listed in `README.md`. The stable
run mapping is:

- `run1`: `python "Detecting Premature Problem Solving in Counseling Responses/src/make_submission.py" --run-id run1`
- `run2`: `python "Detecting Premature Problem Solving in Counseling Responses/src/make_submission.py" --run-id run2`
- `run3`: `python "Detecting Premature Problem Solving in Counseling Responses/src/make_submission.py" --run-id run3`

