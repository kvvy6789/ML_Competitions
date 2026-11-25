# Phase 2b Form Draft - Does the Reply Match the Question?

## Task

I predicted how fully a reply answered a home-improvement question. A reply got label `1` if it covered every request and important constraint, or label `0` if it left something important unanswered. The label measured coverage regardless of whether the advice was correct, safe, or polite.

## Methodology

The dataset was unusually small: 12 labeled training examples, four labeled
validation examples, and eight hidden test examples. The training set contained
seven Fully answered and five Partially answered items. I preserved the supplied
validation set for final comparisons and used four-fold stratified
cross-validation on the 12 training examples for train-only diagnostics. All
models used seed `5102026`. Because a single validation error changes accuracy
by 25 percentage points, I treated validation scores as exploratory rather than
precise estimates.

I parsed each record at the supplied `REPLY` marker to obtain question and reply
views. Text preprocessing lowercased the text, normalized Unicode accents, and
used sublinear TF-IDF. Word models used unigrams and bigrams. The combined
word/character model also used character 3-5-grams. Balanced class weights were
used for discriminative classifiers.

**Majority Baseline.** The majority-class model predicted Fully answered for
every validation item. It achieved 0.750 accuracy but only 0.429 macro-F1,
because it missed the only Partially answered validation item.

**Full-text TF-IDF Logistic Regression.** I modeled the title, question, and
reply together and tested `C` in {0.1, 0.5, 1, 2, 5, 10}. These models also
collapsed to the majority class on validation.

**Reply-only TF-IDF Logistic Regression.** I used only reply text to reduce
question-topic noise. I tested the same regularization grid. This family had
the strongest train-only cross-validation accuracy among the sparse text
models (0.667), but still predicted the majority class for all four validation
examples. The `C=0.1` configuration became `run3` as a conservative lexical
baseline.

**Question-only Diagnostic Model.** This model intentionally excluded replies
to test whether question topic alone was predictive. It achieved only 0.250
validation accuracy, supporting the need to model the relation between the
question and reply rather than topic alone.

**Coverage-emphasis TF-IDF Models.** I represented the question and reply with
explicit markers and duplicated the reply so reply phrases received more
weight. Logistic regression and word+character linear SVM variants still
predicted the majority class on validation.

**Engineered Coverage Logistic Regression.** I created model-derived numeric
features based on the task definition: number of question marks, question and
reply lengths, reply-to-question length ratio, word counts, question/reply
lexical overlap, conjunction/request indicators in the question, and answer
language indicators in the reply. A balanced, standardized logistic regression
with `C=0.1` classified all four validation examples correctly and became
`run1`. Its train-only cross-validation accuracy was only 0.417, so the perfect
validation score should not be interpreted as a stable estimate.

**Engineered Coverage Tree Models.** I tested shallow decision trees and a
300-tree random forest over the same coverage features. The random forest also
classified all four validation examples correctly but had only 0.333 train-only
cross-validation accuracy. It became `run2` as a structurally different model,
not because four validation examples prove it is superior.

**TF-IDF SVD + Trained MLP.** For the neural-network comparison, I reduced TF-IDF to six dense SVD dimensions, standardized them, and trained MLP
classifiers with hidden layers `(8)`, `(16)`, and `(16, 8)`, Adam optimization,
L2 penalty 0.001, and a fixed seed. The best MLP produced majority-class
validation predictions (0.750 accuracy and 0.429 macro-F1). I documented the
neural approach but did not select it for a leaderboard run.

Once I settled on the settings for each run, I trained it again on all 16 labeled examples
and generated one numeric prediction for each of the eight test IDs. The code
checks the sample-submission ID order and restricts labels to 0 or 1.

## Runs

The rows below list the prepared runs. Keep the runs you submitted and enter their Kaggle public scores.

```text
run1 | Engineered Coverage Logistic Regression, C=0.1 | validation accuracy 1.000, macro-F1 1.000 | TBD
run2 | Engineered Coverage Random Forest | validation accuracy 1.000, macro-F1 1.000 | TBD
run3 | Reply-only TF-IDF Logistic Regression, C=0.1 | validation accuracy 0.750, macro-F1 0.429 | TBD
```

Host baseline score(s): `TBD`.

## Reflection

The two models built around coverage features got all four validation examples right. Counts and word overlap gave the models a way to look at both sides of the exchange. For example, a question with several requests may need a reply that addresses several distinct points. I’d be cautious about the result because both models did much worse in cross-validation on the 12 training examples. I also learned how easy it was to overread accuracy here. Predicting Fully answered every time got three out of four examples right, yet missed the only Partially answered example.

## Code

Public repository link: https://github.com/kvvy6789/ML_Competitions/tree/master/Does%20the%20Reply%20Answer%20the%20Questions

The code for this task is under `Does the Reply Answer the Questions/`. Run:

```powershell
python "Does the Reply Answer the Questions/src/run_experiments.py"
```

This reproduces the experiment table, run manifest, and `run1.csv` through
`run3.csv` with seed `5102026`.

