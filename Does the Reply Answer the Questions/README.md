# ARI 510: Does the Reply Match the Question?

Reproducible experiments for the binary reply-coverage competition.

- `0`: Partially answered
- `1`: Fully answered

The course dataset is restricted and is not included in the project package.
Place the supplied files under `Does the Reply Answer the Questions/data/`.

Run all experiments and create three candidate submissions:

```powershell
python "Does the Reply Answer the Questions/src/run_experiments.py"
```

The script uses seed `5102026`, four-fold stratified cross-validation on the 12
training examples, final evaluation on the four supplied validation examples,
and refitting on all 16 labeled examples for test prediction.

