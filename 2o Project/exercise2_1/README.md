# Exercise 2.1 — Logistic regression

Predicts whether a student is admitted to a university from the scores of
two exams, with logistic regression trained by minimising the cross-entropy
cost with its analytic gradient.

## Files

| File | Purpose |
|---|---|
| `My_logisticRegression.m` | MATLAB main script (also runs in GNU Octave) |
| `sigmoid.m`, `costFunction.m`, `predict.m` | Logistic function, cost and gradient, 0/1 prediction |
| `plotData.m`, `plotDecisionBoundary.m` | Plots of the data and of the decision boundary |
| `logistic_regression.py` | Python port of the same pipeline (all helpers in one file) |
| `exam_scores_data1.txt` | Data: exam 1 score, exam 2 score, label (1 = admitted) |

## Run

MATLAB / Octave, from this folder (pauses after every part; press enter):

```
My_logisticRegression
```

Python (needs `numpy`, `scipy` and `matplotlib`):

```
python logistic_regression.py
python logistic_regression.py --exam1 60 --exam2 70 --save-dir figures
```

`--exam1` / `--exam2` are the scores of the student whose admission
probability is predicted (default 45 and 85). The Python script does not
pause: it shows all figures together at the end, and `--save-dir` also
saves them as PNG files.

## Pipeline

1. Plot the two classes.
2. Add the intercept column and evaluate the cost and the gradient at
   theta = 0.
3. Minimise the cost (MATLAB: `fminunc`; Python: `scipy.optimize.minimize`,
   BFGS, with the analytic gradient in both).
4. Plot the decision boundary, predict one student and compute the
   accuracy on the training set.

## Results

| Quantity | MATLAB / Octave | Python |
|---|---|---|
| Cost at theta = 0 | 0.693147 | 0.693147 |
| Gradient at theta = 0 | -0.100000, -12.009217, -11.262842 | same |
| Cost at the optimum | 0.203498 | 0.203498 |
| theta | -25.161272, 0.206233, 0.201470 | -25.161333, 0.206232, 0.201472 |
| P(admitted) for scores 45 and 85 | 0.776289 | 0.776291 |
| Training accuracy | 89 % | 89 % |

## Notes

- The two optimisers stop at slightly different points of the same flat
  minimum, so theta agrees to about five significant digits; the cost, the
  decision boundary and the accuracy are the same.
- The accuracy is measured on the training set, so it is an optimistic
  estimate of the accuracy on new students.
