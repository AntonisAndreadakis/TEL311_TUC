# Exercise 2.2 — Regularised logistic regression

Predicts whether a microchip passes quality assurance from the results of
two tests. The classes are not linearly separable, so the two features are
mapped to all polynomial terms up to degree 6 (28 features) and the
logistic-regression cost is regularised with parameter lambda.

## Files

| File | Purpose |
|---|---|
| `ex2_2_regularizedLogisticRegression.m` | MATLAB main script (also runs in GNU Octave) |
| `mapFeature.m` | Polynomial feature mapping |
| `sigmoid.m`, `costFunctionReg.m`, `predict.m` | Logistic function, regularised cost and gradient, 0/1 prediction |
| `plotData.m`, `plotDecisionBoundary.m` | Plots of the data and of the decision boundary |
| `regularized_logistic_regression.py` | Python port of the same pipeline (all helpers in one file) |
| `ex2_2_data2.txt` | Data: test 1, test 2, label (1 = accepted) |

## Run

MATLAB / Octave, from this folder (pauses twice; press enter):

```
ex2_2_regularizedLogisticRegression
```

Python (needs `numpy`, `scipy` and `matplotlib`):

```
python regularized_logistic_regression.py
python regularized_logistic_regression.py --lambdas 0 0.1 1 --save-dir figures
```

The Python script does not pause: it shows all figures together at the
end, and `--save-dir` also saves them as PNG files.

## Pipeline

1. Plot the two classes and map the features to the 28 polynomial terms.
2. Check the cost and the gradient at theta = 0 (lambda = 1) and at
   theta = 1 (lambda = 10).
3. For lambda = 0, 1, 10, 100: minimise the cost (MATLAB: `fminunc`;
   Python: `scipy.optimize.minimize`, BFGS, with the analytic gradient in
   both, at most 400 iterations), plot the decision boundary and print the
   accuracy on the training set. The intercept is not regularised.

## Results

Cost 0.693147 at theta = 0 (lambda = 1) and 3.164509 at theta = 1
(lambda = 10) in both versions; the gradients agree to all printed digits.

Training accuracy (%):

| lambda | MATLAB (expected) | Octave 8.4 | Python |
|---|---|---|---|
| 0 | 89 | 86.44 | 88.14 |
| 1 | 83.1 | 83.05 | 83.05 |
| 10 | 74.6 | 74.58 | 74.58 |
| 100 | 61 | 61.02 | 61.02 |

## Notes

- For lambda > 0 the cost has a unique minimum and all optimisers reach it
  (final cost 0.529003, 0.648216 and 0.686484 for lambda = 1, 10, 100).
- For lambda = 0 nothing limits the size of theta: the cost keeps
  decreasing slowly while theta grows, so the result depends on where the
  optimiser stops. Octave's `fminunc` stops after 157 iterations at cost
  0.2837; SciPy's BFGS uses all 400 iterations and reaches cost 0.2635.
  The accuracy therefore differs by one or two training points out of 118.
- Larger lambda gives a smoother boundary: lambda = 0 overfits (a very
  irregular boundary), lambda = 1 gives a sensible one, and lambda = 100
  underfits (61 % accuracy).
