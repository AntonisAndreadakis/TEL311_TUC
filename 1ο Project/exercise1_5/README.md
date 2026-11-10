# Exercise 1.5 — Bayes decision boundaries of two Gaussian classes

Draws contours of two 2-D Gaussian class conditional densities and the
Bayes decision boundaries for the priors P(w1) = 0.1, 0.25, 0.5, 0.75, 0.9,
first with different covariance matrices and then with a common one.

```
m1 = [3 3]   S1 = [1.2 -0.4; -0.4 1.2]
m2 = [6 6]   S2 = [1.2  0.4;  0.4 1.2]
```

## Files

| File | Purpose |
|---|---|
| `exercise1_5.m` | MATLAB script (also runs in GNU Octave) |
| `exercise1_5.py` | Python port of the same script |

## Run

MATLAB / Octave, from this folder. Octave needs the `statistics` package
for `mvnpdf` (`pkg load statistics`):

```
exercise1_5
```

Python (needs `numpy` and `matplotlib`):

```
python exercise1_5.py
python exercise1_5.py --priors 0.1 0.25 0.5 0.75 0.9 --save-dir figures
```

`--dt` is the grid step (default 0.01), `--priors` the values of P(w1) and
`--save-dir` saves every figure as PNG.

## Pipeline

1. Evaluate both densities on the grid [-3, 12] x [-3, 12] and draw the
   contours at 0.0001, 0.001, 0.01, 0.05, 0.1.
2. Figure 1, S1 != S2: the boundary `g1(x) = g2(x)` is the hyperbola
   `1.25*x*y = 22.5 - 2*ln(P(w2)/P(w1))`.
3. Figure 2, S1 = S2 (both equal to S2): the boundary is the straight line
   `3.75*(x + y) = 33.75 - 2*ln(P(w2)/P(w1))`.

## Key results

Identical in Octave 8.4 and Python. Both densities have a peak of 0.140674.

| P(w1) | S1 != S2: `1.25*x*y =` | S1 = S2: `3.75*(x + y) =` |
|---|---|---|
| 0.1 | 18.1056 | 29.3556 |
| 0.25 | 20.3028 | 31.5528 |
| 0.5 | 22.5000 | 33.7500 |
| 0.75 | 24.6972 | 35.9472 |
| 0.9 | 26.8944 | 38.1444 |

## Notes

- With different covariance matrices the boundary is a quadratic curve;
  here the squared terms cancel and a hyperbola remains. With a common
  covariance matrix the quadratic terms cancel and the boundary is a line.
- A larger P(w1) moves the boundary away from m1, towards m2: the region
  of the more probable class grows.
- The MATLAB script prints nothing. The Python script prints the table
  above and checks every curve: |g1(x) - g2(x)| is below 1e-9 on all its
  points.
