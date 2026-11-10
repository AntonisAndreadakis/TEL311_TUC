# Exercise 2.7 — Support Vector Machines on an artificial dataset

Trains a linear SVM by solving its dual quadratic program

    minimise    1/2 * lambda' * H * lambda - sum(lambda),   H(i,j) = y_i * y_j * x_i * x_j'
    subject to  sum(lambda .* y) = 0,   0 <= lambda (<= C)

and recovers `w = sum_i lambda_i * y_i * x_i` and the bias `w0` from the
support vectors that lie on the margin.

- **Part 1** — hard margin SVM on the first 50 examples of `twofeature1.txt`
  (linearly separable).
- **Part 2** — soft margin SVM (slack variables, `0 <= lambda <= C`) on all
  51 examples; the last one is an outlier.
- **Part 3** — `twofeature2.txt` (100 examples, one class surrounds the
  other) with the extra feature `x1^2 + x2^2`, so that a plane in 3-D gives a
  circular boundary in the original plane.

## Files

| File | Purpose |
|---|---|
| `svm_exercise1.m`, `svm_exercise2.m`, `svm_exercise3.m` | MATLAB version (dual QP with `quadprog`) |
| `svm_exercise1.py`, `svm_exercise2.py`, `svm_exercise3.py` | Python ports, one per MATLAB script |
| `svm_dual.py` | Shared Python code: data loading, dual QP (`cvxopt`), `w`, `w0`, plots |
| `twofeature1.txt`, `twofeature2.txt` | Data: `label x1 x2` per line, label in {+1, -1} |

## Run

MATLAB, from this folder (needs the Optimization Toolbox for `quadprog`;
`C` is set at the top of `svm_exercise2.m` / `svm_exercise3.m`):

```
svm_exercise1
svm_exercise2
svm_exercise3
```

Python (needs `numpy`, `matplotlib` and `cvxopt`):

```
python svm_exercise1.py
python svm_exercise2.py --C 10000
python svm_exercise3.py --C 1
```

The defaults of `--C` are the values in the MATLAB scripts. Add
`--save-dir DIR` to also save the figure as PNG in `DIR`.

## Results

Python (`cvxopt`) numbers; support vector indices are 1-based.

**Part 1** (hard margin, 50 examples): support vectors 12, 13, 48 with
lambda = 6.3044, 1.6355, 7.9398; w = [2.5414, 3.0694], w0 = -16.316236,
margin width 2/||w|| = 0.501891.

**Part 2** (soft margin, 51 examples):

| C | Support vectors | w | w0 | Margin width | Misclassified |
|---|---|---|---|---|---|
| 0.01 | 42 | [0.1842, 0.2181] | -1.2426 | 7.0058 | 3 |
| 0.1 | 28 | [0.7240, 1.0396] | -5.2272 | 1.5787 | 1 |
| 1 | 12 | [1.4067, 2.1332] | -10.3450 | 0.7827 | 1 |
| 10 | 7 | [2.0522, 4.1681] | -18.2454 | 0.4305 | 1 |
| 100 | 3 | [4.6838, 13.0958] | -53.1565 | 0.1438 | 0 |
| 1000 | 3 | [4.6838, 13.0958] | -53.1565 | 0.1438 | 0 |
| 10000 | 3 | [4.6838, 13.0958] | -53.1565 | 0.1438 | 0 |

**Part 3** (extra feature `x1^2 + x2^2`, 100 examples):

| C | Support vectors | w | w0 | Misclassified |
|---|---|---|---|---|
| 0.01 | 100 | [-0.0080, 0.0143, -0.5108] | 0.7002 | 33 |
| 0.1 | 61 | [-0.0299, -0.1550, -1.7687] | 1.1723 | 6 |
| 1 | 27 | [-0.3886, -0.1391, -3.5127] | 1.8639 | 3 |
| 10 | 15 | [-0.7833, -0.3624, -6.0749] | 2.9842 | 3 |
| 100 | 10 | [-0.7149, -0.4661, -7.4458] | 3.6608 | 3 |
| 1000 | 10 | [-0.7149, -0.4661, -7.4458] | 3.6608 | 3 |
| 10000 | 10 | [-0.7149, -0.4661, -7.4458] | 3.6608 | 3 |

## Notes

- A small C buys a wide margin at the price of training errors; as C grows
  the margin shrinks and fewer samples are support vectors.
- In part 2 the 51 examples are still linearly separable, but only with a
  very narrow margin: from C = 100 on the solution is the hard margin one
  (the outlier, example 51, becomes a support vector) and no longer changes.
  Compare with part 1, where leaving the outlier out gives a margin 3.5
  times wider.
- In part 3 the classes overlap, so 3 training errors remain for every
  large C; from C = 100 on, w and w0 no longer change (only the multipliers
  of the samples that violate the margin grow with C).
- When every support vector is at the bound lambda = C (part 2 with
  C = 0.01, part 3 with C = 0.01) no sample lies exactly on the margin and
  w0 is not unique; both versions take the middle of the interval allowed
  by the KKT conditions.
- A multiplier counts as non-zero when it exceeds `1e-4 * min(1, C)`
  (`1e-4` in part 1). `cvxopt` is an interior point solver, so its
  multipliers are exact to about 1e-9 rather than exactly 0 or C.
- The Python part 3 draws the boundary as the zero contour of
  `w * [x1, x2, x1^2 + x2^2]' + w0`; the MATLAB script marks the grid
  points where the absolute value is below 0.01. Same curve.
