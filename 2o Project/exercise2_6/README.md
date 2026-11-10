# Exercise 2.6 — SVM solved analytically with the KKT conditions

Two small 2-D problems where the maximum margin line is found by hand with
Lagrange multipliers and the Karush-Kuhn-Tucker conditions; the scripts
evaluate the solution and draw the samples, the support vectors, the
separating line and the margins.

- **Part A** — omega_1 (+1): [3,1], [3,-1], [5,1], [5,-1];
  omega_2 (-1): [1,0], [0,1], [0,-1], [-1,0]. Linearly separable.
- **Part B** — omega_1 (+1): [2,2], [2,-2], [-2,-2], [-2,2];
  omega_2 (-1): [1,1], [1,-1], [-1,-1], [-1,1]. omega_2 lies inside the
  square of omega_1, so the samples are first mapped with
  `Phi(x) = x - 4*||x||_2 - 4` (the scalar `4*||x||_2 + 4` is subtracted
  from both coordinates), which makes them linearly separable.

## Files

| File | Purpose |
|---|---|
| `ex_2_6a.m`, `ex_2_6b.m` | MATLAB version (also runs in GNU Octave) |
| `ex_2_6a.py`, `ex_2_6b.py` | Python ports, one per MATLAB script |

## Run

MATLAB / Octave, from this folder:

```
ex_2_6a
ex_2_6b
```

Python (needs `numpy` and `matplotlib`):

```
python ex_2_6a.py
python ex_2_6b.py
```

Add `--save-dir DIR` to also save the figures as PNG in `DIR`.

## Method

The KKT conditions of the hard margin SVM are

    w = sum_i lambda_i * y_i * x_i,   sum_i lambda_i * y_i = 0,
    lambda_i >= 0,   y_i * (w * x_i' + w0) = 1 for every support vector.

- **A**: the support vectors are [3,1], [3,-1] (+1) and [1,0] (-1). Solving
  the conditions on these three samples gives lambda = [1/4, 1/4, 1/2].
- **B**: after the transformation the closest samples of the two classes
  are Phi([2,2]) and Phi([-1,-1]). With one support vector per class,
  `lambda = 2 / ||xp - xn||^2`, `w = lambda * (xp - xn)` and
  `w0 = -w * (xp + xn)' / 2`: the line is the perpendicular bisector of
  the segment between them.

## Results

| Part | Support vectors | lambda | w | w0 | Margin width 2/\|\|w\|\| |
|---|---|---|---|---|---|
| A | [3,1], [3,-1], [1,0] | 1/4, 1/4, 1/2 | [1, 0] | -2 | 2 |
| B | Phi([2,2]) = [-13.3137, -13.3137], Phi([-1,-1]) = [-10.6569, -10.6569] | 0.141666, 0.141666 | [-0.376385, -0.376385] | -9.02216 | 3.75736 |

Part A: separating line x1 = 2 with margins x1 = 1 and x1 = 3.

Part B: transformed samples

| x | Phi(x) | | x | Phi(x) |
|---|---|---|---|---|
| [2, 2] | [-13.3137, -13.3137] | | [1, 1] | [-8.6569, -8.6569] |
| [2, -2] | [-13.3137, -17.3137] | | [1, -1] | [-8.6569, -10.6569] |
| [-2, -2] | [-17.3137, -17.3137] | | [-1, -1] | [-10.6569, -10.6569] |
| [-2, 2] | [-17.3137, -13.3137] | | [-1, 1] | [-10.6569, -8.6569] |

## Notes

- MATLAB and Python print identical numbers.
- Both solutions were checked by solving the hard margin dual QP
  numerically on the same samples: the solver returns the same support
  vectors, multipliers, w and w0.
- In part B the smallest `y * (w * x' + w0)` over all eight samples is
  exactly 1, so every sample is on or outside its margin.
