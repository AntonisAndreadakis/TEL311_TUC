# Exercise 2.5 — Bayesian estimation of a Gaussian mean

Estimates the mean mu of Gaussian measurements with known standard
deviation sigma = 1.25 from n = 1..25 measurements, with the Gaussian prior
N(mu_0 = 0, sigma_0^2), and plots the posterior density of mu and the
predictive density of a new measurement.

## Files

| File | Purpose |
|---|---|
| `ex_2_5.m` | MATLAB version (also runs in GNU Octave with the `statistics` package, for `normpdf`) |
| `ex_2_5.py` | Python port of the same pipeline |
| `data_2_5.txt` | The 25 measurements, as text `H(25)=[ ... ]` |

## Run

MATLAB / Octave, from this folder (in Octave run `pkg load statistics` first):

```
ex_2_5
```

Python (needs `numpy` and `matplotlib`):

```
python ex_2_5.py
python ex_2_5.py --factors 10 1 0.1 0.01 --save-dir figures
```

`--sigma`, `--mu0`, `--posterior-factor` (question A) and `--factors`
(question B) set the parameters; a factor f means sigma_0^2 = f * sigma^2.
The figures are shown together at the end, and `--save-dir` also saves them
as PNG files.

## Pipeline

1. Running sample mean of the first n measurements, n = 1..25.
2. Question A: posterior `p(mu|H(n)) = N(mu_n, sigma_n^2)` for
   sigma_0^2 = 10 sigma^2, with
   `mu_n = (n sigma_0^2 mean_n + sigma^2 mu_0) / (n sigma_0^2 + sigma^2)` and
   `sigma_n^2 = sigma_0^2 sigma^2 / (n sigma_0^2 + sigma^2)`.
3. Question B: predictive density `p(x|H(n)) = N(mu_n, sigma^2 + sigma_n^2)`
   for sigma_0^2 = 10, 1, 0.1 and 0.01 times sigma^2, one figure each.

## Results

Values at n = 25 (identical in MATLAB / Octave and Python); the sample mean
of the 25 measurements is 4.867364.

| sigma_0^2 / sigma^2 | mu_n | std of p(mu\|H(n)) | std of p(x\|H(n)) |
|---|---|---|---|
| 10 | 4.847972 | 0.249501 | 1.274657 |
| 1 | 4.680158 | — | 1.273812 |
| 0.1 | 3.476689 | — | 1.267731 |
| 0.01 | 0.973473 | — | 1.254990 |

## Notes

- With a broad prior (sigma_0^2 = 10 sigma^2) the posterior becomes narrow
  and settles close to the sample mean as n grows.
- The smaller sigma_0^2 is, the more the estimate stays near the prior mean
  mu_0 = 0: for sigma_0^2 = 0.01 sigma^2 the mean after 25 measurements is
  still only 0.97.
- The standard deviation of the predictive density tends to sigma = 1.25,
  because the uncertainty about mu vanishes but the measurement noise
  remains.
- In the Python figures the colour of a curve shows n (dark: small n).
