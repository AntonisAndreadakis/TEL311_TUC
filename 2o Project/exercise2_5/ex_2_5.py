"""
Bayesian estimation of the mean of a Gaussian (Python port of ex_2_5.m).

The n = 25 measurements H(n) = {x_1, ..., x_n} are assumed to come from
N(mu, sigma^2) with known sigma = 1.25. The unknown mean has the Gaussian
prior p(mu) = N(mu_0, sigma_0^2).

Pipeline
--------
1. Read the measurements from `data_2_5.txt` (text of the form
   "H(25)=[ v1 v2 ... ]").
2. Compute the running sample mean of the first k measurements, k = 1..n.
3. Question A: plot the posterior of the mean for n = 1..25 with
   sigma_0^2 = 10 sigma^2:
       p(mu|H(n)) = N(mu_n, sigma_n^2)
       mu_n      = (n sigma_0^2 mean_n + sigma^2 mu_0) / (n sigma_0^2 + sigma^2)
       sigma_n^2 = sigma_0^2 sigma^2 / (n sigma_0^2 + sigma^2)
4. Question B: plot the predictive density for n = 1..25 and for
   sigma_0^2 = 10, 1, 0.1 and 0.01 times sigma^2 (one figure each):
       p(x|H(n)) = N(mu_n, sigma^2 + sigma_n^2)
5. Print mu_n and the standard deviations at n = 25 for every case, and the
   sample mean of all measurements.

The MATLAB script has no helper functions; its formulas are the functions
`running_mean`, `posterior_parameters` and `normal_pdf` (in place of
normpdf) here.

Usage
-----
    python ex_2_5.py [--sigma 1.25] [--mu0 0] [--posterior-factor 10]
                     [--factors 10 1 0.1 0.01] [--save-dir DIR]

All figures are shown together at the end; with `--save-dir` they are also
written as PNG files in that folder.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_FILE = Path(__file__).parent / "data_2_5.txt"


def read_measurements(filename=DATA_FILE):
    """Read the measurements: all the numbers after the '[' of the file."""
    text = Path(filename).read_text()
    body = text[text.index("[") + 1:]
    numbers = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", body)
    return np.array(numbers, dtype=float)


def running_mean(h):
    """Sample mean of the first k measurements, for k = 1..n."""
    return np.cumsum(h) / np.arange(1, len(h) + 1)


def posterior_parameters(n, mean_n, var, mu0, var0):
    """Mean and variance of the posterior p(mu|H(n)) = N(mu_n, s_n).

    n      : number of measurements (scalar or array)
    mean_n : sample mean of those n measurements
    var    : known variance sigma^2 of the measurements
    mu0    : prior mean,  var0 : prior variance sigma_0^2
    """
    mu_n = (n * var0 * mean_n + var * mu0) / (n * var0 + var)
    s_n = var0 * var / (n * var0 + var)
    return mu_n, s_n


def normal_pdf(x, mu, std):
    """Density of N(mu, std^2) at x (MATLAB's normpdf)."""
    return np.exp(-0.5 * ((x - mu) / std) ** 2) / (std * np.sqrt(2.0 * np.pi))


def plot_densities(x, means, stds, xlabel, ylabel, title):
    """New figure with one Gaussian density per (mean, std) pair, i.e. one
    curve per n = 1..len(means)."""
    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(means)))
    for mu, std, color in zip(means, stds, colors):
        ax.plot(x, normal_pdf(x, mu, std), linewidth=1, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    n = len(means)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(1, max(n, 2)))
    fig.colorbar(sm, ax=ax, label="n")
    return ax


def save_figures(save_dir, names):
    """Save every open figure as PNG in save_dir, using the given names."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for num, name in zip(plt.get_fignums(), names):
        plt.figure(num).savefig(save_dir / f"{name}.png", dpi=150,
                                bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian estimation of the mean of a Gaussian")
    parser.add_argument("--sigma", type=float, default=1.25,
                        help="known standard deviation of the measurements (default 1.25)")
    parser.add_argument("--mu0", type=float, default=0.0,
                        help="prior mean of mu (default 0)")
    parser.add_argument("--posterior-factor", type=float, default=10.0,
                        help="question A: sigma_0^2 = factor * sigma^2 (default 10)")
    parser.add_argument("--factors", type=float, nargs="+",
                        default=[10, 1, 0.1, 0.01],
                        help="question B: sigma_0^2 = factor * sigma^2 "
                             "(default 10 1 0.1 0.01)")
    parser.add_argument("--save-dir", default=None,
                        help="folder in which every figure is saved as PNG")
    args = parser.parse_args()

    h = read_measurements()
    n = len(h)
    var = args.sigma ** 2            # known variance sigma^2
    y = running_mean(h)              # sample mean of the first k measurements
    counts = np.arange(1, n + 1)
    x = np.linspace(-5, 12, 1000)

    # ------------------- Question A: p(mu|H(n)) -------------------
    var0 = args.posterior_factor * var
    mu_n, s_n = posterior_parameters(counts, y, var, args.mu0, var0)
    plot_densities(
        x, mu_n, np.sqrt(s_n), r"$\mu$", r"$p(\mu|H(n))$",
        rf"$p(\mu|H(n))$, n = 1..{n}, "
        rf"$\sigma_0^2$ = {args.posterior_factor:g}$\sigma^2$")
    print(f"A: sigma0^2 = {args.posterior_factor:g}*sigma^2, n = {n}: "
          f"mu_n = {mu_n[-1]:f}, sigma_n = {np.sqrt(s_n[-1]):f}")
    figure_names = [f"posterior_mu_factor_{args.posterior_factor:g}"]

    # ------------------- Question B: p(x|H(n)) -------------------
    for factor in args.factors:
        var0 = factor * var
        mu_n, s_n = posterior_parameters(counts, y, var, args.mu0, var0)
        plot_densities(
            x, mu_n, np.sqrt(var + s_n), "x", r"$p(x|H(n))$",
            rf"$p(x|H(n))$, n = 1..{n}, $\sigma_0^2$ = {factor:g}$\sigma^2$")
        print(f"B: sigma0^2 = {factor:g}*sigma^2, n = {n}: "
              f"mean = {mu_n[-1]:f}, std = {np.sqrt(var + s_n[-1]):f}")
        figure_names.append(f"predictive_x_factor_{factor:g}")

    print(f"Sample mean of all {n} measurements: {y[-1]:f}")

    if args.save_dir:
        save_figures(args.save_dir, figure_names)
    plt.show()


if __name__ == "__main__":
    main()
