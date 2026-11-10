"""
Gaussian contours and Bayes decision boundaries (Python port of
exercise1_5.m).

Two classes with 2-D Gaussian class conditional densities

    m1 = [3 3],  S1 = [1.2 -0.4; -0.4 1.2]
    m2 = [6 6],  S2 = [1.2  0.4;  0.4 1.2]

and priors P(w1), P(w2) = 1 - P(w1).

Pipeline
--------
1. Evaluate the two densities on the grid [-3, 12] x [-3, 12] (step 0.01)
   and draw some of their contours.
2. S1 != S2: the decision boundary g1(x) = g2(x), with
       gi(x) = -0.5*(x-mi)'*inv(Si)*(x-mi) - 0.5*ln|Si| + ln P(wi),
   is the hyperbola
       1.25*x*y = 22.5 - 2*ln(P(w2)/P(w1)).
   It is drawn for P(w1) = 0.1, 0.25, 0.5, 0.75, 0.9.
3. S1 == S2 (both equal to S2): the quadratic terms cancel and the boundary
   is the straight line
       3.75*(x + y) = 33.75 - 2*ln(P(w2)/P(w1)),
   drawn for the same priors in a second figure.

Differences with respect to the MATLAB script
---------------------------------------------
* The MATLAB script prints nothing. This script prints the equation of
  every boundary and checks it numerically: the largest |g1(x) - g2(x)| over
  the points of the drawn curve must be zero up to rounding.
* The density is computed by the function mvn_pdf of this file (MATLAB uses
  mvnpdf of the Statistics toolbox).

Usage
-----
    python exercise1_5.py [--dt 0.01] [--priors 0.1 0.25 0.5 0.75 0.9]
                          [--save-dir DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

M1 = np.array([3.0, 3.0])
M2 = np.array([6.0, 6.0])
S1 = np.array([[1.2, -0.4], [-0.4, 1.2]])
S2 = np.array([[1.2, 0.4], [0.4, 1.2]])

# contour levels of the class conditional densities (their peak is ~0.14)
LEVELS = [0.0001, 0.001, 0.01, 0.05, 0.1]
COLORS = "kmgcy"  # one colour per prior, as in the MATLAB script


def mvn_pdf(X, mu, sigma):
    """Multivariate normal density at the rows of X (mirrors MATLAB's
    mvnpdf(X, mu, sigma))."""
    d = len(mu)
    diff = X - mu
    maha = np.sum((diff @ np.linalg.inv(sigma)) * diff, axis=1)
    norm_const = np.sqrt((2 * np.pi) ** d * np.linalg.det(sigma))
    return np.exp(-0.5 * maha) / norm_const


def discriminant(X, mu, sigma, prior):
    """Bayes discriminant function of a Gaussian class at the rows of X:
    g(x) = -0.5*(x-mu)'*inv(sigma)*(x-mu) - 0.5*ln|sigma| + ln(prior)."""
    diff = X - mu
    maha = np.sum((diff @ np.linalg.inv(sigma)) * diff, axis=1)
    return -0.5 * maha - 0.5 * np.log(np.linalg.det(sigma)) + np.log(prior)


def boundary_unequal(x, pw1):
    """Decision boundary for S1 != S2: the hyperbola
    1.25*x*y = 22.5 - 2*ln(P(w2)/P(w1)). Returns y for every x; the curve
    is not defined at x = 0 (NaN)."""
    x = np.where(np.abs(x) < 1e-9, np.nan, x)
    return (22.5 - 2 * np.log((1 - pw1) / pw1)) / (1.25 * x)


def boundary_equal(x, pw1):
    """Decision boundary for S1 == S2 (= S2): the straight line
    3.75*(x + y) = 33.75 - 2*ln(P(w2)/P(w1)). Returns y for every x."""
    return (33.75 - 2 * np.log((1 - pw1) / pw1) - 3.75 * x) / 3.75


def boundary_residual(x, y, sigma1, sigma2, pw1):
    """Largest |g1 - g2| over the points (x, y) of a boundary curve."""
    ok = np.isfinite(y)
    pts = np.column_stack([x[ok], y[ok]])
    g1 = discriminant(pts, M1, sigma1, pw1)
    g2 = discriminant(pts, M2, sigma2, 1 - pw1)
    return np.max(np.abs(g1 - g2))


def plot_case(num, x1, x2, Y1R, Y2R, boundary, priors, title):
    """One figure: contours of the two densities (red: class 1, blue:
    class 2) and the decision boundary of every prior."""
    fig, ax = plt.subplots(num=num, figsize=(7, 7))
    ax.contour(x1, x2, Y1R, levels=LEVELS, colors="r", linewidths=1)
    ax.contour(x1, x2, Y2R, levels=LEVELS, colors="b", linewidths=1)
    ax.grid(True)
    handles = []
    for i, pw1 in enumerate(priors):
        color = COLORS[i % len(COLORS)]
        ax.plot(x1, boundary(x1, pw1), color, linewidth=1)
        handles.append(Line2D([], [], color=color, linewidth=1,
                              label=f"$P(w_1)={pw1:g}$"))
    ax.axis([-3, 12, -3, 12])
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend(handles=handles, loc="upper right")


def finish_figures(save_dir, prefix):
    """Save every open figure as PNG in save_dir (if given) and show them."""
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for num in plt.get_fignums():
            plt.figure(num).savefig(save_dir / f"{prefix}_fig{num}.png", dpi=120)
        print(f"\nFigures saved in {save_dir}")
    plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Gaussian contours and Bayes decision boundaries")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="grid step (default: 0.01)")
    parser.add_argument("--priors", type=float, nargs="+",
                        default=[0.1, 0.25, 0.5, 0.75, 0.9], metavar="P",
                        help="values of P(w1) (default: 0.1 0.25 0.5 0.75 0.9)")
    parser.add_argument("--save-dir", default=None, metavar="DIR",
                        help="save every figure as PNG in this folder")
    args = parser.parse_args(argv)
    if not 0 < args.dt <= 1:
        parser.error("--dt must be in (0, 1]")
    if any(not 0 < p < 1 for p in args.priors):
        parser.error("--priors values must be in (0, 1)")

    # Grid -3:DT:12 on both axes
    n = int(round(15 / args.dt)) + 1
    x1 = -3 + args.dt * np.arange(n)
    x2 = x1.copy()
    X1, X2 = np.meshgrid(x1, x2)
    grid = np.column_stack([X1.ravel(), X2.ravel()])

    Y2R = mvn_pdf(grid, M2, S2).reshape(X1.shape)

    # ---------------- S1 != S2 ----------------
    Y1R = mvn_pdf(grid, M1, S1).reshape(X1.shape)
    print(f"Peak of p(x|w1) = {Y1R.max():.6f}, peak of p(x|w2) = {Y2R.max():.6f}")
    print("\nS1 != S2: decision boundary 1.25*x*y = 22.5 - 2*ln(P(w2)/P(w1))")
    for pw1 in args.priors:
        rhs = 22.5 - 2 * np.log((1 - pw1) / pw1)
        res = boundary_residual(x1, boundary_unequal(x1, pw1), S1, S2, pw1)
        print(f"  P(w1) = {pw1:<5g} 1.25*x*y = {rhs:.4f}"
              f"   (max |g1-g2| on the curve: {res:.1e})")
    plot_case(1, x1, x2, Y1R, Y2R, boundary_unequal, args.priors,
              r"$\Sigma_1 \neq \Sigma_2$ (red: class 1, blue: class 2)")

    # ---------------- S1 == S2 (both equal to S2) ----------------
    Y1R = mvn_pdf(grid, M1, S2).reshape(X1.shape)
    print("\nS1 == S2: decision boundary 3.75*(x + y) = 33.75 - 2*ln(P(w2)/P(w1))")
    for pw1 in args.priors:
        rhs = 33.75 - 2 * np.log((1 - pw1) / pw1)
        res = boundary_residual(x1, boundary_equal(x1, pw1), S2, S2, pw1)
        print(f"  P(w1) = {pw1:<5g} 3.75*(x + y) = {rhs:.4f}"
              f"   (max |g1-g2| on the curve: {res:.1e})")
    plot_case(2, x1, x2, Y1R, Y2R, boundary_equal, args.priors,
              r"$\Sigma_1 = \Sigma_2$ (red: class 1, blue: class 2)")

    finish_figures(args.save_dir, "ex1_5")


if __name__ == "__main__":
    main()
