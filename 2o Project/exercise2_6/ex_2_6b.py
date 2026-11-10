"""
Exercise 2.6 B: SVM after a nonlinear transformation, solved analytically
with the KKT conditions (Python port of ex_2_6b.m).

Pipeline
--------
1. Define the 2-D training samples of the two classes
       omega_1 (+1): [2, 2], [2, -2], [-2, -2], [-2, 2]
       omega_2 (-1): [1, 1], [1, -1], [-1, -1], [-1, 1]
   and plot them. omega_2 lies inside the square of omega_1, so no line
   separates the two classes in the original space.
2. Apply Phi(x) = x - 4*||x||_2 - 4, with ||x||_2^2 = x1^2 + x2^2 (the
   scalar 4*||x||_2 + 4 is subtracted from both coordinates of x) and
   print the transformed samples.
3. In the transformed space the two closest samples of the two classes,
   Phi([2, 2]) and Phi([-1, -1]), are the support vectors. With two
   support vectors xp (+1), xn (-1) the KKT conditions give
       lambda_p = lambda_n = lambda,   w = lambda * (xp - xn),
       w * xp' + w0 = 1,   w * xn' + w0 = -1
       =>  lambda = 2 / ||xp - xn||^2,   w0 = -w * (xp + xn)' / 2
   (the line is the perpendicular bisector of the segment xp - xn).
4. Print the support vectors, lambda, w, w0, the margin width 2/||w|| and
   the smallest y * (w * x' + w0) over all samples (must be >= 1).
5. Plot the transformed samples, the support vectors, the separating line
   and the margins.

Usage
-----
    python ex_2_6b.py [--save-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

X_POS = np.array([[2, 2], [2, -2], [-2, -2], [-2, 2]], dtype=float)  # omega_1 (+1)
X_NEG = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=float)  # omega_2 (-1)


def phi(X):
    """Phi(x) = x - 4*||x||_2 - 4, applied to every row of X."""
    return X - (4 * np.linalg.norm(X, axis=1, keepdims=True) + 4)


def kkt_solution(xp, xn):
    """Maximum margin line for one support vector per class.

    Returns (lam, w, w0) for the support vectors xp (+1) and xn (-1).
    """
    lam = 2 / np.sum((xp - xn) ** 2)
    w = lam * (xp - xn)
    w0 = -w @ (xp + xn) / 2
    return lam, w, w0


def plot_classes(ax, pos, neg):
    """Samples of omega_1 as blue crosses and of omega_2 as red circles."""
    ax.plot(pos[:, 0], pos[:, 1], "b+", ms=10, mew=2, label=r"$\omega_1$ (+1)")
    ax.plot(neg[:, 0], neg[:, 1], "ro", ms=10, mew=2, fillstyle="none",
            label=r"$\omega_2$ (-1)")
    ax.set_box_aspect(1)
    ax.grid(True)


def plot_original():
    """B.1: the original samples."""
    fig, ax = plt.subplots()
    plot_classes(ax, X_POS, X_NEG)
    ax.axis([-3, 3, -3, 3])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2,
              fontsize="small")
    ax.set_title("Exercise 2.6 B: original samples")
    return fig


def plot_transformed(P_pos, P_neg, xp, xn, w, w0):
    """B.2 / B.3: transformed samples, support vectors, line and margins."""
    fig, ax = plt.subplots()
    plot_classes(ax, P_pos, P_neg)
    ax.plot([xp[0], xn[0]], [xp[1], xn[1]], "ks", ms=16, fillstyle="none",
            label="Support vectors")
    # Lines w*x' + w0 = c  =>  x2 = (c - w0 - w[0]*x1) / w[1]
    t = np.linspace(-20, -6, 10)
    ax.plot(t, (0 - w0 - w[0] * t) / w[1], "k-", lw=2, label="Separating line")
    ax.plot(t, (1 - w0 - w[0] * t) / w[1], "b--", label="Margin (+1)")
    ax.plot(t, (-1 - w0 - w[0] * t) / w[1], "r--", label="Margin (-1)")
    ax.axis([-20, -6, -20, -6])
    ax.set_xlabel(r"$\Phi_1(x)$")
    ax.set_ylabel(r"$\Phi_2(x)$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=3,
              fontsize="small")
    ax.set_title("Exercise 2.6 B: transformed samples")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Exercise 2.6 B: SVM with KKT on transformed samples")
    parser.add_argument("--save-dir", default=None,
                        help="also save the figures as PNG in this folder")
    args = parser.parse_args(argv)

    # B.2: transformed samples
    P_pos, P_neg = phi(X_POS), phi(X_NEG)
    print("Phi(x) for omega_1:")
    print(P_pos)
    print("Phi(x) for omega_2:")
    print(P_neg)

    # B.3: support vectors Phi([2, 2]) and Phi([-1, -1])
    xp, xn = P_pos[0], P_neg[2]
    lam, w, w0 = kkt_solution(xp, xn)
    print(f"Support vectors: [{xp[0]:g} {xp[1]:g}] (+1), "
          f"[{xn[0]:g} {xn[1]:g}] (-1)")
    print(f"lambda = {lam:g}, w = [{w[0]:g} {w[1]:g}], w0 = {w0:g}, "
          f"margin width = {2 / np.linalg.norm(w):g}")
    # All samples must satisfy y * (w*x' + w0) >= 1
    y_times_g = np.concatenate([P_pos @ w + w0, -(P_neg @ w + w0)])
    print(f"min y*(w*x'+w0) over all samples: {y_times_g.min():g}")

    figures = [("ex_2_6b_original", plot_original()),
               ("ex_2_6b_transformed", plot_transformed(P_pos, P_neg, xp, xn, w, w0))]
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures:
            fig.savefig(save_dir / f"{name}.png", dpi=150, bbox_inches="tight")
            print(f"Saved {save_dir / (name + '.png')}")
    plt.show()


if __name__ == "__main__":
    main()
