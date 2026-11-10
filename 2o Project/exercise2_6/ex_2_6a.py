"""
Exercise 2.6 A: SVM solved analytically with the KKT conditions
(Python port of ex_2_6a.m).

Pipeline
--------
1. Define the 2-D training samples of the two classes
       omega_1 (+1): [3, 1], [3, -1], [5, 1], [5, -1]
       omega_2 (-1): [1, 0], [0, 1], [0, -1], [-1, 0]
2. The closest samples of the two classes, [3, 1], [3, -1] (omega_1) and
   [1, 0] (omega_2), are the support vectors. The KKT conditions
       w = sum_i lambda_i * y_i * x_i,   sum_i lambda_i * y_i = 0,
       y_i * (w * x_i' + w0) = 1
   on these three samples give lambda = [1/4, 1/4, 1/2], w = [1, 0],
   w0 = -2: the separating line x1 = 2 with margins x1 = 1 and x1 = 3.
3. Print w, w0 and the margin width 2/||w||.
4. Plot the samples, the support vectors, the separating line and the
   margins.

Usage
-----
    python ex_2_6a.py [--save-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

X_POS = np.array([[3, 1], [3, -1], [5, 1], [5, -1]], dtype=float)   # omega_1 (+1)
X_NEG = np.array([[1, 0], [0, 1], [0, -1], [-1, 0]], dtype=float)   # omega_2 (-1)


def kkt_solution():
    """Separating line from the multipliers found with the KKT conditions.

    Returns (Xsup, w, w0): the support vectors (one per row), the weight
    vector and the bias.
    """
    Xsup = np.array([X_POS[0], X_POS[1], X_NEG[0]])
    ysup = np.array([1.0, 1.0, -1.0])
    lam = np.array([1 / 4, 1 / 4, 1 / 2])
    w = (lam * ysup) @ Xsup             # w = sum_i lambda_i * y_i * x_i
    w0 = np.mean(ysup - Xsup @ w)       # y_i * (w * x_i' + w0) = 1
    return Xsup, w, w0


def plot_solution(Xsup, w, w0):
    """Samples, support vectors, separating line and margins."""
    fig, ax = plt.subplots()
    ax.plot(X_POS[:, 0], X_POS[:, 1], "b+", ms=10, mew=2, label=r"$\omega_1$ (+1)")
    ax.plot(X_NEG[:, 0], X_NEG[:, 1], "ro", ms=10, mew=2, fillstyle="none",
            label=r"$\omega_2$ (-1)")
    ax.plot(Xsup[:, 0], Xsup[:, 1], "ks", ms=16, fillstyle="none",
            label="Support vectors")
    # w[1] = 0, so the lines w*x' + w0 = c are vertical: x1 = (c - w0) / w[0]
    t = np.array([-2.5, 2.5])
    x_line = (0 - w0) / w[0]
    ax.plot([x_line, x_line], t, "k-", lw=2,
            label=f"Separating line x1 = {x_line:g}")
    ax.plot([(1 - w0) / w[0]] * 2, t, "b--", label="Margin (+1)")
    ax.plot([(-1 - w0) / w[0]] * 2, t, "r--", label="Margin (-1)")
    ax.axis([-2, 6, -2.5, 2.5])
    ax.grid(True)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=3,
              fontsize="small")
    ax.set_title("Exercise 2.6 A")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description="Exercise 2.6 A: SVM with KKT")
    parser.add_argument("--save-dir", default=None,
                        help="also save the figure as PNG in this folder")
    args = parser.parse_args(argv)

    Xsup, w, w0 = kkt_solution()
    print(f"w = [{w[0]:g} {w[1]:g}], w0 = {w0:g}, "
          f"margin width = {2 / np.linalg.norm(w):g}")

    fig = plot_solution(Xsup, w, w0)
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "ex_2_6a.png", dpi=150, bbox_inches="tight")
        print(f"Saved {save_dir / 'ex_2_6a.png'}")
    plt.show()


if __name__ == "__main__":
    main()
