"""
Shared code for exercise 2.7: linear SVM trained through its dual QP.

The MATLAB scripts (svm_exercise1/2/3.m) solve the dual problem with
`quadprog`; here the same problem is given to `cvxopt.solvers.qp`:

    minimise    1/2 * lambda' * H * lambda - sum(lambda)
    subject to  sum(lambda * y) = 0
                0 <= lambda            (hard margin, C = None)
                0 <= lambda <= C       (soft margin)

with H(i, j) = y_i * y_j * x_i * x_j'. From the multipliers:

    w  = sum_i lambda_i * y_i * x_i
    w0 = mean(y_k - x_k * w) over the support vectors that lie exactly on
         the margin (0 < lambda_k < C)

Samples are stored one per row (X is n x d), labels y are +1 / -1.
"""

from pathlib import Path

import numpy as np
from cvxopt import matrix, solvers

DATA_DIR = Path(__file__).parent


def load_data(filename, n=None):
    """Read a 'label x1 x2' text file from this folder.

    Returns (X, y) for the first n examples (all of them if n is None).
    """
    data = np.loadtxt(DATA_DIR / filename)
    if n is not None:
        data = data[:n]
    return data[:, 1:3], data[:, 0]


def add_norm_feature(X):
    """(x1, x2) -> (x1, x2, ||x||^2), with ||x||^2 = x1^2 + x2^2."""
    return np.column_stack([X, (X ** 2).sum(axis=1)])


def solve_dual(X, y, C=None):
    """Lagrange multipliers of the SVM dual problem.

    C = None gives the hard margin problem (no upper bound on lambda),
    otherwise 0 <= lambda <= C.
    """
    n = len(y)
    H = np.outer(y, y) * (X @ X.T)
    f = -np.ones(n)
    if C is None:                       # -lambda <= 0
        G, h = -np.eye(n), np.zeros(n)
    else:                               # -lambda <= 0 and lambda <= C
        G = np.vstack([-np.eye(n), np.eye(n)])
        h = np.concatenate([np.zeros(n), C * np.ones(n)])
    options = {"show_progress": False,
               "abstol": 1e-10, "reltol": 1e-10, "feastol": 1e-10}
    sol = solvers.qp(matrix(H), matrix(f), matrix(G), matrix(h),
                     matrix(y.reshape(1, n)), matrix(0.0), options=options)
    if sol["status"] != "optimal":
        raise RuntimeError(f"QP solver stopped with status '{sol['status']}'")
    return np.array(sol["x"]).ravel()


def fit(X, y, C=None):
    """Train the SVM and return a dict with everything the scripts print.

    Keys: lam (all multipliers), sv (0-based support vector indices),
    w, w0, width (2 / ||w||) and misclassified (training errors).
    """
    lam = solve_dual(X, y, C)
    # numerical tolerance for lambda = 0 or lambda = C
    tol = 1e-4 if C is None else 1e-4 * min(1.0, C)
    sv = np.flatnonzero(lam > tol)
    w = X.T @ (lam * y)
    s = X @ w
    if C is None:
        on_margin = sv
    else:
        on_margin = np.flatnonzero((lam > tol) & (lam < C - tol))
    if len(on_margin):
        # y_k * (x_k * w + w0) = 1  =>  w0 = y_k - x_k * w
        w0 = np.mean(y[on_margin] - s[on_margin])
    else:
        # All the support vectors are at the bound lambda = C (very small
        # C): w0 is not unique, take the middle of the interval allowed by
        # the KKT conditions.
        at_c = lam > C - tol
        w0 = (np.max(-1 - s[(y == -1) & at_c])
              + np.min(1 - s[(y == 1) & at_c])) / 2
    return {
        "lam": lam,
        "sv": sv,
        "w": w,
        "w0": w0,
        "width": 2 / np.linalg.norm(w),
        "misclassified": int(np.sum(np.sign(s + w0) != y)),
    }


def fmt(values, spec):
    """Format a sequence the way MATLAB's sprintf('<spec> ', v) does."""
    return " ".join(format(v, spec) for v in values)


def plot_samples(ax, X, y, sv):
    """Samples of the two classes as dots, support vectors circled."""
    ax.plot(X[y == 1, 0], X[y == 1, 1], "b.", label="y = +1")
    ax.plot(X[y == -1, 0], X[y == -1, 1], "r.", label="y = -1")
    Xs, ys = X[sv], y[sv]
    ax.plot(Xs[ys == 1, 0], Xs[ys == 1, 1], "bo", fillstyle="none",
            label="Support vectors (+1)")
    ax.plot(Xs[ys == -1, 0], Xs[ys == -1, 1], "ro", fillstyle="none",
            label="Support vectors (-1)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def plot_line_and_margins(ax, X, w, w0):
    """Decision line x*w + w0 = 0 and the margins x*w + w0 = +1 / -1.

    x*w + w0 = c  =>  x2 = -(w1*x1 + w0 - c) / w2
    """
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
    ax.plot(x1, -(w[0] * x1 + w0) / w[1], "--k", label="Decision boundary")
    ax.plot(x1, -(w[0] * x1 + w0 - 1) / w[1], "b", label="Margin (+1)")
    ax.plot(x1, -(w[0] * x1 + w0 + 1) / w[1], "r", label="Margin (-1)")
    ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)


def save_figures(figures, save_dir):
    """Save every (name, figure) pair as <save_dir>/<name>.png."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures:
        fig.savefig(save_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        print(f"Saved {save_dir / (name + '.png')}")
