"""
Logistic Regression (Python port of My_logisticRegression.m and its helpers).

Predicts whether a student is admitted to a university from the scores of
two exams.

Pipeline
--------
1. Read `exam_scores_data1.txt` (exam 1 score, exam 2 score, label 0/1) and
   plot the two classes.
2. Add the intercept column and evaluate the cross-entropy cost J(theta)
   and its analytic gradient at theta = 0.
3. Minimise J(theta) with `scipy.optimize.minimize` (BFGS with the analytic
   gradient), which takes the place of MATLAB's `fminunc`.
4. Plot the linear decision boundary theta' * x = 0.
5. Predict the admission probability of one student (default: scores 45 and
   85) and compute the accuracy on the training set.

MATLAB helper -> Python function
--------------------------------
    sigmoid.m              -> sigmoid
    costFunction.m         -> cost_function
    predict.m              -> predict
    plotData.m             -> plot_data
    plotDecisionBoundary.m -> plot_decision_boundary

Usage
-----
    python logistic_regression.py [--exam1 45] [--exam2 85]
                                  [--max-iter 400] [--save-dir DIR]

All figures are shown together at the end; with `--save-dir` they are also
written as PNG files in that folder.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

DATA_FILE = Path(__file__).parent / "exam_scores_data1.txt"


def sigmoid(z):
    """Logistic function f(z) = 1 / (1 + exp(-z)), element-wise (sigmoid.m)."""
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def cost_function(theta, X, y):
    """Cross-entropy cost and its gradient (costFunction.m).

    J(theta)    = 1/m * sum_i ( -y_i ln h_i - (1 - y_i) ln(1 - h_i) )
    dJ/dtheta_j = 1/m * sum_i ( h_i - y_i ) x_ij,   h_i = sigmoid(x_i . theta)

    Returns (J, grad); grad has the same shape as theta.
    """
    m = len(y)
    h = sigmoid(X @ theta)
    J = np.sum(-y * np.log(h) - (1.0 - y) * np.log(1.0 - h)) / m
    grad = X.T @ (h - y) / m
    return J, grad


def predict(theta, X):
    """Predicted labels (0/1) with a threshold of 0.5 on sigmoid(X theta)
    (predict.m)."""
    return (sigmoid(X @ theta) >= 0.5).astype(float)


def plot_data(X, y, labels=("Admitted", "Not admitted")):
    """New figure with + for the positive and yellow o for the negative
    examples (plotData.m). X is an m x 2 matrix. Returns the axes."""
    _, ax = plt.subplots()
    pos, neg = y == 1, y == 0
    ax.plot(X[pos, 0], X[pos, 1], "k+", markeredgewidth=2, markersize=7,
            label=labels[0])
    ax.plot(X[neg, 0], X[neg, 1], "ko", markerfacecolor="y", markersize=7,
            label=labels[1])
    return ax


def plot_decision_boundary(theta, X, y):
    """Plot the data and the line theta_0 + theta_1 x_1 + theta_2 x_2 = 0
    (the linear branch of plotDecisionBoundary.m). X is m x 3 with a leading
    column of ones. Returns the axes."""
    ax = plot_data(X[:, 1:3], y)
    # Two points are enough to define the line
    plot_x = np.array([X[:, 1].min() - 2, X[:, 1].max() + 2])
    plot_y = (-1.0 / theta[2]) * (theta[1] * plot_x + theta[0])
    ax.plot(plot_x, plot_y, label="Decision Boundary")
    ax.axis([30, 100, 30, 100])
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
        description="Logistic regression on the exam-scores data")
    parser.add_argument("--exam1", type=float, default=45.0,
                        help="exam 1 score of the student to predict (default 45)")
    parser.add_argument("--exam2", type=float, default=85.0,
                        help="exam 2 score of the student to predict (default 85)")
    parser.add_argument("--max-iter", type=int, default=400,
                        help="maximum number of optimiser iterations (default 400)")
    parser.add_argument("--save-dir", default=None,
                        help="folder in which every figure is saved as PNG")
    args = parser.parse_args()

    # ------------------- Load data -------------------
    data = np.loadtxt(DATA_FILE, delimiter=",")
    X, y = data[:, :2], data[:, 2]

    # ------------------- Part 1: plotting -------------------
    print("Plotting data with + indicating (y = 1) examples and o "
          "indicating (y = 0) examples.")
    ax = plot_data(X, y)
    ax.set_xlabel("Exam 1 score")
    ax.set_ylabel("Exam 2 score")
    ax.legend(loc="upper right")

    # ------------------- Part 2: cost and gradient -------------------
    m, n = X.shape
    X = np.hstack([np.ones((m, 1)), X])  # intercept term
    initial_theta = np.zeros(n + 1)

    cost, grad = cost_function(initial_theta, X, y)
    print(f"\nCost at initial theta (zeros): {cost:f}")
    print("Gradient at initial theta (zeros): ")
    for g in grad:
        print(f" {g:f} ")

    # ------------------- Part 3: optimisation -------------------
    # The line search may try a far-away theta where h is exactly 0 or 1
    # (log(0) = -inf, 0 * inf = nan); such steps are simply rejected, as in
    # MATLAB, so the floating-point warnings are silenced.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        res = minimize(cost_function, initial_theta, args=(X, y), jac=True,
                       method="BFGS", options={"maxiter": args.max_iter})
    theta, cost = res.x, res.fun

    print(f"\nCost at theta found by scipy.optimize.minimize: {cost:f}")
    print("theta: ")
    for t in theta:
        print(f" {t:f} ")

    ax = plot_decision_boundary(theta, X, y)
    ax.set_xlabel("Exam 1 score")
    ax.set_ylabel("Exam 2 score")
    ax.legend(loc="upper right")

    # ------------------- Part 4: prediction and accuracy -------------------
    prob = sigmoid(np.array([1.0, args.exam1, args.exam2]) @ theta)
    print(f"\nFor a student with scores {args.exam1:g} and {args.exam2:g}, "
          f"we predict an admission probability of {prob:f}\n")

    p = predict(theta, X)
    print(f"Train Accuracy: {np.mean(p == y) * 100:f}")

    if args.save_dir:
        save_figures(args.save_dir, ["data", "decision_boundary"])
    plt.show()


if __name__ == "__main__":
    main()
