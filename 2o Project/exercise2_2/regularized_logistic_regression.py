"""
Regularised Logistic Regression (Python port of
ex2_2_regularizedLogisticRegression.m and its helpers).

Predicts whether a microchip passes quality assurance from the results of
two tests. The classes are not linearly separable, so the two features are
mapped to all polynomial terms up to degree 6 and the cost is regularised.

Pipeline
--------
1. Read `ex2_2_data2.txt` (test 1, test 2, label 0/1) and plot the classes.
2. Map the two features to the 28 polynomial terms of degree <= 6
   (the first one is the constant 1, i.e. the intercept).
3. Evaluate the regularised cost and gradient at theta = 0 (lambda = 1) and
   at theta = 1 (lambda = 10) as a check.
4. For every lambda (default 0, 1, 10, 100) minimise the cost with
   `scipy.optimize.minimize` (BFGS with the analytic gradient), which takes
   the place of MATLAB's `fminunc`, plot the decision boundary and print
   the accuracy on the training set.

MATLAB helper -> Python function
--------------------------------
    sigmoid.m              -> sigmoid
    mapFeature.m           -> map_feature
    costFunctionReg.m      -> cost_function_reg
    predict.m              -> predict
    plotData.m             -> plot_data
    plotDecisionBoundary.m -> plot_decision_boundary

Usage
-----
    python regularized_logistic_regression.py [--lambdas 0 1 10 100]
                                              [--max-iter 400] [--save-dir DIR]

All figures are shown together at the end; with `--save-dir` they are also
written as PNG files in that folder.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

DATA_FILE = Path(__file__).parent / "ex2_2_data2.txt"
DEGREE = 6  # highest polynomial degree of the feature mapping

# Training accuracies quoted in the MATLAB script for lambda = 0, 1, 10, 100
EXPECTED_ACCURACY = {0: 89, 1: 83.1, 10: 74.6, 100: 61}


def sigmoid(z):
    """Logistic function g(z) = 1 / (1 + exp(-z)), element-wise (sigmoid.m)."""
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def map_feature(x1, x2, degree=DEGREE):
    """Map two features to all polynomial terms up to `degree` (mapFeature.m).

    Columns: 1, x1, x2, x1^2, x1 x2, x2^2, x1^3, ..., x1 x2^5, x2^6
    (28 columns for degree 6). x1 and x2 are scalars or equally long vectors.
    """
    x1 = np.atleast_1d(np.asarray(x1, dtype=float))
    x2 = np.atleast_1d(np.asarray(x2, dtype=float))
    columns = [np.ones_like(x1)]
    for i in range(1, degree + 1):
        for j in range(i + 1):
            columns.append(x1 ** (i - j) * x2 ** j)
    return np.column_stack(columns)


def cost_function_reg(theta, X, y, lam):
    """Regularised cross-entropy cost and its gradient (costFunctionReg.m).

    J = 1/m * sum_i ( -y_i ln h_i - (1 - y_i) ln(1 - h_i) )
        + lam / (2m) * sum_{j>=1} theta_j^2
    grad_j = 1/m * sum_i ( h_i - y_i ) x_ij + lam / m * theta_j   (j >= 1)

    theta_0 (the intercept) is not regularised. Returns (J, grad).
    """
    m = len(y)
    h = sigmoid(X @ theta)
    J = ((-y @ np.log(h) - (1.0 - y) @ np.log(1.0 - h)) / m
         + lam / (2.0 * m) * np.sum(theta[1:] ** 2))
    reg = np.concatenate(([0.0], theta[1:]))
    grad = X.T @ (h - y) / m + lam / m * reg
    return J, grad


def predict(theta, X):
    """Predicted labels (0/1) with a threshold of 0.5 on sigmoid(X theta)
    (predict.m)."""
    return (sigmoid(X @ theta) >= 0.5).astype(float)


def plot_data(X, y, labels=("y = 1", "y = 0")):
    """New figure with + for the positive and yellow o for the negative
    examples (plotData.m). X is an m x 2 matrix. Returns the axes."""
    _, ax = plt.subplots()
    pos, neg = y == 1, y == 0
    ax.plot(X[pos, 0], X[pos, 1], "k+", markeredgewidth=2, markersize=10,
            label=labels[0])
    ax.plot(X[neg, 0], X[neg, 1], "ko", markerfacecolor="y", markersize=10,
            label=labels[1])
    return ax


def plot_decision_boundary(theta, X, y):
    """Plot the data and the contour theta' * map_feature(u, v) = 0 on a
    50 x 50 grid over [-1, 1.5]^2 (the non-linear branch of
    plotDecisionBoundary.m). X is the mapped m x 28 matrix. Returns the axes."""
    ax = plot_data(X[:, 1:3], y)
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    uu, vv = np.meshgrid(u, v)  # z[j, i] belongs to (u[i], v[j])
    z = (map_feature(uu.ravel(), vv.ravel()) @ theta).reshape(uu.shape)
    ax.contour(u, v, z, levels=[0.0], linewidths=2, colors="tab:blue")
    # contour() creates no legend entry, so add a matching proxy line
    ax.plot([], [], color="tab:blue", linewidth=2, label="Decision boundary")
    return ax


def save_figures(save_dir, names):
    """Save every open figure as PNG in save_dir, using the given names."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for num, name in zip(plt.get_fignums(), names):
        plt.figure(num).savefig(save_dir / f"{name}.png", dpi=150,
                                bbox_inches="tight")


def print_first_five(values):
    """Print the first five gradient values, one per line, as in MATLAB."""
    for g in values[:5]:
        print(f" \t{g:f} ")


def main():
    parser = argparse.ArgumentParser(
        description="Regularised logistic regression on the microchip data")
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0, 1, 10, 100],
                        help="regularisation parameters to try (default 0 1 10 100)")
    parser.add_argument("--max-iter", type=int, default=400,
                        help="maximum number of optimiser iterations (default 400)")
    parser.add_argument("--save-dir", default=None,
                        help="folder in which every figure is saved as PNG")
    args = parser.parse_args()

    # ------------------- Load data -------------------
    data = np.loadtxt(DATA_FILE, delimiter=",")
    X, y = data[:, :2], data[:, 2]

    ax = plot_data(X, y)
    ax.set_xlabel("Microchip Test 1")
    ax.set_ylabel("Microchip Test 2")
    ax.legend()
    figure_names = ["data"]

    # ------------------- Part 1: regularised cost and gradient -------------
    X = map_feature(X[:, 0], X[:, 1])  # also adds the column of ones
    initial_theta = np.zeros(X.shape[1])

    cost, grad = cost_function_reg(initial_theta, X, y, 1.0)
    print(f"Cost at initial theta with lambda = 1 (zeros): {cost:f}")
    print("Expected cost (approx): 0.693")
    print("Gradient at initial theta (zeros) - first five values only:")
    print_first_five(grad)
    print("Expected gradients (approx) - first five values only:")
    print(" \t0.0085\n \t0.0188\n \t0.0001\n \t0.0503\n \t0.0115")

    test_theta = np.ones(X.shape[1])
    cost, grad = cost_function_reg(test_theta, X, y, 10.0)
    print(f"\nCost at test theta (with lambda = 10): {cost:f}")
    print("Expected cost (approx): 3.16")
    print("Gradient at test theta - first five values only:")
    print_first_five(grad)
    print("Expected gradients (approx) - first five values only:")
    print(" \t0.3460\n \t0.1614\n \t0.1948\n \t0.2269\n \t0.0922\n")

    # ------------------- Part 2: regularisation and accuracies -------------
    for lam in args.lambdas:
        initial_theta = np.zeros(X.shape[1])
        # Without regularisation the line search may reach a theta where h
        # is exactly 0 or 1 (log(0)); such steps are rejected, so the
        # floating-point warnings are silenced.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            res = minimize(cost_function_reg, initial_theta, args=(X, y, lam),
                           jac=True, method="BFGS",
                           options={"maxiter": args.max_iter})
        theta = res.x

        ax = plot_decision_boundary(theta, X, y)
        ax.set_title(f"lambda = {lam:g}")
        ax.set_xlabel("Microchip Test 1")
        ax.set_ylabel("Microchip Test 2")
        ax.legend()
        figure_names.append(f"boundary_lambda_{lam:g}")

        p = predict(theta, X)
        print(f"lambda = {lam:g}: Train Accuracy: {np.mean(p == y) * 100:f}"
              f"  (cost {res.fun:f}, {res.nit} iterations)")
        if lam in EXPECTED_ACCURACY:
            print(f"Expected accuracy (with lambda = {lam:g}): "
                  f"{EXPECTED_ACCURACY[lam]:g} (approx)")

    if args.save_dir:
        save_figures(args.save_dir, figure_names)
    plt.show()


if __name__ == "__main__":
    main()
