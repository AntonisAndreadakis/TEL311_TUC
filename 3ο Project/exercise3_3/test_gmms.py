"""
Gaussian Mixture Model fitted with EM on the Fisher iris data, compared
with k-means (Python port of test_gmms.m and its helpers).

Pipeline
--------
1. Load the Fisher iris data (150 flowers, 4 measurements, 3 species).
2. Fit a 3-component GMM with the EM algorithm. EM starts from a k-means
   clustering: every cluster gives the weight, mean and covariance of one
   component.
3. Assign every flower to its most probable component.
4. Cluster the same data with k-means alone, for comparison.
5. Match the cluster numbers to the species and print both accuracies.
6. Plot petal length against petal width, coloured by the assigned
   species, with the wrongly assigned flowers circled.

Differences with respect to the MATLAB code
--------------------------------------------
* MATLAB calls the toolbox function kmeans(X, K, 'Replicates', 5). Here
  k-means is implemented in this file with the same defaults: k-means++
  seeding, squared Euclidean distance, at most 100 iterations, and the
  replicate with the lowest total within-cluster distance is kept.
* Cluster and class numbers are 0-based inside the code.
* The random numbers come from numpy, so individual k-means replicates
  differ from MATLAB's. The best of 5 replicates is almost always the same
  partition, and from the same partition EM gives the same model.

Usage
-----
    python test_gmms.py [--replicates 5] [--max-iters 200] [--tol 1e-6]
                        [--seed 0] [--save-dir DIR]

Needs numpy, scipy, matplotlib and scikit-learn (only for the iris data,
the counterpart of MATLAB's `load fisheriris`).
"""

import argparse
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular

K_CLASSES = 3
CLASS_NAMES = ("setosa", "versicolor", "virginica")


def squared_distances(X, centroids):
    """(N X K) squared Euclidean distance of every example to every
    centroid (helper of kmeans)."""
    return ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)


def kmeans_plus_plus(X, K, rng):
    """k-means++ seeding, the default 'Start' of MATLAB's kmeans: the first
    centre is a random example, every next one is drawn with probability
    proportional to the squared distance to the nearest chosen centre."""
    N = X.shape[0]
    centroids = np.empty((K, X.shape[1]))
    centroids[0] = X[rng.integers(N)]
    for k in range(1, K):
        d2 = squared_distances(X, centroids[:k]).min(axis=1)
        total = d2.sum()
        prob = d2 / total if total > 0 else np.full(N, 1.0 / N)
        centroids[k] = X[rng.choice(N, p=prob)]
    return centroids


def kmeans(X, K, replicates, rng, max_iters=100):
    """k-means clustering with several restarts (counterpart of MATLAB's
    kmeans(X, K, 'Replicates', replicates)).

    Returns (labels, centroids, sumd): the 0-based cluster of every example,
    the (K X d) centroids and the total within-cluster squared distance of
    the best replicate."""
    best = None
    for _ in range(replicates):
        centroids = kmeans_plus_plus(X, K, rng)
        labels = np.full(X.shape[0], -1)
        for _ in range(max_iters):
            d2 = squared_distances(X, centroids)
            new_labels = d2.argmin(axis=1)
            # An empty cluster takes the example that lies furthest from
            # its centroid (MATLAB's 'EmptyAction', 'singleton')
            for k in range(K):
                if not (new_labels == k).any():
                    far = d2[np.arange(X.shape[0]), new_labels].argmax()
                    new_labels[far] = k
                    d2[far, k] = 0.0
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        sumd = ((X - centroids[labels]) ** 2).sum()
        if best is None or sumd < best[2]:
            best = (labels, centroids, sumd)
    return best


def gauss_log_pdf(X, mu, Sigma):
    """Log-density of a multivariate Gaussian at every row of X
    (gaussLogPdf.m). X is (N X d), mu (d,), Sigma (d X d); returns (N,).

    Working with logarithms avoids the underflow of the plain density, and
    the Cholesky factor avoids computing the inverse and the determinant."""
    d = X.shape[1]
    Xc = X - mu
    L = np.linalg.cholesky(Sigma)                 # Sigma = L L'
    z = solve_triangular(L, Xc.T, lower=True)     # sum(z^2) = Mahalanobis^2
    maha = (z ** 2).sum(axis=0)
    log_det = 2.0 * np.log(np.diag(L)).sum()
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det + maha)


def init_from_labels(X, labels, K, reg=1e-6):
    """Initial GMM from a hard clustering: every cluster becomes one
    component (the initialisation block of fitGMMs.m)."""
    N, d = X.shape
    Pm = np.zeros(K)
    M = np.zeros((K, d))
    S = np.zeros((K, d, d))
    for k in range(K):
        Xk = X[labels == k]
        Pm[k] = Xk.shape[0] / N
        M[k] = Xk.mean(axis=0)
        S[k] = np.cov(Xk, rowvar=False) + reg * np.eye(d)
    return Pm, M, S


def em(X, Pm, M, S, max_iters=200, tol=1e-6, reg=1e-6):
    """EM iterations of fitGMMs.m, starting from the model (Pm, M, S).

    Returns (Pm, M, S, log_lik, n_iters); log_lik is the data log-likelihood
    of the returned model."""
    N, d = X.shape
    K = M.shape[0]
    ridge = reg * np.eye(d)   # keeps every covariance invertible
    prev_log_lik = -np.inf
    log_lik = -np.inf
    for its in range(1, max_iters + 1):
        # ---- E-step: responsibilities G[n, k] = P(component k | x_n) ----
        log_joint = np.empty((N, K))
        for k in range(K):
            log_joint[:, k] = np.log(Pm[k]) + gauss_log_pdf(X, M[k], S[k])
        # log-sum-exp over the components, for numerical stability
        max_log = log_joint.max(axis=1, keepdims=True)
        log_px = max_log + np.log(np.exp(log_joint - max_log).sum(axis=1, keepdims=True))
        G = np.exp(log_joint - log_px)

        # ---- Convergence check on the data log-likelihood ----
        log_lik = log_px.sum()
        if abs(log_lik - prev_log_lik) < tol * abs(log_lik):
            break
        prev_log_lik = log_lik

        # ---- M-step: maximum-likelihood updates weighted by G ----
        Nk = G.sum(axis=0)                 # effective samples per component
        Pm = Nk / N
        M = (G.T @ X) / Nk[:, None]
        S = np.empty((K, d, d))
        for k in range(K):
            Xc = X - M[k]
            Sk = (Xc.T @ (Xc * G[:, [k]])) / Nk[k]
            S[k] = (Sk + Sk.T) / 2.0 + ridge   # keep it exactly symmetric
    return Pm, M, S, log_lik, its


def fit_gmms(X, K, rng, replicates=5, max_iters=200, tol=1e-6):
    """Fit a Gaussian Mixture Model with the EM algorithm (fitGMMs.m).

    Returns (Pm, M, S): the (K,) mixing probabilities, the (K X d) mean
    vectors and the (K X d X d) covariance matrices."""
    labels, _, _ = kmeans(X, K, replicates, rng)
    Pm, M, S = init_from_labels(X, labels, K)
    Pm, M, S, _, _ = em(X, Pm, M, S, max_iters=max_iters, tol=tol)
    return Pm, M, S


def classify_with_gmms(x, Pm, M, S):
    """Assign one example x (d,) to its most probable GMM component
    (classify_with_gmms.m). Returns the 0-based component index."""
    K = M.shape[0]
    log_post = np.empty(K)   # log(Pm[k] * N(x | M[k], S[k]))
    for k in range(K):
        log_post[k] = np.log(Pm[k]) + gauss_log_pdf(x[None, :], M[k], S[k])[0]
    return int(log_post.argmax())


def cluster_accuracy(clusters, labels, K):
    """Accuracy (%) of a clustering against the true labels
    (clusterAccuracy.m).

    Cluster numbers are arbitrary, so every one-to-one assignment of
    cluster numbers to class numbers is tried and the best one is kept.
    Returns (acc, mapped), mapped being the clusters renamed to classes."""
    acc = -1.0
    mapped = clusters
    # reversed: the same order as MATLAB's perms(1:K)
    for perm in reversed(list(permutations(range(K)))):
        candidate = np.array(perm)[clusters]
        a = 100.0 * np.mean(candidate == labels)
        if a > acc:
            acc = a
            mapped = candidate
    return acc, mapped


def plot_result(X, Y, labels, title):
    """Petal length against petal width, coloured by the assigned class,
    with the wrongly assigned examples circled (plot block of test_gmms.m)."""
    fig, ax = plt.subplots()
    for k, (name, colour) in enumerate(zip(CLASS_NAMES, "rgb")):
        members = Y == k
        ax.plot(X[members, 2], X[members, 3], ".", color=colour,
                markersize=15, label=name)
    wrong = Y != labels
    ax.plot(X[wrong, 2], X[wrong, 3], "ko", markersize=10,
            markerfacecolor="none", label="wrong class")
    ax.set_xlabel("petal length (cm)", fontsize=12, fontweight="bold")
    ax.set_ylabel("petal width (cm)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    return fig


def main(argv=None):
    """Run the script; argv defaults to the command line (sys.argv)."""
    parser = argparse.ArgumentParser(
        description="GMM fitted with EM on the Fisher iris data, versus k-means")
    parser.add_argument("--replicates", type=int, default=5,
                        help="number of k-means restarts")
    parser.add_argument("--max-iters", type=int, default=200,
                        help="upper bound on EM iterations")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="relative log-likelihood change that stops EM")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for the k-means restarts")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="save every figure as PNG in this folder")
    args = parser.parse_args(argv)
    if args.replicates < 1 or args.max_iters < 1:
        parser.error("--replicates and --max-iters must be at least 1")
    rng = np.random.default_rng(args.seed)

    # imported here so that the functions above work without scikit-learn
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    iris_labels = iris.target   # 0: setosa, 1: versicolor, 2: virginica
    K = K_CLASSES

    # ---- GMM fitted with EM ----
    Pm, M, S = fit_gmms(X, K, rng, replicates=args.replicates,
                        max_iters=args.max_iters, tol=args.tol)
    Y_hat = np.array([classify_with_gmms(x, Pm, M, S) for x in X])

    # ---- kmeans, for comparison ----
    c, _, _ = kmeans(X, K, args.replicates, rng)

    # ---- Accuracy ----
    # Cluster numbers are arbitrary, so they are matched to the classes first
    gmm_accuracy, Y_gmm = cluster_accuracy(Y_hat, iris_labels, K)
    kmeans_accuracy, Y_kmeans = cluster_accuracy(c, iris_labels, K)
    print(f"\n\tGMM accuracy = {gmm_accuracy:.2f}%.")
    print(f"\n\tkmeans accuracy = {kmeans_accuracy:.2f}%.")

    # ---- Plots: petal length vs petal width, coloured by assigned class ----
    figures = {
        "gmm.png": plot_result(
            X, Y_gmm, iris_labels,
            f"fitted GMM (accuracy {gmm_accuracy:.2f}%)"),
        "kmeans.png": plot_result(
            X, Y_kmeans, iris_labels,
            f"fitted kmeans (accuracy {kmeans_accuracy:.2f}%)"),
    }
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(args.save_dir / name, dpi=150)
        print(f"\n\tFigures saved in {args.save_dir}")

    plt.show()


if __name__ == "__main__":
    main()
