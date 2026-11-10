"""
Linear Discriminant Analysis vs PCA (Python port of ex1_3_lda.m and its
helpers).

Pipeline
--------
Part 1 - two classes in 2-D (ex1_3_data1.mat)
1. Load the 14 samples X and their classes c, standardize the features and
   plot the two classes.
2. Fisher's linear discriminant: v = inv(Sw) * (mu1 - mu2), scaled to unit
   norm. Project every sample on v (1-D), map it back on the line of v and
   draw the lines that connect every sample with its projection.
3. Do the same with PCA (first principal component) in a second figure, to
   compare the two projections.

Part 2 - Fisher iris (150 samples, 4 features, 3 classes)
4. Standardize the features and plot the first two of them.
5. Multiclass LDA: eigenvectors of inv(Sw) * Sb (solved as the generalized
   problem Sb * v = lambda * Sw * v) for the NewDim = 2 largest eigenvalues.
6. Project the samples on the two LDA vectors and plot them.

Differences with respect to the MATLAB script
---------------------------------------------
* No pauses: every figure is created and all of them are shown at the end.
* The MATLAB script prints only progress messages. This script also prints
  the vectors it computes (Fisher direction, first principal component, LDA
  projection vectors) so that the two versions can be compared.
* Eigenvectors are only defined up to their sign, so the principal
  components and the LDA projection vectors (and the projections on them)
  may come out with the opposite sign than in MATLAB/Octave.
* Iris data: MATLAB has fisheriris.mat built in. Here the file
  fisheriris.mat is read if it exists next to this script; otherwise the
  same data set is taken from scikit-learn (the only use of scikit-learn).

Usage
-----
    python ex1_3_lda.py [-k 1] [--new-dim 2] [--save-dir DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.linalg import eigh

DATA_DIR = Path(__file__).parent
DATA_2D = DATA_DIR / "ex1_3_data1.mat"
DATA_IRIS = DATA_DIR / "fisheriris.mat"  # optional, see load_fisher_iris
SPECIES = ("setosa", "versicolor", "virginica")  # classes 0, 1, 2


def feature_normalize(X):
    """Standardize the columns of X (mirrors featureNormalize.m).

    Returns (X_norm, mu, sigma). sigma is the sample standard deviation
    (divisor m - 1), as MATLAB's std().
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    return (X - mu) / sigma, mu, sigma


def my_pca(X):
    """PCA by eigendecomposition of the covariance matrix (mirrors the
    myPCA.m of this exercise, which uses cov(X), i.e. divisor m - 1).

    Returns (U, S): the eigenvectors in the columns of U and the
    eigenvalues, in descending order, on the diagonal of S.
    """
    R = np.cov(X, rowvar=False)
    eigenval, eigenvec = np.linalg.eigh(R)  # R is symmetric
    ind = np.argsort(eigenval)[::-1]
    return eigenvec[:, ind], np.diag(eigenval[ind])


def fisher_linear_discriminant(X1, X2):
    """Fisher's linear discriminant for two classes (mirrors
    fisherLinearDiscriminant.m).

    X1, X2: the samples of each class, one per row. Returns the unit vector
    v of the direction that best separates the classes.
    """
    mu1 = X1.mean(axis=0)
    mu2 = X2.mean(axis=0)
    S1 = (X1 - mu1).T @ (X1 - mu1)  # scatter matrix of class 1
    S2 = (X2 - mu2).T @ (X2 - mu2)  # scatter matrix of class 2
    Sw = S1 + S2                    # within class scatter matrix
    v = np.linalg.solve(Sw, mu1 - mu2)  # inv(Sw) * (mu1 - mu2)
    return v / np.linalg.norm(v)


def my_lda(samples, labels, new_dim):
    """Multiclass LDA (mirrors myLDA.m).

    samples: one sample per row, labels: the class of every sample,
    new_dim: dimension of the reduced feature vector (<= classes - 1).
    Returns A (features x new_dim): one unit projection vector per column.
    """
    num_samples, num_features = samples.shape
    if num_samples != len(labels):
        raise ValueError("Number of Samples are not the same with the "
                         "Number of Labels.")
    classes = np.unique(labels)

    P = np.zeros(len(classes))                    # class prior probabilities
    mu = np.zeros((len(classes), num_features))   # class means
    Sw = np.zeros((num_features, num_features))   # within class scatter
    for i, cls in enumerate(classes):
        Xi = samples[labels == cls]
        P[i] = Xi.shape[0] / num_samples
        mu[i] = Xi.mean(axis=0)
        Sw += P[i] * np.cov(Xi, rowvar=False, bias=True)  # cov(Xi, 1)

    m0 = P @ mu  # global mean

    Sb = np.zeros((num_features, num_features))   # between class scatter
    for i in range(len(classes)):
        d = (mu[i] - m0)[:, None]
        Sb += P[i] * (d @ d.T)

    # Eigendecomposition of inv(Sw) * Sb, solved as the generalized
    # symmetric problem Sb * v = lambda * Sw * v
    eigenval, eigenvec = eigh(Sb, Sw)
    ind = np.argsort(eigenval)[::-1]
    A = eigenvec[:, ind[:new_dim]]

    # normalize every projection vector to unit length
    return A / np.linalg.norm(A, axis=0)


def project_data_lda(X, v):
    """Project the samples X (one per row) on the projection vectors v
    (mirrors projectDataLDA.m). v holds one vector per column; a single
    vector may also be given as a 1-D array. Returns Z (samples x NewDim)."""
    v = np.asarray(v)
    if v.ndim == 1:
        v = v[:, None]
    return X @ v


def recover_data_lda(Z, v):
    """Map the projections Z back to the original space, on the line (or
    subspace) spanned by the unit vectors v (mirrors recoverDataLDA.m)."""
    v = np.asarray(v)
    if v.ndim == 1:
        v = v[:, None]
    return Z @ v.T


def project_data_pca(X, U, K):
    """Project the samples on the first K eigenvectors (mirrors
    projectDataPCA.m)."""
    return X @ U[:, :K]


def recover_data_pca(Z, U, K):
    """Map the PCA projections back to the original space (mirrors
    recoverDataPCA.m)."""
    return Z @ U[:, :K].T


def draw_line(ax, p1, p2, *args, **kwargs):
    """Draw a line from point p1 to point p2 (mirrors drawLine.m)."""
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], *args, **kwargs)


def load_fisher_iris():
    """Fisher iris data (replaces MATLAB's load('fisheriris.mat')).

    Returns (meas, labels): the 150x4 measurements and the class of every
    sample (0: setosa, 1: versicolor, 2: virginica). Reads fisheriris.mat
    from the folder of this script if it is there, otherwise takes the data
    from scikit-learn.
    """
    if DATA_IRIS.exists():
        data = loadmat(DATA_IRIS)
        meas = np.asarray(data["meas"], dtype=float)
        species = [str(np.ravel(s)[0]) for s in np.ravel(data["species"])]
        labels = np.array([SPECIES.index(s) for s in species])
        return meas, labels
    try:
        from sklearn.datasets import load_iris
    except ImportError as exc:
        raise SystemExit(
            "The iris data were not found: put fisheriris.mat next to this "
            "script or install scikit-learn.") from exc
    iris = load_iris()
    assert tuple(iris.target_names) == SPECIES
    return iris.data.astype(float), iris.target.astype(int)


def finish_figures(save_dir, prefix):
    """Save every open figure as PNG in save_dir (if given) and show them."""
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for num in plt.get_fignums():
            plt.figure(num).savefig(save_dir / f"{prefix}_fig{num}.png", dpi=120)
        print(f"\nFigures saved in {save_dir}")
    plt.show()


def fmt(values):
    """Format a vector for printing."""
    return " ".join(f"{v: .6f}" for v in np.ravel(values))


def two_class_example(K):
    """Part 1: Fisher's linear discriminant vs PCA on the 2-D samples."""
    print("Visualizing example dataset for LDA.\n")
    data = loadmat(DATA_2D)
    X = data["X"].astype(float)
    c = data["c"].ravel()

    X_norm, mu, sigma = feature_normalize(X)
    X1 = X_norm[c == 1]
    X2 = X_norm[c == 2]

    # ---------------- 2-class Fisher linear discriminant ----------------
    fig, ax = plt.subplots(num=1)
    ax.plot(X1[:, 0], X1[:, 1], "bo", fillstyle="none")
    ax.plot(X2[:, 0], X2[:, 1], "rs", fillstyle="none")
    ax.axis([-2.1, 2.1, -2.1, 2.1])
    ax.set_aspect("equal")
    ax.set_title("Fisher linear discriminant")

    v = fisher_linear_discriminant(X1, X2)
    print(f"Fisher direction: v = {fmt(v)}")
    draw_line(ax, -5 * v, 5 * v, "-g", linewidth=1)

    # Project on the direction of v and map back on the line of v
    Z1 = project_data_lda(X1, v)
    Z2 = project_data_lda(X2, v)
    X1_rec = recover_data_lda(Z1, v)
    X2_rec = recover_data_lda(Z2, v)
    print(f"Projections of class 1: {fmt(Z1)}")
    print(f"Projections of class 2: {fmt(Z2)}")

    print("\nDisplaying LDA on example dataset.\n")
    ax.plot(X1_rec[:, 0], X1_rec[:, 1], "bo")
    for x, x_rec in zip(X1, X1_rec):
        draw_line(ax, x, x_rec, "--k", linewidth=1)
    ax.plot(X2_rec[:, 0], X2_rec[:, 1], "rs")
    for x, x_rec in zip(X2, X2_rec):
        draw_line(ax, x, x_rec, "--k", linewidth=1)

    # ---------------- Principal component analysis ----------------
    print("\nRunning PCA on example dataset.\n")
    U, S = my_pca(X_norm)
    Z = project_data_pca(X_norm, U, K)
    X_rec = recover_data_pca(Z, U, K)
    print(f"First principal component: U(:,1) = {fmt(U[:, 0])}")
    print(f"Eigenvalues: {fmt(np.diag(S))}")

    print("\nVisualizing example dataset for PCA.\n")
    fig, ax = plt.subplots(num=2)
    ax.axis([-2.1, 2.1, -2.1, 2.1])
    ax.set_aspect("equal")
    ax.set_title(f"PCA (K = {K})")
    draw_line(ax, -2.3 * U[:, 0], 2.3 * U[:, 0], "-g", linewidth=1)
    ax.plot(X1[:, 0], X1[:, 1], "bo", fillstyle="none")
    ax.plot(X2[:, 0], X2[:, 1], "rs", fillstyle="none")
    ax.plot(X_rec[c == 1, 0], X_rec[c == 1, 1], "bo")
    ax.plot(X_rec[c == 2, 0], X_rec[c == 2, 1], "rs")
    for x, x_rec in zip(X_norm, X_rec):
        draw_line(ax, x, x_rec, "--k", linewidth=1)


def iris_example(new_dim):
    """Part 2: multiclass LDA on the Fisher iris data."""
    meas, iris_labels = load_fisher_iris()
    meas_norm, mu, sigma = feature_normalize(meas)
    styles = ("bo", "rs", "g+")

    fig, ax = plt.subplots(num=3)
    for cls, style in enumerate(styles):
        Xc = meas_norm[iris_labels == cls]
        ax.plot(Xc[:, 0], Xc[:, 1], style, fillstyle="none",
                label=SPECIES[cls])
    ax.set_xlabel("sepal length (standardized)")
    ax.set_ylabel("sepal width (standardized)")
    ax.set_title("Iris: first two features")
    ax.legend()

    v = my_lda(meas_norm, iris_labels, new_dim)
    print("LDA projection vectors on the iris data (one per column):")
    for row in v:
        print("   " + fmt(row))

    meas_reduced = project_data_lda(meas_norm, v)
    print("Projection of the first sample of every class:")
    for cls in range(len(SPECIES)):
        first = meas_reduced[iris_labels == cls][0]
        print(f"   {SPECIES[cls]:<11s}{fmt(first)}")

    fig, ax = plt.subplots(num=4)
    for cls, style in enumerate(styles):
        Zc = meas_reduced[iris_labels == cls]
        # with new_dim = 1 the samples are drawn on a horizontal line
        y = Zc[:, 1] if new_dim > 1 else np.zeros(len(Zc))
        ax.plot(Zc[:, 0], y, style, fillstyle="none", label=SPECIES[cls])
    ax.set_xlabel("LDA 1")
    ax.set_ylabel("LDA 2")
    ax.set_title(f"Iris after LDA (NewDim = {new_dim})")
    ax.legend()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fisher LDA vs PCA on 2-D samples, multiclass LDA on iris")
    parser.add_argument("-k", "--k", type=int, default=1, choices=(1, 2),
                        help="number of principal components kept in the "
                             "2-D example (default: 1)")
    parser.add_argument("--new-dim", type=int, default=2, choices=(1, 2),
                        help="feature dimension after LDA on the iris data "
                             "(default: 2, at most classes - 1 = 2)")
    parser.add_argument("--save-dir", default=None, metavar="DIR",
                        help="save every figure as PNG in this folder")
    args = parser.parse_args(argv)

    two_class_example(args.k)
    iris_example(args.new_dim)
    finish_figures(args.save_dir, "ex1_3")


if __name__ == "__main__":
    main()
