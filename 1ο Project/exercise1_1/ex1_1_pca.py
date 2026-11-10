"""
Principal Component Analysis (Python port of ex1_1_pca.m and its helpers).

Pipeline
--------
Part 1 - PCA on 2-D samples (ex1_1_data1.mat)
1. Load the 50 two-dimensional samples and plot them.
2. Standardize every feature (zero mean, unit standard deviation).
3. Compute the principal components: eigenvectors / eigenvalues of the
   covariance matrix Sigma = (1/m) * X' * X, sorted by descending eigenvalue,
   and draw them on top of the samples.
4. Print the share of the total variance explained by every component.
5. Project the samples on the first K = 1 component, map them back to 2-D
   and draw the lines that connect every sample with its approximation.

Part 2 - PCA on faces (ex1_1_faces.mat, 5000 images of 32x32 pixels)
6. Show the first 100 faces.
7. Standardize, run PCA and show the first 36 principal components
   (the "eigenfaces").
8. For K = 100, 10, 50, 200: project on the first K eigenfaces, recover the
   images and show them next to the originals.

Differences with respect to the MATLAB script
---------------------------------------------
* No pauses: every figure is created and all of them are shown at the end.
* Every plot gets its own figure. The MATLAB script draws the first 100
  faces and then the eigenfaces on figure 2, replacing the plot of the
  normalized 2-D samples.
* An eigenvector is only defined up to its sign, so U(:,1) and the
  projections Z may come out with the opposite sign than in MATLAB/Octave.
  The recovered data X_rec = Z * U' does not depend on the sign.
* The share of the variance that is lost for every K is also printed for
  the faces (the MATLAB script only prints the size of Z).

Usage
-----
    python ex1_1_pca.py [-k 1] [--face-k 100 10 50 200] [--save-dir DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

DATA_DIR = Path(__file__).parent
DATA_2D = DATA_DIR / "ex1_1_data1.mat"
DATA_FACES = DATA_DIR / "ex1_1_faces.mat"


def feature_normalize(X):
    """Standardize the columns of X (mirrors featureNormalize.m).

    Returns (X_norm, mu, sigma). sigma is the sample standard deviation
    (divisor m - 1), as MATLAB's std().
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    return (X - mu) / sigma, mu, sigma


def my_pca(X):
    """PCA by eigendecomposition of the covariance matrix (mirrors myPCA.m).

    X must have zero-mean columns. Returns (U, S): the eigenvectors in the
    columns of U and the eigenvalues, in descending order, on the diagonal
    of S.
    """
    m = X.shape[0]
    R = (X.T @ X) / m  # covariance matrix Sigma = (1/m) * X' * X
    eigenval, eigenvec = np.linalg.eigh(R)  # R is symmetric
    ind = np.argsort(eigenval)[::-1]
    return eigenvec[:, ind], np.diag(eigenval[ind])


def project_data(X, U, K):
    """Project the samples on the first K eigenvectors (mirrors
    projectData.m): z_i = U_K' * x_i for every row x_i of X."""
    return X @ U[:, :K]


def recover_data(Z, U, K):
    """Map the projections back to the original space (mirrors
    recoverData.m): x_rec_i = U_K * z_i for every row z_i of Z."""
    return Z @ U[:, :K].T


def draw_line(ax, p1, p2, *args, **kwargs):
    """Draw a line from point p1 to point p2 (mirrors drawLine.m)."""
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], *args, **kwargs)


def display_data(ax, X, example_width=None):
    """Show the rows of X as grayscale images in a grid (mirrors
    displayData.m). Every row is scaled by its largest absolute value.
    Returns the displayed array."""
    m, n = X.shape
    if example_width is None:
        example_width = int(round(np.sqrt(n)))
    example_height = n // example_width

    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))
    pad = 1  # padding between the images

    display_array = -np.ones((pad + display_rows * (example_height + pad),
                              pad + display_cols * (example_width + pad)))
    for curr_ex in range(m):
        j, i = divmod(curr_ex, display_cols)
        r0 = pad + j * (example_height + pad)
        c0 = pad + i * (example_width + pad)
        # MATLAB stores the pixels column by column (order="F")
        patch = X[curr_ex].reshape((example_height, example_width), order="F")
        display_array[r0:r0 + example_height, c0:c0 + example_width] = \
            patch / np.max(np.abs(X[curr_ex]))

    ax.imshow(display_array, cmap="gray", vmin=-1, vmax=1)
    ax.set_aspect("equal")
    ax.axis("off")
    return display_array


def finish_figures(save_dir, prefix):
    """Save every open figure as PNG in save_dir (if given) and show them."""
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for num in plt.get_fignums():
            plt.figure(num).savefig(save_dir / f"{prefix}_fig{num}.png", dpi=120)
        print(f"\nFigures saved in {save_dir}")
    plt.show()


def pca_2d(K):
    """Part 1: PCA on the 2-D samples."""
    print("Reading and Visualizing initial 2D dataset.\n")
    X = loadmat(DATA_2D)["X"]

    fig, ax = plt.subplots(num=1)
    ax.plot(X[:, 0], X[:, 1], "bo", fillstyle="none")
    ax.axis([0.5, 6.5, 2, 8])
    ax.set_aspect("equal")
    ax.set_title("Samples and principal components")

    X_norm, mu, sigma = feature_normalize(X)
    U, S = my_pca(X_norm)

    # Eigenvectors centered at the mean of the samples: the directions of
    # maximum variation of the dataset
    draw_line(ax, mu, mu + 1.5 * S[0, 0] * U[:, 0], "-k", linewidth=2)
    draw_line(ax, mu, mu + 1.5 * S[1, 1] * U[:, 1], "-k", linewidth=2)

    print("Top eigenvector: ")
    print(f" U(:,1) = {U[0, 0]:f} {U[1, 0]:f} ")
    print("\n(you should expect to see -0.707107 -0.707107, up to the sign)")

    # Variance contribution of each principal component
    pc_variance = np.diag(S) / np.sum(np.diag(S))
    print(" PCvariance = " + " ".join(f"{v:f}" for v in pc_variance) + " ")
    print("\n(you should expect to see 0.8678 and 0.1322)")

    print("\nDimension reduction of dataset samples.\n")
    fig, ax = plt.subplots(num=2)
    ax.plot(X_norm[:, 0], X_norm[:, 1], "bo", fillstyle="none")
    ax.axis([-4, 3, -4, 3])
    ax.set_aspect("equal")
    ax.set_title(f"Normalized samples and their projection (K = {K})")

    Z = project_data(X_norm, U, K)
    print(f"Projection of the first example: {Z[0, 0]:f}")
    print("\n(this value should be about 1.481274, up to the sign)\n")

    X_rec = recover_data(Z, U, K)
    print(f"Approximation of the first example: {X_rec[0, 0]:f} {X_rec[0, 1]:f}")
    print("\n(this value should be about  -1.047419 -1.047419)\n")

    # Lines connecting the projected points to the original points
    ax.plot(X_rec[:, 0], X_rec[:, 1], "ro", fillstyle="none")
    for x, x_rec in zip(X_norm, X_rec):
        draw_line(ax, x, x_rec, "--k", linewidth=1)


def pca_faces(face_ks):
    """Part 2: PCA on the face images."""
    print("\nLoading face dataset.\n")
    X = loadmat(DATA_FACES)["X"]

    fig, ax = plt.subplots(num=3)
    display_data(ax, X[:100])
    ax.set_title("First 100 faces")

    print("\nRunning PCA on face dataset.\n")
    X_norm, mu, sigma = feature_normalize(X)
    U, S = my_pca(X_norm)

    fig, ax = plt.subplots(num=4)
    display_data(ax, U[:, :36].T)
    ax.set_title("First 36 principal components (eigenfaces)")

    print("\nDimension reduction for face dataset.\n")
    eigenval = np.diag(S)
    for K in face_ks:
        Z = project_data(X_norm, U, K)
        print(f"K = {K}: the projected data Z has a size of: "
              f"{Z.shape[0]} {Z.shape[1]} ")

        X_rec = recover_data(Z, U, K)
        lost = 1.0 - eigenval[:K].sum() / eigenval.sum()
        print(f"        variance lost: {100 * lost:.2f}%")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.2))
        display_data(ax1, X_norm[:100])
        ax1.set_title("Original faces")
        display_data(ax2, X_rec[:100])
        ax2.set_title(f"Recovered faces (K = {K})")
        fig.tight_layout()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PCA on 2-D samples and on face images")
    parser.add_argument("-k", "--k", type=int, default=1, choices=(1, 2),
                        help="number of components kept for the 2-D samples "
                             "(default: 1)")
    parser.add_argument("--face-k", type=int, nargs="+",
                        default=[100, 10, 50, 200],
                        help="numbers of eigenfaces used for the "
                             "reconstructions (default: 100 10 50 200)")
    parser.add_argument("--save-dir", default=None, metavar="DIR",
                        help="save every figure as PNG in this folder")
    args = parser.parse_args(argv)
    if any(not 1 <= k <= 1024 for k in args.face_k):
        parser.error("--face-k values must be between 1 and 1024")

    pca_2d(args.k)
    pca_faces(args.face_k)
    finish_figures(args.save_dir, "ex1_1")


if __name__ == "__main__":
    main()
