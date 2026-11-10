"""Tests of the Python versions of the 1st set of exercises
(1.1 PCA, 1.3 LDA, 1.4 MNIST aspect ratio and Bayes rule, 1.5 Bayes
decision boundaries).

The reference numbers were checked against the MATLAB code running in GNU
Octave 8.4. Eigenvectors are compared up to their sign.
"""

import numpy as np
import pytest
from scipy.io import loadmat

from conftest import SET1, load_exercise

pca = load_exercise(SET1 / "exercise1_1" / "ex1_1_pca.py")
lda = load_exercise(SET1 / "exercise1_3" / "ex1_3_lda.py")
ex4 = load_exercise(SET1 / "exercise1_4" / "exercise1_4.py")
ex5 = load_exercise(SET1 / "exercise1_5" / "exercise1_5.py")

PRIORS = [0.1, 0.25, 0.5, 0.75, 0.9]


def same_up_to_sign(a, b, tol=1e-6):
    """True if the vectors a and b are equal or opposite."""
    a, b = np.ravel(a), np.ravel(b)
    return np.allclose(a, b, atol=tol) or np.allclose(a, -b, atol=tol)


def close_figures():
    """Close the figures a main() left open."""
    import matplotlib.pyplot as plt
    plt.close("all")


# ------------------------------------------------------------------
# Exercise 1.1 - PCA
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def pca_2d():
    """Normalized 2-D samples and their principal components."""
    X = loadmat(pca.DATA_2D)["X"]
    X_norm, mu, sigma = pca.feature_normalize(X)
    U, S = pca.my_pca(X_norm)
    return X, X_norm, mu, sigma, U, S


@pytest.fixture(scope="module")
def pca_faces():
    """Normalized faces and their principal components."""
    X = loadmat(pca.DATA_FACES)["X"]
    X_norm, _, _ = pca.feature_normalize(X)
    U, S = pca.my_pca(X_norm)
    return X_norm, U, S


def test_feature_normalize_gives_zero_mean_and_unit_std(pca_2d):
    """Every normalized feature has mean 0 and sample std 1."""
    X, X_norm, mu, sigma, _, _ = pca_2d
    assert X_norm.mean(axis=0) == pytest.approx([0, 0], abs=1e-12)
    assert X_norm.std(axis=0, ddof=1) == pytest.approx([1, 1])
    assert mu == pytest.approx(X.mean(axis=0))
    assert sigma == pytest.approx(X.std(axis=0, ddof=1))
    assert X_norm * sigma + mu == pytest.approx(X)


def test_pca_2d_top_eigenvector(pca_2d):
    """The first principal component is +-[0.707107 0.707107]."""
    U = pca_2d[4]
    assert same_up_to_sign(U[:, 0], [0.707107, 0.707107])


def test_pca_2d_variance_fractions(pca_2d):
    """The two components explain 86.78% and 13.22% of the variance."""
    S = pca_2d[5]
    fractions = np.diag(S) / np.trace(S)
    assert fractions == pytest.approx([0.867765, 0.132235], abs=1e-6)


def test_pca_eigenvectors_orthonormal_and_eigenvalues_sorted(pca_2d):
    """U is orthonormal, S is diagonal with descending eigenvalues."""
    X_norm, U, S = pca_2d[1], pca_2d[4], pca_2d[5]
    assert U.T @ U == pytest.approx(np.eye(2), abs=1e-12)
    eigenval = np.diag(S)
    assert np.all(np.diff(eigenval) <= 0)
    assert S == pytest.approx(np.diag(eigenval))
    # U and S diagonalize the covariance matrix (1/m) X'X
    R = X_norm.T @ X_norm / X_norm.shape[0]
    assert R @ U == pytest.approx(U @ S, abs=1e-12)


def test_project_data_first_example(pca_2d):
    """The first sample projects to +-1.481274 with K = 1."""
    X_norm, U = pca_2d[1], pca_2d[4]
    Z = pca.project_data(X_norm, U, 1)
    assert Z.shape == (50, 1)
    assert abs(Z[0, 0]) == pytest.approx(1.481274, abs=1e-6)


def test_recover_data_first_example(pca_2d):
    """The first sample is approximated by (-1.047419, -1.047419)."""
    X_norm, U = pca_2d[1], pca_2d[4]
    X_rec = pca.recover_data(pca.project_data(X_norm, U, 1), U, 1)
    assert X_rec.shape == X_norm.shape
    assert X_rec[0] == pytest.approx([-1.047419, -1.047419], abs=1e-6)


def test_recover_of_full_projection_is_identity(pca_2d):
    """With K equal to the number of features nothing is lost."""
    X_norm, U = pca_2d[1], pca_2d[4]
    X_rec = pca.recover_data(pca.project_data(X_norm, U, 2), U, 2)
    assert X_rec == pytest.approx(X_norm, abs=1e-12)


def test_display_data_grid_layout():
    """Four 2x2 images give a 7x7 grid with scaled, column-major patches."""
    import matplotlib.pyplot as plt
    X = np.array([[1.0, 2, 3, 4], [4, 4, 4, 4], [-2, 0, 0, 0], [0, 0, 0, 8]])
    fig, ax = plt.subplots()
    grid = pca.display_data(ax, X)
    plt.close(fig)
    assert grid.shape == (7, 7)
    assert grid[0] == pytest.approx(-np.ones(7))            # padding
    assert grid[1:3, 1:3] == pytest.approx(np.array([[0.25, 0.75], [0.5, 1.0]]))
    assert grid[4:6, 1:3] == pytest.approx(np.array([[-1.0, 0], [0, 0]]))


@pytest.mark.slow
def test_faces_reconstruction_error_values(pca_faces):
    """The relative reconstruction error of the faces for every K."""
    X_norm, U, _ = pca_faces
    expected = {10: 0.327276, 50: 0.132095, 100: 0.068068, 200: 0.026928}
    total = np.linalg.norm(X_norm) ** 2
    errors = []
    for K, value in expected.items():
        X_rec = pca.recover_data(pca.project_data(X_norm, U, K), U, K)
        errors.append(np.linalg.norm(X_norm - X_rec) ** 2 / total)
        assert errors[-1] == pytest.approx(value, abs=1e-6)
    assert np.all(np.diff(errors) < 0)  # decreases as K grows


@pytest.mark.slow
def test_faces_eigenvalues(pca_faces):
    """Largest eigenvalues of the faces and lost variance from them."""
    S = pca_faces[2]
    eigenval = np.diag(S)
    assert eigenval[:3] == pytest.approx([336.9576, 130.9126, 54.6826],
                                         abs=1e-3)
    assert eigenval.sum() == pytest.approx(1023.7952, abs=1e-3)
    assert 1 - eigenval[:100].sum() / eigenval.sum() == pytest.approx(
        0.068068, abs=1e-6)


@pytest.mark.slow
def test_ex1_1_main_runs_and_saves_figures(tmp_path):
    """The whole script runs headless and saves its 8 figures."""
    pca.main(["--save-dir", str(tmp_path)])
    close_figures()
    assert len(list(tmp_path.glob("*.png"))) == 8


# ------------------------------------------------------------------
# Exercise 1.3 - LDA
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_classes():
    """Normalized samples of the two classes of the 2-D example."""
    data = loadmat(lda.DATA_2D)
    X_norm, _, _ = lda.feature_normalize(data["X"].astype(float))
    c = data["c"].ravel()
    return X_norm[c == 1], X_norm[c == 2]


@pytest.fixture(scope="module")
def iris():
    """Normalized iris measurements and their labels (0, 1, 2)."""
    pytest.importorskip("sklearn")
    meas, labels = lda.load_fisher_iris()
    meas_norm, _, _ = lda.feature_normalize(meas)
    return meas_norm, labels


def fisher_criterion(v, X1, X2):
    """J(v) = (difference of projected means)^2 / projected scatter."""
    z1, z2 = X1 @ v, X2 @ v
    scatter = np.sum((z1 - z1.mean()) ** 2) + np.sum((z2 - z2.mean()) ** 2)
    return (z1.mean() - z2.mean()) ** 2 / scatter


def scatter_matrices(samples, labels):
    """Within and between class scatter matrices, as defined in myLDA."""
    m0 = np.zeros(samples.shape[1])
    Sw = np.zeros((samples.shape[1],) * 2)
    stats = []
    for cls in np.unique(labels):
        Xi = samples[labels == cls]
        p = len(Xi) / len(samples)
        Sw += p * np.cov(Xi, rowvar=False, bias=True)
        m0 += p * Xi.mean(axis=0)
        stats.append((p, Xi.mean(axis=0)))
    Sb = sum(p * np.outer(mu - m0, mu - m0) for p, mu in stats)
    return Sw, Sb


def test_fisher_direction_reference_value(two_classes):
    """The Fisher direction is the unit vector +-[-0.560202 0.828356]."""
    v = lda.fisher_linear_discriminant(*two_classes)
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert same_up_to_sign(v, [-0.560202, 0.828356])


def test_fisher_direction_maximizes_fisher_criterion(two_classes):
    """No random direction has a larger Fisher criterion than v."""
    X1, X2 = two_classes
    best = fisher_criterion(lda.fisher_linear_discriminant(X1, X2), X1, X2)
    angles = np.random.default_rng(0).uniform(0, np.pi, 500)
    for a in angles:
        w = np.array([np.cos(a), np.sin(a)])
        assert fisher_criterion(w, X1, X2) <= best + 1e-12


def test_fisher_projection_separates_the_classes(two_classes):
    """After the projection on v the two classes do not overlap."""
    X1, X2 = two_classes
    v = lda.fisher_linear_discriminant(X1, X2)
    Z1 = lda.project_data_lda(X1, v)
    Z2 = lda.project_data_lda(X2, v)
    assert Z1.shape == (7, 1) and Z2.shape == (7, 1)
    assert Z1.min() > Z2.max() or Z2.min() > Z1.max()
    assert np.sort(np.abs(Z1.ravel()))[[0, -1]] == pytest.approx(
        [0.082624, 1.492271], abs=1e-6)


def test_recover_data_lda_lies_on_the_line_of_v(two_classes):
    """Recovered samples are the orthogonal projections on the line of v."""
    X1, _ = two_classes
    v = lda.fisher_linear_discriminant(*two_classes)
    X_rec = lda.recover_data_lda(lda.project_data_lda(X1, v), v)
    assert X_rec.shape == X1.shape
    assert X_rec[:, 0] * v[1] - X_rec[:, 1] * v[0] == pytest.approx(
        np.zeros(7), abs=1e-12)                 # parallel to v
    assert (X1 - X_rec) @ v == pytest.approx(np.zeros(7), abs=1e-12)


def test_pca_of_exercise_1_3(two_classes):
    """First principal component +-[0.7071 0.7071], eigenvalues by cov()."""
    X_norm = np.vstack(two_classes)
    U, S = lda.my_pca(X_norm)
    assert same_up_to_sign(U[:, 0], [0.707107, 0.707107])
    assert np.diag(S) == pytest.approx([1.465032, 0.534968], abs=1e-6)
    Z = lda.project_data_pca(X_norm, U, 1)
    X_rec = lda.recover_data_pca(Z, U, 1)
    assert X_rec[:, 0] == pytest.approx(X_rec[:, 1])    # on the line y = x
    assert lda.recover_data_pca(lda.project_data_pca(X_norm, U, 2), U, 2) \
        == pytest.approx(X_norm, abs=1e-12)


def test_my_lda_iris_reference_vectors(iris):
    """The LDA projection vectors of the iris data, up to their sign."""
    A = lda.my_lda(*iris, 2)
    assert same_up_to_sign(A[:, 0], [0.151288, 0.147333, -0.855985, -0.471905])
    assert same_up_to_sign(A[:, 1], [0.006936, 0.327861, -0.571705, 0.752072])


@pytest.mark.parametrize("new_dim", [1, 2])
def test_my_lda_honours_new_dim(iris, new_dim):
    """A has NewDim unit columns and the projection has NewDim features."""
    meas_norm, labels = iris
    A = lda.my_lda(meas_norm, labels, new_dim)
    assert A.shape == (4, new_dim)
    assert np.linalg.norm(A, axis=0) == pytest.approx(np.ones(new_dim))
    assert lda.project_data_lda(meas_norm, A).shape == (150, new_dim)


def test_my_lda_does_not_depend_on_label_values(iris):
    """Labels 0..2, 1..3 or any other distinct values give the same A."""
    meas_norm, labels = iris
    A = lda.my_lda(meas_norm, labels, 2)
    assert lda.my_lda(meas_norm, labels + 1, 2) == pytest.approx(A)
    assert lda.my_lda(meas_norm, np.array([7, -3, 40])[labels], 2) \
        == pytest.approx(A)


def test_my_lda_vectors_are_eigenvectors_of_inv_sw_sb(iris):
    """inv(Sw)*Sb*a = lambda*a for the two largest eigenvalues."""
    meas_norm, labels = iris
    A = lda.my_lda(meas_norm, labels, 2)
    Sw, Sb = scatter_matrices(meas_norm, labels)
    M = np.linalg.solve(Sw, Sb)
    top = np.sort(np.linalg.eigvals(M).real)[::-1][:2]
    for j in range(2):
        a = A[:, j]
        lam = a @ M @ a  # a has unit norm
        assert M @ a == pytest.approx(lam * a, abs=1e-8)
        assert lam == pytest.approx(top[j])


def test_my_lda_rejects_wrong_number_of_labels(iris):
    """Samples and labels of different length raise an error."""
    meas_norm, labels = iris
    with pytest.raises(ValueError):
        lda.my_lda(meas_norm, labels[:-1], 2)


def test_iris_classes_are_separated_on_first_lda_axis(iris):
    """Setosa is completely apart from the other two classes on LDA 1."""
    meas_norm, labels = iris
    z = lda.project_data_lda(meas_norm, lda.my_lda(meas_norm, labels, 2))[:, 0]
    setosa, rest = z[labels == 0], z[labels != 0]
    assert setosa.min() > rest.max() or setosa.max() < rest.min()


def test_ex1_3_main_runs_and_saves_figures(tmp_path):
    """The whole script runs headless and saves its 4 figures."""
    pytest.importorskip("sklearn")
    lda.main(["--save-dir", str(tmp_path)])
    close_figures()
    assert len(list(tmp_path.glob("*.png"))) == 4


# ------------------------------------------------------------------
# Exercise 1.4 - aspect ratio and Bayes rule
# ------------------------------------------------------------------

def image_with(rows, cols, shape=(28, 28)):
    """Blank image with ink in the rectangle rows x cols (slices)."""
    image = np.zeros(shape)
    image[rows, cols] = 255
    return image


def test_aspect_ratio_of_single_pixel_is_one():
    """A single pixel is a 1x1 rectangle."""
    assert ex4.compute_aspect_ratio(image_with(5, 9)) == pytest.approx(1.0)


def test_aspect_ratio_of_horizontal_bar():
    """A bar 1 pixel high and 3 wide has aspect ratio 3."""
    image = image_with(slice(4, 5), slice(10, 13))
    assert ex4.compute_aspect_ratio(image) == pytest.approx(3.0)


def test_aspect_ratio_of_vertical_bar():
    """A bar 3 pixels high and 1 wide has aspect ratio 1/3."""
    image = image_with(slice(10, 13), slice(4, 5))
    assert ex4.compute_aspect_ratio(image) == pytest.approx(1 / 3)


def test_aspect_ratio_uses_the_enclosing_rectangle():
    """Two far pixels define the rectangle: width 10, height 20."""
    image = np.zeros((28, 28), dtype=np.uint8)
    image[2, 5] = 1
    image[21, 14] = 200
    assert ex4.compute_aspect_ratio(image) == pytest.approx(0.5)


def test_aspect_ratio_of_blank_image_is_nan():
    """A blank image has no enclosing rectangle."""
    assert np.isnan(ex4.compute_aspect_ratio(np.zeros((28, 28))))


@pytest.mark.parametrize("value, expected", [
    (0.0, 0),      # the minimum falls in the first bin
    (0.5, 0),
    (1.0, 1),      # an interior edge belongs to the bin on its right
    (1.5, 1),
    (2.0, 2),
    (2.999, 2),
    (3.0, 2),      # the maximum falls in the last bin
    (-1.0, 0),     # values outside the range go to the nearest bin
    (9.0, 2),
])
def test_find_bin_edge_cases(value, expected):
    """Bins [0,1), [1,2), [2,3]; index 0-based (MATLAB findBin minus 1)."""
    assert ex4.find_bin(value, np.array([0.0, 1.0, 2.0, 3.0])) == expected


@pytest.fixture(scope="module")
def mnist_stats():
    """Aspect ratios of MNIST and the bin counts of the digits 1 and 2,
    computed with the helpers of the script as its main() does."""
    data = loadmat(ex4.DATA_FILE)
    images = data["trainX"].reshape(-1, 28, 28)
    labels = data["trainY"].ravel().astype(int)
    ratios = np.array([ex4.compute_aspect_ratio(im) for im in images])
    both = (labels == 1) | (labels == 2)
    bin_ends = np.linspace(ratios[both].min(), ratios[both].max(), 4)
    counts = np.zeros((2, 3), dtype=int)
    for row, digit in enumerate((1, 2)):
        for ratio in ratios[labels == digit]:
            counts[row, ex4.find_bin(ratio, bin_ends)] += 1
    return ratios, bin_ends, counts


def test_mnist_aspect_ratio_range(mnist_stats):
    """Minimum 0.1 and maximum 2.2222, with no blank image."""
    ratios, bin_ends, _ = mnist_stats
    assert not np.any(np.isnan(ratios))
    assert ratios.min() == pytest.approx(0.1)
    assert ratios.max() == pytest.approx(20 / 9)
    assert bin_ends == pytest.approx([0.1, 0.807407, 1.514815, 2.222222],
                                     abs=1e-6)


def test_mnist_bin_counts(mnist_stats):
    """Histogram counts (Low, Medium, High) of the digits 1 and 2."""
    counts = mnist_stats[2]
    assert counts[0].tolist() == [6393, 349, 0]
    assert counts[1].tolist() == [1473, 4402, 83]


def test_mnist_probabilities(mnist_stats):
    """Priors, likelihoods, evidence and posteriors from the counts."""
    counts = mnist_stats[2]
    n = counts.sum(axis=1)
    priors = n / n.sum()
    likelihoods = counts / n[:, None]
    evidence = counts.sum(axis=0) / n.sum()
    posteriors = likelihoods[:, 0] * priors / evidence[0]

    assert priors == pytest.approx([0.5309, 0.4691], abs=5e-5)
    assert likelihoods.sum(axis=1) == pytest.approx([1, 1])
    assert likelihoods[0] == pytest.approx([0.9482, 0.0518, 0.0], abs=5e-5)
    assert likelihoods[1] == pytest.approx([0.2472, 0.7388, 0.0139], abs=5e-5)
    assert evidence.sum() == pytest.approx(1)
    assert evidence == pytest.approx([0.6194, 0.3741, 0.0065], abs=5e-5)
    assert posteriors == pytest.approx([0.8127, 0.1873], abs=5e-5)
    assert posteriors.sum() == pytest.approx(1)
    # total probability: the evidence is the prior-weighted likelihood
    assert priors @ likelihoods == pytest.approx(evidence)


def test_ex1_4_main_prints_reference_values(tmp_path, capsys):
    """The whole script runs headless, prints the results, saves 3 figures."""
    ex4.main(["--save-dir", str(tmp_path)])
    close_figures()
    out = capsys.readouterr().out
    assert "min aspect ratio = 0.1000, max aspect ratio = 2.2222" in out
    assert "C1: L = 6393, M = 349, H = 0" in out
    assert "C2: L = 1473, M = 4402, H = 83" in out
    assert "P(C1) = 0.5309, P(C2) = 0.4691" in out
    assert "P(C1|X=L) = 0.8127, P(C2|X=L) = 0.1873" in out
    assert len(list(tmp_path.glob("*.png"))) == 3


def test_ex1_4_main_rejects_equal_digits():
    """The two classes must be different digits."""
    with pytest.raises(SystemExit):
        ex4.main(["--c1", "3", "--c2", "3"])


# ------------------------------------------------------------------
# Exercise 1.5 - Bayes decision boundaries
# ------------------------------------------------------------------

@pytest.mark.parametrize("mu, sigma", [(ex5.M1, ex5.S1), (ex5.M2, ex5.S2)])
def test_mvn_pdf_matches_scipy(mu, sigma):
    """mvn_pdf equals scipy.stats.multivariate_normal.pdf."""
    from scipy.stats import multivariate_normal
    pts = np.random.default_rng(1).uniform(-3, 12, size=(200, 2))
    assert ex5.mvn_pdf(pts, mu, sigma) == pytest.approx(
        multivariate_normal(mu, sigma).pdf(pts), rel=1e-10, abs=1e-300)


def test_mvn_pdf_peak_value():
    """The peak of both densities is 1/(2*pi*sqrt(1.28)) = 0.140674."""
    assert ex5.mvn_pdf(ex5.M1[None, :], ex5.M1, ex5.S1)[0] == pytest.approx(
        0.140674, abs=1e-6)
    assert ex5.mvn_pdf(ex5.M2[None, :], ex5.M2, ex5.S2)[0] == pytest.approx(
        1 / (2 * np.pi * np.sqrt(1.28)))


def test_discriminant_is_log_of_prior_times_density():
    """g(x) = ln(P * p(x)) + ln(2*pi), so g1 - g2 is the log posterior ratio."""
    pts = np.random.default_rng(2).uniform(0, 9, size=(50, 2))
    g = ex5.discriminant(pts, ex5.M1, ex5.S1, 0.3)
    expected = np.log(0.3 * ex5.mvn_pdf(pts, ex5.M1, ex5.S1)) + np.log(2 * np.pi)
    assert g == pytest.approx(expected)


@pytest.mark.parametrize("pw1", PRIORS)
def test_hyperbola_satisfies_g1_equals_g2(pw1):
    """S1 != S2: every point of the hyperbola has g1(x) = g2(x)."""
    x = np.linspace(-3, 12, 301)        # includes x = 0, where y is NaN
    y = ex5.boundary_unequal(x, pw1)
    assert np.isnan(y[np.abs(x) < 1e-9]).all()
    ok = np.isfinite(y)
    assert ok.sum() == len(x) - 1
    assert 1.25 * x[ok] * y[ok] == pytest.approx(
        22.5 - 2 * np.log((1 - pw1) / pw1))
    pts = np.column_stack([x[ok], y[ok]])
    g1 = ex5.discriminant(pts, ex5.M1, ex5.S1, pw1)
    g2 = ex5.discriminant(pts, ex5.M2, ex5.S2, 1 - pw1)
    assert g1 - g2 == pytest.approx(np.zeros(ok.sum()), abs=1e-7)
    assert ex5.boundary_residual(x, y, ex5.S1, ex5.S2, pw1) < 1e-7


@pytest.mark.parametrize("pw1", PRIORS)
def test_line_satisfies_g1_equals_g2(pw1):
    """S1 == S2: every point of the line has g1(x) = g2(x)."""
    x = np.linspace(-3, 12, 301)
    y = ex5.boundary_equal(x, pw1)
    assert 3.75 * (x + y) == pytest.approx(
        33.75 - 2 * np.log((1 - pw1) / pw1))
    pts = np.column_stack([x, y])
    g1 = ex5.discriminant(pts, ex5.M1, ex5.S2, pw1)
    g2 = ex5.discriminant(pts, ex5.M2, ex5.S2, 1 - pw1)
    assert g1 - g2 == pytest.approx(np.zeros(len(x)), abs=1e-10)


def test_points_off_the_boundary_are_classified_by_side():
    """Below the line class 1 wins, above it class 2 (equal covariances)."""
    x = np.linspace(0, 9, 10)
    y = ex5.boundary_equal(x, 0.5)
    for shift, sign in ((-0.5, 1), (0.5, -1)):
        pts = np.column_stack([x, y + shift])
        diff = (ex5.discriminant(pts, ex5.M1, ex5.S2, 0.5)
                - ex5.discriminant(pts, ex5.M2, ex5.S2, 0.5))
        assert np.all(sign * diff > 0)


def test_equal_priors_equal_covariances_boundary_through_midpoint():
    """With P(w1) = 0.5 and S1 = S2 the line passes through (m1 + m2)/2."""
    mid = (ex5.M1 + ex5.M2) / 2
    assert ex5.boundary_equal(np.array([mid[0]]), 0.5)[0] == pytest.approx(
        mid[1])


def test_larger_prior_moves_the_boundaries_towards_m2():
    """As P(w1) grows both boundaries move away from m1."""
    x = np.array([4.5])
    line = [ex5.boundary_equal(x, p)[0] for p in PRIORS]
    hyperbola = [ex5.boundary_unequal(x, p)[0] for p in PRIORS]
    assert np.all(np.diff(line) > 0)
    assert np.all(np.diff(hyperbola) > 0)


def test_ex1_5_main_runs_and_saves_figures(tmp_path, capsys):
    """The whole script runs headless (coarse grid) and saves 2 figures."""
    ex5.main(["--dt", "0.05", "--save-dir", str(tmp_path)])
    close_figures()
    out = capsys.readouterr().out
    assert "1.25*x*y = 18.1056" in out
    assert "3.75*(x + y) = 38.1444" in out
    assert len(list(tmp_path.glob("*.png"))) == 2
