"""Tests of the Python versions of exercises 3.2 (k-means and image
compression), 3.3 (Gaussian mixture model with EM) and 4.1 (feature
selection inside and outside the cross validation).

The reference numbers were obtained by running the MATLAB code in GNU
Octave; the remaining checks are mathematical properties.
"""

from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.io import loadmat
from scipy.stats import multivariate_normal

from conftest import SET3, SET4, load_exercise

KMEANS_SCRIPT = SET3 / "exercise3_2" / "ex3_kmeans.py"
GMM_SCRIPT = SET3 / "exercise3_3" / "test_gmms.py"
OVERFITTING_SCRIPT = SET4 / "exercise4_1" / "overfitting_test.py"

INITIAL_CENTROIDS = np.array([[3.0, 3.0], [6.0, 2.0], [8.0, 5.0]])


@pytest.fixture(scope="module")
def km():
    """The k-means script of exercise 3.2."""
    return load_exercise(KMEANS_SCRIPT)


@pytest.fixture(scope="module")
def gmm():
    """The GMM script of exercise 3.3."""
    return load_exercise(GMM_SCRIPT)


@pytest.fixture(scope="module")
def ovf():
    """The overfitting script of exercise 4.1 (needs scikit-learn)."""
    pytest.importorskip("sklearn")
    return load_exercise(OVERFITTING_SCRIPT)


@pytest.fixture(scope="module")
def data1():
    """The 300 two-dimensional examples of ex3data1.mat."""
    return loadmat(SET3 / "exercise3_2" / "ex3data1.mat")["X"]


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure a test has opened."""
    yield
    plt.close("all")


def kmeans_cost(X, centroids):
    """Sum of squared distances of every example to its closest centroid."""
    d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return d2.min(axis=1).sum()


def pngs_in(folder):
    """The non-empty PNG files in a folder."""
    return [p for p in folder.glob("*.png") if p.stat().st_size > 0]


# --------------------------------------------------------------------------
# Exercise 3.2: k-means and image compression
# --------------------------------------------------------------------------

def test_find_closest_centroids_first_three_examples(km, data1):
    """The first three examples go to centroids 1, 3, 2 (0-based: 0, 2, 1)."""
    idx = km.find_closest_centroids(data1, INITIAL_CENTROIDS)
    assert idx.shape == (300,)
    assert list(idx[:3]) == [0, 2, 1]
    assert list(idx[:3] + 1) == [1, 3, 2]   # what the script prints


def test_find_closest_centroids_matches_brute_force(km):
    """The vectorised assignment equals a plain double loop on random data."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 3))
    centroids = rng.standard_normal((5, 3))
    expected = []
    for x in X:
        dists = [sum((x[j] - c[j]) ** 2 for j in range(3)) for c in centroids]
        expected.append(dists.index(min(dists)))
    assert list(km.find_closest_centroids(X, centroids)) == expected


def test_find_closest_centroids_tie_goes_to_lowest_index(km):
    """A point at equal distance from two centroids takes the first one."""
    idx = km.find_closest_centroids(np.array([[0.0, 0.0]]),
                                    np.array([[1.0, 0.0], [-1.0, 0.0]]))
    assert idx[0] == 0


def test_compute_centroids_reference_values(km, data1):
    """One update from the initial centroids gives the reference centroids."""
    idx = km.find_closest_centroids(data1, INITIAL_CENTROIDS)
    centroids = km.compute_centroids(data1, idx, 3, np.random.default_rng(0))
    expected = [[2.428301, 3.157924], [5.813503, 2.633656], [7.119387, 3.616684]]
    assert centroids == pytest.approx(np.array(expected), abs=1e-6)


def test_compute_centroids_empty_cluster_has_no_nan(km):
    """A cluster without examples is re-seeded with an example, not NaN."""
    X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0]])
    idx = np.array([0, 0, 2])   # cluster 1 is empty
    centroids = km.compute_centroids(X, idx, 3, np.random.default_rng(0))
    assert np.isfinite(centroids).all()
    assert centroids[0] == pytest.approx([1.0, 0.0])
    assert centroids[2] == pytest.approx([10.0, 10.0])
    assert any(np.array_equal(centroids[1], x) for x in X)


def test_kmeans_init_centroids_are_distinct_examples(km):
    """The initial centroids are K different rows of the data."""
    X = np.arange(40, dtype=float).reshape(20, 2)
    centroids = km.kmeans_init_centroids(X, 6, np.random.default_rng(3))
    assert centroids.shape == (6, 2)
    rows = {tuple(c) for c in centroids}
    assert len(rows) == 6
    assert rows <= {tuple(x) for x in X}


def test_kmeans_cost_is_non_increasing(km, data1):
    """The k-means cost never goes up from one iteration to the next."""
    rng = np.random.default_rng(0)
    costs = [kmeans_cost(data1, INITIAL_CENTROIDS)]
    for n_iters in range(1, 11):
        centroids, _, _ = km.run_kmeans(data1, INITIAL_CENTROIDS, n_iters, rng)
        costs.append(kmeans_cost(data1, centroids))
    assert all(b <= a + 1e-9 for a, b in zip(costs, costs[1:]))
    assert costs[-1] < costs[0]


def test_run_kmeans_ten_iterations_reference_centroids(km, data1):
    """Ten iterations reach the centroids and cluster sizes of the MATLAB run."""
    centroids, idx, fig = km.run_kmeans(data1, INITIAL_CENTROIDS, 10,
                                        np.random.default_rng(0))
    expected = [[1.9539946649, 5.0255700594],
                [3.0436711927, 1.0154104079],
                [6.0336673560, 3.0005251118]]
    assert centroids == pytest.approx(np.array(expected), abs=1e-9)
    assert list(np.bincount(idx)) == [98, 102, 100]
    assert fig is None


def test_run_kmeans_does_not_modify_initial_centroids(km, data1):
    """The array of initial centroids is left untouched."""
    initial = INITIAL_CENTROIDS.copy()
    km.run_kmeans(data1, initial, 3, np.random.default_rng(0))
    assert np.array_equal(initial, INITIAL_CENTROIDS)


def test_compression_gives_at_most_k_colours(km):
    """A compressed image holds at most K different colours."""
    rng = np.random.default_rng(1)
    K = 4
    A = rng.random((12, 10, 3))
    X = A.reshape(-1, 3)
    initial = km.kmeans_init_centroids(X, K, rng)
    centroids, _, _ = km.run_kmeans(X, initial, 10, rng)
    idx = km.find_closest_centroids(X, centroids)
    recovered = centroids[idx].reshape(A.shape)
    assert recovered.shape == A.shape
    assert len(np.unique(recovered.reshape(-1, 3), axis=0)) <= K
    # compression error is smaller than with the initial colours
    assert kmeans_cost(X, centroids) <= kmeans_cost(X, initial)


def test_compression_reproduces_image_that_has_k_colours(km):
    """An image that already has K colours is recovered exactly."""
    rng = np.random.default_rng(2)
    colours = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0],
                        [0.25, 0.25, 1.0], [1.0, 1.0, 0.75]])
    A = colours[rng.integers(4, size=(9, 11))]
    X = A.reshape(-1, 3)
    centroids, _, _ = km.run_kmeans(X, colours, 5, rng)
    idx = km.find_closest_centroids(X, centroids)
    recovered = centroids[idx].reshape(A.shape)
    assert recovered == pytest.approx(A, abs=1e-12)


def test_load_image_values_are_exact_8bit_levels(km):
    """The image is read as doubles in 0..1 that are multiples of 1/255."""
    A = km.load_image(SET3 / "exercise3_2" / "bird_small.png")
    assert A.dtype == np.float64
    assert A.shape == (128, 128, 3)
    assert A.min() >= 0.0 and A.max() <= 1.0
    assert A * 255 == pytest.approx(np.round(A * 255), abs=1e-9)


def test_kmeans_script_runs_and_saves_figures(km, tmp_path):
    """The whole script runs on the small image and writes its two figures."""
    km.main(["--seed", "0", "--image", "bird_small.png",
             "--save-dir", str(tmp_path)])
    assert len(pngs_in(tmp_path)) == 2


@pytest.mark.slow
def test_kmeans_script_runs_on_default_image(km, tmp_path, capsys):
    """The script with its default settings (peppers.png, 16 colours) runs
    and prints the reference numbers of parts 1 and 2."""
    km.main(["--seed", "0", "--save-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "\n 1 3 2\n" in out
    assert " 2.428301 3.157924 \n 5.813503 2.633656 \n 7.119387 3.616684 " in out
    assert len(pngs_in(tmp_path)) == 2


# --------------------------------------------------------------------------
# Exercise 3.3: Gaussian mixture model fitted with EM
# --------------------------------------------------------------------------

def synthetic_clusters(seed=0):
    """Three well-separated 2-D Gaussian clusters with weights 0.5, 0.3, 0.2.
    Returns (X, labels, means, weights)."""
    rng = np.random.default_rng(seed)
    means = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 12.0]])
    sizes = [300, 180, 120]
    X = np.vstack([m + rng.standard_normal((n, 2)) * [1.0, 0.5]
                   for m, n in zip(means, sizes)])
    labels = np.repeat(np.arange(3), sizes)
    return X, labels, means, np.array(sizes) / sum(sizes)


def log_likelihood(gmm, X, Pm, M, S):
    """Data log-likelihood of a mixture, computed with scipy."""
    dens = sum(Pm[k] * multivariate_normal.pdf(X, M[k], S[k])
               for k in range(len(Pm)))
    return np.log(dens).sum()


def test_gauss_log_pdf_matches_scipy(gmm):
    """The log-density equals scipy.stats.multivariate_normal.logpdf."""
    rng = np.random.default_rng(0)
    for d in (1, 2, 4):
        X = rng.standard_normal((25, d))
        mu = rng.standard_normal(d)
        B = rng.standard_normal((d, d))
        Sigma = B @ B.T + 0.5 * np.eye(d)
        expected = multivariate_normal.logpdf(X, mu, Sigma).reshape(-1)
        assert gmm.gauss_log_pdf(X, mu, Sigma) == pytest.approx(expected, abs=1e-10)


def test_gauss_log_pdf_does_not_underflow(gmm):
    """Far from the mean the log-density stays finite (the density is 0)."""
    logp = gmm.gauss_log_pdf(np.array([[100.0, 100.0]]), np.zeros(2), np.eye(2))
    assert np.isfinite(logp[0])
    assert logp[0] == pytest.approx(-np.log(2 * np.pi) - 10000.0)


def test_kmeans_finds_separated_clusters(gmm):
    """The script's own k-means separates well-separated clusters perfectly."""
    X, labels, means, _ = synthetic_clusters()
    found, centroids, sumd = gmm.kmeans(X, 3, 5, np.random.default_rng(0))
    acc, _ = gmm.cluster_accuracy(found, labels, 3)
    assert acc == pytest.approx(100.0)
    assert sumd == pytest.approx(((X - centroids[found]) ** 2).sum())
    order = [np.argmin(((centroids - m) ** 2).sum(axis=1)) for m in means]
    assert centroids[order] == pytest.approx(means, abs=0.2)


def test_kmeans_more_replicates_never_worse(gmm):
    """The best of several restarts is at least as good as the first one."""
    X, _, _, _ = synthetic_clusters(1)
    _, _, one = gmm.kmeans(X, 3, 1, np.random.default_rng(5))
    _, _, many = gmm.kmeans(X, 3, 6, np.random.default_rng(5))
    assert many <= one + 1e-9


def test_fit_gmms_recovers_means_and_weights(gmm):
    """EM on well-separated clusters recovers their means and weights."""
    X, _, means, weights = synthetic_clusters()
    Pm, M, S = gmm.fit_gmms(X, 3, np.random.default_rng(0))
    order = [np.argmin(((M - m) ** 2).sum(axis=1)) for m in means]
    assert sorted(order) == [0, 1, 2]
    assert M[order] == pytest.approx(means, abs=0.2)
    assert Pm[order] == pytest.approx(weights, abs=0.01)
    # the clusters were generated with standard deviations 1 and 0.5
    for k in order:
        assert np.sqrt(np.diag(S[k])) == pytest.approx([1.0, 0.5], abs=0.15)


def test_fit_gmms_returns_valid_parameters(gmm):
    """Weights sum to 1 and every covariance is symmetric positive definite."""
    X, _, _, _ = synthetic_clusters(2)
    Pm, M, S = gmm.fit_gmms(X, 3, np.random.default_rng(1))
    assert Pm.shape == (3,) and M.shape == (3, 2) and S.shape == (3, 2, 2)
    assert (Pm > 0).all()
    assert Pm.sum() == pytest.approx(1.0)
    for Sk in S:
        assert np.array_equal(Sk, Sk.T)
        assert (np.linalg.eigvalsh(Sk) > 0).all()


def test_em_log_likelihood_is_non_decreasing(gmm):
    """The data log-likelihood never decreases from one EM iteration to the
    next (overlapping clusters, poor start, so that EM has work to do)."""
    rng = np.random.default_rng(3)
    X = np.vstack([rng.standard_normal((80, 2)),
                   rng.standard_normal((80, 2)) + [2.5, 1.0],
                   rng.standard_normal((80, 2)) * 0.7 + [0.0, 3.0]])
    Pm, M, S = gmm.init_from_labels(X, np.arange(240) % 3, 3)
    # one iteration at a time: em(max_iters=1) returns the log-likelihood of
    # the model it was given and the model after one M-step
    history = []
    for _ in range(40):
        Pm, M, S, log_lik, _ = gmm.em(X, Pm, M, S, max_iters=1, tol=0.0)
        history.append(log_lik)
    assert all(b >= a - 1e-8 for a, b in zip(history, history[1:]))
    assert history[-1] > history[0] + 1.0


def test_em_reported_log_likelihood_matches_scipy(gmm):
    """The log-likelihood that em() returns is that of the returned model."""
    X, labels, _, _ = synthetic_clusters(4)
    start = gmm.init_from_labels(X, labels, 3)
    Pm, M, S, log_lik, n_iters = gmm.em(X, *start)
    assert 1 <= n_iters <= 200
    assert log_lik == pytest.approx(log_likelihood(gmm, X, Pm, M, S), rel=1e-9)


def test_init_from_labels_matches_sample_statistics(gmm):
    """Every cluster gives its share, its mean and its sample covariance."""
    X, labels, _, weights = synthetic_clusters(5)
    Pm, M, S = gmm.init_from_labels(X, labels, 3)
    assert Pm == pytest.approx(weights)
    for k in range(3):
        Xk = X[labels == k]
        assert M[k] == pytest.approx(Xk.mean(axis=0))
        assert S[k] == pytest.approx(np.cov(Xk, rowvar=False) + 1e-6 * np.eye(2))


def test_classify_with_gmms_picks_most_probable_component(gmm):
    """An example is assigned to the component with the highest posterior."""
    Pm = np.array([0.5, 0.5])
    M = np.array([[0.0, 0.0], [4.0, 0.0]])
    S = np.stack([np.eye(2), np.eye(2)])
    assert gmm.classify_with_gmms(np.array([0.5, 0.0]), Pm, M, S) == 0
    assert gmm.classify_with_gmms(np.array([3.0, 1.0]), Pm, M, S) == 1
    # at the midpoint the larger weight wins
    Pm = np.array([0.2, 0.8])
    assert gmm.classify_with_gmms(np.array([2.0, 0.0]), Pm, M, S) == 1


def test_cluster_accuracy_perfect_for_permuted_clustering(gmm):
    """A perfect clustering with renamed clusters scores 100%."""
    labels = np.array([0, 0, 1, 1, 1, 2, 2, 0, 2, 1])
    for perm in permutations(range(3)):
        clusters = np.array(perm)[labels]
        acc, mapped = gmm.cluster_accuracy(clusters, labels, 3)
        assert acc == pytest.approx(100.0)
        assert np.array_equal(mapped, labels)


def test_cluster_accuracy_invariant_to_cluster_renaming(gmm):
    """Renaming the clusters changes neither the accuracy nor the mapping."""
    rng = np.random.default_rng(0)
    labels = rng.integers(3, size=60)
    clusters = np.where(rng.random(60) < 0.7, labels, rng.integers(3, size=60))
    acc, mapped = gmm.cluster_accuracy(clusters, labels, 3)
    assert acc == pytest.approx(100.0 * np.mean(mapped == labels))
    assert 100.0 / 3 <= acc < 100.0
    for perm in permutations(range(3)):
        acc_p, mapped_p = gmm.cluster_accuracy(np.array(perm)[clusters], labels, 3)
        assert acc_p == pytest.approx(acc)
        assert np.array_equal(mapped_p, mapped)


def test_iris_accuracies_reference_values(gmm):
    """On iris with seed 0: GMM 96.67%, k-means 89.33%, as in MATLAB."""
    datasets = pytest.importorskip("sklearn.datasets")
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    rng = np.random.default_rng(0)
    Pm, M, S = gmm.fit_gmms(X, 3, rng)
    Y_hat = np.array([gmm.classify_with_gmms(x, Pm, M, S) for x in X])
    clusters, _, sumd = gmm.kmeans(X, 3, 5, rng)
    gmm_acc, Y_gmm = gmm.cluster_accuracy(Y_hat, y, 3)
    kmeans_acc, _ = gmm.cluster_accuracy(clusters, y, 3)
    assert gmm_acc == pytest.approx(96.67, abs=0.005)
    assert kmeans_acc == pytest.approx(89.33, abs=0.005)
    # the 5 flowers the GMM gets wrong (1-based, as in MATLAB)
    assert list(np.flatnonzero(Y_gmm != y) + 1) == [69, 71, 73, 78, 84]
    assert np.sort(Pm) == pytest.approx([0.299314, 1 / 3, 0.367353], abs=1e-4)
    assert sumd == pytest.approx(78.851441, abs=1e-5)


def test_gmm_script_runs_and_saves_figures(gmm, tmp_path, capsys):
    """The whole script runs, prints both accuracies and writes two figures."""
    pytest.importorskip("sklearn")
    gmm.main(["--seed", "0", "--save-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "GMM accuracy = 96.67%." in out
    assert "kmeans accuracy = 89.33%." in out
    assert len(pngs_in(tmp_path)) == 2


# --------------------------------------------------------------------------
# Exercise 4.1: feature selection, cross validation and overfitting
# --------------------------------------------------------------------------

def noise_problem(seed, num_examples=25, num_negative=10, num_features=1000):
    """Random features and the labels of the exercise (-1 first, then +1)."""
    rng = np.random.default_rng(seed)
    labels = np.ones(num_examples)
    labels[:num_negative] = -1
    return rng.standard_normal((num_examples, num_features)), labels


def test_similarity_measure_equals_squared_correlation(ovf):
    """The measure is the squared Pearson correlation coefficient."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.standard_normal(24)
        y = rng.standard_normal(24)
        assert ovf.similarity_measure(x, y) == pytest.approx(
            np.corrcoef(x, y)[0, 1] ** 2)


def test_similarity_measure_is_one_for_linear_relation(ovf):
    """Linearly related vectors score 1, whatever the sign of the slope."""
    x = np.random.default_rng(1).standard_normal(30)
    assert ovf.similarity_measure(x, 3.0 * x + 2.0) == pytest.approx(1.0)
    assert ovf.similarity_measure(x, -0.5 * x + 7.0) == pytest.approx(1.0)


def test_similarity_measure_is_symmetric_and_bounded(ovf):
    """r(x, y) = r(y, x), and the value lies between 0 and 1."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(40)
    y = rng.standard_normal(40)
    r = ovf.similarity_measure(x, y)
    assert r == pytest.approx(ovf.similarity_measure(y, x))
    assert 0.0 <= r <= 1.0


def test_select_features_finds_the_informative_features(ovf):
    """Features built from the labels are selected first, best one first."""
    features, labels = noise_problem(3, num_features=50)
    features[:, 7] = 2.0 * labels                      # perfectly correlated
    features[:, 31] = labels + 0.3 * features[:, 31]   # strongly correlated
    selected = ovf.select_features(features, labels, 5)
    assert selected.shape == (5,)
    assert len(set(selected)) == 5
    assert list(selected[:2]) == [7, 31]


def test_select_features_orders_by_decreasing_similarity(ovf):
    """The selected features are the ones with the highest similarity."""
    features, labels = noise_problem(4, num_features=80)
    selected = ovf.select_features(features, labels, 10)
    r = np.array([ovf.similarity_measure(features[:, j], labels)
                  for j in range(80)])
    assert list(r[selected]) == sorted(r, reverse=True)[:10]


def test_classify_example_on_separable_data(ovf):
    """The linear SVM classifies points on either side of a clear gap."""
    train_x = np.array([[-2.0, 0.0], [-3.0, 1.0], [2.0, 0.0], [3.0, -1.0]])
    train_y = np.array([-1.0, -1.0, 1.0, 1.0])
    assert ovf.classify_example(train_x, train_y, np.array([-1.5, 0.3])) == -1
    assert ovf.classify_example(train_x, train_y, np.array([2.5, 0.3])) == 1


def test_leave_one_out_accuracy_is_100_on_separable_data(ovf):
    """Informative features give 100% with every selection procedure."""
    features, labels = noise_problem(5, num_features=30)
    features[:, :5] += 6.0 * labels[:, None]
    for selection in ("none", "inside", "outside"):
        assert ovf.leave_one_out_accuracy(features, labels, 5, selection) \
            == pytest.approx(100.0)


def test_selection_outside_cv_overfits_and_inside_stays_near_chance(ovf):
    """On pure noise (seed 0): selection outside the CV reports 100%, the
    honest procedures stay near chance."""
    features, labels = noise_problem(0)
    none = ovf.leave_one_out_accuracy(features, labels, 100, "none")
    inside = ovf.leave_one_out_accuracy(features, labels, 100, "inside")
    outside = ovf.leave_one_out_accuracy(features, labels, 100, "outside")
    assert outside == pytest.approx(100.0)
    assert 20.0 <= inside <= 80.0
    assert 20.0 <= none <= 80.0
    # the values of the script for --seed 0
    assert (none, inside) == pytest.approx((56.0, 44.0))


def test_selection_inside_cv_never_sees_held_out_subject(ovf, monkeypatch):
    """Inside the CV the features are chosen from the training subjects of
    the fold only; outside the CV they are chosen once from all subjects."""
    features, labels = noise_problem(6, num_examples=12, num_negative=5,
                                     num_features=60)
    original = ovf.select_features
    calls = []

    def recording_select(f, y, num_selected):
        calls.append((f.copy(), y.copy()))
        return original(f, y, num_selected)

    monkeypatch.setattr(ovf, "select_features", recording_select)

    ovf.leave_one_out_accuracy(features, labels, 10, "inside")
    assert len(calls) == 12
    for i, (f, y) in enumerate(calls):
        assert f.shape == (11, 60) and y.shape == (11,)
        assert np.array_equal(f, np.delete(features, i, axis=0))
        assert np.array_equal(y, np.delete(labels, i))
        assert not (f == features[i]).all(axis=1).any()

    calls.clear()
    ovf.leave_one_out_accuracy(features, labels, 10, "outside")
    assert len(calls) == 1
    assert np.array_equal(calls[0][0], features)

    calls.clear()
    ovf.leave_one_out_accuracy(features, labels, 10, "none")
    assert calls == []


def test_overfitting_script_runs_and_saves_figure(ovf, tmp_path, capsys):
    """The whole script runs, prints three accuracies and writes its figure."""
    ovf.main(["--seed", "0", "--save-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert out.count("accuracy: ") == 3
    assert "accuracy: 100.00%" in out
    assert len(pngs_in(tmp_path)) == 1


def test_overfitting_script_rejects_invalid_sizes(ovf):
    """More selected features than features is refused on the command line."""
    with pytest.raises(SystemExit):
        ovf.main(["--num-features", "10", "--num-selected", "20"])
