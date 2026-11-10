"""Tests of the Python versions of exercises 2.1, 2.2, 2.4 and 2.5
(logistic regression, regularised logistic regression, Bernoulli naive
Bayes on digits, Bayesian estimation of a Gaussian mean)."""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.optimize import minimize

from conftest import SET2, load_exercise

logreg = load_exercise(SET2 / "exercise2_1" / "logistic_regression.py")
reglog = load_exercise(SET2 / "exercise2_2" / "regularized_logistic_regression.py")
digits = load_exercise(SET2 / "exercise2_4" / "ex_2_4.py")
bayes = load_exercise(SET2 / "exercise2_5" / "ex_2_5.py")


# --------------------------------------------------------------------------
# Helpers and fixtures
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def close_figures():
    """Every test starts and ends without open matplotlib figures."""
    plt.close("all")
    yield
    plt.close("all")


def numerical_gradient(f, theta, h=1e-6):
    """Central finite differences of the scalar function f at theta."""
    grad = np.zeros_like(theta)
    for j in range(len(theta)):
        step = np.zeros_like(theta)
        step[j] = h
        grad[j] = (f(theta + step) - f(theta - step)) / (2 * h)
    return grad


def fit(cost, theta0, *args):
    """Minimise a (cost, gradient) function as the scripts do (BFGS)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return minimize(cost, theta0, args=args, jac=True, method="BFGS",
                        options={"maxiter": 400})


def run_main(module, monkeypatch, tmp_path, *extra_args):
    """Run module.main() with --save-dir tmp_path; return the PNGs written."""
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--save-dir", str(tmp_path), *extra_args])
    module.main()
    return sorted(tmp_path.glob("*.png"))


@pytest.fixture(scope="module")
def exam_data():
    """Exercise 2.1 data: X with the intercept column (100 x 3) and y."""
    data = np.loadtxt(logreg.DATA_FILE, delimiter=",")
    X = np.hstack([np.ones((len(data), 1)), data[:, :2]])
    return X, data[:, 2]


@pytest.fixture(scope="module")
def exam_fit(exam_data):
    """Optimiser result of exercise 2.1, started at theta = 0."""
    X, y = exam_data
    return fit(logreg.cost_function, np.zeros(3), X, y)


@pytest.fixture(scope="module")
def chip_data():
    """Exercise 2.2 data: polynomial features (118 x 28) and y."""
    data = np.loadtxt(reglog.DATA_FILE, delimiter=",")
    return reglog.map_feature(data[:, 0], data[:, 1]), data[:, 2]


@pytest.fixture(scope="module")
def digit_data():
    """Exercise 2.4: (train, test, B) with B the Bernoulli parameters."""
    train, test = digits.load_digits()
    return train, test, digits.train_bernoulli(train)


@pytest.fixture(scope="module")
def digit_confusion(digit_data):
    """Row-normalised confusion matrix of the test set (10 x 10)."""
    _, test, B = digit_data
    cm = np.zeros((10, 10))
    for label in range(10):
        pred = digits.classify(test[label], B)
        cm[label] = np.bincount(pred, minlength=10) / len(pred)
    return cm


@pytest.fixture(scope="module")
def measurements():
    """Exercise 2.5: the 25 measurements."""
    return bayes.read_measurements()


# --------------------------------------------------------------------------
# Exercise 2.1 - logistic regression
# --------------------------------------------------------------------------

@pytest.mark.parametrize("module", [logreg, reglog], ids=["2.1", "2.2"])
def test_sigmoid_known_values(module):
    """sigmoid(0) = 0.5, sigmoid(ln 3) = 0.75, and the tails go to 0 and 1."""
    assert module.sigmoid(0) == pytest.approx(0.5)
    assert module.sigmoid(np.log(3)) == pytest.approx(0.75)
    assert module.sigmoid(50) == pytest.approx(1.0)
    assert module.sigmoid(-50) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("module", [logreg, reglog], ids=["2.1", "2.2"])
def test_sigmoid_elementwise_and_symmetric(module):
    """sigmoid works element-wise on matrices and sigmoid(-z) = 1 - sigmoid(z)."""
    z = np.array([[-2.0, -0.5, 0.0], [0.5, 2.0, 7.0]])
    s = module.sigmoid(z)
    assert s.shape == z.shape
    assert np.all((s > 0) & (s < 1))
    assert module.sigmoid(-z) == pytest.approx(1 - s)


def test_cost_at_zero_theta_is_ln2(exam_data):
    """At theta = 0 every h is 0.5, so the cost is ln 2 = 0.693147."""
    X, y = exam_data
    cost, _ = logreg.cost_function(np.zeros(3), X, y)
    assert cost == pytest.approx(np.log(2))
    assert cost == pytest.approx(0.693147, abs=1e-6)


def test_gradient_at_zero_theta_reference(exam_data):
    """Gradient at theta = 0 equals the values given in the assignment."""
    X, y = exam_data
    _, grad = logreg.cost_function(np.zeros(3), X, y)
    assert grad.shape == (3,)
    assert grad == pytest.approx([-0.1, -12.009217, -11.262842], abs=1e-6)


def test_gradient_matches_finite_differences(exam_data):
    """The analytic gradient agrees with central finite differences."""
    X, y = exam_data
    theta = np.array([-0.5, 0.01, -0.005])
    _, grad = logreg.cost_function(theta, X, y)
    numeric = numerical_gradient(
        lambda t: logreg.cost_function(t, X, y)[0], theta)
    assert grad == pytest.approx(numeric, rel=1e-5, abs=1e-7)


def test_optimum_cost_and_theta(exam_fit):
    """The optimiser reaches cost 0.203498 and the reference theta."""
    assert exam_fit.fun == pytest.approx(0.203498, abs=1e-6)
    assert exam_fit.x == pytest.approx([-25.1613, 0.20623, 0.20147], rel=1e-3)


def test_gradient_vanishes_at_optimum(exam_data, exam_fit):
    """The gradient is (numerically) zero at the fitted theta."""
    X, y = exam_data
    _, grad = logreg.cost_function(exam_fit.x, X, y)
    assert np.max(np.abs(grad)) < 1e-3


def test_admission_probability_45_85(exam_fit):
    """A student with scores 45 and 85 is admitted with probability 0.776."""
    prob = logreg.sigmoid(np.array([1.0, 45.0, 85.0]) @ exam_fit.x)
    assert prob == pytest.approx(0.776, abs=1e-3)


def test_training_accuracy_89_percent(exam_data, exam_fit):
    """The fitted model classifies 89 of the 100 training points correctly."""
    X, y = exam_data
    assert np.mean(logreg.predict(exam_fit.x, X) == y) * 100 == pytest.approx(89.0)


def test_predict_threshold_at_half():
    """predict gives 1 exactly when theta'x >= 0 (sigmoid >= 0.5)."""
    X = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 2.0]])
    theta = np.array([0.0, 1.0])
    assert logreg.predict(theta, X).tolist() == [0.0, 1.0, 1.0]
    assert reglog.predict(theta, X).tolist() == [0.0, 1.0, 1.0]


# --------------------------------------------------------------------------
# Exercise 2.2 - regularised logistic regression
# --------------------------------------------------------------------------

def test_map_feature_shape_and_columns(chip_data):
    """map_feature gives 28 columns: 1, x1, x2, x1^2, x1 x2, ..., x2^6."""
    X, _ = chip_data
    assert X.shape == (118, 28)
    out = reglog.map_feature(2.0, 3.0)
    assert out.shape == (1, 28)
    assert out[0, :6] == pytest.approx([1, 2, 3, 4, 6, 9])
    assert out[0, -1] == pytest.approx(3.0 ** 6)
    assert out[0, -7] == pytest.approx(2.0 ** 6)


def test_reg_cost_at_zero_theta(chip_data):
    """At theta = 0 the regularised cost is ln 2 for any lambda."""
    X, y = chip_data
    for lam in (0.0, 1.0, 100.0):
        cost, _ = reglog.cost_function_reg(np.zeros(28), X, y, lam)
        assert cost == pytest.approx(0.693147, abs=1e-6)


def test_reg_gradient_at_zero_theta_reference(chip_data):
    """First five gradient values at theta = 0, lambda = 1 (reference)."""
    X, y = chip_data
    _, grad = reglog.cost_function_reg(np.zeros(28), X, y, 1.0)
    assert grad.shape == (28,)
    assert grad[:5] == pytest.approx(
        [0.008475, 0.018788, 0.000078, 0.050345, 0.011501], abs=1e-6)


def test_reg_cost_and_gradient_at_ones_lambda_10(chip_data):
    """Cost 3.164509 and reference gradient at theta = 1, lambda = 10."""
    X, y = chip_data
    cost, grad = reglog.cost_function_reg(np.ones(28), X, y, 10.0)
    assert cost == pytest.approx(3.164509, abs=1e-6)
    assert grad[:5] == pytest.approx(
        [0.346045, 0.161352, 0.194796, 0.226863, 0.092186], abs=1e-6)


def test_intercept_is_not_regularised(chip_data):
    """lambda changes neither the gradient of theta_0 nor the cost of a
    theta whose only non-zero entry is the intercept."""
    X, y = chip_data
    theta = np.zeros(28)
    theta[0] = 2.0
    cost0, grad0 = reglog.cost_function_reg(theta, X, y, 0.0)
    cost9, grad9 = reglog.cost_function_reg(theta, X, y, 9.0)
    assert cost9 == pytest.approx(cost0)
    assert grad9 == pytest.approx(grad0)
    _, g0 = reglog.cost_function_reg(np.ones(28), X, y, 0.0)
    _, g9 = reglog.cost_function_reg(np.ones(28), X, y, 9.0)
    assert g9[0] == pytest.approx(g0[0])
    assert g9[1:] == pytest.approx(g0[1:] + 9.0 / len(y))


def test_lambda_zero_equals_unregularised_cost(chip_data):
    """With lambda = 0 the cost and gradient equal those of exercise 2.1."""
    X, y = chip_data
    theta = np.linspace(-0.3, 0.3, 28)
    cost_r, grad_r = reglog.cost_function_reg(theta, X, y, 0.0)
    cost_p, grad_p = logreg.cost_function(theta, X, y)
    assert cost_r == pytest.approx(cost_p)
    assert grad_r == pytest.approx(grad_p)


@pytest.mark.parametrize("lam", [0.0, 1.0, 10.0])
def test_reg_gradient_matches_finite_differences(chip_data, lam):
    """The analytic regularised gradient agrees with finite differences."""
    X, y = chip_data
    theta = np.linspace(-0.5, 0.5, 28)
    _, grad = reglog.cost_function_reg(theta, X, y, lam)
    numeric = numerical_gradient(
        lambda t: reglog.cost_function_reg(t, X, y, lam)[0], theta)
    assert grad == pytest.approx(numeric, rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("lam, cost, accuracy", [
    (1.0, 0.529003, 83.050847),
    (10.0, 0.648216, 74.576271),
    (100.0, 0.686484, 61.016949),
])
def test_regularised_optimum(chip_data, lam, cost, accuracy):
    """For lambda > 0 the unique minimum gives the reference cost/accuracy."""
    X, y = chip_data
    res = fit(reglog.cost_function_reg, np.zeros(28), X, y, lam)
    assert res.fun == pytest.approx(cost, abs=2e-6)
    assert np.mean(reglog.predict(res.x, X) == y) * 100 == pytest.approx(
        accuracy, abs=1e-4)


def test_unregularised_fit_accuracy_range(chip_data):
    """For lambda = 0 the result depends on where the optimiser stops, so
    only a range is checked: accuracy 84-92 %, cost below the lambda = 1 one."""
    X, y = chip_data
    res = fit(reglog.cost_function_reg, np.zeros(28), X, y, 0.0)
    assert 84.0 <= np.mean(reglog.predict(res.x, X) == y) * 100 <= 92.0
    assert res.fun < 0.35


def test_larger_lambda_shrinks_theta(chip_data):
    """The norm of the regularised part of theta decreases as lambda grows."""
    X, y = chip_data
    norms = [np.linalg.norm(
        fit(reglog.cost_function_reg, np.zeros(28), X, y, lam).x[1:])
        for lam in (1.0, 10.0, 100.0)]
    assert norms[0] > norms[1] > norms[2]


# --------------------------------------------------------------------------
# Exercise 2.4 - Bernoulli naive Bayes
# --------------------------------------------------------------------------

def test_digits_data_shapes_and_binary(digit_data):
    """10 training and 10 test sets of 500 x 784 pixels with values 0/1."""
    train, test, _ = digit_data
    assert len(train) == len(test) == 10
    for part in train + test:
        assert part.shape == (500, 784)
        assert set(np.unique(part)) <= {0.0, 1.0}


def test_bernoulli_parameters_are_probabilities(digit_data):
    """The ML estimates form a 10 x 784 matrix of values in [0, 1] equal to
    the per-pixel mean of the training images."""
    train, _, B = digit_data
    assert B.shape == (10, 784)
    assert B.min() >= 0.0 and B.max() <= 1.0
    assert B[3] == pytest.approx(train[3].sum(axis=0) / 500)


def test_to_image_is_row_major():
    """to_image puts the first 28 values of the vector in the first row."""
    img = digits.to_image(np.arange(784))
    assert img.shape == (28, 28)
    assert img[0, 5] == 5 and img[1, 0] == 28


def test_log_likelihood_small_example():
    """log p(x|y) = sum x ln p + (1 - x) ln(1 - p) on a hand-made example."""
    B = np.array([[0.9, 0.2], [0.5, 0.5]])
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    ll = digits.log_likelihoods(X, B)
    assert ll.shape == (2, 2)
    assert ll[0] == pytest.approx([np.log(0.9 * 0.8), np.log(0.25)])
    assert ll[1] == pytest.approx([np.log(0.1 * 0.2), np.log(0.25)])
    assert digits.classify(X, B).tolist() == [0, 1]


def test_log_likelihood_finite_with_extreme_parameters():
    """Clamping keeps the log-likelihood finite when p is exactly 0 or 1."""
    B = np.array([[0.0, 1.0], [1.0, 0.0]])
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    ll = digits.log_likelihoods(X, B)
    assert np.all(np.isfinite(ll))
    assert digits.classify(X, B).tolist() == [1, 0]


def test_digits_test_accuracy(digit_confusion):
    """The classifier reaches accuracy 0.7982 on the 5000 test images."""
    assert np.trace(digit_confusion) / 10 == pytest.approx(0.7982, abs=1e-3)


def test_confusion_matrix_rows_sum_to_one(digit_confusion):
    """Every row of the confusion matrix is a distribution over predictions."""
    assert digit_confusion.shape == (10, 10)
    assert np.all(digit_confusion >= 0)
    assert digit_confusion.sum(axis=1) == pytest.approx(np.ones(10))


def test_confusion_matrix_diagonal_dominant(digit_confusion):
    """Every digit is most often classified as itself (and more than half
    of the time)."""
    assert np.argmax(digit_confusion, axis=1).tolist() == list(range(10))
    assert np.all(np.diag(digit_confusion) > 0.5)


def test_training_images_fit_their_own_model(digit_data):
    """Accuracy on the training images of a digit is high (digit 1 > 0.9)."""
    train, _, B = digit_data
    assert np.mean(digits.classify(train[1], B) == 1) > 0.9


# --------------------------------------------------------------------------
# Exercise 2.5 - Bayesian estimation of a Gaussian mean
# --------------------------------------------------------------------------

SIGMA2 = 1.25 ** 2


def test_measurements_loaded(measurements):
    """The data file gives 25 values with sample mean 4.867364."""
    assert measurements.shape == (25,)
    assert measurements[0] == pytest.approx(3.1803)
    assert measurements.mean() == pytest.approx(4.867364, abs=1e-6)


def test_running_mean(measurements):
    """running_mean[k-1] is the mean of the first k values."""
    assert bayes.running_mean(np.array([1.0, 3.0, 8.0])) == pytest.approx(
        [1.0, 2.0, 4.0])
    y = bayes.running_mean(measurements)
    assert y[0] == pytest.approx(measurements[0])
    assert y[-1] == pytest.approx(measurements.mean())


def test_posterior_at_n25(measurements):
    """Posterior for sigma0^2 = 10 sigma^2, n = 25: mean 4.847972, std 0.249501."""
    mu_n, s_n = bayes.posterior_parameters(
        25, measurements.mean(), SIGMA2, 0.0, 10 * SIGMA2)
    assert mu_n == pytest.approx(4.847972, abs=1e-6)
    assert np.sqrt(s_n) == pytest.approx(0.249501, abs=1e-6)


@pytest.mark.parametrize("factor, mean, std", [
    (10, 4.847972, 1.274657),
    (1, 4.680158, 1.273812),
    (0.1, 3.476689, 1.267731),
    (0.01, 0.973473, 1.254990),
])
def test_predictive_density_at_n25(measurements, factor, mean, std):
    """Mean and std of p(x|H(25)) for the four prior variances."""
    mu_n, s_n = bayes.posterior_parameters(
        25, measurements.mean(), SIGMA2, 0.0, factor * SIGMA2)
    assert mu_n == pytest.approx(mean, abs=1e-6)
    assert np.sqrt(SIGMA2 + s_n) == pytest.approx(std, abs=1e-6)


def test_posterior_variance_decreases_with_n(measurements):
    """sigma_n^2 decreases strictly with n and stays below both the prior
    variance and sigma^2 / n."""
    n = np.arange(1, 26)
    _, s_n = bayes.posterior_parameters(
        n, bayes.running_mean(measurements), SIGMA2, 0.0, 10 * SIGMA2)
    assert np.all(np.diff(s_n) < 0)
    assert np.all(s_n < 10 * SIGMA2)
    assert np.all(s_n < SIGMA2 / n)


def test_posterior_precisions_add():
    """1/sigma_n^2 = n/sigma^2 + 1/sigma_0^2."""
    _, s_n = bayes.posterior_parameters(7, 3.0, 2.0, 1.0, 5.0)
    assert 1 / s_n == pytest.approx(7 / 2.0 + 1 / 5.0)


def test_small_prior_variance_pulls_mean_to_mu0(measurements):
    """The smaller sigma_0^2, the closer mu_n is to mu_0; mu_n always lies
    between mu_0 and the sample mean."""
    mean = measurements.mean()
    mus = [bayes.posterior_parameters(25, mean, SIGMA2, 0.0, f * SIGMA2)[0]
           for f in (10, 1, 0.1, 0.01)]
    assert all(a > b for a, b in zip(mus, mus[1:]))
    assert all(0.0 < mu < mean for mu in mus)


def test_posterior_limits():
    """Very broad prior -> sample mean; very narrow prior -> prior mean."""
    mu_broad, _ = bayes.posterior_parameters(25, 4.0, SIGMA2, 1.0, 1e12)
    mu_narrow, s_narrow = bayes.posterior_parameters(25, 4.0, SIGMA2, 1.0, 1e-12)
    assert mu_broad == pytest.approx(4.0)
    assert mu_narrow == pytest.approx(1.0)
    assert s_narrow == pytest.approx(1e-12, rel=1e-6)


def test_normal_pdf_properties():
    """normal_pdf has the right peak value, is symmetric and integrates to 1."""
    assert bayes.normal_pdf(0.0, 0.0, 1.0) == pytest.approx(1 / np.sqrt(2 * np.pi))
    assert bayes.normal_pdf(1.0, 3.0, 2.0) == pytest.approx(
        bayes.normal_pdf(5.0, 3.0, 2.0))
    x = np.linspace(-20.0, 26.0, 20001)
    pdf = bayes.normal_pdf(x, 3.0, 2.0)
    assert np.sum(pdf) * (x[1] - x[0]) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# Smoke tests: every script runs from start to end and saves its figures
# --------------------------------------------------------------------------

def test_main_logistic_regression(monkeypatch, tmp_path, capsys):
    """2.1 main() finishes, saves 2 PNGs and prints the 89 % accuracy."""
    pngs = run_main(logreg, monkeypatch, tmp_path)
    assert [p.name for p in pngs] == ["data.png", "decision_boundary.png"]
    assert "Train Accuracy: 89.000000" in capsys.readouterr().out


def test_main_regularized_logistic_regression(monkeypatch, tmp_path, capsys):
    """2.2 main() finishes and saves the data plot and one boundary per lambda."""
    pngs = run_main(reglog, monkeypatch, tmp_path, "--lambdas", "1", "100")
    assert {p.name for p in pngs} == {
        "data.png", "boundary_lambda_1.png", "boundary_lambda_100.png"}
    out = capsys.readouterr().out
    assert "lambda = 1: Train Accuracy: 83.05" in out
    assert "lambda = 100: Train Accuracy: 61.01" in out


def test_main_naive_bayes(monkeypatch, tmp_path, capsys):
    """2.4 main() finishes, saves 3 PNGs and prints accuracy 0.7982."""
    pngs = run_main(digits, monkeypatch, tmp_path)
    assert {p.name for p in pngs} == {
        "sample_digit.png", "bernoulli_models.png", "confusion_matrix.png"}
    assert "Accuracy: 0.7982" in capsys.readouterr().out


def test_main_bayesian_estimation(monkeypatch, tmp_path, capsys):
    """2.5 main() finishes, saves 5 PNGs and prints the n = 25 posterior."""
    pngs = run_main(bayes, monkeypatch, tmp_path)
    assert len(pngs) == 5
    out = capsys.readouterr().out
    assert "mu_n = 4.847972, sigma_n = 0.249501" in out
    assert "Sample mean of all 25 measurements: 4.867364" in out
