"""Tests of the Python versions of exercises 2.6 and 2.7 (SVM).

2.6: the analytic KKT solutions. 2.7: the dual QP solved with cvxopt,
checked against the verified reference values and the KKT conditions.
The 2.7 tests are skipped when cvxopt is not installed.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from conftest import SET2, load_exercise

EX26 = SET2 / "exercise2_6"
EX27 = SET2 / "exercise2_7"

# cvxopt is an interior point solver: results are good to about 1e-8,
# the reference values are known to 4-6 decimals
SOLVER_TOL = 1e-6
REF_TOL = 1e-4


# --------------------------------------------------------------- helpers

@pytest.fixture(scope="module")
def ex_a():
    return load_exercise(EX26 / "ex_2_6a.py")


@pytest.fixture(scope="module")
def ex_b():
    return load_exercise(EX26 / "ex_2_6b.py")


@pytest.fixture(scope="module")
def svm_dual():
    pytest.importorskip("cvxopt")
    return load_exercise(EX27 / "svm_dual.py")


@pytest.fixture(autouse=True)
def close_figures():
    """Do not let figures pile up between tests."""
    yield
    plt.close("all")


def run_main(script, tmp_path, *extra):
    """Run a script's main() saving its figures in tmp_path; return the PNGs."""
    load_exercise(script).main([*extra, "--save-dir", str(tmp_path)])
    return sorted(tmp_path.glob("*.png"))


def margins(X, y, w, w0):
    """y * (w . x + w0) for every sample."""
    return y * (X @ w + w0)


# ----------------------------------------------------------- exercise 2.6

def test_2_6a_weights_bias_and_width(ex_a):
    """Part A: the KKT solution is w = [1, 0], w0 = -2, width 2."""
    _, w, w0 = ex_a.kkt_solution()
    assert w == pytest.approx([1.0, 0.0], abs=1e-12)
    assert w0 == pytest.approx(-2.0, abs=1e-12)
    assert 2 / np.linalg.norm(w) == pytest.approx(2.0)


def test_2_6a_all_samples_outside_margin(ex_a):
    """Part A: every sample satisfies y * (w.x + w0) >= 1."""
    _, w, w0 = ex_a.kkt_solution()
    assert np.all(ex_a.X_POS @ w + w0 >= 1 - 1e-12)
    assert np.all(-(ex_a.X_NEG @ w + w0) >= 1 - 1e-12)


def test_2_6a_support_vectors_lie_on_margin(ex_a):
    """Part A: [3,1], [3,-1], [1,0] are the support vectors, |w.x + w0| = 1."""
    Xsup, w, w0 = ex_a.kkt_solution()
    assert Xsup.tolist() == [[3, 1], [3, -1], [1, 0]]
    assert Xsup @ w + w0 == pytest.approx([1.0, 1.0, -1.0], abs=1e-12)


def test_2_6b_phi_of_the_samples(ex_b):
    """Part B: Phi(x) = x - 4*||x|| - 4 on both classes."""
    a, b = 4 * np.sqrt(8) + 4, 4 * np.sqrt(2) + 4     # 15.3137, 9.6569
    assert ex_b.phi(ex_b.X_POS) == pytest.approx(ex_b.X_POS - a)
    assert ex_b.phi(ex_b.X_NEG) == pytest.approx(ex_b.X_NEG - b)
    assert ex_b.phi(np.array([[2.0, 2.0]])) == pytest.approx(
        np.array([[-13.3137, -13.3137]]), abs=1e-4)
    assert ex_b.phi(np.array([[-1.0, -1.0]])) == pytest.approx(
        np.array([[-10.6569, -10.6569]]), abs=1e-4)


def test_2_6b_lambda_weights_bias_and_width(ex_b):
    """Part B: lambda = 0.141666, w = -0.376385*[1, 1], w0 = -9.02216."""
    xp, xn = ex_b.phi(ex_b.X_POS)[0], ex_b.phi(ex_b.X_NEG)[2]
    lam, w, w0 = ex_b.kkt_solution(xp, xn)
    assert lam == pytest.approx(0.141666, abs=1e-6)
    assert w == pytest.approx([-0.376385, -0.376385], abs=1e-6)
    assert w0 == pytest.approx(-9.02216, abs=1e-5)
    assert 2 / np.linalg.norm(w) == pytest.approx(3.75736, abs=1e-5)
    # the margin width is the distance between the two support vectors
    assert 2 / np.linalg.norm(w) == pytest.approx(np.linalg.norm(xp - xn))


def test_2_6b_feasible_with_support_vectors_on_margin(ex_b):
    """Part B: all transformed samples have y*f >= 1, the two SVs exactly 1."""
    P_pos, P_neg = ex_b.phi(ex_b.X_POS), ex_b.phi(ex_b.X_NEG)
    _, w, w0 = ex_b.kkt_solution(P_pos[0], P_neg[2])
    m_pos, m_neg = P_pos @ w + w0, -(P_neg @ w + w0)
    assert np.all(m_pos >= 1 - 1e-9) and np.all(m_neg >= 1 - 1e-9)
    assert m_pos[0] == pytest.approx(1.0)
    assert m_neg[2] == pytest.approx(1.0)
    # the other six samples are strictly outside the margin
    assert np.sum(np.r_[m_pos, m_neg] > 1 + 1e-6) == 6


def test_2_6a_script_runs_and_saves_figure(tmp_path, capsys):
    """Smoke test: ex_2_6a.py runs to the end and writes its PNG."""
    pngs = run_main(EX26 / "ex_2_6a.py", tmp_path)
    assert [p.name for p in pngs] == ["ex_2_6a.png"]
    assert "w = [1 0], w0 = -2, margin width = 2" in capsys.readouterr().out


def test_2_6b_script_runs_and_saves_figures(tmp_path, capsys):
    """Smoke test: ex_2_6b.py runs to the end and writes its two PNGs."""
    pngs = run_main(EX26 / "ex_2_6b.py", tmp_path)
    assert [p.name for p in pngs] == ["ex_2_6b_original.png",
                                      "ex_2_6b_transformed.png"]
    assert "lambda = 0.141666" in capsys.readouterr().out


def test_2_6_solutions_agree_with_numerical_qp(ex_a, ex_b, svm_dual):
    """The analytic 2.6 solutions equal the hard margin QP solutions."""
    y = np.r_[np.ones(4), -np.ones(4)]

    model = svm_dual.fit(np.vstack([ex_a.X_POS, ex_a.X_NEG]), y)
    _, w, w0 = ex_a.kkt_solution()
    assert model["w"] == pytest.approx(w, abs=SOLVER_TOL)
    assert model["w0"] == pytest.approx(w0, abs=SOLVER_TOL)
    assert model["sv"].tolist() == [0, 1, 4]
    assert model["lam"][model["sv"]] == pytest.approx([0.25, 0.25, 0.5],
                                                      abs=SOLVER_TOL)

    P_pos, P_neg = ex_b.phi(ex_b.X_POS), ex_b.phi(ex_b.X_NEG)
    model = svm_dual.fit(np.vstack([P_pos, P_neg]), y)
    lam, w, w0 = ex_b.kkt_solution(P_pos[0], P_neg[2])
    assert model["w"] == pytest.approx(w, abs=SOLVER_TOL)
    assert model["w0"] == pytest.approx(w0, abs=1e-5)
    assert model["sv"].tolist() == [0, 6]
    assert model["lam"][model["sv"]] == pytest.approx([lam, lam], abs=SOLVER_TOL)


# ------------------------------------------- exercise 2.7: svm_dual module

# Two points per class; the closest pair is (2,0) / (0,0), so the maximum
# margin line is x1 = 1: w = [1, 0], w0 = -1, lambda = 1/2 on that pair.
TINY_X = np.array([[2.0, 0.0], [3.0, 1.0], [0.0, 0.0], [-1.0, 1.0]])
TINY_Y = np.array([1.0, 1.0, -1.0, -1.0])


def test_svm_dual_tiny_set_hard_margin_known_answer(svm_dual):
    """Hard margin on a 4-point set with a hand-computed solution."""
    model = svm_dual.fit(TINY_X, TINY_Y)
    assert model["sv"].tolist() == [0, 2]
    assert model["lam"] == pytest.approx([0.5, 0.0, 0.5, 0.0], abs=SOLVER_TOL)
    assert model["w"] == pytest.approx([1.0, 0.0], abs=SOLVER_TOL)
    assert model["w0"] == pytest.approx(-1.0, abs=SOLVER_TOL)
    assert model["width"] == pytest.approx(2.0, abs=SOLVER_TOL)
    assert model["misclassified"] == 0


def test_svm_dual_large_C_equals_hard_margin(svm_dual):
    """Separable data: the soft margin solution with large C is the hard one."""
    hard = svm_dual.fit(TINY_X, TINY_Y)
    soft = svm_dual.fit(TINY_X, TINY_Y, C=100.0)
    assert soft["w"] == pytest.approx(hard["w"], abs=SOLVER_TOL)
    assert soft["w0"] == pytest.approx(hard["w0"], abs=SOLVER_TOL)
    assert soft["sv"].tolist() == hard["sv"].tolist()


def test_svm_dual_small_C_caps_the_multipliers(svm_dual):
    """C below the hard margin lambda (1/2): multipliers stop at C, margin widens."""
    model = svm_dual.fit(TINY_X, TINY_Y, C=0.1)
    assert model["lam"].max() == pytest.approx(0.1, abs=SOLVER_TOL)
    assert model["width"] > 2.0
    assert np.isfinite(model["w0"])


@pytest.mark.parametrize("C", [None, 0.05, 1.0, 1000.0])
def test_svm_dual_multipliers_satisfy_dual_constraints(svm_dual, C):
    """sum(lambda*y) = 0, 0 <= lambda <= C and w = sum(lambda*y*x)."""
    X, y = svm_dual.load_data("twofeature1.txt")
    if C is None:
        X, y = X[:-1], y[:-1]           # separable part only
    lam = svm_dual.solve_dual(X, y, C)
    scale = max(1.0, lam.max())
    assert lam @ y == pytest.approx(0.0, abs=1e-7 * scale)
    assert lam.min() >= -1e-7 * scale
    if C is not None:
        assert lam.max() <= C * (1 + 1e-7)
    model = svm_dual.fit(X, y, C)
    assert model["w"] == pytest.approx((model["lam"] * y) @ X)
    assert model["width"] == pytest.approx(2 / np.linalg.norm(model["w"]))


def test_svm_dual_load_data_shapes_and_labels(svm_dual):
    """The data files give (n, 2) samples with labels in {-1, +1}."""
    X1, y1 = svm_dual.load_data("twofeature1.txt")
    X2, y2 = svm_dual.load_data("twofeature2.txt")
    assert X1.shape == (51, 2) and y1.shape == (51,)
    assert X2.shape == (100, 2) and y2.shape == (100,)
    assert set(y1) == {-1.0, 1.0} and set(y2) == {-1.0, 1.0}
    Xn, yn = svm_dual.load_data("twofeature1.txt", n=50)
    assert Xn.shape == (50, 2) and np.array_equal(Xn, X1[:50])
    assert np.array_equal(yn, y1[:50])


def test_svm_dual_add_norm_feature(svm_dual):
    """(x1, x2) -> (x1, x2, x1^2 + x2^2)."""
    Xa = svm_dual.add_norm_feature(np.array([[3.0, 4.0], [0.5, -1.0]]))
    assert Xa == pytest.approx(np.array([[3.0, 4.0, 25.0], [0.5, -1.0, 1.25]]))


# ------------------------------------------------- exercise 2.7, part 1

@pytest.fixture(scope="module")
def part1(svm_dual):
    X, y = svm_dual.load_data("twofeature1.txt")
    X, y = X[:-1], y[:-1]
    return X, y, svm_dual.fit(X, y)


def test_part1_support_vectors(part1):
    """Hard margin, 50 examples: support vectors 12, 13, 48 (1-based)."""
    _, _, model = part1
    assert (model["sv"] + 1).tolist() == [12, 13, 48]
    assert model["lam"][model["sv"]] == pytest.approx(
        [6.3044, 1.6355, 7.9398], abs=REF_TOL)


def test_part1_weights_bias_and_width(part1):
    """Hard margin: w = [2.5414, 3.0694], w0 = -16.316236, width 0.501891."""
    _, _, model = part1
    assert model["w"] == pytest.approx([2.5414, 3.0694], abs=REF_TOL)
    assert model["w0"] == pytest.approx(-16.316236, abs=1e-5)
    assert model["width"] == pytest.approx(0.501891, abs=1e-6)


def test_part1_separates_with_full_margin(part1):
    """Hard margin: no training error, y*f >= 1 everywhere, = 1 on the SVs."""
    X, y, model = part1
    m = margins(X, y, model["w"], model["w0"])
    assert model["misclassified"] == 0
    assert m.min() >= 1 - 1e-6
    assert m[model["sv"]] == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------- exercise 2.7, part 2

PART2_REF = {   # C: (number of SVs, w, w0, width, misclassified)
    0.01: (42, [0.1842, 0.2181], -1.2426, 7.0058, 3),
    0.1: (28, [0.7240, 1.0396], -5.2272, 1.5787, 1),
    1: (12, [1.4067, 2.1332], -10.3450, 0.7827, 1),
    10000: (3, [4.6838, 13.0958], -53.156465, 0.143800, 0),
}


@pytest.mark.parametrize("C", sorted(PART2_REF))
def test_part2_reference_values(svm_dual, C):
    """Soft margin, 51 examples: SV count, w, w0, width, errors for each C."""
    n_sv, w, w0, width, n_err = PART2_REF[C]
    X, y = svm_dual.load_data("twofeature1.txt")
    model = svm_dual.fit(X, y, C=C)
    assert len(model["sv"]) == n_sv
    assert model["w"] == pytest.approx(w, abs=REF_TOL)
    assert model["w0"] == pytest.approx(w0, abs=REF_TOL)
    assert model["width"] == pytest.approx(width, abs=REF_TOL)
    assert model["misclassified"] == n_err


def test_part2_large_C_support_vectors_include_the_outlier(svm_dual):
    """C = 10000: support vectors 5, 43, 51 (1-based), all on the margin."""
    X, y = svm_dual.load_data("twofeature1.txt")
    model = svm_dual.fit(X, y, C=10000)
    assert (model["sv"] + 1).tolist() == [5, 43, 51]
    m = margins(X, y, model["w"], model["w0"])
    assert m[model["sv"]] == pytest.approx(1.0, abs=1e-5)
    assert m.min() >= 1 - 1e-5


def test_part2_margin_shrinks_as_C_grows(svm_dual):
    """A larger C never gives a wider margin or more support vectors."""
    X, y = svm_dual.load_data("twofeature1.txt")
    models = [svm_dual.fit(X, y, C=C) for C in (0.01, 0.1, 1, 10, 100)]
    widths = [m["width"] for m in models]
    counts = [len(m["sv"]) for m in models]
    assert widths == sorted(widths, reverse=True)
    assert counts == sorted(counts, reverse=True)


@pytest.mark.parametrize("C", [0.1, 1, 10])
def test_part2_complementary_slackness(svm_dual, C):
    """lambda = 0 => y*f >= 1;  0 < lambda < C => y*f = 1;  lambda = C => y*f <= 1."""
    X, y = svm_dual.load_data("twofeature1.txt")
    model = svm_dual.fit(X, y, C=C)
    lam, tol = model["lam"], 1e-4 * min(1, C)
    m = margins(X, y, model["w"], model["w0"])
    free = (lam > tol) & (lam < C - tol)
    assert free.any()
    assert m[free] == pytest.approx(1.0, abs=1e-5)
    assert np.all(m[lam <= tol] >= 1 - 1e-5)
    assert np.all(m[lam >= C - tol] <= 1 + 1e-5)


# ------------------------------------------------- exercise 2.7, part 3

PART3_REF = {   # C: (number of SVs, w, w0, misclassified)
    0.1: (61, [-0.0299, -0.1550, -1.7687], 1.1723, 6),
    1: (27, [-0.3886, -0.1391, -3.5127], 1.863946, 3),
    100: (10, [-0.7149, -0.4661, -7.4458], 3.6608, 3),
}


@pytest.mark.parametrize("C", sorted(PART3_REF))
def test_part3_reference_values(svm_dual, C):
    """Extra feature x1^2+x2^2, 100 examples: SV count, w, w0, errors per C."""
    n_sv, w, w0, n_err = PART3_REF[C]
    X, y = svm_dual.load_data("twofeature2.txt")
    Xa = svm_dual.add_norm_feature(X)
    model = svm_dual.fit(Xa, y, C=C)
    assert len(model["sv"]) == n_sv
    assert model["w"] == pytest.approx(w, abs=REF_TOL)
    assert model["w0"] == pytest.approx(w0, abs=REF_TOL)
    assert model["misclassified"] == n_err


def test_part3_boundary_is_a_circle_around_the_positive_class(svm_dual):
    """C = 1: the weight of ||x||^2 is negative, so f > 0 inside a circle."""
    X, y = svm_dual.load_data("twofeature2.txt")
    model = svm_dual.fit(svm_dual.add_norm_feature(X), y, C=1)
    w, w0 = model["w"], model["w0"]
    assert w[2] < 0
    # w3*(x1^2+x2^2) + w1*x1 + w2*x2 + w0 = 0: centre and radius
    centre = -w[:2] / (2 * w[2])
    radius = np.sqrt(centre @ centre - w0 / w[2])
    assert np.linalg.norm(centre) < 0.1
    assert radius == pytest.approx(0.73, abs=0.02)
    assert w0 > 0                       # the origin is classified +1


def test_part3_all_at_bound_fallback_for_tiny_C(svm_dual):
    """C = 0.01: every sample is at lambda = C; w0 comes from the KKT interval."""
    X, y = svm_dual.load_data("twofeature2.txt")
    model = svm_dual.fit(svm_dual.add_norm_feature(X), y, C=0.01)
    assert len(model["sv"]) == 100
    assert model["lam"] == pytest.approx(0.01, abs=1e-8)
    assert model["w0"] == pytest.approx(0.700244, abs=1e-5)
    assert model["misclassified"] == 33


# ---------------------------------------------- exercise 2.7: the scripts

def test_svm_exercise1_script_runs_and_saves_figure(svm_dual, tmp_path, capsys):
    """Smoke test: svm_exercise1.py prints the solution and writes its PNG."""
    pngs = run_main(EX27 / "svm_exercise1.py", tmp_path)
    out = capsys.readouterr().out
    assert [p.name for p in pngs] == ["svm_exercise1.png"]
    assert "Support vector indices: 12 13 48" in out
    assert "Width of the margin: 0.501891" in out


def test_svm_exercise2_script_runs_and_saves_figure(svm_dual, tmp_path, capsys):
    """Smoke test: svm_exercise2.py with its default C = 10000."""
    pngs = run_main(EX27 / "svm_exercise2.py", tmp_path)
    out = capsys.readouterr().out
    assert [p.name for p in pngs] == ["svm_exercise2_C10000.png"]
    assert "C = 10000" in out
    assert "Support vector indices: 5 43 51" in out
    assert "Misclassified training examples: 0" in out


def test_svm_exercise2_script_accepts_C(svm_dual, tmp_path, capsys):
    """svm_exercise2.py --C 0.1 uses the given C."""
    pngs = run_main(EX27 / "svm_exercise2.py", tmp_path, "--C", "0.1")
    out = capsys.readouterr().out
    assert [p.name for p in pngs] == ["svm_exercise2_C0.1.png"]
    assert "C = 0.1" in out
    assert "Misclassified training examples: 1" in out


def test_svm_exercise3_script_runs_and_saves_figure(svm_dual, tmp_path, capsys):
    """Smoke test: svm_exercise3.py with its default C = 1."""
    pngs = run_main(EX27 / "svm_exercise3.py", tmp_path)
    out = capsys.readouterr().out
    assert [p.name for p in pngs] == ["svm_exercise3_C1.png"]
    assert "Number of support vectors: 27" in out
    assert "Misclassified training examples: 3" in out


@pytest.mark.parametrize("script", ["svm_exercise2.py", "svm_exercise3.py"])
def test_scripts_reject_non_positive_C(svm_dual, script):
    """--C 0 is refused by argparse (exit code 2), nothing is solved."""
    with pytest.raises(SystemExit) as excinfo:
        load_exercise(EX27 / script).main(["--C", "0"])
    assert excinfo.value.code == 2
