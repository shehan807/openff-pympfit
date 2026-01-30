import numpy as np

from pympfit.mpfit.core import _regular_solid_harmonic, build_A_matrix, build_b_vector


class TestBuildAMatrix:

    def test_build_a_matrix(self):
        A_expected = np.array([[1.5, 1.5], [1.5, 2.0]])  # noqa: N806

        xyzmult = np.array([[0.0, 0.0, 0.0]])
        xyzcharge = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        r1, r2 = 0.5, 2.0
        maxl = 1

        A = np.zeros((2, 2))  # noqa: N806
        A = build_A_matrix(0, xyzmult, xyzcharge, r1, r2, maxl, A)  # noqa: N806

        assert A.shape == (2, 2)
        assert np.allclose(A, A.T)
        assert np.allclose(A, A_expected)


class TestBuildBVector:

    def test_build_b_vector(self):
        b_expected = np.array([1.5, 1.5])

        xyzmult = np.array([[0.0, 0.0, 0.0]])
        xyzcharge = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        r1, r2 = 0.5, 2.0
        maxl = 1

        multipoles = np.zeros((1, 2, 2, 2))
        multipoles[0, 0, 0, 0] = 1.0

        b = np.zeros(2)
        b = build_b_vector(0, xyzmult, xyzcharge, r1, r2, maxl, multipoles, b)

        assert b.shape == (2,)
        assert np.allclose(b, b_expected)


class TestRegularSolidHarmonic:

    def test_known_values(self):
        rsh = _regular_solid_harmonic

        assert np.isclose(rsh(0, 0, 0, 0, 0, 0), 1.0)
        assert np.isclose(rsh(0, 0, 0, 1, 2, 3), 1.0)

        assert np.isclose(rsh(1, 0, 0, 0, 0, 1.0), 1.0)
        assert np.isclose(rsh(1, 0, 0, 0, 0, 2.5), 2.5)

        assert np.isclose(rsh(1, 1, 0, 1.0, 0, 0), 1.0)
        assert np.isclose(rsh(1, 1, 0, 3.0, 0, 0), 3.0)

        assert np.isclose(rsh(1, 1, 1, 0, 1.0, 0), 1.0)
        assert np.isclose(rsh(1, 1, 1, 0, 4.0, 0), 4.0)

        assert np.isclose(rsh(4, 0, 0, 0, 0, 1.0), 1.0)
        assert np.isclose(rsh(4, 0, 0, 0, 0, 2.0), 16.0)
