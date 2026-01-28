import numpy as np

from openff_pympfit.mpfit.core import (
    _regular_solid_harmonic,
    _regular_solid_harmonic_depr,
    build_A_matrix,
    build_b_vector,
)


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

    def _rsh(self, l, m, cs, x, y, z):
        """Helper: call vectorized _regular_solid_harmonic with scalar inputs."""
        result = _regular_solid_harmonic(
            l, m, cs,
            np.array([x], dtype=np.float64),
            np.array([y], dtype=np.float64),
            np.array([z], dtype=np.float64),
        )
        return result[0]

    def test_known_values(self):
        rsh = self._rsh

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

    def test_vectorized(self):
        """Verify vectorized version matches scalar version across multiple points."""
        from openff_pympfit.mpfit.core import _regular_solid_harmonic_depr

        rng = np.random.default_rng(42)
        x = rng.standard_normal(20)
        y = rng.standard_normal(20)
        z = rng.standard_normal(20)

        for l in range(5):
            for m in range(l + 1):
                for cs in ([0] if m == 0 else [0, 1]):
                    vec_result = _regular_solid_harmonic(l, m, cs, x, y, z)
                    scalar_results = np.array([
                        _regular_solid_harmonic_depr(l, m, cs, x[i], y[i], z[i])
                        for i in range(len(x))
                    ])
                    np.testing.assert_allclose(
                        vec_result, scalar_results, rtol=1e-12,
                        err_msg=f"Mismatch at l={l}, m={m}, cs={cs}",
                    )
