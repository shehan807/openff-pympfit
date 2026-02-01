import numpy as np
import pytest

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


RSH_EXPECTED = [
    (0, 0, 0, 1.0),
    # l=1: R_10 = z, R_11c = x, R_11s = y
    (1, 0, 0, 3.0),
    (1, 1, 0, 1.0),
    (1, 1, 1, 2.0),
    # l=2:
    #       R_20  = (3z^2-r^2)/2, R_21c = sqrt(3)*xz, R_21s = sqrt(3)*yz,
    #       R_22c = sqrt(3)/2*(x^2-y^2), R_22s = sqrt(3)*xy
    (2, 0, 0, 6.5),
    (2, 1, 0, 5.196152422706632),
    (2, 1, 1, 10.392304845413264),
    (2, 2, 0, -2.598076211353316),
    (2, 2, 1, 3.464101615137754),
    # l=3:
    #       R_30  = z(2z^2-3x^2-3y^2)/2,
    #       R_31c = sqrt(6)/4 * x(4z^2-x^2-y^2),
    #       R_31s = sqrt(6)/4 * y(4z^2-x^2-y^2),
    #       R_32c = sqrt(15)/2 * (x^2-y^2)*z,
    #       R_32s = sqrt(15) * xyz,
    #       R_33c = sqrt(10)/4 * (x^3-3xy^2),
    #       R_33s = sqrt(10)/4 * (3x^2y-y^3)
    (3, 0, 0, 4.5),
    (3, 1, 0, 18.98354550656964),
    (3, 1, 1, 37.96709101313928),
    (3, 2, 0, -17.428425057933374),
    (3, 2, 1, 23.237900077244507),
    (3, 3, 0, -8.696263565463045),
    (3, 3, 1, -1.5811388300841898),
    # l=4:
    #   R_40  = 1/8*(8z^4 - 24(x^2+y^2)z^2 + 3(x^2+y^2)^2)
    #   R_41c = sqrt(10)/4*(4xz^3 - 3xz(x^2+y^2))
    #   R_41s = sqrt(10)/4*(4yz^3 - 3yz(x^2+y^2))
    #   R_42c = sqrt(5)/4*(x^2-y^2)*(6z^2-x^2-y^2)
    #   R_42s = sqrt(5)/4*xy*(6z^2-x^2-y^2)
    #   R_43c = sqrt(70)/4*z*(x^3-3xy^2)
    #   R_43s = sqrt(70)/4*z*(3x^2y-y^3)
    #   R_44c = sqrt(35)/8*(x^4-6x^2y^2+y^4)
    #   R_44s = sqrt(35)/8*xy*(x^2-y^2)
    (4, 0, 0, -44.625),
    (4, 1, 0, 49.80587314765198),
    (4, 1, 1, 99.61174629530396),
    (4, 2, 0, -82.17549817311728),
    (4, 2, 1, 54.78366544874485),
    (4, 3, 0, -69.02445218906124),
    (4, 3, 1, -12.549900398011133),
    (4, 4, 0, -5.176569810212164),
    (4, 4, 1, -4.437059837324712),
]


class TestRegularSolidHarmonic:

    @pytest.mark.parametrize(
        "l, m, cs, expected",
        RSH_EXPECTED,
        ids=[f"l{l}_m{m}_cs{cs}" for l, m, cs, _ in RSH_EXPECTED],
    )
    def test_known_values(self, l, m, cs, expected):
        """Test each (l,m,cs) at (x,y,z) = (1,2,3) against analytical value."""
        result = float(_regular_solid_harmonic(l, m, cs, 1.0, 2.0, 3.0))
        assert np.isclose(result, expected, rtol=1e-12)

    @pytest.mark.parametrize(
        "l, m, cs, expected",
        RSH_EXPECTED,
        ids=[f"l{l}_m{m}_cs{cs}" for l, m, cs, _ in RSH_EXPECTED],
    )
    def test_vector_matches_scalar(self, l, m, cs, expected):

        rng = np.random.default_rng(42)

        x = rng.standard_normal(8)
        y = rng.standard_normal(8)
        z = rng.standard_normal(8)

        vec_result = _regular_solid_harmonic(l, m, cs, x, y, z)
        scalar_results = np.array(
            [
                float(_regular_solid_harmonic(l, m, cs, x[i], y[i], z[i]))
                for i in range(len(x))
            ]
        )
        assert np.allclose(vec_result, scalar_results, rtol=1e-12)

    @pytest.mark.parametrize(
        "l, m, cs, expected",
        RSH_EXPECTED,
        ids=[f"l{l}_m{m}_cs{cs}" for l, m, cs, _ in RSH_EXPECTED],
    )
    def test_origin(self, l, m, cs, expected):
        result = float(_regular_solid_harmonic(l, m, cs, 0, 0, 0))
        if l == 0:
            assert np.isclose(result, 1.0)
        else:
            assert np.isclose(result, 0.0, atol=1e-12)

    @pytest.mark.parametrize(
        "l, m, cs, expected",
        RSH_EXPECTED,
        ids=[f"l{l}_m{m}_cs{cs}" for l, m, cs, _ in RSH_EXPECTED],
    )
    def test_near_origin(self, l, m, cs, expected):
        eps = 1e-15
        result = float(_regular_solid_harmonic(l, m, cs, eps, eps, eps))
        assert np.isfinite(result)
        if l > 0:
            assert abs(result) < 1e-5

    def test_vector_with_origin_point(self):

        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 0.5, 1.0])
        z = np.array([0.0, 1.5, 0.5])

        result = _regular_solid_harmonic(0, 0, 0, x, y, z)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        assert np.isclose(result[0], 1.0)
