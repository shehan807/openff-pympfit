import numpy as np
import pytest

from pympfit.mpfit.solvers import MPFITSolverError, MPFITSVDSolver


class TestMPFITSVDSolver:
    """Test MPFITSVDSolver produces valid charges."""

    def test_solve(self):
        A_site = np.array(  # noqa: N806
            [[1.0 / 3.0, 2.0 / 3.0], [3.0 / 3.0, 5.0 / 3.0]]
        )
        b_site = np.array([-1.0, -2.0])

        design_matrix = np.empty(2, dtype=object)
        design_matrix[0] = A_site
        design_matrix[1] = A_site

        reference_values = np.empty(2, dtype=object)
        reference_values[0] = b_site
        reference_values[1] = b_site

        quse_masks = np.empty(2, dtype=object)
        quse_masks[0] = np.array([True, True])
        quse_masks[1] = np.array([True, True])

        charges = MPFITSVDSolver().solve(
            design_matrix,
            reference_values,
            ancillary_arrays={"quse_masks": quse_masks},
        )

        assert charges.shape == (2, 1)
        assert np.allclose(charges, np.array([[6.0], [-6.0]]), atol=0.001)

    def test_solve_with_different_masks(self):
        """Test that quse_masks correctly control which atoms receive charges."""
        A_site_0 = np.array([[1.0]])  # noqa: N806
        b_site_0 = np.array([2.0])

        A_site_1 = np.array([[1.0]])  # noqa: N806
        b_site_1 = np.array([3.0])

        design_matrix = np.empty(2, dtype=object)
        design_matrix[0] = A_site_0
        design_matrix[1] = A_site_1

        reference_values = np.empty(2, dtype=object)
        reference_values[0] = b_site_0
        reference_values[1] = b_site_1

        quse_masks = np.empty(2, dtype=object)
        quse_masks[0] = np.array([True, False])
        quse_masks[1] = np.array([False, True])

        charges = MPFITSVDSolver().solve(
            design_matrix,
            reference_values,
            ancillary_arrays={"quse_masks": quse_masks},
        )

        assert charges.shape == (2, 1)
        assert np.allclose(charges, np.array([[2.0], [3.0]]), atol=0.001)

    def test_solve_error(self):
        """Test that mismatched dimensions raise an error."""
        A_site = np.array([[1.0, 2.0], [3.0, 4.0]])  # noqa: N806
        b_site = np.array([1.0, 2.0, 3.0])

        design_matrix = np.empty(1, dtype=object)
        design_matrix[0] = A_site

        reference_values = np.empty(1, dtype=object)
        reference_values[0] = b_site

        quse_masks = np.empty(1, dtype=object)
        quse_masks[0] = np.array([True])

        with pytest.raises(ValueError, match="matmul"):
            MPFITSVDSolver().solve(
                design_matrix,
                reference_values,
                ancillary_arrays={"quse_masks": quse_masks},
            )

    def test_solve_no_quse(self):
        """Test that missing quse_masks raises MPFITSolverError."""
        A_site = np.array([[1.0, 2.0], [3.0, 4.0]])  # noqa: N806
        b_site = np.array([1.0, 2.0])

        design_matrix = np.empty(1, dtype=object)
        design_matrix[0] = A_site

        reference_values = np.empty(1, dtype=object)
        reference_values[0] = b_site

        with pytest.raises(MPFITSolverError, match="quse_masks"):
            MPFITSVDSolver().solve(
                design_matrix,
                reference_values,
                ancillary_arrays=None,
            )
