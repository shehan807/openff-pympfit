import numpy as np
import pytest

from openff_pympfit.mpfit.solvers import MPFITSolverError, MPFITSVDSolver


class TestMPFITSVDSolver:
    """Test MPFITSVDSolver produces valid charges."""

    def test_solve(self):
        site_A = np.array([[1.0 / 3.0, 2.0 / 3.0], [3.0 / 3.0, 5.0 / 3.0]])
        site_b = np.array([-1.0, -2.0])

        design_matrix = np.empty(2, dtype=object)
        design_matrix[0] = site_A
        design_matrix[1] = site_A

        reference_values = np.empty(2, dtype=object)
        reference_values[0] = site_b
        reference_values[1] = site_b

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
        site_0_A = np.array([[1.0]])
        site_0_b = np.array([2.0])  

        site_1_A = np.array([[1.0]])
        site_1_b = np.array([3.0])  

        design_matrix = np.empty(2, dtype=object)
        design_matrix[0] = site_0_A
        design_matrix[1] = site_1_A

        reference_values = np.empty(2, dtype=object)
        reference_values[0] = site_0_b
        reference_values[1] = site_1_b

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
        site_A = np.array([[1.0, 2.0], [3.0, 4.0]])
        site_b = np.array([1.0, 2.0, 3.0])  

        design_matrix = np.empty(1, dtype=object)
        design_matrix[0] = site_A

        reference_values = np.empty(1, dtype=object)
        reference_values[0] = site_b

        quse_masks = np.empty(1, dtype=object)
        quse_masks[0] = np.array([True])

        with pytest.raises(ValueError):
            MPFITSVDSolver().solve(
                design_matrix,
                reference_values,
                ancillary_arrays={"quse_masks": quse_masks},
            )

    def test_solve_no_quse(self):
        """Test that missing quse_masks raises MPFITSolverError."""
        site_A = np.array([[1.0, 2.0], [3.0, 4.0]])
        site_b = np.array([1.0, 2.0])

        design_matrix = np.empty(1, dtype=object)
        design_matrix[0] = site_A

        reference_values = np.empty(1, dtype=object)
        reference_values[0] = site_b

        with pytest.raises(MPFITSolverError, match="quse_masks"):
            MPFITSVDSolver().solve(
                design_matrix,
                reference_values,
                ancillary_arrays=None,
            )
