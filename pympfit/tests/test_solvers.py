import functools

import numpy as np
import pytest

from pympfit.mpfit.solvers import (
    ConstrainedMPFITSolver,
    ConstrainedMPFITState,
    ConstrainedSciPySolver,
    MPFITSolverError,
    MPFITSVDSolver,
    _find_twin,
    build_quse_matrix,
    count_parameters,
    expandcharge,
)


def _make_dummy_state(atomtype=("O1", "H1", "H1"), molecule_charges=(0.0,)):
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    rvdw = np.full(3, 5.0)
    multipoles = np.zeros((3, 1, 1, 2))
    multipoles[0, 0, 0, 0] = -0.8
    multipoles[1, 0, 0, 0] = 0.4
    multipoles[2, 0, 0, 0] = 0.4
    return ConstrainedMPFITState(
        xyzmult=xyz,
        xyzcharge=xyz,
        multipoles=multipoles,
        quse=build_quse_matrix(xyz, xyz, rvdw),
        atomtype=atomtype,
        rvdw=rvdw,
        lmax=np.zeros(3),
        r1=0.5,
        r2=2.0,
        maxl=0,
        atom_counts=(3,),
        molecule_charges=molecule_charges,
    )


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


class TestConstrainedMPFITSolver:

    def test_helpers(self):
        state = _make_dummy_state(atomtype=("O1", "H1", "H1"))
        assert _find_twin(state.atomtype, 0) is None
        assert _find_twin(state.atomtype, 1) is None
        assert _find_twin(state.atomtype, 2) == 1
        assert count_parameters(state) == 8
        _, qstore = expandcharge(np.zeros(8), state)
        assert qstore.shape == (3,)
        assert np.isclose(qstore[1], qstore[2])

    def test_loss(self):
        state = _make_dummy_state()
        n_params = count_parameters(state)
        p0 = np.zeros(n_params)

        loss = ConstrainedMPFITSolver.loss(p0, state, conchg=1.0)
        assert isinstance(loss, float)

        expected_loss = 4.0 * np.pi * 1.5 * (0.8**2 + 0.4**2 + 0.4**2)
        assert np.isclose(loss, expected_loss)

    def test_jacobian(self):
        state = _make_dummy_state()
        n_params = count_parameters(state)
        p0 = np.random.default_rng(42).normal(0, 0.1, n_params)
        conchg = 1.0

        jacobian = ConstrainedMPFITSolver.jacobian(p0, state, conchg)
        assert jacobian.shape == (n_params,)

        loss_func = functools.partial(
            ConstrainedMPFITSolver.loss,
            state=state,
            conchg=conchg,
        )
        h = 0.0001
        expected_jacobian = np.array(
            [
                (
                    loss_func(p0 + h * np.eye(n_params)[k])
                    - loss_func(p0 - h * np.eye(n_params)[k])
                )
                / (2.0 * h)
                for k in range(n_params)
            ]
        )

        assert np.allclose(jacobian, expected_jacobian)


class TestConstrainedSciPySolver:

    def test_solve(self, monkeypatch):
        state = _make_dummy_state(atomtype=("O1", "H1", "H1"))
        n_params = count_parameters(state)
        monkeypatch.setattr(
            ConstrainedSciPySolver,
            "initial_guess",
            lambda _self, _s: np.zeros(n_params),
        )
        qstore = ConstrainedSciPySolver(conchg=10.0).solve(state)
        assert qstore.shape == (3,)
        assert np.isclose(qstore[1], qstore[2], atol=1e-6)
        assert np.isclose(np.sum(qstore), 0.0, atol=1e-4)
