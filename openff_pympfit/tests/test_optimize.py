import numpy as np
import pytest
from openff.toolkit import Molecule
from openff.units import unit

from openff_pympfit import GDMASettings, MoleculeGDMARecord
from openff_pympfit.optimize import MPFITObjective, MPFITObjectiveTerm


def _make_gdma_settings(mpfit_atom_radius: float) -> GDMASettings:
    """Create GDMA settings with specified atom radius (to control quse_mask)."""
    return GDMASettings(
        limit=4,
        mpfit_atom_radius=mpfit_atom_radius,
        mpfit_inner_radius=0.001,
        mpfit_outer_radius=10.0,
    )


def _make_hcl_record(mpfit_atom_radius: float) -> MoleculeGDMARecord:
    molecule = Molecule.from_smiles("[H]Cl")

    conformer = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.27, 0.0, 0.0],
            ]
        )
        * unit.angstrom
    )

    multipoles = np.zeros((2, 25))
    multipoles[0, 0] = 0.18
    multipoles[1, 0] = -0.18

    return MoleculeGDMARecord.from_molecule(
        molecule, conformer, multipoles, _make_gdma_settings(mpfit_atom_radius)
    )


def _make_water_record(mpfit_atom_radius: float) -> MoleculeGDMARecord:
    molecule = Molecule.from_smiles("O")

    conformer = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ]
        )
        * unit.angstrom
    )

    multipoles = np.zeros((3, 25))
    multipoles[0, 0] = -0.8
    multipoles[1, 0] = 0.4
    multipoles[2, 0] = 0.4

    return MoleculeGDMARecord.from_molecule(
        molecule, conformer, multipoles, _make_gdma_settings(mpfit_atom_radius)
    )


# Radius values to control quse_mask
RADIUS_SMALL = 0.5  # Only includes self-site (1x1 matrices)
RADIUS_MODERATE = 2.0  # Includes some neighbors (variable matrix sizes)
RADIUS_LARGE = 100.0  # Includes all atoms (nxn matrixes)


class TestComputeObjectiveTerms:

    @pytest.mark.parametrize(
        "mpfit_atom_radius", [RADIUS_SMALL, RADIUS_MODERATE, RADIUS_LARGE]
    )
    def test_hcl(self, mpfit_atom_radius):
        record = _make_hcl_record(mpfit_atom_radius)
        n_atoms = 2

        objective_terms_generator = MPFITObjective.compute_objective_terms(
            gdma_records=[record],
            return_quse_masks=True,
        )
        objective_terms = list(objective_terms_generator)

        assert len(objective_terms) == 1
        term, masks_dict = objective_terms[0]
        assert isinstance(term, MPFITObjectiveTerm)

        a_mat = term.atom_charge_design_matrix
        assert a_mat.dtype == np.dtype("O")
        assert len(a_mat) == n_atoms

        b = term.reference_values
        assert b.dtype == np.dtype("O")
        assert len(b) == n_atoms

        quse_masks = masks_dict["quse_masks"]
        assert len(quse_masks) == n_atoms

        # A[i] shape should matche quse_mask[i] count
        for i in range(n_atoms):
            n_charges = np.count_nonzero(quse_masks[i])
            assert a_mat[i].shape == (n_charges, n_charges)
            assert len(b[i]) == n_charges

        if mpfit_atom_radius == RADIUS_LARGE:
            for i in range(n_atoms):
                assert a_mat[i].shape == (n_atoms, n_atoms)
                assert np.all(quse_masks[i])
        elif mpfit_atom_radius in (RADIUS_SMALL, RADIUS_MODERATE):
            for i in range(n_atoms):
                assert a_mat[i].shape == (1, 1)
                assert quse_masks[i][i] is True
                assert np.count_nonzero(quse_masks[i]) == 1

    @pytest.mark.parametrize(
        "mpfit_atom_radius", [RADIUS_SMALL, RADIUS_MODERATE, RADIUS_LARGE]
    )
    def test_water(self, mpfit_atom_radius):
        record = _make_water_record(mpfit_atom_radius)
        n_atoms = 3

        objective_terms_generator = MPFITObjective.compute_objective_terms(
            gdma_records=[record],
            return_quse_masks=True,
        )
        objective_terms = list(objective_terms_generator)

        assert len(objective_terms) == 1
        term, masks_dict = objective_terms[0]

        assert isinstance(term, MPFITObjectiveTerm)

        a_mat = term.atom_charge_design_matrix
        assert a_mat.dtype == np.dtype("O")
        assert len(a_mat) == n_atoms

        b = term.reference_values
        assert b.dtype == np.dtype("O")
        assert len(b) == n_atoms

        quse_masks = masks_dict["quse_masks"]
        assert len(quse_masks) == n_atoms

        for i in range(n_atoms):
            n_charges = np.count_nonzero(quse_masks[i])
            assert a_mat[i].shape == (n_charges, n_charges)
            assert len(b[i]) == n_charges

        if mpfit_atom_radius == RADIUS_LARGE:
            for i in range(n_atoms):
                assert a_mat[i].shape == (n_atoms, n_atoms)
                assert np.all(quse_masks[i])
        elif mpfit_atom_radius == RADIUS_SMALL:
            for i in range(n_atoms):
                assert a_mat[i].shape == (1, 1)
                assert quse_masks[i][i] is True
                assert np.count_nonzero(quse_masks[i]) == 1
        elif mpfit_atom_radius == RADIUS_MODERATE:
            assert a_mat[0].shape == (3, 3)
            assert np.all(quse_masks[0])
            assert a_mat[1].shape == (2, 2)

            assert quse_masks[1][0] is True
            assert quse_masks[1][1] is True
            assert quse_masks[1][2] is False

            assert a_mat[2].shape == (2, 2)
            assert quse_masks[2][0] is True
            assert quse_masks[2][1] is False
            assert quse_masks[2][2] is True

    @pytest.mark.parametrize(
        "mpfit_atom_radius", [RADIUS_SMALL, RADIUS_MODERATE, RADIUS_LARGE]
    )
    def test_methanol(self, make_methanol_record, mpfit_atom_radius):
        record = make_methanol_record(mpfit_atom_radius)
        n_atoms = 6

        objective_terms_generator = MPFITObjective.compute_objective_terms(
            gdma_records=[record],
            return_quse_masks=True,
        )
        objective_terms = list(objective_terms_generator)

        assert len(objective_terms) == 1
        term, masks_dict = objective_terms[0]

        assert isinstance(term, MPFITObjectiveTerm)

        a_mat = term.atom_charge_design_matrix
        assert a_mat.dtype == np.dtype("O")
        assert len(a_mat) == n_atoms

        b = term.reference_values
        assert b.dtype == np.dtype("O")
        assert len(b) == n_atoms

        quse_masks = masks_dict["quse_masks"]
        assert len(quse_masks) == n_atoms

        for i in range(n_atoms):
            n_charges = np.count_nonzero(quse_masks[i])
            assert a_mat[i].shape == (n_charges, n_charges)
            assert len(b[i]) == n_charges

        if mpfit_atom_radius == RADIUS_LARGE:
            for i in range(n_atoms):
                assert a_mat[i].shape == (n_atoms, n_atoms)
                assert np.all(quse_masks[i])
        elif mpfit_atom_radius == RADIUS_SMALL:
            for i in range(n_atoms):
                assert a_mat[i].shape == (1, 1)
                assert quse_masks[i][i] is True
                assert np.count_nonzero(quse_masks[i]) == 1

    def test_without_quse_masks_flag(self):
        """Test that return_quse_masks=False returns only the term."""
        record = _make_hcl_record(RADIUS_LARGE)

        objective_terms_generator = MPFITObjective.compute_objective_terms(
            gdma_records=[record],
            return_quse_masks=False,
        )
        objective_terms = list(objective_terms_generator)

        assert len(objective_terms) == 1
        assert isinstance(objective_terms[0], MPFITObjectiveTerm)
