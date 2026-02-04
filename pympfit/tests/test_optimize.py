import numpy as np
import pytest
from openff.toolkit import Molecule
from openff.units import unit

from pympfit import GDMASettings, MoleculeGDMARecord
from pympfit.optimize import MPFITObjective, MPFITObjectiveTerm


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


torch = pytest.importorskip("torch")


class TestPredict:

    @pytest.mark.parametrize(
        "distance1,distance2",
        [
            (0.3, 0.7),
            (0.5, 1.5),
            (0.7, 2.7),
            (1.0, 5.0),
        ],
    )
    def test_predict_shape_consistency(self, meoh_gdma_sto3g, distance1, distance2):
        """Test that predict() returns consistent shapes across different distances."""
        from openff.recharge.charges.vsite import (
            BondChargeSiteParameter,
            VirtualSiteCollection,
        )

        vsite_collection = VirtualSiteCollection(
            parameters=[
                BondChargeSiteParameter(
                    smirks="[#6:1]-[#8:2]",
                    name="EP",
                    distance=0.5,
                    charge_increments=(0.0, 0.0),
                    sigma=0.0,
                    epsilon=0.0,
                    match="all-permutations",
                )
            ]
        )
        vsite_coord_keys = [("[#6:1]-[#8:2]", "BondCharge", "EP", "distance")]

        objective_term, metadata = next(
            MPFITObjective.compute_objective_terms(
                [meoh_gdma_sto3g],
                vsite_collection=vsite_collection,
                _vsite_coordinate_parameter_keys=vsite_coord_keys,
                return_quse_masks=True,
            )
        )

        n_atoms = 6
        charge_params = torch.ones((n_atoms, 1), dtype=torch.float64) * 0.1

        pred1 = objective_term.predict(
            charge_params, torch.tensor([[distance1]], dtype=torch.float64)
        )
        pred2 = objective_term.predict(
            charge_params, torch.tensor([[distance2]], dtype=torch.float64)
        )

        # Number of predictions must match reference_values for loss() to work
        assert len(pred1) == len(pred2), (
            f"Lengths differ: {len(pred1)} vs {len(pred2)} "
            f"at distances {distance1}, {distance2}"
        )
        assert len(pred1) == objective_term.reference_values.shape[0]

        # Inner arrays may have different sizes due to quse_mask changes
        # Just verify predictions are not identical (content differs)
        predictions_differ = False
        for p1, p2 in zip(pred1, pred2, strict=False):
            p1_arr = p1.detach().numpy().flatten()
            p2_arr = p2.detach().numpy().flatten()
            # Compare sums as a simple difference check (handles different sizes)
            if not np.isclose(p1_arr.sum(), p2_arr.sum()):
                predictions_differ = True

        assert predictions_differ, "Predictions should differ at different distances"

    @pytest.mark.parametrize(
        "vsite_increments",
        [
            (0.0, 0.0),
            (0.1, 0.1),
            (0.2, -0.1),
            (-0.15, 0.25),
        ],
    )
    def test_predict_charge_conservation(self, meoh_gdma_sto3g, vsite_increments):
        """Test that charge conservation is maintained in predict().

        The vsite_charge_assignment_matrix encodes charge redistribution:
        when a vsite has charge, atoms bonded to it are adjusted to conserve
        total molecular charge.
        """
        from openff.recharge.charges.vsite import (
            BondChargeSiteParameter,
            VirtualSiteCollection,
        )

        vsite_collection = VirtualSiteCollection(
            parameters=[
                BondChargeSiteParameter(
                    smirks="[#6:1]-[#8:2]",
                    name="EP",
                    distance=0.5,
                    charge_increments=(0.1, 0.1),
                    sigma=0.0,
                    epsilon=0.0,
                    match="all-permutations",
                )
            ]
        )

        vsite_charge_keys = [
            ("[#6:1]-[#8:2]", "BondCharge", "EP", 0),
            ("[#6:1]-[#8:2]", "BondCharge", "EP", 1),
        ]
        vsite_coord_keys = [("[#6:1]-[#8:2]", "BondCharge", "EP", "distance")]

        objective_term, metadata = next(
            MPFITObjective.compute_objective_terms(
                [meoh_gdma_sto3g],
                vsite_collection=vsite_collection,
                _vsite_charge_parameter_keys=vsite_charge_keys,
                _vsite_coordinate_parameter_keys=vsite_coord_keys,
                return_quse_masks=True,
            )
        )

        # verify assignment matrix structure (columns sum to 0)
        assignment_matrix = objective_term.vsite_charge_assignment_matrix
        column_sums = assignment_matrix.sum(axis=0)
        assert np.allclose(
            column_sums, 0.0
        ), f"Columns should sum to 0, got {column_sums}"

        # verify fixed charges sum to 0 (required for conservation)
        fixed_charges_sum = objective_term.vsite_fixed_charges.sum()
        assert np.isclose(
            fixed_charges_sum, 0.0, atol=1e-10
        ), f"Fixed charges should sum to 0, got {fixed_charges_sum}"

        # verify actual charges sum to formal charge
        n_atoms = 6
        atom_charges = np.array([[0.1], [-0.2], [0.05], [0.05], [0.0], [0.0]])
        trainable_vsite = np.array([[vsite_increments[0]], [vsite_increments[1]]])

        # charge adjustment
        charge_adjustment = (
            assignment_matrix @ trainable_vsite + objective_term.vsite_fixed_charges
        )
        atom_adjustment = charge_adjustment[:n_atoms]
        vsite_charges = charge_adjustment[n_atoms:]

        final_atom_charges = atom_charges + atom_adjustment
        all_charges = np.vstack([final_atom_charges, vsite_charges])
        total_charge = all_charges.sum()

        assert np.isclose(
            total_charge, 0.0, atol=1e-10
        ), f"Total charge should be 0 (formal charge), got {total_charge}"

    @pytest.mark.parametrize(
        "molecule_name,smirks,n_atoms,n_vsites",
        [
            ("methanol", "[#6:1]-[#8:2]", 6, 1),  # BondCharge on C-O
            ("water", "[#8:1]-[#1:2]", 3, 2),  # BondCharge on O-H (2 matches)
        ],
    )
    def test_predict_molecules(
        self, molecule_name, smirks, n_atoms, n_vsites, meoh_gdma_sto3g
    ):
        """Parametrized test for predict() across different molecules.

        Tests both methanol (C-O bond) and water (O-H bonds) following
        the openff-recharge vsite test patterns.
        """
        from openff.recharge.charges.vsite import (
            BondChargeSiteParameter,
            VirtualSiteCollection,
        )

        if molecule_name == "methanol":
            gdma_record = meoh_gdma_sto3g
        elif molecule_name == "water":
            gdma_record = _make_water_record(RADIUS_LARGE)
        else:
            raise ValueError(f"Unknown molecule: {molecule_name}")

        vsite_collection = VirtualSiteCollection(
            parameters=[
                BondChargeSiteParameter(
                    smirks=smirks,
                    name="EP",
                    distance=0.5,
                    charge_increments=(0.1, 0.1),
                    sigma=0.0,
                    epsilon=0.0,
                    match="all-permutations",
                )
            ]
        )
        vsite_coord_keys = [(smirks, "BondCharge", "EP", "distance")]

        objective_term, metadata = next(
            MPFITObjective.compute_objective_terms(
                [gdma_record],
                vsite_collection=vsite_collection,
                _vsite_coordinate_parameter_keys=vsite_coord_keys,
                return_quse_masks=True,
            )
        )

        assert metadata["n_vsites"] == n_vsites
        assert objective_term.reference_values.shape[0] == n_atoms

        charge_params = torch.ones((n_atoms, 1), dtype=torch.float64) * 0.1
        pred = objective_term.predict(
            charge_params, torch.tensor([[0.5]] * n_vsites, dtype=torch.float64)
        )

        assert len(pred) == n_atoms

        # Test predictions change with different distances
        pred2 = objective_term.predict(
            charge_params, torch.tensor([[1.0]] * n_vsites, dtype=torch.float64)
        )

        predictions_differ = any(
            not np.isclose(p1.detach().numpy().sum(), p2.detach().numpy().sum())
            for p1, p2 in zip(pred, pred2, strict=False)
        )
        assert predictions_differ, "Predictions should differ at different distances"
