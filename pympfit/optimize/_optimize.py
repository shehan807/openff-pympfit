from collections.abc import Generator

import numpy as np
from openff.recharge.charges.vsite import (
    VirtualSiteChargeKey,
    VirtualSiteCollection,
    VirtualSiteGeometryKey,
)
from openff.recharge.optimize._optimize import Objective, ObjectiveTerm
from openff.units import unit

from pympfit.gdma.storage import MoleculeGDMARecord


class MPFITObjectiveTerm(ObjectiveTerm):
    """Store precalculated values for multipole moment fitting.

    Computes the difference between a reference set of distributed multipole
    moments and a set computed using fixed partial charges.

    Attributes
    ----------
    gdma_record : MoleculeGDMARecord | None
        Reference to the source GDMA record. 
    quse_masks : np.ndarray | None
        Boolean masks indicating which charges are included for each multipole
        site. 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gdma_record: MoleculeGDMARecord | None = None
        self.quse_masks: np.ndarray | None = None

    @classmethod
    def _objective(cls) -> type["MPFITObjective"]:
        return MPFITObjective

    def predict(self, charge_parameters, vsite_coordinate_parameters=None):
        """Predict multipole moment contributions for given charges and vsite positions.

        This method is designed for Bayesian inference and rebuilds A matrices
        with new vsite positions, computing A @ q for each multipole site.

        Parameters
        ----------
        charge_parameters : torch.Tensor
            Charge values with shape (n_atom_charges + n_vsite_charges, 1).
        vsite_coordinate_parameters : torch.Tensor, optional
            Virtual site local frame coordinates (distance, angles) being sampled.
            Shape (n_trainable_coords, 1).

        Returns
        -------
        list[torch.Tensor]
            Predicted multipole contributions for each atom site, matching the
            shape of ``reference_values``.

        Raises
        ------
        ValueError
            If called on a term without virtual sites (use SVD solver instead).
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "predict() requires PyTorch for differentiable inference. "
                "Install with: pip install torch sphericart[torch]"
            ) from None

        from openff.recharge.charges.vsite import VirtualSiteGenerator
        from openff.recharge.utilities.tensors import append_zero

        from pympfit.mpfit.core_torch import build_A_matrix_torch

        if self.vsite_local_coordinate_frame is None:
            raise ValueError(
                "predict() requires virtual sites. For atom-only charge fitting, "
                "use the SVD solver directly via generate_mpfit_charge_parameter()."
            )

        n_vsites = self.vsite_local_coordinate_frame.shape[1]
        if n_vsites == 0:
            raise ValueError(
                "predict() requires virtual sites. For atom-only charge fitting, "
                "use the SVD solver directly via generate_mpfit_charge_parameter()."
            )

        settings = self.gdma_record.gdma_settings
        r1 = settings.mpfit_inner_radius
        r2 = settings.mpfit_outer_radius
        max_rank = settings.limit
        bohr_conformer_np = unit.convert(
            self.gdma_record.conformer, unit.angstrom, unit.bohr
        )
        bohr_conformer = torch.from_numpy(bohr_conformer_np)
        n_atoms = bohr_conformer.shape[0]

        # compute new vsite Cartesian positions from local frame 
        trainable = append_zero(vsite_coordinate_parameters.flatten())[
            self.vsite_coord_assignment_matrix
        ]
        vsite_fixed_coords_t = torch.from_numpy(self.vsite_fixed_coords)
        vsite_local_coords = vsite_fixed_coords_t + trainable

        # Convert local coords (distance, angles) to Cartesian positions
        vsite_local_frame_t = torch.from_numpy(self.vsite_local_coordinate_frame)
        vsite_coords_angstrom = VirtualSiteGenerator.convert_local_coordinates(
            vsite_local_coords, vsite_local_frame_t, backend="torch"
        )
        angstrom_to_bohr = unit.convert(1.0, unit.angstrom, unit.bohr)
        vsite_coords_bohr = vsite_coords_angstrom * angstrom_to_bohr

        augmented_coords = torch.cat([bohr_conformer, vsite_coords_bohr], dim=0)

        n_trainable_vsite_charges = self.vsite_charge_assignment_matrix.shape[1]
        atom_charges = charge_parameters[:n_atoms]

        if n_trainable_vsite_charges > 0:
            # redistribute vsite charge increments to parent atoms
            trainable_vsite_charges = charge_parameters[n_atoms:]
            vsite_charge_matrix_t = torch.from_numpy(self.vsite_charge_assignment_matrix)
            vsite_fixed_charges_t = torch.from_numpy(self.vsite_fixed_charges)
            charge_adjustment = (
                vsite_charge_matrix_t @ trainable_vsite_charges
                + vsite_fixed_charges_t
            )
        else:
            charge_adjustment = torch.from_numpy(self.vsite_fixed_charges)

        atom_adjustment = charge_adjustment[:n_atoms]
        vsite_charges = charge_adjustment[n_atoms:]

        final_atom_charges = atom_charges + atom_adjustment
        all_charges = torch.cat([final_atom_charges, vsite_charges], dim=0)

        predictions = []
        for i in range(n_atoms):
            # Use stored quse_mask to ensure consistent shape with reference_values
            quse_mask = torch.from_numpy(self.quse_masks[i].astype(bool))

            masked_coords = augmented_coords[quse_mask]
            masked_charges = all_charges[quse_mask]

            site_A = build_A_matrix_torch(
                i, bohr_conformer, masked_coords, r1, r2, max_rank
            )

            site_pred = site_A @ masked_charges
            predictions.append(site_pred)

        return predictions


class MPFITObjective(Objective):
    """Compute contributions to the MPFIT least squares objective function.

    Contains helper functions for capturing the deviation of multipole moments
    computed using molecular partial charges from GDMA calculations.
    """

    @classmethod
    def _objective_term(cls) -> type[MPFITObjectiveTerm]:
        return MPFITObjectiveTerm

    @classmethod
    def extract_arrays(
        cls,
        gdma_record: MoleculeGDMARecord,
    ) -> dict:
        """Extract numerical arrays from a single GDMA record."""
        from openff.toolkit import Molecule

        from pympfit.mpfit.core import _convert_flat_to_hierarchical

        molecule = Molecule.from_mapped_smiles(
            gdma_record.tagged_smiles, allow_undefined_stereo=True
        )
        settings = gdma_record.gdma_settings
        bohr_conformer = unit.convert(gdma_record.conformer, unit.angstrom, unit.bohr)
        multipoles = _convert_flat_to_hierarchical(
            gdma_record.multipoles, molecule.n_atoms, settings.limit
        )
        return {
            "bohr_conformer": bohr_conformer,
            "multipoles": multipoles,
            "rvdw": np.full(molecule.n_atoms, settings.mpfit_atom_radius),
            "lmax": np.full(molecule.n_atoms, settings.limit, dtype=float),
            "r1": settings.mpfit_inner_radius,
            "r2": settings.mpfit_outer_radius,
            "maxl": settings.limit,
            "n_atoms": molecule.n_atoms,
        }

    @classmethod
    def compute_objective_terms(
        cls,
        gdma_records: list[MoleculeGDMARecord],
        vsite_collection: VirtualSiteCollection | None = None,
        _vsite_charge_parameter_keys: list[VirtualSiteChargeKey] | None = None,
        _vsite_coordinate_parameter_keys: list[VirtualSiteGeometryKey] | None = None,
        return_quse_masks: bool = False,
    ) -> Generator[tuple[MPFITObjectiveTerm, dict] | MPFITObjectiveTerm, None, None]:
        """Pre-calculates the terms that contribute to the total objective function."""
        from pympfit.mpfit.core import build_A_matrix, build_b_vector

        for gdma_record in gdma_records:
            arrays = cls.extract_arrays(gdma_record)
            bohr_conformer = arrays["bohr_conformer"]
            multipoles = arrays["multipoles"]
            rvdw = arrays["rvdw"]
            r1 = arrays["r1"]
            r2 = arrays["r2"]
            max_rank = arrays["maxl"]
            n_atoms = arrays["n_atoms"]

            if vsite_collection is not None:
                from openff.recharge.charges.vsite import VirtualSiteGenerator
                from openff.toolkit import Molecule

                molecule = Molecule.from_mapped_smiles(
                    gdma_record.tagged_smiles, allow_undefined_stereo=True
                )
                conformer_angstrom = gdma_record.conformer

                vsite_positions = VirtualSiteGenerator.generate_positions(
                    molecule, vsite_collection, conformer_angstrom * unit.angstrom
                )
                vsite_positions_bohr = unit.convert(
                    vsite_positions.m_as(unit.angstrom), unit.angstrom, unit.bohr
                )
                n_vsites = vsite_positions_bohr.shape[0]

                (vsite_charge_assignment_matrix, vsite_fixed_charges) = (
                    cls._compute_vsite_charge_terms(
                        molecule, vsite_collection, _vsite_charge_parameter_keys or []
                    )
                )
                (vsite_coord_assignment_matrix, vsite_fixed_coords,
                 vsite_local_coordinate_frame) = (
                    cls._compute_vsite_coord_terms(
                        molecule, conformer_angstrom, vsite_collection,
                        _vsite_coordinate_parameter_keys or []
                    )
                )

                augmented_coords_bohr = np.vstack([bohr_conformer, vsite_positions_bohr])
                rvdw = np.concatenate([rvdw, np.full(n_vsites, arrays["r1"])])
            else:
                n_vsites = 0
                augmented_coords_bohr = bohr_conformer
                vsite_charge_assignment_matrix = None
                vsite_fixed_charges = None
                vsite_coord_assignment_matrix = None
                vsite_fixed_coords = None
                vsite_local_coordinate_frame = None

            atom_charge_design_matrices = []

            reference_values = []
            quse_masks = []

            for i in range(n_atoms):
                rqm = np.linalg.norm(augmented_coords_bohr - bohr_conformer[i], axis=1)
                quse_mask = rqm < rvdw[i]

                quse_masks.append(quse_mask)

                qsites = np.count_nonzero(quse_mask)

                site_A = np.zeros((qsites, qsites))
                site_b = np.zeros(qsites)

                masked_charge_conformer = augmented_coords_bohr[quse_mask]

                if masked_charge_conformer.shape[0] == 0:
                    masked_charge_conformer = augmented_coords_bohr
                    quse_masks[-1] = np.ones(n_atoms + n_vsites, dtype=bool)

                site_A = build_A_matrix(
                    i,
                    bohr_conformer,
                    masked_charge_conformer,
                    r1,
                    r2,
                    max_rank,
                    site_A,
                )
                site_b = build_b_vector(
                    i,
                    bohr_conformer,
                    masked_charge_conformer,
                    r1,
                    r2,
                    max_rank,
                    multipoles,
                    site_b,
                )

                atom_charge_design_matrices.append(site_A)
                reference_values.append(site_b)

            atom_charge_design_matrix = np.array(
                atom_charge_design_matrices, dtype=object
            )
            reference_values = np.array(reference_values, dtype=object)
            quse_masks = np.array(quse_masks, dtype=object)

            # Base class assertion requires all vsite fields to be non-None
            _GRID_PLACEHOLDER = np.empty((0, 3)) if n_vsites > 0 else None

            objective_term = cls._objective_term()(
                atom_charge_design_matrix,
                vsite_charge_assignment_matrix,
                vsite_fixed_charges,
                vsite_coord_assignment_matrix,
                vsite_fixed_coords,
                vsite_local_coordinate_frame,
                _GRID_PLACEHOLDER,
                reference_values,
            )

            objective_term.gdma_record = gdma_record
            objective_term.quse_masks = quse_masks

            if return_quse_masks:
                yield objective_term, {"quse_masks": quse_masks, "n_vsites": n_vsites}
            else:
                yield objective_term
