from collections.abc import Generator
from typing import Union

import numpy as np
from openff.recharge.charges.vsite import (
    VirtualSiteChargeKey,
    VirtualSiteCollection,
    VirtualSiteGeometryKey,
)
from openff.recharge.optimize._optimize import Objective, ObjectiveTerm
from openff.units import unit

from openff_pympfit.gdma.storage import MoleculeGDMARecord
from openff_pympfit.mbis.storage import MoleculeMBISRecord

# Type alias for records that can be used with MPFIT
MultipoleRecord = Union[MoleculeGDMARecord, MoleculeMBISRecord]


class MPFITObjectiveTerm(ObjectiveTerm):
    """Store precalculated values for multipole moment fitting.

    Computes the difference between a reference set of distributed multipole
    moments and a set computed using fixed partial charges.
    See the ``predict`` and ``loss`` functions for more details.
    """

    @classmethod
    def _objective(cls) -> type["MPFITObjective"]:
        return MPFITObjective


def _get_settings_from_record(record: MultipoleRecord):
    """Extract settings from either GDMA or MBIS record.

    Returns a tuple of (max_rank, r1, r2, default_atom_radius).
    """
    if isinstance(record, MoleculeGDMARecord):
        settings = record.gdma_settings
        return (
            settings.limit,
            settings.mpfit_inner_radius,
            settings.mpfit_outer_radius,
            settings.mpfit_atom_radius,
        )
    elif isinstance(record, MoleculeMBISRecord):
        settings = record.mbis_settings
        return (
            settings.limit,
            settings.mpfit_inner_radius,
            settings.mpfit_outer_radius,
            settings.mpfit_atom_radius,
        )
    else:
        raise TypeError(f"Unknown record type: {type(record)}")


class MPFITObjective(Objective):
    """Compute contributions to the MPFIT least squares objective function.

    Contains helper functions for capturing the deviation of multipole moments
    computed using molecular partial charges from GDMA calculations.
    """

    @classmethod
    def _objective_term(cls) -> type[MPFITObjectiveTerm]:
        return MPFITObjectiveTerm

    @classmethod
    def compute_objective_terms(
        cls,
        records: list[MultipoleRecord],
        vsite_collection: VirtualSiteCollection | None = None,
        _vsite_charge_parameter_keys: list[VirtualSiteChargeKey] | None = None,
        _vsite_coordinate_parameter_keys: list[VirtualSiteGeometryKey] | None = None,
        return_quse_masks: bool = False,
    ) -> Generator[tuple[MPFITObjectiveTerm, dict] | MPFITObjectiveTerm, None, None]:
        """Pre-calculates the terms that contribute to the total objective function.

        This is an adaptation of the original compute_objective_terms method for MPFIT,
        which works with multipole moments instead of ESP data.

        Parameters
        ----------
        records
            The multipole records (either GDMA or MBIS) to compute objective terms for.

        For complete documentation, see the original method in the Objective class.
        Note: BCC parameters are not applicable for MPFIT and have been removed.
        """
        from openff.toolkit import Molecule

        from openff_pympfit.mpfit.core import (
            _convert_flat_to_hierarchical,
            build_A_matrix,
            build_b_vector,
        )

        for record in records:
            molecule: Molecule = Molecule.from_mapped_smiles(
                record.tagged_smiles, allow_undefined_stereo=True
            )
            conformer = record.conformer

            # Get MPFIT settings from the record (works for both GDMA and MBIS)
            max_rank, r1, r2, default_atom_radius = _get_settings_from_record(record)

            # Convert the flat multipoles to hierarchical format
            flat_multipoles = record.multipoles
            num_sites = flat_multipoles.shape[0]
            multipoles = _convert_flat_to_hierarchical(
                flat_multipoles, num_sites, max_rank
            )

            atom_charge_design_matrices = []

            # Convert conformer from Angstroms to Bohrs
            bohr_conformer = unit.convert(conformer, unit.angstrom, unit.bohr)

            # Create rvdw array using the configured default atom radius
            rvdw = np.full(molecule.n_atoms, default_atom_radius)

            # Prepare the reference values and quse_masks
            reference_values = []
            quse_masks = []

            # Process each atom site
            for i in range(molecule.n_atoms):
                # Calculate distances from current multipole site to all atoms
                rqm = np.linalg.norm(bohr_conformer[i] - bohr_conformer, axis=1)
                # Create mask for atoms within rvdw
                quse_mask = rqm < rvdw[i]

                # Store the mask for later use by the solver
                quse_masks.append(quse_mask)

                qsites = np.count_nonzero(quse_mask)

                # Build the A matrix for this site's multipoles
                site_A = np.zeros((qsites, qsites))
                site_b = np.zeros(qsites)

                # Apply the mask to get charge positions to use
                masked_charge_conformer = bohr_conformer[quse_mask]

                # If no charges are within range, use all charges
                if masked_charge_conformer.shape[0] == 0:
                    masked_charge_conformer = bohr_conformer
                    # Update the mask to include all atoms
                    quse_masks[-1] = np.ones(molecule.n_atoms, dtype=bool)

                # Use the multipole site coordinates and masked charge coordinates
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

            # We don't currently support virtual sites for MPFIT
            if vsite_collection is not None:
                raise NotImplementedError("Virtual sites are not supported for MPFIT")

            atom_charge_design_matrix = np.array(
                atom_charge_design_matrices, dtype=object
            )
            reference_values = np.array(reference_values, dtype=object)
            quse_masks = np.array(quse_masks, dtype=object)

            objective_term = cls._objective_term()(
                atom_charge_design_matrix,
                None,  # vsite_charge_assignment_matrix
                None,  # vsite_fixed_charges
                None,  # vsite_coord_assignment_matrix
                None,  # vsite_fixed_coords
                None,  # vsite_local_coordinate_frame
                None,  # grid_coordinates not needed for MPFIT
                reference_values,
            )

            if return_quse_masks:
                # Return the quse_masks along with the objective term
                yield objective_term, {"quse_masks": quse_masks}
            else:
                yield objective_term
