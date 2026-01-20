from typing import TYPE_CHECKING
from collections.abc import Generator

import numpy
from openff.units import unit

from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeGenerator,
)
from openff.recharge.charges.qc import QCChargeGenerator, QCChargeSettings
from openff.recharge.charges.vsite import (
    VirtualSiteChargeKey,
    VirtualSiteCollection,
    VirtualSiteGenerator,
    VirtualSiteGeometryKey,
)
from openff_pympfit.gdma.storage import MoleculeGDMARecord
from openff.recharge.utilities.tensors import (
    TensorType,
    append_zero,
    concatenate,
    to_numpy,
    to_torch,
)

from openff.recharge.optimize._optimize import ObjectiveTerm, Objective

if TYPE_CHECKING:
    from openff.toolkit import Molecule

class MPFITObjectiveTerm(ObjectiveTerm):
    """A class that stores precalculated values used to compute the difference between
    a reference set of distributed multipole moments and a set computed using a set of fixed
    partial charges.

    See the ``predict`` and ``loss`` functions for more details.
    """

    @classmethod
    def _objective(cls) -> type["MPFITObjective"]:
        return MPFITObjective


class MPFITObjective(Objective):
    """A utility class which contains helper functions for computing the
    contributions to a least squares objective function which captures the
    deviation of multipole moments computed using molecular partial charges and the  
    multipole moments from GDMA calculations."""

    @classmethod
    def _objective_term(cls) -> type[MPFITObjectiveTerm]:
        return MPFITObjectiveTerm

    @classmethod
    def _flatten_charges(cls) -> bool:
        return False

    @classmethod
    def _compute_design_matrix_precursor(
        cls, grid_coordinates: numpy.ndarray, conformer: numpy.ndarray
    ):
        """For MPFIT, the design matrix precursor is calculated differently than for ESP or
        electric fields. Instead of grid coordinates, we use the molecular coordinates and
        build a matrix that maps charges to multipole moments.
        
        The implementation needs to construct the A matrix as described in 
        J. Comp. Chem. Vol. 12, No. 8, 913-917 (1991)
        """
        from openff_pympfit.mpfit.core import _regular_solid_harmonic, build_A_matrix

        # For MPFIT, we build the design matrix directly in compute_objective_terms
        # This is a placeholder to satisfy the interface
        return numpy.ones((1, conformer.shape[0]))

    @classmethod
    def _electrostatic_property(cls, record: MoleculeGDMARecord) -> numpy.ndarray:
        from openff_pympfit.mpfit.core import _convert_flat_to_hierarchical
        # Convert flat multipoles to hierarchical format for the solver
        # Determine the number of sites and max rank from the multipoles array
        flat_multipoles = record.multipoles
        num_sites = flat_multipoles.shape[0]
        max_rank = 4  # Default max rank for GDMA

        # Convert from flat to hierarchical format
        return _convert_flat_to_hierarchical(flat_multipoles, num_sites, max_rank)
        
    @classmethod
    def compute_objective_terms(
        cls,
        gdma_records: list[MoleculeGDMARecord],
        charge_collection: None | (QCChargeSettings | LibraryChargeCollection) = None,
        charge_parameter_keys: list[tuple[str, tuple[int, ...]]] | None = None,
        vsite_collection: VirtualSiteCollection | None = None,
        vsite_charge_parameter_keys: list[VirtualSiteChargeKey] | None = None,
        vsite_coordinate_parameter_keys: list[VirtualSiteGeometryKey] | None = None,
        return_quse_masks: bool = False,
    ) -> Generator[tuple[MPFITObjectiveTerm, dict] | MPFITObjectiveTerm, None, None]:
        """Pre-calculates the terms that contribute to the total objective function.

        This is an adaptation of the original compute_objective_terms method for MPFIT,
        which works with multipole moments instead of ESP data.

        For complete documentation, see the original method in the Objective class.
        Note: BCC parameters are not applicable for MPFIT and have been removed.
        """
        from openff_pympfit.mpfit.core import build_A_matrix, build_b_vector, _convert_flat_to_hierarchical
        from openff.toolkit import Molecule

        for gdma_record in gdma_records:
            molecule: Molecule = Molecule.from_mapped_smiles(
                gdma_record.tagged_smiles, allow_undefined_stereo=True
            )
            conformer = gdma_record.conformer
            
            # Get MPFIT settings from the record
            gdma_settings = gdma_record.gdma_settings
            max_rank = gdma_settings.limit
            r1 = gdma_settings.mpfit_inner_radius
            r2 = gdma_settings.mpfit_outer_radius
            default_atom_radius = gdma_settings.mpfit_atom_radius

            # Convert the flat multipoles to hierarchical format
            flat_multipoles = gdma_record.multipoles
            num_sites = flat_multipoles.shape[0]
            multipoles = _convert_flat_to_hierarchical(flat_multipoles, num_sites, max_rank)

            fixed_atom_charges = numpy.zeros((molecule.n_atoms, 1))
            atom_charge_design_matrices = []

            # We'll use the molecular coordinates as our design matrix precursor
            # This is just a placeholder to match the interface
            design_matrix_precursor = cls._compute_design_matrix_precursor(
                None, conformer
            )

            if charge_collection is None:
                pass
            elif isinstance(charge_collection, QCChargeSettings):
                assert (
                    charge_parameter_keys is None
                ), "charges generated using `QCChargeSettings` cannot be trained"

                fixed_atom_charges += QCChargeGenerator.generate(
                    molecule, [conformer * unit.angstrom], charge_collection
                )

            elif isinstance(charge_collection, LibraryChargeCollection):
                (
                    library_assignment_matrix,
                    library_fixed_charges,
                ) = cls._compute_library_charge_terms(
                    molecule,
                    charge_collection,
                    charge_parameter_keys,
                )

                fixed_atom_charges += library_fixed_charges

                # Convert conformer from Angstroms to Bohrs once
                bohr_conformer = unit.convert(conformer, unit.angstrom, unit.bohr)
                
                # Create rvdw array using the configured default atom radius
                rvdw = numpy.full(molecule.n_atoms, default_atom_radius)
                
                # Build the A matrix that maps charges to multipole moments
                A_matrix = numpy.zeros((molecule.n_atoms, library_assignment_matrix.shape[1]))
                # Compute the reference values (b vector)
                if cls._flatten_charges():
                    fixed_atom_charges = fixed_atom_charges.flatten()

                # Prepare the reference values and quse_masks
                reference_values = []
                quse_masks = []

                # Process each atom site
                for i in range(molecule.n_atoms):
                    # Calculate distances from current multipole site to all atoms
                    rqm = numpy.linalg.norm(bohr_conformer[i] - bohr_conformer, axis=1)
                    # Create mask for atoms within rvdw
                    quse_mask = rqm < rvdw[i]

                    # Store the mask for later use by the solver
                    quse_masks.append(quse_mask)

                    qsites = numpy.count_nonzero(quse_mask)

                    # Build the A matrix for this site's multipoles
                    site_A = numpy.zeros((qsites, qsites))
                    site_b = numpy.zeros(qsites)

                    # Apply the mask to get charge positions to use
                    masked_charge_conformer = bohr_conformer[quse_mask]

                    # If no charges are within range, use all charges
                    if masked_charge_conformer.shape[0] == 0:
                        masked_charge_conformer = bohr_conformer
                        # Update the mask to include all atoms
                        quse_masks[-1] = numpy.ones(molecule.n_atoms, dtype=bool)
                    
                    # Use the multipole site coordinates and masked charge coordinates
                    site_A = build_A_matrix(i, bohr_conformer, masked_charge_conformer, r1, r2, max_rank, site_A)
                    site_b = build_b_vector(i, bohr_conformer, masked_charge_conformer, r1, r2, max_rank, multipoles, site_b)
                
                    atom_charge_design_matrices.append(site_A)
                    reference_values.append(site_b)
            else:
                raise NotImplementedError()

            # We don't currently support virtual sites for MPFIT
            if vsite_collection is not None:
                raise NotImplementedError("Virtual sites are not supported for MPFIT")

            atom_charge_design_matrix = numpy.array(atom_charge_design_matrices, dtype=object)
            reference_values = numpy.array(reference_values, dtype=object)
            quse_masks = numpy.array(quse_masks, dtype=object)


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
                yield objective_term, {'quse_masks': quse_masks}
            else:
                yield objective_term
