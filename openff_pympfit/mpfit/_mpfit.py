import warnings
from typing import TYPE_CHECKING

import numpy as np
from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeParameter,
)
from openff.recharge.utilities.toolkits import (
    get_atom_symmetries,
    molecule_to_tagged_smiles,
)
from openff.toolkit.utils.exceptions import AtomMappingWarning
from openff.units import unit

from openff_pympfit.gdma.storage import MoleculeGDMARecord
from openff_pympfit.mpfit.solvers import MPFITSolver
from openff_pympfit.optimize import MPFITObjective

if TYPE_CHECKING:
    from openff.toolkit import Molecule


def _generate_dummy_values(smiles: str) -> list[float]:
    """Generate a list of dummy values for a ``LibraryChargeParameter``.

    The values sum to the correct total charge.
    """
    from openff.toolkit import Molecule

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AtomMappingWarning)
        molecule: Molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

    total_charge = molecule.total_charge.m_as(unit.elementary_charge)
    per_atom_charge = total_charge / molecule.n_atoms

    n_values = len(set(molecule.properties["atom_map"].values()))
    return [per_atom_charge] * n_values


def molecule_to_mpfit_library_charge(
    molecule: "Molecule",
    symmetrize_hydrogens: bool = False,
    symmetrize_other_atoms: bool = False,
) -> LibraryChargeParameter:
    """Create a library charge parameter from a molecule with equivalence groups.

    Each atom is assigned a map index representing which equivalence group it is in.

    Parameters
    ----------
    molecule
        The molecule to create the SMILES pattern from.
    symmetrize_hydrogens
        Whether all topologically symmetric hydrogens in the molecule should be
        assigned an equivalent charge.
    symmetrize_other_atoms
        Whether all topologically symmetric heavy atoms in the molecule should be
        assigned an equivalent charge.

    Returns
    -------
        The library charge with atoms assigned their correct symmetry groups.
    """
    atom_symmetries = get_atom_symmetries(molecule)
    max_symmetry_group = max(atom_symmetries) + 1

    hydrogen_atoms = [
        i for i, atom in enumerate(molecule.atoms) if atom.atomic_number == 1
    ]
    other_atoms = [
        i for i, atom in enumerate(molecule.atoms) if atom.atomic_number != 1
    ]

    atoms_not_to_symmetrize = ([] if symmetrize_hydrogens else hydrogen_atoms) + (
        [] if symmetrize_other_atoms else other_atoms
    )

    for index in atoms_not_to_symmetrize:
        atom_symmetries[index] = max_symmetry_group
        max_symmetry_group += 1

    symmetry_groups = sorted(set(atom_symmetries))

    atom_indices = [symmetry_groups.index(group) + 1 for group in atom_symmetries]
    tagged_smiles = molecule_to_tagged_smiles(molecule, atom_indices)

    return LibraryChargeParameter(
        smiles=tagged_smiles,
        value=_generate_dummy_values(tagged_smiles),
        provenance={
            "hydrogen-indices": sorted({atom_indices[i] - 1 for i in hydrogen_atoms}),
            "other-indices": sorted({atom_indices[i] - 1 for i in other_atoms}),
        },
    )


def generate_mpfit_charge_parameter(
    gdma_records: list[MoleculeGDMARecord], solver: MPFITSolver | None
) -> LibraryChargeParameter:
    """Generate point charges that reproduce the distributed multipole analysis data.

    Parameters
    ----------
    gdma_records
        The records containing the distributed multipole data.
    solver
        The solver to use when finding the charges that minimize the MPFIT loss
        function. By default, the SVD solver is used.

    Returns
    -------
        The charges generated for the molecule.
    """
    from openff.toolkit import Molecule

    from openff_pympfit.mpfit.solvers import MPFITSVDSolver

    solver = MPFITSVDSolver() if solver is None else solver

    # Ensure all records are for the same molecule
    unique_smiles = {
        Molecule.from_mapped_smiles(
            record.tagged_smiles, allow_undefined_stereo=True
        ).to_smiles(mapped=False)
        for record in gdma_records
    }
    if len(unique_smiles) != 1:
        msg = "all GDMA records must be generated for the same molecule"
        raise ValueError(msg)

    molecule = Molecule.from_smiles(
        next(iter(unique_smiles)), allow_undefined_stereo=True
    )

    # Create the charge parameter
    mpfit_parameter = molecule_to_mpfit_library_charge(
        molecule,
        symmetrize_hydrogens=False,
        symmetrize_other_atoms=False,
    )

    # Generate the design matrices and reference data
    objective_terms_and_masks = list(
        MPFITObjective.compute_objective_terms(
            gdma_records,
            charge_collection=LibraryChargeCollection(parameters=[mpfit_parameter]),
            charge_parameter_keys=[
                (mpfit_parameter.smiles, tuple(range(len(mpfit_parameter.value))))
            ],
            return_quse_masks=True,
        )
    )

    # Separate objective terms and quse_masks
    objective_terms = [term for term, _ in objective_terms_and_masks]
    quse_masks_list = [mask["quse_masks"] for _, mask in objective_terms_and_masks]

    # Validate that quse_masks are identical across all conformers
    # quse_mask depends on interatomic distances, which can vary across conformers
    # If masks differ, per-site stacking would combine incompatible subproblems
    if len(quse_masks_list) > 1:
        reference_masks = quse_masks_list[0]
        for conf_idx, masks in enumerate(quse_masks_list[1:], start=2):
            for site_idx, (ref_mask, mask) in enumerate(zip(reference_masks, masks)):
                if not np.array_equal(ref_mask, mask):
                    msg = (
                        f"quse_mask mismatch at site {site_idx} between conformer 1 "
                        f"and conformer {conf_idx}. This can occur when conformers have "
                        f"significantly different geometries. Consider using conformers "
                        f"with more similar structures or increasing mpfit_atom_radius."
                    )
                    raise ValueError(msg)

    # Combine at per-site level: stack each site's matrices across conformers
    # This properly handles multi-conformer fitting by solving a single
    # least-squares problem per site with data from all conformers
    n_sites = len(objective_terms[0].atom_charge_design_matrix)
    combined_design_matrices = []
    combined_reference_values = []

    for site_idx in range(n_sites):
        site_As = [term.atom_charge_design_matrix[site_idx] for term in objective_terms]
        site_bs = [term.reference_values[site_idx] for term in objective_terms]
        combined_design_matrices.append(np.vstack(site_As))
        combined_reference_values.append(np.concatenate(site_bs))

    # Solve the fitting problem
    mpfit_charges = solver.solve(
        np.array(combined_design_matrices, dtype=object),
        np.array(combined_reference_values, dtype=object),
        ancillary_arrays={"quse_masks": quse_masks_list[0]},
    )

    mpfit_parameter.value = mpfit_charges.flatten().tolist()

    return mpfit_parameter
