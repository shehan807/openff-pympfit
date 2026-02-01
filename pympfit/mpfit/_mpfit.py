import warnings
from typing import TYPE_CHECKING

import numpy as np
from openff.recharge.charges.library import LibraryChargeParameter
from openff.recharge.utilities.toolkits import molecule_to_tagged_smiles
from openff.toolkit.utils.exceptions import AtomMappingWarning
from openff.units import unit

from pympfit.gdma.storage import MoleculeGDMARecord
from pympfit.mpfit.solvers import MPFITSolver
from pympfit.optimize import MPFITObjective

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


def molecule_to_mpfit_library_charge(molecule: "Molecule") -> LibraryChargeParameter:
    """Create a library charge parameter from a molecule.

    Parameters
    ----------
    molecule
        The molecule to create the SMILES pattern from.

    Returns
    -------
        The library charge parameter with one charge value per atom.
    """
    hydrogen_indices = [
        i for i, atom in enumerate(molecule.atoms) if atom.atomic_number == 1
    ]
    other_indices = [
        i for i, atom in enumerate(molecule.atoms) if atom.atomic_number != 1
    ]

    atom_indices = list(range(1, molecule.n_atoms + 1))
    tagged_smiles = molecule_to_tagged_smiles(molecule, atom_indices)

    return LibraryChargeParameter(
        smiles=tagged_smiles,
        value=_generate_dummy_values(tagged_smiles),
        provenance={
            "hydrogen-indices": hydrogen_indices,
            "other-indices": other_indices,
        },
    )


def _fit_single_conformer(
    gdma_record: MoleculeGDMARecord,
    solver: "MPFITSolver",
) -> np.ndarray:
    """Fit charges for a single conformer.

    Parameters
    ----------
    gdma_record
        The GDMA record for this conformer.
    solver
        The solver to use for fitting.

    Returns
    -------
        Array of fitted charges with shape (n_atoms,).
    """
    # Generate objective term for this single conformer
    objective_terms_and_masks = list(
        MPFITObjective.compute_objective_terms(
            [gdma_record],
            return_quse_masks=True,
        )
    )

    term, mask_dict = objective_terms_and_masks[0]
    quse_masks = mask_dict["quse_masks"]

    # Solve for this conformer
    charges = solver.solve(
        np.array(term.atom_charge_design_matrix, dtype=object),
        np.array(term.reference_values, dtype=object),
        ancillary_arrays={"quse_masks": quse_masks},
    )

    return charges.flatten()


def generate_mpfit_charge_parameter(
    gdma_records: list[MoleculeGDMARecord], solver: MPFITSolver | None
) -> LibraryChargeParameter:
    """Generate point charges that reproduce the distributed multipole analysis data.

    For multiple conformers, charges are fit independently for each conformer
    and then averaged.

    Parameters
    ----------
    gdma_records
        The records containing the distributed multipole data. If multiple
        records are provided, charges are fit independently for each and
        averaged.
    solver
        The solver to use when finding the charges that minimize the MPFIT loss
        function. By default, the SVD solver is used.

    Returns
    -------
        The charges generated for the molecule.
    """
    from openff.toolkit import Molecule

    from pympfit.mpfit.solvers import MPFITSVDSolver

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
    mpfit_parameter = molecule_to_mpfit_library_charge(molecule)

    # Fit each conformer independently and average the results
    all_charges = []
    for record in gdma_records:
        charges = _fit_single_conformer(record, solver)
        all_charges.append(charges)

    # Average charges across conformers
    averaged_charges = np.mean(all_charges, axis=0)

    mpfit_parameter.value = averaged_charges.tolist()

    return mpfit_parameter
