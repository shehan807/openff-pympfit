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
    from openff.recharge.charges.vsite import VirtualSiteCollection
    from openff.toolkit import Molecule

    from pympfit.mpfit.solvers import ConstrainedMPFITSolver


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
    vsite_collection: "VirtualSiteCollection | None" = None,
    fit_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit charges for a single conformer.

    Parameters
    ----------
    gdma_record
        The GDMA record for this conformer.
    solver
        The solver to use for fitting.
    vsite_collection
        Optional virtual site collection defining extra charge sites.
    fit_limit
        Optional maximum multipole rank for fitting. When provided and less
        than the GDMA expansion rank, the multipole tensor is truncated so
        only terms up to this rank are used.

    Returns
    -------
        Tuple of (atom_charges, vsite_charges). vsite_charges is None if
        no vsites are present.
    """
    from openff.toolkit import Molecule

    # Generate objective term for this single conformer
    objective_terms_and_masks = list(
        MPFITObjective.compute_objective_terms(
            [gdma_record],
            vsite_collection=vsite_collection,
            return_quse_masks=True,
            fit_limit=fit_limit,
        )
    )

    term, mask_dict = objective_terms_and_masks[0]
    quse_masks = mask_dict["quse_masks"]
    n_vsites = mask_dict.get("n_vsites", 0)

    # Solve for this conformer
    all_charges = solver.solve(
        np.array(term.atom_charge_design_matrix, dtype=object),
        np.array(term.reference_values, dtype=object),
        ancillary_arrays={"quse_masks": quse_masks, "n_vsites": n_vsites},
    )

    # Split into atom and vsite charges
    molecule = Molecule.from_mapped_smiles(
        gdma_record.tagged_smiles, allow_undefined_stereo=True
    )
    n_atoms = molecule.n_atoms
    atom_charges = all_charges[:n_atoms].flatten()
    vsite_charges = all_charges[n_atoms:].flatten() if n_vsites > 0 else None

    return atom_charges, vsite_charges


def generate_mpfit_charge_parameter(
    gdma_records: list[MoleculeGDMARecord],
    solver: MPFITSolver | None = None,
    vsite_collection: "VirtualSiteCollection | None" = None,
    fit_limit: int | None = None,
) -> LibraryChargeParameter | tuple[LibraryChargeParameter, np.ndarray]:
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
    vsite_collection
        Optional virtual site collection defining extra charge sites beyond
        atoms. When provided, charges are fit at both atom and vsite positions.
    fit_limit
        Optional maximum multipole rank for fitting. When provided and less
        than the GDMA expansion rank, the multipole tensor is truncated so
        only terms up to this rank are used. This allows running GDMA once
        at a high rank (e.g., limit=8) and fitting charges at multiple lower
        ranks (e.g., fit_limit=2, 4) without rerunning GDMA.

    Returns
    -------
        When vsite_collection is None: LibraryChargeParameter with atom charges.
        When vsite_collection is provided: Tuple of (LibraryChargeParameter,
        vsite_charges) where vsite_charges is a numpy array of shape (n_vsites,).
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
    all_atom_charges = []
    all_vsite_charges = []
    for record in gdma_records:
        atom_charges, vsite_charges = _fit_single_conformer(
            record, solver, vsite_collection, fit_limit=fit_limit
        )
        all_atom_charges.append(atom_charges)
        if vsite_charges is not None:
            all_vsite_charges.append(vsite_charges)

    # Average atom charges across conformers
    averaged_atom_charges = np.mean(all_atom_charges, axis=0)
    mpfit_parameter.value = averaged_atom_charges.tolist()

    # Return tuple only when vsites are present (backward compatible)
    if vsite_collection is not None and all_vsite_charges:
        averaged_vsite_charges = np.mean(all_vsite_charges, axis=0)
        return mpfit_parameter, averaged_vsite_charges

    return mpfit_parameter


def generate_global_atom_type_labels(
    molecules: list["Molecule"],
    radius: int = 2,
    equivalize_between_methyl_carbons: bool = True,
    equivalize_between_methyl_hydrogens: bool = True,
    equivalize_between_other_heavy_atoms: bool = True,
    equivalize_between_other_hydrogen_atoms: bool = True,
) -> list[list[str]]:
    """Generate consistent atom type labels across multiple molecules.

    This function implements a hybrid, hierarchical atom typing approach, in which:

    1. **Within-molecule** equivalence is set via
       ``get_atom_symmetries`` for topologically equivalent
       atoms for the same molecule
    2. **Cross-molecule** equivalence is set via Morgan
       fingerprints, where local chemical environments
       establish equivalence.

    Two atoms receive the same label only if they share both the same
    Morgan environment hash **and** the same within-molecule symmetry
    group (relative to their Morgan group).

    Parameters
    ----------
    molecules
        The molecules to generate labels for.
    radius
        The Morgan fingerprint radius (in bonds) controlling cross-molecule
        equivalence scope. Smaller values (2-3) match shared substructures
        (e.g., imidazolium ring across EMIM/BMIM). Larger values require
        more of the surrounding structure to match.
    equivalize_between_methyl_carbons
        Whether topologically symmetric methyl(ene) carbons (matched by
        ``[#6X4H3,#6H4,#6X4H2:1]``) in **different** molecules should be
        assigned an equivalent charge.
    equivalize_between_methyl_hydrogens
        Whether topologically symmetric methyl(ene) hydrogens (attached to
        a methyl(ene) carbon) in **different** molecules should be assigned
        an equivalent charge.
    equivalize_between_other_heavy_atoms
        Whether topologically symmetric heavy atoms that are not
        methyl(ene) carbons in **different** molecules should be assigned
        an equivalent charge.
    equivalize_between_other_hydrogen_atoms
        Whether topologically symmetric hydrogens that are not methyl(ene)
        hydrogens in **different** molecules should be assigned an
        equivalent charge.

    Returns
    -------
        A list of label lists, one per molecule. Each inner list contains
        one string label per atom, ordered by atom index.
    """
    from collections import defaultdict

    from openff.recharge.utilities.toolkits import get_atom_symmetries
    from openff.units.elements import SYMBOLS
    from rdkit.Chem import rdFingerprintGenerator

    # Global registries for cross-molecule label assignment
    env_to_label_map: dict[tuple, str] = {}
    element_counters: defaultdict[str, int] = defaultdict(int)
    unique_atom_counter = 0
    all_molecule_labels: list[list[str]] = []

    for molecule in molecules:
        mol_labels: list[str] = []

        atom_symmetries = get_atom_symmetries(molecule)

        # Atom classification ("within" molecule)
        methyl_carbons = {
            i for i, in molecule.chemical_environment_matches("[#6X4H3,#6H4,#6X4H2:1]")
        }
        methyl_hydrogens = {
            atom.molecule_atom_index
            for i in methyl_carbons
            for atom in molecule.atoms[i].bonded_atoms
            if atom.atomic_number == 1
        }
        other_heavy_atoms = {
            i
            for i, atom in enumerate(molecule.atoms)
            if atom.atomic_number != 1 and i not in methyl_carbons
        }
        other_hydrogen_atoms = {
            i
            for i, atom in enumerate(molecule.atoms)
            if atom.atomic_number == 1 and i not in methyl_hydrogens
        }

        # Cross-molecule hashing via Morgan fingerprints
        rdkit_mol = molecule.to_rdkit()
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)

        atom_morgan_hashes = []
        for i in range(molecule.n_atoms):
            fp = morgan_gen.GetSparseCountFingerprint(rdkit_mol, fromAtoms=[i])
            atom_morgan_hashes.append(tuple(sorted(fp.GetNonzeroElements().items())))

        # have atoms shared within molecules take precendence
        sym_group_to_atoms: defaultdict[int, list[int]] = defaultdict(list)
        for i, sym in enumerate(atom_symmetries):
            sym_group_to_atoms[sym].append(i)

        representative_hash = list(atom_morgan_hashes)
        for atom_indices in sym_group_to_atoms.values():
            rep = atom_morgan_hashes[atom_indices[0]]
            for idx in atom_indices:
                representative_hash[idx] = rep

        # Assign labels per atom
        for i, atom in enumerate(molecule.atoms):
            element = SYMBOLS[atom.atomic_number]
            is_methyl_c = i in methyl_carbons
            is_methyl_h = i in methyl_hydrogens

            # Determine if cross-molecule equivalence applies
            if is_methyl_c:
                should_equivalize = equivalize_between_methyl_carbons
            elif is_methyl_h:
                should_equivalize = equivalize_between_methyl_hydrogens
            elif i in other_heavy_atoms:
                should_equivalize = equivalize_between_other_heavy_atoms
            elif i in other_hydrogen_atoms:
                should_equivalize = equivalize_between_other_hydrogen_atoms
            else:
                should_equivalize = False

            if should_equivalize:
                env_key = (element, representative_hash[i])
                if env_key not in env_to_label_map:
                    element_counters[element] += 1
                    env_to_label_map[env_key] = f"{element}{element_counters[element]}"
                mol_labels.append(env_to_label_map[env_key])
            else:
                # unique label across all molecules
                unique_atom_counter += 1
                mol_labels.append(f"{element}_u{unique_atom_counter}")

        all_molecule_labels.append(mol_labels)

    return all_molecule_labels


def generate_constrained_mpfit_charge_parameter(
    gdma_records: list[MoleculeGDMARecord],
    molecules: list["Molecule"],
    solver: "ConstrainedMPFITSolver | None" = None,
    atom_type_labels: list[list[str]] | None = None,
    radius: int = 2,
    equivalize_between_methyl_carbons: bool = True,
    equivalize_between_methyl_hydrogens: bool = True,
    equivalize_between_other_heavy_atoms: bool = True,
    equivalize_between_other_hydrogen_atoms: bool = True,
) -> list[LibraryChargeParameter]:
    """Fit constrained charges for one or more molecules.

    When multiple molecules are provided, atoms in **different** molecules
    that share the same atom type label are constrained to carry the same
    charge. For a single molecule, within-molecule symmetry constraints
    are still applied. Labels can be generated automatically via Morgan
    fingerprints and within-molecule symmetry, or supplied directly.

    Parameters
    ----------
    gdma_records
        One GDMA record per molecule (one conformer each).
    molecules
        The molecules corresponding to each GDMA record, in the same order.
        The formal charge of each molecule is read from
        ``molecule.total_charge``.
    solver
        The constrained solver to use. Defaults to ``ConstrainedSciPySolver``.
    atom_type_labels
        Per-molecule atom type labels. If ``None``, labels are generated
        automatically via ``generate_global_atom_type_labels``.
    radius
        Morgan fingerprint radius passed to
        ``generate_global_atom_type_labels`` when ``atom_type_labels``
        is ``None``.
    equivalize_between_methyl_carbons
        Whether topologically symmetric methyl(ene) carbons (matched by
        ``[#6X4H3,#6H4,#6X4H2:1]``) in **different** molecules should be
        assigned an equivalent charge.
    equivalize_between_methyl_hydrogens
        Whether topologically symmetric methyl(ene) hydrogens (attached to
        a methyl(ene) carbon) in **different** molecules should be assigned
        an equivalent charge.
    equivalize_between_other_heavy_atoms
        Whether topologically symmetric heavy atoms that are not
        methyl(ene) carbons in **different** molecules should be assigned
        an equivalent charge.
    equivalize_between_other_hydrogen_atoms
        Whether topologically symmetric hydrogens that are not methyl(ene)
        hydrogens in **different** molecules should be assigned an
        equivalent charge.

    Returns
    -------
        One ``LibraryChargeParameter`` per molecule.
    """
    from pympfit.mpfit.solvers import (
        ConstrainedMPFITSolver,
        ConstrainedMPFITState,
        ConstrainedSciPySolver,
        build_quse_matrix,
    )

    if len(gdma_records) != len(molecules):
        msg = (
            f"gdma_records has {len(gdma_records)} entries, "
            f"but molecules has {len(molecules)}"
        )
        raise ValueError(msg)

    if solver is None:
        solver = ConstrainedSciPySolver()
    elif not isinstance(solver, ConstrainedMPFITSolver):
        msg = (
            f"solver must be a ConstrainedMPFITSolver instance, "
            f"got {type(solver).__name__}"
        )
        raise TypeError(msg)

    if atom_type_labels is None:
        atom_type_labels = generate_global_atom_type_labels(
            molecules,
            radius=radius,
            equivalize_between_methyl_carbons=equivalize_between_methyl_carbons,
            equivalize_between_methyl_hydrogens=equivalize_between_methyl_hydrogens,
            equivalize_between_other_heavy_atoms=equivalize_between_other_heavy_atoms,
            equivalize_between_other_hydrogen_atoms=equivalize_between_other_hydrogen_atoms,
        )

    if len(atom_type_labels) != len(molecules):
        msg = (
            f"atom_type_labels has {len(atom_type_labels)} entries, "
            f"but molecules has {len(molecules)}"
        )
        raise ValueError(msg)

    flat_labels = tuple(
        label for mol_labels in atom_type_labels for label in mol_labels
    )

    mol_charges = tuple(
        mol.total_charge.m_as(unit.elementary_charge) for mol in molecules
    )

    all_xyz = []
    all_multipoles = []
    all_rvdw = []
    all_lmax = []
    atom_counts = []

    for record in gdma_records:
        arrays = MPFITObjective.extract_arrays(record)
        all_xyz.append(arrays["bohr_conformer"])
        all_multipoles.append(arrays["multipoles"])
        all_rvdw.append(arrays["rvdw"])
        all_lmax.append(arrays["lmax"])
        atom_counts.append(arrays["n_atoms"])

    settings = gdma_records[0].gdma_settings

    xyzcharge = np.vstack(all_xyz)
    xyzmult = np.vstack(all_xyz)
    rvdw = np.concatenate(all_rvdw)

    state = ConstrainedMPFITState(
        xyzmult=xyzmult,
        xyzcharge=xyzcharge,
        multipoles=np.vstack(all_multipoles),
        quse=build_quse_matrix(xyzmult, xyzcharge, rvdw),
        atomtype=flat_labels,
        rvdw=rvdw,
        lmax=np.concatenate(all_lmax),
        r1=settings.mpfit_inner_radius,
        r2=settings.mpfit_outer_radius,
        maxl=settings.limit,
        atom_counts=tuple(atom_counts),
        molecule_charges=mol_charges,
    )

    qstore = solver.solve(state)

    # split charges by molecule → LibraryChargeParameter ──
    parameters = []
    offset = 0
    for molecule, n in zip(molecules, atom_counts, strict=False):
        mol_charges_arr = qstore[offset : offset + n]
        param = molecule_to_mpfit_library_charge(molecule)
        param.value = mol_charges_arr.tolist()
        parameters.append(param)
        offset += n

    return parameters
