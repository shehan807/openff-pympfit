"""Study conformational dependence of MPFIT charges for TFSI anion.

TFSI (bis(trifluoromethanesulfonyl)imide) is a common ionic liquid anion
with flexibility around the S-N-S backbone. This script examines how
fitted charges vary across conformers.

Key question: Do charges change significantly across low-energy conformers?
"""

import numpy as np
from openff.recharge.utilities.molecule import extract_conformers
from openff.toolkit.topology import Molecule
from openff.units import unit
from openff.units.elements import SYMBOLS

from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
from openff_pympfit.gdma.storage import MoleculeGDMARecord
from openff_pympfit.mpfit import generate_mpfit_charge_parameter
from openff_pympfit.mpfit.solvers import MPFITSVDSolver


def main():
    # TFSI anion SMILES
    smiles = "C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F"

    print("=" * 70)
    print("TFSI CONFORMATIONAL DEPENDENCE STUDY")
    print("=" * 70)
    print(f"\nSMILES: {smiles}")

    molecule = Molecule.from_smiles(smiles)
    print(f"Atoms: {molecule.n_atoms}")
    print(f"Total charge: {molecule.total_charge}")

    # Print atom labels for reference
    print("\nAtom labels:")
    for i, atom in enumerate(molecule.atoms):
        print(f"  {i:2d}: {SYMBOLS[atom.atomic_number]}")

    # Generate conformers
    print("\n" + "-" * 70)
    print("CONFORMER GENERATION")
    print("-" * 70)
    molecule.generate_conformers(n_conformers=50, rms_cutoff=0.5 * unit.angstrom)
    print(f"Conformers generated: {molecule.n_conformers}")

    # Limit to first few for reasonable runtime
    n_to_test = min(5, molecule.n_conformers)
    if molecule.n_conformers > n_to_test:
        molecule._conformers = molecule._conformers[:n_to_test]
    print(f"Conformers to analyze: {n_to_test}")

    conformers = extract_conformers(molecule)
    qc_settings = GDMASettings()
    mpfit_solver = MPFITSVDSolver(svd_threshold=1.0e-4)

    # Generate GDMA data for each conformer (WITHOUT Psi4 minimization)
    print("\n" + "-" * 70)
    print("GDMA CALCULATIONS (minimize=False, preserving conformer geometry)")
    print("-" * 70)

    gdma_records = []
    for i, conf in enumerate(conformers):
        print(f"  Conformer {i + 1}/{len(conformers)}...", end=" ", flush=True)
        coords, mp = Psi4GDMAGenerator.generate(
            molecule, conf, qc_settings, minimize=False  # Keep original geometry!
        )
        record = MoleculeGDMARecord.from_molecule(molecule, coords, mp, qc_settings)
        gdma_records.append(record)
        print("done")

    # Fit charges for each conformer individually
    print("\n" + "-" * 70)
    print("INDIVIDUAL CONFORMER CHARGES")
    print("-" * 70)

    all_charges = []
    for i, record in enumerate(gdma_records):
        result = generate_mpfit_charge_parameter([record], mpfit_solver)
        charges = np.array(result.value)
        all_charges.append(charges)
        print(f"\nConformer {i + 1}:")
        print(f"  Sum: {sum(charges):.4f}")

    all_charges = np.array(all_charges)

    # Multi-conformer fit
    print("\n" + "-" * 70)
    print("MULTI-CONFORMER FIT (all conformers combined)")
    print("-" * 70)

    try:
        multi_result = generate_mpfit_charge_parameter(gdma_records, mpfit_solver)
        multi_charges = np.array(multi_result.value)
        print(f"Sum: {sum(multi_charges):.4f}")
    except ValueError as e:
        print(f"Multi-conformer fit failed: {e}")
        multi_charges = None

    # Per-atom analysis
    print("\n" + "=" * 70)
    print("PER-ATOM CHARGE ANALYSIS")
    print("=" * 70)

    # Header
    header = f"{'Atom':<8}"
    for i in range(len(all_charges)):
        header += f"{'Conf ' + str(i + 1):>10}"
    header += f"{'Mean':>10}{'Std':>10}{'Range':>10}"
    if multi_charges is not None:
        header += f"{'Multi':>10}"
    print(header)
    print("-" * len(header))

    # Per-atom data
    for atom_idx in range(molecule.n_atoms):
        element = SYMBOLS[molecule.atoms[atom_idx].atomic_number]
        row = f"{element}{atom_idx:<7}"

        atom_charges = all_charges[:, atom_idx]
        for q in atom_charges:
            row += f"{q:>10.4f}"

        mean_q = np.mean(atom_charges)
        std_q = np.std(atom_charges)
        range_q = np.max(atom_charges) - np.min(atom_charges)

        row += f"{mean_q:>10.4f}{std_q:>10.4f}{range_q:>10.4f}"

        if multi_charges is not None:
            row += f"{multi_charges[atom_idx]:>10.4f}"

        print(row)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Charge variation by atom type
    atom_types = {}
    for atom_idx in range(molecule.n_atoms):
        element = SYMBOLS[molecule.atoms[atom_idx].atomic_number]
        if element not in atom_types:
            atom_types[element] = []
        atom_charges = all_charges[:, atom_idx]
        atom_types[element].append(np.max(atom_charges) - np.min(atom_charges))

    print("\nCharge range by element (max - min across conformers):")
    for element, ranges in sorted(atom_types.items()):
        print(f"  {element}: mean range = {np.mean(ranges):.4f}, max range = {np.max(ranges):.4f}")

    # Overall statistics
    print(f"\nOverall max |Δq| across all atoms and conformers: {np.max(all_charges) - np.min(all_charges):.4f}")

    if multi_charges is not None:
        max_diff_from_multi = np.max(np.abs(all_charges - multi_charges), axis=1)
        print(f"\nMax |Δq| from multi-conformer fit:")
        for i, diff in enumerate(max_diff_from_multi):
            print(f"  Conformer {i + 1}: {diff:.4f}")


if __name__ == "__main__":
    main()
