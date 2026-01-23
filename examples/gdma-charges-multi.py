"""Multi-conformer test comparing single vs multi-conformer charge fitting.

This script demonstrates that:
1. Single conformer fitting works
2. Multi-conformer fitting with identical conformers gives the same result
3. Multi-conformer fitting with different conformers gives slightly different results
"""

import numpy as np

# From openff-recharge (dependency)
from openff.recharge.utilities.molecule import extract_conformers
from openff.toolkit.topology import Molecule

from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
from openff_pympfit.gdma.storage import MoleculeGDMARecord

# From openff-pympfit
from openff_pympfit.mpfit import generate_mpfit_charge_parameter
from openff_pympfit.mpfit.solvers import MPFITSVDSolver


def main():
    qc_data_settings = GDMASettings()
    mpfit_solver = MPFITSVDSolver(svd_threshold=1.0e-4)

    # 1,2-difluoroethane - gauche vs anti conformers with ~4 kJ/mol difference
    molecule: Molecule = Molecule.from_smiles("FCCF")

    # Generate multiple conformers
    from openff.units import unit
    molecule.generate_conformers(n_conformers=20, rms_cutoff=0.5 * unit.angstrom)
    # Keep only first 3 for testing
    if molecule.n_conformers > 3:
        molecule._conformers = molecule._conformers[:3]
    print(f"Generated {molecule.n_conformers} conformer(s)")
    conformers = extract_conformers(molecule)

    # Generate GDMA data for each conformer
    print("\nGenerating GDMA data for conformers...")
    gdma_records = []
    for i, input_conformer in enumerate(conformers):
        print(f"  Processing conformer {i+1}/{len(conformers)}...", end=" ")
        conformer, mp = Psi4GDMAGenerator.generate(
            molecule, input_conformer, qc_data_settings, minimize=True
        )
        record = MoleculeGDMARecord.from_molecule(
            molecule, conformer, mp, qc_data_settings
        )
        gdma_records.append(record)
        print(f"done (multipoles shape: {mp.shape})")

    # Test 1: Single conformer
    print(f"\n{'='*70}")
    print("TEST 1: Single conformer fitting")
    print(f"{'='*70}")
    single_result = generate_mpfit_charge_parameter([gdma_records[0]], mpfit_solver)
    single_charges = np.array(single_result.value)
    print(f"Charges: {np.round(single_charges, 4)}")

    # Test 2: Multi-conformer with IDENTICAL records (should give same result)
    print(f"\n{'='*70}")
    print("TEST 2: Multi-conformer with 3 IDENTICAL records (same conformer x3)")
    print(f"{'='*70}")
    identical_result = generate_mpfit_charge_parameter(
        [gdma_records[0]] * 3, mpfit_solver
    )
    identical_charges = np.array(identical_result.value)
    print(f"Charges: {np.round(identical_charges, 4)}")
    print(f"Max diff from single: {np.max(np.abs(identical_charges - single_charges)):.2e}")

    # Test 3: Multi-conformer with DIFFERENT records (may give different result)
    if len(gdma_records) > 1:
        print(f"\n{'='*70}")
        print(f"TEST 3: Multi-conformer with {len(gdma_records)} DIFFERENT conformers")
        print(f"{'='*70}")
        multi_result = generate_mpfit_charge_parameter(gdma_records, mpfit_solver)
        multi_charges = np.array(multi_result.value)
        print(f"Charges: {np.round(multi_charges, 4)}")
        print(f"Max diff from single: {np.max(np.abs(multi_charges - single_charges)):.4f}")

        # Summary comparison
        print(f"\n{'='*70}")
        print("SUMMARY: Charge comparison across fitting methods")
        print(f"{'='*70}")
        print(f"{'Atom':<6} {'Single':<12} {'Identical x3':<12} {'Multi-conf':<12} {'Δ(multi-single)':<15}")
        print("-" * 70)
        from openff.units.elements import SYMBOLS
        for i, atom in enumerate(molecule.atoms):
            element = SYMBOLS[atom.atomic_number]
            delta = multi_charges[i] - single_charges[i]
            print(f"{element}{i:<5} {single_charges[i]:>11.4f} {identical_charges[i]:>11.4f} {multi_charges[i]:>11.4f} {delta:>14.4f}")

        print("-" * 70)
        print(f"Total: {sum(single_charges):>11.4f} {sum(identical_charges):>11.4f} {sum(multi_charges):>11.4f}")
    else:
        print("\nNote: Only 1 conformer generated, skipping multi-conformer comparison.")


if __name__ == "__main__":
    main()
