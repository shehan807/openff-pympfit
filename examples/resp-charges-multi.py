"""Compare RESP multi-conformer charge fitting for the same molecules tested with MPFIT.

Tests:
1. TFSI (neutral) - should collapse to similar conformers
2. Ethylene glycol - gauche vs extended conformers
3. 1,2-difluoroethane - gauche vs anti conformers

This allows comparison with gdma-charges-multi.py to see if RESP is more
robust than MPFIT for multi-conformer fitting.
"""

import numpy as np
from openff.recharge.utilities.molecule import extract_conformers
from openff.toolkit.topology import Molecule
from openff.units import unit

from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import MSKGridSettings
from openff.recharge.charges.resp import generate_resp_charge_parameter
from openff.recharge.charges.resp._resp import (
    molecule_to_resp_library_charge,
    generate_resp_systems_of_equations,
)
from openff.recharge.charges.resp.solvers import IterativeSolver
from openff.recharge.charges.library import LibraryChargeCollection


def generate_resp_no_symmetry(qc_data_records, molecule):
    """Generate RESP charges WITHOUT symmetry constraints.

    This tests whether RESP stability comes from:
    - Hyperbolic restraints (a, b parameters)
    - Or symmetry equivalencing
    """
    solver = IterativeSolver()

    # Create charge parameter with NO equivalencing (each atom independent)
    resp_parameter = molecule_to_resp_library_charge(
        molecule,
        equivalize_within_methyl_carbons=False,
        equivalize_within_methyl_hydrogens=False,
        equivalize_within_other_heavy_atoms=False,
        equivalize_within_other_hydrogen_atoms=False,
    )

    # Single stage fit with no equivalencing between conformers either
    a = 0.0005
    b = 0.1

    (
        design_matrix,
        reference_esp,
        constraint_matrix,
        constraint_vector,
        restraint_indices,
        charge_map,
    ) = generate_resp_systems_of_equations(
        resp_parameter,
        qc_data_records,
        equivalize_between_methyl_carbons=False,
        equivalize_between_methyl_hydrogens=False,
        equivalize_between_other_heavy_atoms=False,
        equivalize_between_other_hydrogen_atoms=False,
        fix_methyl_carbons=False,
        fix_methyl_hydrogens=False,
        fix_other_heavy_atoms=False,
        fix_other_hydrogen_atoms=False,
    )

    resp_charges = solver.solve(
        design_matrix,
        reference_esp,
        constraint_matrix,
        constraint_vector,
        a,
        b,
        restraint_indices,
        len(qc_data_records),
    )

    # Map charges back to atoms
    charges = np.zeros(molecule.n_atoms)
    for array_index, value_index in charge_map.items():
        # value_index maps to the atom
        charges[value_index] = float(resp_charges[array_index].item(0))

    return charges


def test_molecule(name: str, smiles: str, minimize: bool = True):
    """Test RESP multi-conformer fitting for a single molecule."""

    print(f"\n{'='*70}")
    print(f"RESP TEST: {name}")
    print(f"SMILES: {smiles}")
    print(f"minimize={minimize}")
    print(f"{'='*70}")

    molecule = Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=20, rms_cutoff=0.5 * unit.angstrom)

    # Keep only first 3 conformers
    if molecule.n_conformers > 3:
        molecule._conformers = molecule._conformers[:3]

    print(f"Conformers: {molecule.n_conformers}")

    if molecule.n_conformers < 2:
        print("Only 1 conformer, skipping multi-conformer test")
        return

    conformers = extract_conformers(molecule)

    # Use similar settings to MPFIT (PBE0/def2-SVP)
    esp_settings = ESPSettings(
        method="pbe0",
        basis="def2-SVP",
        grid_settings=MSKGridSettings(density=1.0),
    )

    # Generate ESP data for each conformer
    print("\nGenerating ESP data...")
    esp_records = []
    for i, conf in enumerate(conformers):
        print(f"  Conformer {i+1}/{len(conformers)}...", end=" ", flush=True)

        final_conformer, grid, esp, field = Psi4ESPGenerator.generate(
            molecule, conf, esp_settings, minimize=minimize
        )

        record = MoleculeESPRecord.from_molecule(
            molecule, final_conformer, grid, esp, field, esp_settings
        )
        esp_records.append(record)
        print(f"done (grid points: {grid.shape[0]})")

    # Test 1: Single conformer
    print(f"\n{'-'*70}")
    print("TEST 1: Single conformer fitting")
    print(f"{'-'*70}")
    single_result = generate_resp_charge_parameter([esp_records[0]], None)
    single_charges = np.array(single_result.value)
    print(f"Charges: {np.round(single_charges, 4)}")
    print(f"Sum: {sum(single_charges):.6f}")

    # Test 2: Multi-conformer with IDENTICAL records
    print(f"\n{'-'*70}")
    print("TEST 2: Multi-conformer with 3 IDENTICAL records")
    print(f"{'-'*70}")
    identical_result = generate_resp_charge_parameter([esp_records[0]] * 3, None)
    identical_charges = np.array(identical_result.value)
    print(f"Charges: {np.round(identical_charges, 4)}")
    print(f"Max diff from single: {np.max(np.abs(identical_charges - single_charges)):.2e}")

    # =========================================================================
    # NO SYMMETRY TESTS (to isolate effect of restraints vs equivalencing)
    # =========================================================================

    # Test 3: Single conformer WITHOUT symmetry
    print(f"\n{'-'*70}")
    print("TEST 3: Single conformer - NO SYMMETRY (restraints only)")
    print(f"{'-'*70}")
    single_nosym = generate_resp_no_symmetry([esp_records[0]], molecule)
    print(f"Charges: {np.round(single_nosym, 4)}")
    print(f"Sum: {sum(single_nosym):.6f}")

    # Test 4: Multi-conformer WITHOUT symmetry
    print(f"\n{'-'*70}")
    print(f"TEST 4: Multi-conformer ({len(esp_records)} conf) - NO SYMMETRY (restraints only)")
    print(f"{'-'*70}")
    multi_nosym = generate_resp_no_symmetry(esp_records, molecule)
    print(f"Charges: {np.round(multi_nosym, 4)}")
    print(f"Max diff from single (no sym): {np.max(np.abs(multi_nosym - single_nosym)):.4f}")
    print(f"Sum: {sum(multi_nosym):.6f}")

    # Summary comparison
    print(f"\n{'-'*70}")
    print("SUMMARY: Per-atom charges (NO SYMMETRY)")
    print(f"{'-'*70}")
    print(f"{'Atom':<6} {'Single':<12} {'Multi-conf':<12} {'Δ(multi-single)':<15}")
    print("-" * 55)

    from openff.units.elements import SYMBOLS
    for i, atom in enumerate(molecule.atoms):
        element = SYMBOLS[atom.atomic_number]
        delta = multi_nosym[i] - single_nosym[i]
        print(f"{element}{i:<5} {single_nosym[i]:>11.4f} {multi_nosym[i]:>11.4f} {delta:>14.4f}")

    print("-" * 55)
    print(f"Total: {sum(single_nosym):>11.4f} {sum(multi_nosym):>11.4f}")
    print(f"\nMax |Δq| (no symmetry): {np.max(np.abs(multi_nosym - single_nosym)):.4f}")

    return {
        "single_nosym": single_nosym,
        "multi_nosym": multi_nosym,
        "max_diff_nosym": np.max(np.abs(multi_nosym - single_nosym))
    }


def main():
    print("="*70)
    print("RESP MULTI-CONFORMER CHARGE FITTING COMPARISON")
    print("="*70)
    print("\nThis tests the same molecules as gdma-charges-multi.py (MPFIT)")
    print("to compare robustness of RESP vs MPFIT for multi-conformer fitting.\n")

    # Test molecules (same as MPFIT tests)
    molecules = [
        ("1,2-difluoroethane", "FCCF"),
        ("Ethylene glycol", "OCCO"),
        # ("TFSI (neutral)", "C(F)(F)(F)S(=O)(=O)NS(=O)(=O)C(F)(F)F"),  # Slow, uncomment if desired
    ]

    results = {}
    for name, smiles in molecules:
        try:
            results[name] = test_molecule(name, smiles, minimize=True)
        except Exception as e:
            print(f"ERROR for {name}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY: Max |Δq| (single vs multi-conformer) - NO SYMMETRY")
    print("="*70)
    for name, result in results.items():
        if result:
            print(f"  {name}: {result['max_diff_nosym']:.4f}")


if __name__ == "__main__":
    main()
