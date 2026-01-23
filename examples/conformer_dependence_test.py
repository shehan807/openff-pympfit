"""Test conformational dependence of MPFIT charges across various molecules.

Tests two categories:
1. Conformer-DEPENDENT molecules (should show meaningful charge differences)
2. Conformer-INDEPENDENT molecules (should show minimal/no charge differences)
"""

import numpy as np
from openff.recharge.utilities.molecule import extract_conformers
from openff.toolkit.topology import Molecule
from openff.units import unit

from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
from openff_pympfit.gdma.storage import MoleculeGDMARecord
from openff_pympfit.mpfit import generate_mpfit_charge_parameter
from openff_pympfit.mpfit.solvers import MPFITSVDSolver


# Conformer-DEPENDENT molecules (expect significant charge variation)
CONFORMER_DEPENDENT = {
    "1,2-difluoroethane": "FCCF",
    "1,2-dichloroethane": "ClCCCl",
    "ethanol": "CCO",
    "ethylene glycol": "OCCO",
    "2-fluoroethanol": "OCCF",
    "1,2-dimethoxyethane": "COCCOC",
    "acetamide": "CC(=O)N",
}

# Conformer-INDEPENDENT molecules (expect minimal charge variation)
CONFORMER_INDEPENDENT = {
    "methane": "C",
    "ethane": "CC",
    "neopentane": "CC(C)(C)C",
    "benzene": "c1ccccc1",
    "carbon dioxide": "O=C=O",
    "acetylene": "C#C",
    "cyclohexane": "C1CCCCC1",
}


def test_molecule(name: str, smiles: str, n_conformers: int = 3):
    """Test a single molecule and return max charge difference."""
    print(f"\n  Testing {name} ({smiles})...", end=" ", flush=True)

    try:
        molecule = Molecule.from_smiles(smiles)
        molecule.generate_conformers(n_conformers=20, rms_cutoff=0.5 * unit.angstrom)

        # Limit conformers
        if molecule.n_conformers > n_conformers:
            molecule._conformers = molecule._conformers[:n_conformers]

        if molecule.n_conformers < 2:
            print(f"only {molecule.n_conformers} conformer(s), skipping")
            return None, "insufficient_conformers"

        conformers = extract_conformers(molecule)
        qc_settings = GDMASettings()
        mpfit_solver = MPFITSVDSolver(svd_threshold=1.0e-4)

        # Generate GDMA records for each conformer
        gdma_records = []
        for conf in conformers:
            coords, mp = Psi4GDMAGenerator.generate(
                molecule, conf, qc_settings, minimize=False
            )
            record = MoleculeGDMARecord.from_molecule(molecule, coords, mp, qc_settings)
            gdma_records.append(record)

        # Single conformer charges
        single_result = generate_mpfit_charge_parameter([gdma_records[0]], mpfit_solver)
        single_charges = np.array(single_result.value)

        # Multi-conformer charges
        try:
            multi_result = generate_mpfit_charge_parameter(gdma_records, mpfit_solver)
            multi_charges = np.array(multi_result.value)
            max_diff = np.max(np.abs(multi_charges - single_charges))
            print(f"max Δq = {max_diff:.4f} ({molecule.n_conformers} conformers)")
            return max_diff, "success"
        except ValueError as e:
            if "quse_mask mismatch" in str(e):
                print(f"quse_mask mismatch (conformers too different)")
                return None, "mask_mismatch"
            raise

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return None, "error"


def main():
    print("=" * 70)
    print("CONFORMATIONAL DEPENDENCE TEST")
    print("=" * 70)

    results = {"dependent": {}, "independent": {}}

    # Test conformer-DEPENDENT molecules
    print("\n" + "=" * 70)
    print("CONFORMER-DEPENDENT MOLECULES (expect significant Δq)")
    print("=" * 70)
    for name, smiles in CONFORMER_DEPENDENT.items():
        max_diff, status = test_molecule(name, smiles)
        results["dependent"][name] = (max_diff, status)

    # Test conformer-INDEPENDENT molecules
    print("\n" + "=" * 70)
    print("CONFORMER-INDEPENDENT MOLECULES (expect minimal Δq)")
    print("=" * 70)
    for name, smiles in CONFORMER_INDEPENDENT.items():
        max_diff, status = test_molecule(name, smiles)
        results["independent"][name] = (max_diff, status)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nConformer-DEPENDENT (expect Δq > 0.01):")
    for name, (max_diff, status) in results["dependent"].items():
        if status == "success":
            flag = "✓" if max_diff > 0.01 else "⚠ LOW"
            print(f"  {name:25s}: max Δq = {max_diff:.4f} {flag}")
        else:
            print(f"  {name:25s}: {status}")

    print("\nConformer-INDEPENDENT (expect Δq < 0.01):")
    for name, (max_diff, status) in results["independent"].items():
        if status == "success":
            flag = "✓" if max_diff < 0.01 else "⚠ HIGH"
            print(f"  {name:25s}: max Δq = {max_diff:.4f} {flag}")
        else:
            print(f"  {name:25s}: {status}")


if __name__ == "__main__":
    main()
