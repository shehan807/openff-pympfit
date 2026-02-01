"""Verify constrained MPFIT charges reproduce the QM ESP for EMIM.

Runs GDMA + constrained charge fitting on EMIM, then computes the
point-charge ESP at reference grid points and compares against the
pbe0/def2-SVP QM ESP.
"""

import numpy as np
from openff.toolkit import Molecule
from openff.recharge.utilities.molecule import extract_conformers
from openff.units import unit
from openff.units.elements import SYMBOLS

from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
from openff_pympfit.gdma.storage import MoleculeGDMARecord
from openff_pympfit.mpfit._legacy_constrained_final import (
    setup_from_gdma_records,
    optimize_constrained,
    count_parameters,
    generate_atom_type_labels_from_symmetry,
)

# Conversion factor: 1 bohr = 0.529177 angstrom
BOHR_TO_ANGSTROM = 0.529177210903

# Reference ESP data (pbe0/def2-SVP, MSK grid)
ESP_DIR = "/Users/shehanparmar/Desktop/dev/work/MPFIT_Project/pympfit/pympfit/tests/data/esp"


def main():
    # Use the same SMILES as the ESP generation script
    smiles = "CCN1C=C[N+](=C1)C"  # EMIM
    print("=" * 70)
    print("  EMIM — Constrained MPFIT + ESP Verification")
    print("=" * 70)

    # 1. Load ESP reference conformer and use it for GDMA (same geometry)
    print("\n--- Generating GDMA record ---")
    ref_conf = np.load(f"{ESP_DIR}/emim_conformer.npy")  # angstrom
    mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    conformer = unit.Quantity(ref_conf, unit.angstrom)
    gdma_settings = GDMASettings()  # defaults: pbe0/def2-SVP

    # minimize=False: use the exact ESP reference geometry
    conformer, multipoles = Psi4GDMAGenerator.generate(
        mol, conformer, gdma_settings, minimize=False,
    )
    record = MoleculeGDMARecord.from_molecule(
        mol, conformer, multipoles, gdma_settings,
    )
    print(f"  Atoms: {mol.n_atoms}")

    # 2. Assign symmetry-based atom type labels
    labels = generate_atom_type_labels_from_symmetry(mol)
    print(f"  Labels: {labels}")

    # 3. Setup and optimize
    print("\n--- Constrained MPFIT ---")
    state = setup_from_gdma_records(record, labels)
    state.molecule_charge = 1.0  # EMIM is a +1 cation
    state.conchg = 1.0
    n_params = count_parameters(state)
    print(f"  Parameters: {n_params}")

    result = optimize_constrained(state)
    charges = result["qstore"]
    scipy_res = result["scipy_result"]

    print(f"  Converged: {result['success']}")
    print(f"  Iterations: {scipy_res.nit}")
    print(f"  Objective: {result['objective']:.8e}")
    print(f"  Total charge: {np.sum(charges):+.6f}")

    # 4. Print per-atom charges
    print("\n--- Fitted Charges ---")
    print(f"  {'Atom':>4} {'Elem':>4} {'Label':<10} {'Charge':>12}")
    print(f"  {'-' * 34}")
    for i in range(mol.n_atoms):
        elem = SYMBOLS[mol.atoms[i].atomic_number]
        print(f"  {i:>4} {elem:>4} {labels[i]:<10} {charges[i]:>+12.6f}")

    # 5. Load reference ESP data
    print("\n--- ESP Comparison ---")
    ref_grid = np.load(f"{ESP_DIR}/emim_grid.npy")       # (N_grid, 3) angstrom
    ref_esp = np.load(f"{ESP_DIR}/emim_esp.npy").flatten()  # (N_grid,) hartree/e
    ref_conf = np.load(f"{ESP_DIR}/emim_conformer.npy")   # (N_atoms, 3) angstrom

    print(f"  Grid points: {ref_grid.shape[0]}")
    print(f"  QM ESP range: [{ref_esp.min():.6f}, {ref_esp.max():.6f}]")

    # 6. Get GDMA-optimized conformer in angstrom
    gdma_conformer = unit.convert(record.conformer, unit.angstrom, unit.angstrom)
    print(f"  GDMA conformer shape: {gdma_conformer.shape}")

    # Check if conformers match (they may differ due to geometry optimization)
    conf_diff = np.max(np.abs(gdma_conformer - ref_conf))
    print(f"  Max conformer difference: {conf_diff:.6f} angstrom")
    if conf_diff > 0.5:
        print("  WARNING: Conformers differ significantly. ESP comparison may not be meaningful.")
        print("  Using GDMA conformer for point-charge ESP calculation.")

    # 7. Compute point-charge ESP at grid points (Coulomb's law)
    # V(r) = Σ_i q_i / |r - r_i|   (in atomic units: hartree/e when distances in bohr)
    coord = gdma_conformer  # angstrom
    diff = ref_grid[:, np.newaxis, :] - coord[np.newaxis, :, :]  # (N_grid, N_atoms, 3)
    distances = np.linalg.norm(diff, axis=2)  # (N_grid, N_atoms) angstrom
    # Convert to atomic units: divide by bohr_to_angstrom
    calc_esp = np.sum(charges[np.newaxis, :] / distances, axis=1) * BOHR_TO_ANGSTROM

    print(f"  Calc ESP range: [{calc_esp.min():.6f}, {calc_esp.max():.6f}]")

    # 8. Compare
    esp_diff = calc_esp - ref_esp
    rmse = np.sqrt(np.mean(esp_diff ** 2))
    max_err = np.max(np.abs(esp_diff))
    rel_rmse = rmse / np.std(ref_esp)

    print(f"\n  RMSE:     {rmse:.6f} hartree/e")
    print(f"  Max err:  {max_err:.6f} hartree/e")
    print(f"  Rel RMSE: {rel_rmse:.4f} (relative to ESP std dev)")

    # Convert to kcal/mol for chemical intuition (1 hartree = 627.509 kcal/mol)
    rmse_kcal = rmse * 627.509
    max_kcal = max_err * 627.509
    print(f"\n  RMSE:     {rmse_kcal:.2f} kcal/mol")
    print(f"  Max err:  {max_kcal:.2f} kcal/mol")

    if rel_rmse < 0.1:
        print("\n  RESULT: GOOD — point charges reproduce QM ESP well")
    elif rel_rmse < 0.3:
        print("\n  RESULT: ACCEPTABLE — moderate ESP reproduction")
    else:
        print("\n  RESULT: POOR — charges do not reproduce QM ESP well")


if __name__ == "__main__":
    main()
