"""Direct comparison of MPFIT vs RESP charges for the same conformers.

For each conformer:
1. Run QM optimization (minimize=True)
2. Compute GDMA multipoles -> MPFIT charges
3. Compute ESP grid -> RESP charges
4. Verify geometries are identical
5. Compare charges
"""

import numpy as np
from openff.toolkit.topology import Molecule
from openff.recharge.utilities.molecule import extract_conformers
from openff.units import unit

# GDMA/MPFIT imports
from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
from openff_pympfit.gdma.storage import MoleculeGDMARecord
from openff_pympfit.mpfit import generate_mpfit_charge_parameter
from openff_pympfit.mpfit.solvers import MPFITSVDSolver

# ESP/RESP imports
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import MSKGridSettings
from openff.recharge.charges.resp._resp import (
    molecule_to_resp_library_charge,
    generate_resp_systems_of_equations,
)
from openff.recharge.charges.resp.solvers import IterativeSolver


def resp_no_symmetry(esp_records, molecule):
    """RESP without symmetry constraints."""
    solver = IterativeSolver()
    resp_param = molecule_to_resp_library_charge(
        molecule,
        equivalize_within_methyl_carbons=False,
        equivalize_within_methyl_hydrogens=False,
        equivalize_within_other_heavy_atoms=False,
        equivalize_within_other_hydrogen_atoms=False,
    )
    design_matrix, ref_esp, constraint_matrix, constraint_vector, restraint_indices, charge_map = \
        generate_resp_systems_of_equations(
            resp_param, esp_records,
            equivalize_between_methyl_carbons=False,
            equivalize_between_methyl_hydrogens=False,
            equivalize_between_other_heavy_atoms=False,
            equivalize_between_other_hydrogen_atoms=False,
            fix_methyl_carbons=False, fix_methyl_hydrogens=False,
            fix_other_heavy_atoms=False, fix_other_hydrogen_atoms=False,
        )
    resp_charges = solver.solve(design_matrix, ref_esp, constraint_matrix, constraint_vector,
                                 0.0005, 0.1, restraint_indices, len(esp_records))
    charges = np.zeros(molecule.n_atoms)
    for arr_idx, val_idx in charge_map.items():
        charges[val_idx] = float(resp_charges[arr_idx].item(0))
    return charges


def test_molecule(name, smiles, n_conf=3):
    print(f"\n{'='*70}")
    print(f"{name} ({smiles})")
    print(f"{'='*70}")

    molecule = Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=20, rms_cutoff=0.5 * unit.angstrom)
    if molecule.n_conformers > n_conf:
        molecule._conformers = molecule._conformers[:n_conf]
    print(f"Conformers: {molecule.n_conformers}")

    conformers = extract_conformers(molecule)
    gdma_settings = GDMASettings()
    esp_settings = ESPSettings(method="pbe0", basis="def2-SVP", grid_settings=MSKGridSettings())
    mpfit_solver = MPFITSVDSolver(svd_threshold=1e-4)

    all_mpfit = []
    all_resp = []
    gdma_records = []
    esp_records = []

    for i, conf in enumerate(conformers):
        print(f"\n--- Conformer {i+1} ---")

        # GDMA calculation
        gdma_coords, multipoles = Psi4GDMAGenerator.generate(molecule, conf, gdma_settings, minimize=True)
        gdma_record = MoleculeGDMARecord.from_molecule(molecule, gdma_coords, multipoles, gdma_settings)
        gdma_records.append(gdma_record)

        # ESP calculation
        esp_coords, grid, esp, field = Psi4ESPGenerator.generate(molecule, conf, esp_settings, minimize=True)
        esp_record = MoleculeESPRecord.from_molecule(molecule, esp_coords, grid, esp, field, esp_settings)
        esp_records.append(esp_record)

        # Check geometry match
        gdma_xyz = np.array(gdma_coords.m_as(unit.angstrom))
        esp_xyz = np.array(esp_coords.m_as(unit.angstrom))
        geom_diff = np.max(np.abs(gdma_xyz - esp_xyz))
        print(f"Geometry max diff: {geom_diff:.6f} Å")

        # MPFIT charges (single conformer)
        mpfit_result = generate_mpfit_charge_parameter([gdma_record], mpfit_solver)
        mpfit_charges = np.array(mpfit_result.value)

        # RESP charges (single conformer, no symmetry)
        resp_charges = resp_no_symmetry([esp_record], molecule)

        all_mpfit.append(mpfit_charges)
        all_resp.append(resp_charges)

        print(f"MPFIT: {np.round(mpfit_charges, 4)}, sum={sum(mpfit_charges):.4f}")
        print(f"RESP:  {np.round(resp_charges, 4)}, sum={sum(resp_charges):.4f}")

    # Global multi-conformer fit
    print(f"\n--- Global Fit (all {len(conformers)} conformers) ---")
    try:
        global_mpfit_result = generate_mpfit_charge_parameter(gdma_records, mpfit_solver)
        global_mpfit = np.array(global_mpfit_result.value)
        print(f"MPFIT: {np.round(global_mpfit, 4)}, sum={sum(global_mpfit):.4f}")
    except Exception as e:
        print(f"MPFIT global fit failed: {e}")
        global_mpfit = None

    global_resp = resp_no_symmetry(esp_records, molecule)
    print(f"RESP:  {np.round(global_resp, 4)}, sum={sum(global_resp):.4f}")

    # Summary
    all_mpfit = np.array(all_mpfit)
    all_resp = np.array(all_resp)

    print(f"\n{'='*70}")
    print("SUMMARY: Per-conformer charges and global fit")
    print(f"{'='*70}")

    from openff.units.elements import SYMBOLS

    # Header
    header = f"{'Atom':<6}"
    for i in range(len(all_mpfit)):
        header += f" {'MPFIT'+str(i+1):>8}"
    if global_mpfit is not None:
        header += f" {'MPFITglob':>9}"
    for i in range(len(all_resp)):
        header += f" {'RESP'+str(i+1):>8}"
    header += f" {'RESPglob':>9}"
    print(header)
    print("-" * len(header))

    for j in range(molecule.n_atoms):
        element = SYMBOLS[molecule.atoms[j].atomic_number]
        row = f"{element}{j:<5}"
        for i in range(len(all_mpfit)):
            row += f" {all_mpfit[i, j]:>8.4f}"
        if global_mpfit is not None:
            row += f" {global_mpfit[j]:>9.4f}"
        for i in range(len(all_resp)):
            row += f" {all_resp[i, j]:>8.4f}"
        row += f" {global_resp[j]:>9.4f}"
        print(row)

    print("-" * len(header))

    # Per-conformer range
    print(f"\n{'Atom':<6} {'MPFIT range':<15} {'RESP range':<15}")
    print("-" * 40)
    for j in range(molecule.n_atoms):
        element = SYMBOLS[molecule.atoms[j].atomic_number]
        mpfit_range = np.ptp(all_mpfit[:, j])
        resp_range = np.ptp(all_resp[:, j])
        print(f"{element}{j:<5} {mpfit_range:>14.4f} {resp_range:>14.4f}")

    print("-" * 40)
    print(f"Max:   {np.max(np.ptp(all_mpfit, axis=0)):>14.4f} {np.max(np.ptp(all_resp, axis=0)):>14.4f}")


if __name__ == "__main__":
    test_molecule("Ethylene glycol", "OCCO", n_conf=2)
    test_molecule("1,2-difluoroethane", "FCCF", n_conf=2)
    test_molecule("TFSI anion", "C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F", n_conf=2)
