"""Test different SVD thresholds to see the effect."""

import numpy as np
from openff.toolkit.topology import Molecule
from openff.recharge.utilities.molecule import extract_conformers
from openff.units import unit

from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
from openff_pympfit.gdma.storage import MoleculeGDMARecord
from openff_pympfit.optimize import MPFITObjective
from openff.recharge.charges.library import LibraryChargeCollection
from openff_pympfit.mpfit._mpfit import molecule_to_mpfit_library_charge


def solve_svd(A, b, threshold=1e-4):
    """Solve A @ q = b using SVD with threshold."""
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    S_thresh = S.copy()
    S_thresh[threshold > S_thresh] = 0.0
    inv_S = np.zeros_like(S_thresh)
    mask_S = S_thresh != 0
    inv_S[mask_S] = 1.0 / S_thresh[mask_S]
    q = (Vh.T * inv_S) @ (U.T @ b)
    return q, S


def main():
    smiles = "FCCF"
    print(f"Debugging: {smiles}")

    molecule = Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=20, rms_cutoff=0.5 * unit.angstrom)
    molecule._conformers = molecule._conformers[:2]

    conformers = extract_conformers(molecule)
    gdma_settings = GDMASettings()

    # Generate GDMA records
    gdma_records = []
    for i, conf in enumerate(conformers):
        print(f"Generating GDMA for conformer {i+1}...")
        coords, multipoles = Psi4GDMAGenerator.generate(molecule, conf, gdma_settings, minimize=True)
        record = MoleculeGDMARecord.from_molecule(molecule, coords, multipoles, gdma_settings)
        gdma_records.append(record)

    # Create charge parameter
    mpfit_parameter = molecule_to_mpfit_library_charge(molecule)

    # Get objective terms
    all_terms = []
    for record in gdma_records:
        terms_and_masks = list(MPFITObjective.compute_objective_terms(
            [record],
            charge_collection=LibraryChargeCollection(parameters=[mpfit_parameter]),
            charge_parameter_keys=[(mpfit_parameter.smiles, tuple(range(len(mpfit_parameter.value))))],
            return_quse_masks=True,
        ))
        term, mask_dict = terms_and_masks[0]
        all_terms.append((term, mask_dict["quse_masks"]))

    term1, masks1 = all_terms[0]
    term2, masks2 = all_terms[1]

    # Focus on Site 1 (the problematic carbon site)
    site_idx = 1
    A1 = term1.atom_charge_design_matrix[site_idx]
    b1 = term1.reference_values[site_idx]
    A2 = term2.atom_charge_design_matrix[site_idx]
    b2 = term2.reference_values[site_idx]

    A_stacked = np.vstack([A1, A2])
    b_stacked = np.concatenate([b1, b2])

    # Get singular values
    _, S_stacked = solve_svd(A_stacked, b_stacked)
    print(f"\nStacked singular values: {S_stacked}")

    # Test different thresholds
    print("\n" + "=" * 70)
    print("EFFECT OF SVD THRESHOLD ON STACKED SOLUTION")
    print("=" * 70)

    q1, _ = solve_svd(A1, b1, threshold=1e-4)
    print(f"\nReference q1 (single conformer): {np.round(q1.flatten(), 4)}")

    thresholds = [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5]
    for thresh in thresholds:
        q_stacked, _ = solve_svd(A_stacked, b_stacked, threshold=thresh)
        diff_from_q1 = np.max(np.abs(q_stacked.flatten() - q1.flatten()))
        res = np.linalg.norm(A_stacked @ q_stacked - b_stacked)
        print(f"\nThreshold {thresh:.0e}:")
        print(f"  q_stacked = {np.round(q_stacked.flatten(), 4)}")
        print(f"  Max diff from q1: {diff_from_q1:.4f}")
        print(f"  Residual: {res:.6f}")

    # Also test relative threshold (threshold * S[0])
    print("\n" + "=" * 70)
    print("RELATIVE THRESHOLD (threshold * S[0])")
    print("=" * 70)

    S0 = S_stacked[0]
    rel_thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    for rel_thresh in rel_thresholds:
        abs_thresh = rel_thresh * S0
        q_stacked, _ = solve_svd(A_stacked, b_stacked, threshold=abs_thresh)
        diff_from_q1 = np.max(np.abs(q_stacked.flatten() - q1.flatten()))
        res = np.linalg.norm(A_stacked @ q_stacked - b_stacked)
        print(f"\nRelative {rel_thresh:.0e} (absolute {abs_thresh:.4f}):")
        print(f"  q_stacked = {np.round(q_stacked.flatten(), 4)}")
        print(f"  Max diff from q1: {diff_from_q1:.4f}")
        print(f"  Residual: {res:.6f}")


if __name__ == "__main__":
    main()
