"""Verify the stacking math is correct by computing residuals."""

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


def debug_site(A1, b1, A2, b2, site_name):
    """Debug a single site with detailed analysis."""
    print(f"\n{'='*60}")
    print(f"SITE: {site_name}")
    print(f"{'='*60}")

    print(f"\nA1 shape: {A1.shape}, A2 shape: {A2.shape}")
    print(f"b1 shape: {b1.shape}, b2 shape: {b2.shape}")

    # Solve individually
    q1, S1 = solve_svd(A1, b1)
    q2, S2 = solve_svd(A2, b2)

    print(f"\n--- Individual solutions ---")
    print(f"q1 = {np.round(q1.flatten(), 4)}")
    print(f"q2 = {np.round(q2.flatten(), 4)}")
    print(f"S1 = {S1}")
    print(f"S2 = {S2}")

    # Residuals for individual solutions
    res1_with_q1 = np.linalg.norm(A1 @ q1 - b1)
    res2_with_q2 = np.linalg.norm(A2 @ q2 - b2)
    print(f"\nResidual ||A1 @ q1 - b1|| = {res1_with_q1:.6e}")
    print(f"Residual ||A2 @ q2 - b2|| = {res2_with_q2:.6e}")

    # Stack and solve
    A_stacked = np.vstack([A1, A2])
    b_stacked = np.concatenate([b1, b2])
    q_stacked, S_stacked = solve_svd(A_stacked, b_stacked)

    print(f"\n--- Stacked solution ---")
    print(f"q_stacked = {np.round(q_stacked.flatten(), 4)}")
    print(f"S_stacked = {S_stacked}")
    print(f"Condition number = {S_stacked[0]/S_stacked[-1]:.2f}")

    # Residuals for stacked solution
    res1_with_stacked = np.linalg.norm(A1 @ q_stacked - b1)
    res2_with_stacked = np.linalg.norm(A2 @ q_stacked - b2)
    res_total_stacked = np.linalg.norm(A_stacked @ q_stacked - b_stacked)

    print(f"\nResidual ||A1 @ q_stacked - b1|| = {res1_with_stacked:.6e}")
    print(f"Residual ||A2 @ q_stacked - b2|| = {res2_with_stacked:.6e}")
    print(f"Residual ||A_stacked @ q_stacked - b_stacked|| = {res_total_stacked:.6e}")

    # What if we used q1 on the stacked system?
    res_total_with_q1 = np.linalg.norm(A_stacked @ q1 - b_stacked)
    print(f"\nResidual ||A_stacked @ q1 - b_stacked|| = {res_total_with_q1:.6e}")

    # Compare: is q_stacked actually better than q1 for the stacked system?
    print(f"\nIs q_stacked better than q1 for stacked system?")
    print(f"  q_stacked residual: {res_total_stacked:.6e}")
    print(f"  q1 residual:        {res_total_with_q1:.6e}")
    print(f"  Difference:         {res_total_with_q1 - res_total_stacked:.6e}")

    # Check the actual matrices
    print(f"\n--- Matrix analysis ---")
    print(f"A1:\n{A1}")
    print(f"\nb1: {b1}")
    print(f"\nA2:\n{A2}")
    print(f"\nb2: {b2}")
    print(f"\nA1 - A2 (difference):\n{A1 - A2}")
    print(f"\nb1 - b2: {b1 - b2}")


def main():
    smiles = "FCCF"
    print(f"Debugging: {smiles}")

    molecule = Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=20, rms_cutoff=0.5 * unit.angstrom)
    molecule._conformers = molecule._conformers[:2]
    print(f"Conformers: {molecule.n_conformers}")

    conformers = extract_conformers(molecule)
    gdma_settings = GDMASettings()

    # Generate GDMA records
    gdma_records = []
    for i, conf in enumerate(conformers):
        print(f"\nGenerating GDMA for conformer {i+1}...")
        coords, multipoles = Psi4GDMAGenerator.generate(molecule, conf, gdma_settings, minimize=True)
        record = MoleculeGDMARecord.from_molecule(molecule, coords, multipoles, gdma_settings)
        gdma_records.append(record)

    # Create charge parameter
    mpfit_parameter = molecule_to_mpfit_library_charge(molecule, symmetrize_hydrogens=False, symmetrize_other_atoms=False)

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

    debug_site(A1, b1, A2, b2, f"Site {site_idx} (Carbon)")


if __name__ == "__main__":
    main()
