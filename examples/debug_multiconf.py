"""Debug multi-conformer fitting to understand why charges blow up."""

import numpy as np
from openff.toolkit.topology import Molecule
from openff.recharge.utilities.molecule import extract_conformers
from openff.units import unit

from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
from openff_pympfit.gdma.storage import MoleculeGDMARecord
from openff_pympfit.optimize import MPFITObjective
from openff.recharge.charges.library import LibraryChargeCollection, LibraryChargeParameter
from openff_pympfit.mpfit._mpfit import molecule_to_mpfit_library_charge


def debug_molecule(smiles, n_conf=2):
    print(f"Debugging: {smiles}")
    print("=" * 70)

    molecule = Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=20, rms_cutoff=0.5 * unit.angstrom)
    if molecule.n_conformers > n_conf:
        molecule._conformers = molecule._conformers[:n_conf]
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

    # Get objective terms for each conformer
    print("\n" + "=" * 70)
    print("OBJECTIVE TERMS ANALYSIS")
    print("=" * 70)

    all_objective_terms = []
    for i, record in enumerate(gdma_records):
        terms_and_masks = list(MPFITObjective.compute_objective_terms(
            [record],
            charge_collection=LibraryChargeCollection(parameters=[mpfit_parameter]),
            charge_parameter_keys=[(mpfit_parameter.smiles, tuple(range(len(mpfit_parameter.value))))],
            return_quse_masks=True,
        ))
        term, mask_dict = terms_and_masks[0]
        all_objective_terms.append((term, mask_dict["quse_masks"]))

        print(f"\n--- Conformer {i+1} ---")
        print(f"Number of sites: {len(term.atom_charge_design_matrix)}")
        for site_idx in range(len(term.atom_charge_design_matrix)):
            A = term.atom_charge_design_matrix[site_idx]
            b = term.reference_values[site_idx]
            quse_mask = mask_dict["quse_masks"][site_idx]
            print(f"  Site {site_idx}: A shape={A.shape}, b shape={b.shape}, quse_mask sum={sum(quse_mask)}")

    # Compare A matrices and b vectors between conformers
    print("\n" + "=" * 70)
    print("COMPARING CONFORMERS")
    print("=" * 70)

    term1, masks1 = all_objective_terms[0]
    term2, masks2 = all_objective_terms[1]

    for site_idx in range(len(term1.atom_charge_design_matrix)):
        A1 = term1.atom_charge_design_matrix[site_idx]
        A2 = term2.atom_charge_design_matrix[site_idx]
        b1 = term1.reference_values[site_idx]
        b2 = term2.reference_values[site_idx]

        A_diff = np.max(np.abs(A1 - A2))
        b_diff = np.max(np.abs(b1 - b2))

        print(f"Site {site_idx}: max|A1-A2|={A_diff:.6f}, max|b1-b2|={b_diff:.6f}")

    # Solve each site individually for each conformer
    print("\n" + "=" * 70)
    print("PER-SITE SOLUTIONS (SVD)")
    print("=" * 70)

    svd_threshold = 1e-4

    for conf_idx, (term, masks) in enumerate(all_objective_terms):
        print(f"\n--- Conformer {conf_idx+1} ---")
        for site_idx in range(len(term.atom_charge_design_matrix)):
            A = term.atom_charge_design_matrix[site_idx]
            b = term.reference_values[site_idx]

            U, S, Vh = np.linalg.svd(A, full_matrices=False)
            S_thresh = S.copy()
            S_thresh[svd_threshold > S_thresh] = 0.0
            inv_S = np.zeros_like(S_thresh)
            mask_S = S_thresh != 0
            inv_S[mask_S] = 1.0 / S_thresh[mask_S]
            q = (Vh.T * inv_S) @ (U.T @ b)

            print(f"  Site {site_idx}: q = {np.round(q.flatten(), 4)}, S = {np.round(S, 4)}")

    # Now solve with stacked matrices
    print("\n" + "=" * 70)
    print("STACKED SOLUTIONS (combined conformers)")
    print("=" * 70)

    for site_idx in range(len(term1.atom_charge_design_matrix)):
        A1 = term1.atom_charge_design_matrix[site_idx]
        A2 = term2.atom_charge_design_matrix[site_idx]
        b1 = term1.reference_values[site_idx]
        b2 = term2.reference_values[site_idx]

        A_stacked = np.vstack([A1, A2])
        b_stacked = np.concatenate([b1, b2])

        U, S, Vh = np.linalg.svd(A_stacked, full_matrices=False)
        S_thresh = S.copy()
        S_thresh[svd_threshold > S_thresh] = 0.0
        inv_S = np.zeros_like(S_thresh)
        mask_S = S_thresh != 0
        inv_S[mask_S] = 1.0 / S_thresh[mask_S]
        q_stacked = (Vh.T * inv_S) @ (U.T @ b_stacked)

        # Also compute individual solutions for comparison
        U1, S1, Vh1 = np.linalg.svd(A1, full_matrices=False)
        S1_thresh = S1.copy()
        S1_thresh[svd_threshold > S1_thresh] = 0.0
        inv_S1 = np.zeros_like(S1_thresh)
        mask_S1 = S1_thresh != 0
        inv_S1[mask_S1] = 1.0 / S1_thresh[mask_S1]
        q1 = (Vh1.T * inv_S1) @ (U1.T @ b1)

        print(f"Site {site_idx}:")
        print(f"  Conf1 q: {np.round(q1.flatten(), 4)}")
        print(f"  Stacked q: {np.round(q_stacked.flatten(), 4)}")
        print(f"  Stacked S: {np.round(S, 6)}")
        print(f"  Condition number: {S[0]/S[-1] if S[-1] > 0 else 'inf':.2f}")


if __name__ == "__main__":
    debug_molecule("FCCF", n_conf=2)
