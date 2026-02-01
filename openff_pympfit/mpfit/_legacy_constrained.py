"""
Constrained MPFIT Implementation

Fits partial charges to reproduce GDMA multipoles with atom-type equivalence constraints.
Atoms with the same type label are constrained to have equal total charges.

Per-molecule charge conservation is enforced via soft penalty terms weighted by conchg.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from openff_pympfit.mpfit.core import _regular_solid_harmonic

if TYPE_CHECKING:
    from openff_pympfit.gdma.storage import MoleculeGDMARecord


@dataclass
class ConstrainedMPFITState:
    """Container for constrained MPFIT state variables."""

    maxl: int = 4
    r1: float = 3.78
    r2: float = 9.45
    molecule_charge: float = 0.0
    conchg: float = 0.0

    atomtype: list[str] = field(default_factory=list)
    quse: np.ndarray | None = None
    allcharge: np.ndarray | None = None
    qstore: np.ndarray | None = None
    multipoles: np.ndarray | None = None
    xyzmult: np.ndarray | None = None
    xyzcharge: np.ndarray | None = None
    lmax: np.ndarray | None = None
    rvdw: np.ndarray | None = None

    # Per-molecule tracking for charge conservation
    atom_counts: list[int] = field(default_factory=list)
    molecule_charges: list[float] = field(default_factory=list)


# --- Setup ---

def setup_from_gdma_records(
    gdma_records: list[MoleculeGDMARecord] | MoleculeGDMARecord,
    atom_type_labels: list[str],
) -> ConstrainedMPFITState:
    """Initialize state from one or more GDMA records.

    For multiple molecules, coordinates and multipoles are stacked into a
    combined system.  Atoms with matching labels (even across different
    molecules) are constrained to share the same charge.

    Parameters
    ----------
    gdma_records : list[MoleculeGDMARecord] or MoleculeGDMARecord
        One or more GDMA records.
    atom_type_labels : list[str]
        Atom type labels for all atoms (concatenated across molecules).
        Length must equal the total number of atoms.
    """
    from openff.toolkit import Molecule
    from openff.units import unit
    from openff_pympfit.mpfit.core import _convert_flat_to_hierarchical

    if not isinstance(gdma_records, list):
        gdma_records = [gdma_records]

    all_xyz = []
    all_multipoles = []
    all_rvdw = []
    all_lmax = []
    atom_counts = []
    total_atoms = 0

    for gdma_record in gdma_records:
        molecule = Molecule.from_mapped_smiles(
            gdma_record.tagged_smiles, allow_undefined_stereo=True
        )
        n_atoms = molecule.n_atoms
        total_atoms += n_atoms
        atom_counts.append(n_atoms)
        gdma_settings = gdma_record.gdma_settings

        conformer_bohr = unit.convert(gdma_record.conformer, unit.angstrom, unit.bohr)
        all_xyz.append(conformer_bohr)

        multipoles = _convert_flat_to_hierarchical(
            gdma_record.multipoles, n_atoms, gdma_settings.limit
        )
        all_multipoles.append(multipoles)

        all_rvdw.append(np.full(n_atoms, gdma_settings.mpfit_atom_radius))
        all_lmax.append(np.full(n_atoms, gdma_settings.limit, dtype=float))

    if len(atom_type_labels) != total_atoms:
        raise ValueError(
            f"atom_type_labels has {len(atom_type_labels)} entries, "
            f"but total atoms across all molecules is {total_atoms}"
        )

    gdma_settings = gdma_records[0].gdma_settings

    state = ConstrainedMPFITState()
    state.r1 = gdma_settings.mpfit_inner_radius
    state.r2 = gdma_settings.mpfit_outer_radius
    state.atomtype = atom_type_labels

    state.xyzcharge = np.vstack(all_xyz)
    state.xyzmult = np.vstack(all_xyz)
    state.multipoles = np.vstack(all_multipoles)
    state.rvdw = np.concatenate(all_rvdw)
    state.lmax = np.concatenate(all_lmax)

    state.atom_counts = atom_counts
    state.quse = build_quse_matrix(state.xyzmult, state.xyzcharge, state.rvdw)
    state.allcharge = np.zeros((total_atoms, total_atoms))
    state.qstore = np.zeros(total_atoms)

    return state


def build_quse_matrix(
    xyzmult: np.ndarray,
    xyzcharge: np.ndarray,
    rvdw: np.ndarray,
) -> np.ndarray:
    """Build binary mask: quse[s,i]=1 if atom i affects site s."""
    from scipy.spatial.distance import cdist
    return (cdist(xyzmult, xyzcharge) < rvdw[:, None]).astype(int)


def count_parameters(state: ConstrainedMPFITState) -> int:
    """Count free parameters after applying atom-type equivalence constraints."""
    atomtype = state.atomtype
    quse = state.quse
    n_atoms = len(atomtype)
    n_params = 0

    for i in range(n_atoms):
        n_sites_using = np.sum(quse[:, i])

        if i == 0:
            n_params += n_sites_using
        else:
            twin = next((k for k in range(i) if atomtype[i] == atomtype[k]), None)
            if twin is not None:
                n_params += n_sites_using - 1
            else:
                n_params += n_sites_using

    return n_params


# --- Core Algorithm ---

def expandcharge(p0: np.ndarray, state: ConstrainedMPFITState) -> None:
    """Map reduced parameters to full charges with atom-type constraints."""
    atomtype = state.atomtype
    quse = state.quse
    n_sites = state.xyzmult.shape[0]
    n_atoms = len(atomtype)

    state.allcharge = np.zeros((n_sites, n_atoms))
    state.qstore = np.zeros(n_atoms)
    count = 0

    for i in range(n_atoms):
        charge_sum = 0.0

        if i == 0:
            for j in range(n_sites):
                if quse[j, i] == 1:
                    state.allcharge[j, i] = p0[count]
                    charge_sum += p0[count]
                    count += 1
            state.qstore[i] = charge_sum
        else:
            twin = next((k for k in range(i) if atomtype[i] == atomtype[k]), None)

            if twin is not None:
                count1 = np.sum(quse[:, i])
                count2 = 1
                for j in range(n_sites):
                    if quse[j, i] == 1 and count2 < count1:
                        state.allcharge[j, i] = p0[count]
                        charge_sum += p0[count]
                        count += 1
                        count2 += 1
                    elif quse[j, i] == 1 and count2 == count1:
                        state.allcharge[j, i] = state.qstore[twin] - charge_sum
                        state.qstore[i] = state.qstore[twin]
            else:
                for j in range(n_sites):
                    if quse[j, i] == 1:
                        state.allcharge[j, i] = p0[count]
                        charge_sum += p0[count]
                        count += 1
                state.qstore[i] = charge_sum


def kaisq(p0: np.ndarray, state: ConstrainedMPFITState) -> float:
    """Objective function: Eq. F_optimization summed over sites + charge penalty.

    Computes Σ_a F^a where F^a = Σ_{l,m} (4π/(2l+1)) W_l [Q_lm^a - Σ_i q_i R_lm(r_i)]²
    plus per-molecule charge conservation: λ Σ_mol (Σ_{i∈mol} q_i - Q_mol)²
    """
    expandcharge(p0, state)

    n_sites = state.xyzmult.shape[0]
    maxl = state.maxl
    sumkai = 0.0

    for s in range(n_sites):
        q0 = state.allcharge[s, :]
        rmax = state.rvdw[s] + state.r2
        rminn = state.rvdw[s] + state.r1

        lmax_s = int(state.lmax[s])
        W = np.zeros(maxl + 1)
        for i in range(lmax_s + 1):
            W[i] = (1.0 / (1.0 - 2.0 * i)) * (rmax ** (1 - 2 * i) - rminn ** (1 - 2 * i))

        dx = state.xyzcharge[:, 0] - state.xyzmult[s, 0]
        dy = state.xyzcharge[:, 1] - state.xyzmult[s, 1]
        dz = state.xyzcharge[:, 2] - state.xyzmult[s, 2]

        site_sum = 0.0
        for l in range(lmax_s + 1):
            weight = (4.0 * np.pi) if l == 0 else (4.0 * np.pi / (2.0 * l + 1.0))
            weight *= W[l]

            for m in range(l + 1):
                cs_range = [0] if m == 0 else [0, 1]
                for cs in cs_range:
                    rsh_vals = _regular_solid_harmonic(l, m, cs, dx, dy, dz)
                    sum1 = np.dot(q0, rsh_vals)
                    site_sum += weight * (state.multipoles[s, l, m, cs] - sum1) ** 2

        sumkai += site_sum

    # Per-molecule charge conservation penalties
    sumcon = 0.0
    if state.molecule_charges and state.atom_counts:
        offset = 0
        for n_i, q_target in zip(state.atom_counts, state.molecule_charges):
            mol_q = np.sum(state.qstore[offset:offset + n_i])
            sumcon += state.conchg * (mol_q - q_target) ** 2
            offset += n_i
    else:
        sumchg = np.sum(state.qstore)
        sumcon = state.conchg * (sumchg - state.molecule_charge) ** 2

    return sumkai + sumcon


def dkaisq(p0: np.ndarray, state: ConstrainedMPFITState) -> np.ndarray:
    """Gradient of kaisq with respect to reduced parameters."""
    expandcharge(p0, state)

    n_sites = state.xyzmult.shape[0]
    n_atoms = n_sites
    maxl = state.maxl
    dparam = np.zeros((n_sites, n_atoms))

    for s in range(n_sites):
        q0 = state.allcharge[s, :]
        rmax = state.rvdw[s] + state.r2
        rminn = state.rvdw[s] + state.r1

        lmax_s = int(state.lmax[s])
        W = np.zeros(maxl + 1)
        for i in range(lmax_s + 1):
            W[i] = (1.0 / (1.0 - 2.0 * i)) * (rmax ** (1 - 2 * i) - rminn ** (1 - 2 * i))

        dx = state.xyzcharge[:, 0] - state.xyzmult[s, 0]
        dy = state.xyzcharge[:, 1] - state.xyzmult[s, 1]
        dz = state.xyzcharge[:, 2] - state.xyzmult[s, 2]

        for l in range(lmax_s + 1):
            weight = (4.0 * np.pi) if l == 0 else (4.0 * np.pi / (2.0 * l + 1.0))
            weight *= W[l]

            for m in range(l + 1):
                cs_range = [0] if m == 0 else [0, 1]
                for cs in cs_range:
                    rsh_vals = _regular_solid_harmonic(l, m, cs, dx, dy, dz)
                    sum1 = np.dot(q0, rsh_vals)
                    coeff = 2.0 * weight * (state.multipoles[s, l, m, cs] - sum1)
                    dparam[s, :] -= coeff * rsh_vals

    # Flatten to atom-major order: dparam1[atom_i * n_sites + site_s]
    dparam1 = dparam.T.flatten()

    # Per-molecule charge conservation gradient
    if state.molecule_charges and state.atom_counts:
        offset = 0
        for n_i, q_target in zip(state.atom_counts, state.molecule_charges):
            mol_q = np.sum(state.qstore[offset:offset + n_i])
            grad_val = state.conchg * 2.0 * (mol_q - q_target)
            for a in range(offset, offset + n_i):
                dparam1[a * n_sites:a * n_sites + n_sites] += grad_val
            offset += n_i
    else:
        sumchg = np.sum(state.qstore)
        dparam1 += state.conchg * 2.0 * (sumchg - state.molecule_charge)

    return createdkaisq(dparam1, state)


def createdkaisq(dparam1: np.ndarray, state: ConstrainedMPFITState) -> np.ndarray:
    """Transform full-space gradient to reduced parameter space."""
    atomtype = state.atomtype
    quse = state.quse
    n_atoms = len(atomtype)
    n_sites = n_atoms
    dparam1 = dparam1.copy()

    # Pass 1: For each twinned atom, redistribute the constrained (last)
    # site's gradient to the twin's entries and negatively to the free sites.
    for i in range(1, n_atoms):
        twin = next((k for k in range(i) if atomtype[i] == atomtype[k]), None)
        if twin is not None:
            count1 = np.sum(quse[:, i])
            count2 = 1
            for j in range(n_sites):
                if quse[j, i] == 1 and count2 < count1:
                    count2 += 1
                elif quse[j, i] == 1 and count2 == count1:
                    for k in range(n_sites):
                        dparam1[twin * n_sites + k] += dparam1[i * n_sites + j]
                    for k in range(j):
                        dparam1[i * n_sites + k] -= dparam1[i * n_sites + j]

    # Pass 2: Extract free-parameter gradient entries
    n_params = count_parameters(state)
    dkaisq_out = np.zeros(n_params)
    count = 0

    for i in range(n_atoms):
        if i == 0:
            for j in range(n_sites):
                if quse[j, i] == 1:
                    dkaisq_out[count] = dparam1[i * n_sites + j]
                    count += 1
        else:
            twin = next((k for k in range(i) if atomtype[i] == atomtype[k]), None)
            if twin is not None:
                count1 = np.sum(quse[:, i])
                count2 = 1
                for j in range(n_sites):
                    if quse[j, i] == 1 and count2 < count1:
                        dkaisq_out[count] = dparam1[i * n_sites + j]
                        count2 += 1
                        count += 1
            else:
                for j in range(n_sites):
                    if quse[j, i] == 1:
                        dkaisq_out[count] = dparam1[i * n_sites + j]
                        count += 1

    return dkaisq_out


# --- Optimization ---

def optimize_constrained(
    state: ConstrainedMPFITState,
    p0_init: np.ndarray | None = None,
) -> dict:
    """Run constrained MPFIT optimization using L-BFGS-B."""
    n_params = count_parameters(state)
    if p0_init is None:
        p0_init = np.zeros(n_params)

    result = minimize(
        fun=lambda p: kaisq(p, state),
        x0=p0_init,
        jac=lambda p: dkaisq(p, state),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-8, "gtol": 1e-6},
    )

    expandcharge(result.x, state)

    return {
        "qstore": state.qstore.copy(),
        "allcharge": state.allcharge.copy(),
        "objective": result.fun,
        "success": result.success,
        "scipy_result": result,
    }


def generate_atom_type_labels_from_symmetry(
    molecule,
    equivalize_hydrogens: bool = True,
    equivalize_other_atoms: bool = True,
) -> list[str]:
    """Generate atom type labels based on molecular symmetry."""
    from openff.recharge.utilities.toolkits import get_atom_symmetries
    from openff.units.elements import SYMBOLS

    symmetry_groups = get_atom_symmetries(molecule)
    labels = []

    for i, (atom, group) in enumerate(zip(molecule.atoms, symmetry_groups)):
        element = SYMBOLS[atom.atomic_number]
        is_hydrogen = atom.atomic_number == 1

        if (is_hydrogen and equivalize_hydrogens) or (not is_hydrogen and equivalize_other_atoms):
            labels.append(f"{element}{group}")
        else:
            labels.append(f"{element}_{i}")

    return labels


def fit_constrained_mpfit(
    gdma_records: list[MoleculeGDMARecord] | MoleculeGDMARecord,
    atom_type_labels: list[str],
    molecule_charges: list[float] | float = 0.0,
    conchg: float = 1.0,
) -> dict:
    """Fit constrained MPFIT charges for one or more molecules.

    Parameters
    ----------
    gdma_records : list[MoleculeGDMARecord] or MoleculeGDMARecord
        One or more GDMA records.
    atom_type_labels : list[str]
        Atom type labels for all atoms (concatenated across molecules).
    molecule_charges : list[float] or float
        Target charge for each molecule. If a single float, applied to every
        molecule. If a list, must match the number of GDMA records.
    conchg : float
        Weight for the charge conservation penalty (default 1.0).

    Returns
    -------
    dict with keys: qstore, allcharge, objective, success, scipy_result
    """
    if not isinstance(gdma_records, list):
        gdma_records = [gdma_records]

    state = setup_from_gdma_records(gdma_records, atom_type_labels)
    state.conchg = conchg

    if isinstance(molecule_charges, (int, float)):
        state.molecule_charges = [float(molecule_charges)] * len(gdma_records)
    else:
        if len(molecule_charges) != len(gdma_records):
            raise ValueError(
                f"molecule_charges has {len(molecule_charges)} entries, "
                f"but there are {len(gdma_records)} GDMA records"
            )
        state.molecule_charges = [float(q) for q in molecule_charges]

    state.molecule_charge = sum(state.molecule_charges)
    return optimize_constrained(state)


# --- Test / Main ---

def test_molecule(
    name: str,
    smiles: str,
    labels: list[str],
    expected_charge: int = 0,
) -> dict:
    """Test constrained MPFIT on a single molecule."""
    import time
    from openff.toolkit import Molecule
    from openff.recharge.utilities.molecule import extract_conformers
    from openff.units.elements import SYMBOLS
    from openff_pympfit.gdma import GDMASettings
    from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
    from openff_pympfit.gdma.storage import MoleculeGDMARecord

    timings = {}
    t_total_start = time.time()

    print("\n" + "=" * 70)
    print(f"Testing: {name}")
    print("=" * 70)

    t_start = time.time()
    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    molecule.generate_conformers(n_conformers=1)
    [conformer] = extract_conformers(molecule)
    timings["molecule_setup"] = time.time() - t_start

    print(f"\nSMILES: {smiles}")
    print(f"Atoms: {molecule.n_atoms}, Charge: {molecule.total_charge}")

    print("\n--- Psi4 GDMA ---")
    gdma_settings = GDMASettings()

    t_start = time.time()
    conformer, multipoles = Psi4GDMAGenerator.generate(
        molecule, conformer, gdma_settings, minimize=True,
    )
    gdma_record = MoleculeGDMARecord.from_molecule(
        molecule, conformer, multipoles, gdma_settings,
    )
    timings["psi4_gdma"] = time.time() - t_start

    print(f"  [Psi4: {timings['psi4_gdma']:.2f}s]")

    print(f"\n--- Atom Types ---")
    print(f"Labels: {labels}")

    equiv_classes = {}
    for i, label in enumerate(labels):
        equiv_classes.setdefault(label, []).append(i)

    print("\nEquivalence classes:")
    for label, indices in equiv_classes.items():
        atoms_str = ", ".join(
            f"{i}({SYMBOLS[molecule.atoms[i].atomic_number]})" for i in indices
        )
        print(f"  {label}: [{atoms_str}]")

    t_start = time.time()
    state = setup_from_gdma_records(gdma_record, labels)
    state.molecule_charge = float(expected_charge)
    n_full = int(np.sum(state.quse))
    n_reduced = count_parameters(state)
    timings["mpfit_setup"] = time.time() - t_start

    print(f"\n--- Setup ---")
    print(f"Params: {n_reduced}/{n_full} (saved {n_full - n_reduced})")

    print("\n--- Optimization ---")
    t_start = time.time()
    result = optimize_constrained(state)
    timings["optimization"] = time.time() - t_start

    print(f"  [{timings['optimization']:.2f}s]")

    print("\n--- Final Charges ---")
    for i, (label, q) in enumerate(zip(labels, result["qstore"])):
        print(f"  {i:2d} ({SYMBOLS[molecule.atoms[i].atomic_number]:2s}, {label}): {q:+.6f}")

    all_satisfied = True
    constraint_results = {}
    for label, indices in equiv_classes.items():
        if len(indices) > 1:
            charges = [result["qstore"][i] for i in indices]
            max_diff = max(charges) - min(charges)
            satisfied = max_diff < 1e-10
            if not satisfied:
                all_satisfied = False
            constraint_results[label] = {"max_diff": max_diff, "satisfied": satisfied}

    print("\n--- Constraints ---")
    for label, info in constraint_results.items():
        print(f"  {label}: {info['max_diff']:.2e} [{'PASS' if info['satisfied'] else 'FAIL'}]")
    print(f"\nTotal charge: {np.sum(result['qstore']):.6f} (expected: {expected_charge})")

    timings["total"] = time.time() - t_total_start

    return {
        "name": name, "n_atoms": molecule.n_atoms, "qstore": result["qstore"],
        "labels": labels, "all_satisfied": all_satisfied,
        "total_charge": np.sum(result["qstore"]), "timings": timings,
    }


def main():
    """Test constrained MPFIT on ethanol."""
    from openff.toolkit import Molecule

    print("=" * 70)
    print("Constrained MPFIT — Ethanol Test")
    print("=" * 70)

    molecule = Molecule.from_smiles("CCO")
    labels = generate_atom_type_labels_from_symmetry(molecule)
    test_molecule("Ethanol", "CCO", labels, expected_charge=0)


if __name__ == "__main__":
    main()
