"""Evaluate electrostatic interaction energies using Cartesian multipoles.

This module provides functions to compute multipole interaction tensors and
evaluate electrostatic energies between molecular fragments using multipole
expansions up to quadrupole order.
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from qcelemental import constants

if TYPE_CHECKING:
    import qcelemental as qcel


def compute_multipole_interaction_tensors(
    r_a: NDArray[np.float64],
    r_b: NDArray[np.float64],
) -> tuple[
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Compute interaction tensors for multipole expansions in Cartesian coordinates.

    Calculates the T-tensors used in multipole-multipole interaction energy
    expressions. These tensors encode the geometric dependence of electrostatic
    interactions between multipole moments.

    Parameters
    ----------
    r_a
        Position vector of site A in Cartesian coordinates [Bohr], shape (3,).
    r_b
        Position vector of site B in Cartesian coordinates [Bohr], shape (3,).

    Returns
    -------
    tuple
        A 5-tuple containing:
        - T0: Monopole-monopole interaction tensor (scalar).
        - T1: Monopole-dipole interaction tensor, shape (3,).
        - T2: Monopole-quadrupole and dipole-dipole interaction tensor, shape (3, 3).
        - T3: Dipole-quadrupole interaction tensor, shape (3, 3, 3).
        - T4: Quadrupole-quadrupole interaction tensor, shape (3, 3, 3, 3).

    Notes
    -----
    The T-tensors are defined as derivatives of 1/R with respect to Cartesian
    coordinates and follow standard conventions in molecular electrostatics
    (e.g., Stone, "The Theory of Intermolecular Forces", 2nd ed., 2013).
    """
    d_r = r_b - r_a
    r = np.linalg.norm(d_r)
    delta = np.identity(3)

    # T0: Monopole-monopole (charge-charge) interaction
    t0 = r**-1

    # T1: Monopole-dipole (charge-dipole) interaction
    t1 = (r**-3) * (-1.0 * d_r)

    # T2: Monopole-quadrupole and dipole-dipole interactions
    t2 = (r**-5) * (3 * np.outer(d_r, d_r) - r * r * delta)

    # T3: Dipole-quadrupole interaction
    r_dd = np.multiply.outer(d_r, delta)
    t3 = (r**-7) * (
        -15 * np.multiply.outer(np.outer(d_r, d_r), d_r)
        + r * r * (r_dd + r_dd.transpose(1, 0, 2) + r_dd.transpose(2, 0, 1))
    )

    # T4: Quadrupole-quadrupole interaction
    rr_dd = np.multiply.outer(np.outer(d_r, d_r), delta)
    dddd = np.multiply.outer(delta, delta)
    t4 = (r**-9) * (
        105 * np.multiply.outer(np.outer(d_r, d_r), np.outer(d_r, d_r))
        - 15
        * r
        * r
        * (
            rr_dd
            + rr_dd.transpose(0, 2, 1, 3)
            + rr_dd.transpose(0, 3, 2, 1)
            + rr_dd.transpose(2, 1, 0, 3)
            + rr_dd.transpose(3, 1, 2, 0)
            + rr_dd.transpose(2, 3, 0, 1)
        )
        + 3 * (r**4) * (dddd + dddd.transpose(0, 2, 1, 3) + dddd.transpose(0, 3, 2, 1))
    )

    return t0, t1, t2, t3, t4


def evaluate_interaction_energy(
    r_a: NDArray[np.float64],
    q_a: float,
    mu_a: NDArray[np.float64],
    theta_a: NDArray[np.float64],
    r_b: NDArray[np.float64],
    q_b: float,
    mu_b: NDArray[np.float64],
    theta_b: NDArray[np.float64],
    traceless: bool = False,
) -> float:
    """Evaluate electrostatic interaction energy between two multipole sites.

    Computes the interaction energy between two sites using their multipole
    moments (charge, dipole, quadrupole) in Cartesian representation.

    Parameters
    ----------
    r_a
        Position vector of site A in Cartesian coordinates [Bohr], shape (3,).
    q_a
        Charge at site A [atomic units].
    mu_a
        Dipole moment vector at site A [atomic units], shape (3,).
    theta_a
        Quadrupole moment tensor at site A [atomic units], shape (3, 3).
    r_b
        Position vector of site B in Cartesian coordinates [Bohr], shape (3,).
    q_b
        Charge at site B [atomic units].
    mu_b
        Dipole moment vector at site B [atomic units], shape (3,).
    theta_b
        Quadrupole moment tensor at site B [atomic units], shape (3, 3).
    traceless
        If False (default), ensures quadrupole tensors are made traceless before
        computing the interaction. Set to True if inputs are already traceless.

    Returns
    -------
    float
        Total electrostatic interaction energy [Hartree] including charge-charge,
        charge-dipole, charge-quadrupole, dipole-dipole, dipole-quadrupole, and
        quadrupole-quadrupole contributions.

    Notes
    -----
    The quadrupole tensors must be traceless for correct energy evaluation.
    If traceless=False, this function modifies the input tensors in-place to
    remove their traces.
    """
    t0, t1, t2, t3, t4 = compute_multipole_interaction_tensors(r_a, r_b)

    # Ensure quadrupoles are traceless
    if not traceless:
        trace_a = np.trace(theta_a)
        theta_a[0, 0] -= trace_a / 3.0
        theta_a[1, 1] -= trace_a / 3.0
        theta_a[2, 2] -= trace_a / 3.0
        trace_b = np.trace(theta_b)
        theta_b[0, 0] -= trace_b / 3.0
        theta_b[1, 1] -= trace_b / 3.0
        theta_b[2, 2] -= trace_b / 3.0

    # Charge-charge interaction
    e_qq = np.sum(t0 * q_a * q_b)

    # Charge-dipole interactions
    e_qu = np.sum(t1 * (q_a * mu_b - q_b * mu_a))

    # Charge-quadrupole interactions
    e_q_theta = np.sum(t2 * (q_a * theta_b + q_b * theta_a)) * (1.0 / 3.0)

    # Dipole-dipole interaction
    e_uu = np.sum(t2 * np.outer(mu_a, mu_b)) * (-1.0)

    # Dipole-quadrupole interactions
    e_u_theta = np.sum(
        t3 * (np.multiply.outer(mu_a, theta_b) - np.multiply.outer(mu_b, theta_a))
    ) * (-1.0 / 3.0)

    # Quadrupole-quadrupole interaction
    e_theta_theta = np.sum(t4 * np.multiply.outer(theta_a, theta_b)) * (1.0 / 9.0)

    # Sum all contributions
    e_charge = e_qq
    e_dipole = e_qu + e_uu
    e_quadrupole = e_q_theta + e_u_theta + e_theta_theta

    return e_charge + e_dipole + e_quadrupole


def evaluate_dimer_interaction_energy(
    mol_dimer: "qcel.models.Molecule",
    charges_a: NDArray[np.float64],
    dipoles_a: NDArray[np.float64],
    quadrupoles_a: NDArray[np.float64],
    charges_b: NDArray[np.float64],
    dipoles_b: NDArray[np.float64],
    quadrupoles_b: NDArray[np.float64],
) -> float:
    """Evaluate total electrostatic interaction energy between two molecular fragments.

    Computes the sum of all pairwise atom-atom electrostatic interactions between
    two molecular fragments using their atomic multipole moments.

    Parameters
    ----------
    mol_dimer
        QCElemental Molecule object containing two fragments. Fragment 0 corresponds
        to molecule A, and fragment 1 corresponds to molecule B.
    charges_a
        Atomic charges for molecule A [atomic units], shape (N_A,).
    dipoles_a
        Atomic dipole moment vectors for molecule A [atomic units], shape (N_A, 3).
    quadrupoles_a
        Atomic quadrupole moment tensors for molecule A [atomic units],
        shape (N_A, 3, 3). Must be traceless.
    charges_b
        Atomic charges for molecule B [atomic units], shape (N_B,).
    dipoles_b
        Atomic dipole moment vectors for molecule B [atomic units], shape (N_B, 3).
    quadrupoles_b
        Atomic quadrupole moment tensors for molecule B [atomic units],
        shape (N_B, 3, 3). Must be traceless.

    Returns
    -------
    float
        Total electrostatic interaction energy [kcal/mol].

    Notes
    -----
    The energy is converted from Hartree to kcal/mol using QCElemental's conversion
    constants. All atomic coordinates are obtained from the molecule object in Bohr.
    """
    total_energy = 0.0
    geom_a = mol_dimer.get_fragment(0).geometry
    geom_b = mol_dimer.get_fragment(1).geometry
    n_atoms_a = len(mol_dimer.get_fragment(0).atomic_numbers)
    n_atoms_b = len(mol_dimer.get_fragment(1).atomic_numbers)

    for i in range(n_atoms_a):
        for j in range(n_atoms_b):
            r_a = geom_a[i]
            q_a_i = charges_a[i]
            mu_a_i = dipoles_a[i]
            theta_a_i = quadrupoles_a[i]

            r_b = geom_b[j]
            q_b_j = charges_b[j]
            mu_b_j = dipoles_b[j]
            theta_b_j = quadrupoles_b[j]

            pair_energy = evaluate_interaction_energy(
                r_a, q_a_i, mu_a_i, theta_a_i, r_b, q_b_j, mu_b_j, theta_b_j
            )
            total_energy += pair_energy

    return total_energy * constants.hartree2kcalmol
