import numpy as np
from qcelemental import constants
import qcelemental as qcel


def T_cart(RA, RB):
    dR = RB - RA
    R = np.linalg.norm(dR)
    delta = np.identity(3)
    # E_qq
    T0 = R**-1
    # E_qu
    T1 = (R**-3) * (-1.0 * dR)
    T2 = (R**-5) * (3 * np.outer(dR, dR) - R * R * delta)

    Rdd = np.multiply.outer(dR, delta)
    T3 = (R**-7) * (
        -15 * np.multiply.outer(np.outer(dR, dR), dR)
        + R * R * (Rdd + Rdd.transpose(1, 0, 2) + Rdd.transpose(2, 0, 1))
    )
    RRdd = np.multiply.outer(np.outer(dR, dR), delta)
    dddd = np.multiply.outer(delta, delta)
    # Used for E_QQ
    T4 = (R**-9) * (
        105 * np.multiply.outer(np.outer(dR, dR), np.outer(dR, dR))
        - 15
        * R
        * R
        * (
            RRdd
            + RRdd.transpose(0, 2, 1, 3)
            + RRdd.transpose(0, 3, 2, 1)
            + RRdd.transpose(2, 1, 0, 3)
            + RRdd.transpose(3, 1, 2, 0)
            + RRdd.transpose(2, 3, 0, 1)
        )
        + 3 * (R**4) * (dddd + dddd.transpose(0, 2, 1, 3) + dddd.transpose(0, 3, 2, 1))
    )

    return T0, T1, T2, T3, T4


def eval_interaction(RA, qA, muA, thetaA, RB, qB, muB, thetaB, traceless=False):
    T0, T1, T2, T3, T4 = T_cart(RA, RB)

    # Most inputs will already be traceless, but we can ensure this is the case
    if not traceless:
        traceA = np.trace(thetaA)
        thetaA[0, 0] -= traceA / 3.0
        thetaA[1, 1] -= traceA / 3.0
        thetaA[2, 2] -= traceA / 3.0
        traceB = np.trace(thetaB)
        thetaB[0, 0] -= traceB / 3.0
        thetaB[1, 1] -= traceB / 3.0
        thetaB[2, 2] -= traceB / 3.0

    E_qq = np.sum(T0 * qA * qB)
    E_qu = np.sum(T1 * (qA * muB - qB * muA))
    E_qQ = np.sum(T2 * (qA * thetaB + qB * thetaA)) * (1.0 / 3.0)

    E_uu = np.sum(T2 * np.outer(muA, muB)) * (-1.0)
    E_uQ = np.sum(
        T3 * (np.multiply.outer(muA, thetaB) - np.multiply.outer(muB, thetaA))
    ) * (-1.0 / 3.0)

    E_QQ = np.sum(T4 * np.multiply.outer(thetaA, thetaB)) * (1.0 / 9.0)

    # partial-charge electrostatic energy
    E_q = E_qq

    # dipole correction
    E_u = E_qu + E_uu

    # quadrupole correction
    E_Q = E_qQ + E_uQ + E_QQ

    return E_q + E_u + E_Q


def eval_qcel_dimer(mol_dimer, qA, muA, thetaA, qB, muB, thetaB):
    """
    Evaluate the electrostatic interaction energy between two molecules using
    their multipole moments. Dimensionalities of qA should be [N], muA should
    be [N, 3], and thetaA should be [N, 3, 3]. Same for qB, muB, and thetaB.
    """
    total_energy = 0.0
    RA = mol_dimer.get_fragment(0).geometry
    RB = mol_dimer.get_fragment(1).geometry
    ZA = mol_dimer.get_fragment(0).atomic_numbers
    ZB = mol_dimer.get_fragment(1).atomic_numbers
    for i in range(len(ZA)):
        for j in range(len(ZB)):
            rA = RA[i]
            qA_i = qA[i]
            muA_i = muA[i]
            thetaA_i = thetaA[i]

            rB = RB[j]
            qB_j = qB[j]
            muB_j = muB[j]
            thetaB_j = thetaB[j]

            pair_energy = eval_interaction(
                rA, qA_i, muA_i, thetaA_i, rB, qB_j, muB_j, thetaB_j
            )
            total_energy += pair_energy
    return total_energy * constants.hartree2kcalmol
