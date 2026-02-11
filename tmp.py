import psi4
from pympfit import (
    generate_mpfit_charge_parameter,
    Psi4MBISGenerator,
    MBISSettings,
    MoleculeMBISRecord,
    Psi4GDMAGenerator,
    GDMASettings,
    MoleculeGDMARecord,
    MPFITSVDSolver,
)
from openff.units import unit
from openff.recharge.utilities.molecule import extract_conformers
from openff.toolkit import Molecule
import numpy as np
import qcelemental as qcel
from qcelemental import constants
from pprint import pprint as pp


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


# Create molecule
molecule = Molecule.from_smiles("O")
molecule.generate_conformers(n_conformers=1)
[conformer] = extract_conformers(molecule)
print(molecule)
print(dir(molecule))
print(conformer)
mol_dict = molecule.to_dict()
# conformer is np.array([[X, Y, Z], [X, Y, Z], ...])
print(type(conformer[0, 0]))  # <class 'pint.Quantity'>
shift = 2.0
# convert pint.Quantity to float
mol_str = "0 1\n"
mol_str += "\n".join(
    f"{atom['atomic_number']} {conformer[i, 0].to(unit.angstrom).magnitude} {conformer[i, 1].to(unit.angstrom).magnitude} {conformer[i, 2].to(unit.angstrom).magnitude}"
    for i, atom in enumerate(mol_dict["atoms"])
)
mol_str += "\n--\n0 1\n"
mol_str += "\n".join(
    f"{atom['atomic_number']} {conformer[i, 0].to(unit.angstrom).magnitude + shift} {conformer[i, 1].to(unit.angstrom).magnitude + shift} {conformer[i, 2].to(unit.angstrom).magnitude}"
    for i, atom in enumerate(mol_dict["atoms"])
)
mol_str += "\nunits angstrom"
qcel_mol = qcel.models.Molecule.from_data(mol_str)
print(qcel_mol.to_string("xyz"))
print(qcel_mol.to_string("psi4"))


# Generate GDMA multipoles
settings = GDMASettings(limit=2)
coords, gdma_multipoles = Psi4GDMAGenerator.generate(
    molecule, conformer, settings, n_threads=12, memory=32 * unit.gigabyte
)
# Create record and fit charges
record = MoleculeGDMARecord.from_molecule(molecule, coords, gdma_multipoles, settings)
gdma_charges = generate_mpfit_charge_parameter([record], MPFITSVDSolver())

# Generate MBIS multipoles
settings = MBISSettings(max_radial_moments=4, method="hf", basis="aug-cc-pvdz")
coords, multipoles = Psi4MBISGenerator.generate(
    molecule, conformer, settings, n_threads=12, memory=32 * unit.gigabyte
)
print(f"{coords = }")

conformer_2 = conformer.copy()
conformer_2[:, 0] += shift * unit.angstrom
conformer_2[:, 1] += shift * unit.angstrom
coords_2, multipoles_2 = Psi4MBISGenerator.generate(
    molecule, conformer_2, settings, n_threads=12, memory=32 * unit.gigabyte
)

# Create record and fit charges
record = MoleculeMBISRecord.from_molecule(molecule, coords, multipoles, settings)
charges = generate_mpfit_charge_parameter([record], MPFITSVDSolver())

print("GDMA multipoles: ", gdma_multipoles)
print("GDMA charges: ", gdma_multipoles[:, 0])


print("MBIS multipoles: ", multipoles)
print("MBIS charges: ", multipoles[:, 0])

print("fitted charges: ", charges)

# Difference between MBIS and fitted charges
print("MBIS charges vs. MBIS fitted charges:\n", multipoles[:, 0] - charges.value)
# Difference between GDMA and fitted charges
print(
    "GDMA charges vs. GDMA fitted charges:\n",
    gdma_multipoles[:, 0] - gdma_charges.value,
)
# Difference between GDMA and MBIS charges
print("GDMA charges vs. MBIS charges:\n", gdma_multipoles[:, 0] - multipoles[:, 0])


q = multipoles[:, 0]
mu = multipoles[:, 1:4]
# Need to expand theta compact uppper triangle with diagnol back into (3, 3)


def expand_theta(theta_i):
    N = theta_i.shape[0]
    t = np.zeros((N, 3, 3))
    for i in range(N):
        t[i, 0, 0] = theta_i[i, 0]
        t[i, 0, 1] = theta_i[i, 1]
        t[i, 0, 2] = theta_i[i, 2]
        t[i, 1, 0] = theta_i[i, 1]
        t[i, 1, 1] = theta_i[i, 3]
        t[i, 1, 2] = theta_i[i, 4]
        t[i, 2, 0] = theta_i[i, 2]
        t[i, 2, 1] = theta_i[i, 4]
        t[i, 2, 2] = theta_i[i, 5]
    return t


theta = expand_theta(multipoles[:, 4:10])


print("Charges: ", q)
print("Dipoles: ", mu)
print("Quadrupoles: ", theta)

q2 = multipoles[:, 0]
mu2 = multipoles[:, 1:4]
# theta2 = multipoles[:, 4:13].reshape(-1, 3, 3)
theta2 = expand_theta(multipoles_2[:, 4:10])

print("Charges: ", q2)
print("Dipoles: ", mu2)
print("Quadrupoles: ", theta2)

E_elst = eval_qcel_dimer(qcel_mol, q, mu, theta, q2, mu2, theta2)
print(f"elst mtp energy: {E_elst:.4f} kcal/mol")

# sapt0 energy
psi4.core.be_quiet()
psi4.set_num_threads(12)
psi4.set_memory("32 GB")
psi4.set_options(
    {"basis": "aug-cc-pVDZ", "scf_type": "df", "freeze_core": True, "guess": "sad"}
)
psi4.geometry(mol_str)
psi4.energy("sapt0")
qcvars = psi4.core.variables()
sapt0_elst = qcvars["SAPT0 ELST ENERGY"] * constants.hartree2kcalmol
print(f"SAPT0 elst     : {sapt0_elst:.4f} kcal/mol")

# MBIS monA
psi4.set_options(
    {
        "basis": "aug-cc-pVDZ",
        "scf_type": "df",
        "freeze_core": True,
        "guess": "sad",
        "mbis_radial_points": 99,
        "mbis_spherical_points": 590,
        "mbis_d_convergence": 9,
        "max_radial_moment": 3,
    }
)
psi4.geometry(qcel_mol.get_fragment(0).to_string("psi4"))
_, wfn = psi4.energy("hf", return_wfn=True)
psi4.oeprop(wfn, "mbis_charges")
wfn_vars = wfn.variables()
mbis_monA_q = wfn_vars["MBIS CHARGES"].flatten()
mbis_monA_mu = wfn_vars["MBIS DIPOLES"].reshape(-1, 3)
mbis_monA_theta = wfn_vars["MBIS QUADRUPOLES"].reshape(-1, 3, 3)

# MBIS monB
psi4.geometry(qcel_mol.get_fragment(1).to_string("psi4"))
_, wfn = psi4.energy("hf", return_wfn=True)
psi4.oeprop(wfn, "mbis_charges")
wfn_vars = wfn.variables()
# pp(wfn_vars)
mbis_monB_q = wfn_vars["MBIS CHARGES"].flatten()
mbis_monB_mu = wfn_vars["MBIS DIPOLES"].reshape(-1, 3)
mbis_monB_theta = wfn_vars["MBIS QUADRUPOLES"].reshape(-1, 3, 3)

# Electrostatic interaction energy using monomer MBIS multipoles
E_elst_mbis = eval_qcel_dimer(
    qcel_mol,
    mbis_monA_q,
    mbis_monA_mu,
    mbis_monA_theta,
    mbis_monB_q,
    mbis_monB_mu,
    mbis_monB_theta,
)
print(f"elst mtp energy (monomer MBIS multipoles): {E_elst_mbis:.4f} kcal/mol")

# check qs, mus, and thetas are close to those from the dimer calculation
assert np.isclose(mbis_monA_q, q).all(), f"MBIS monA charges:\n{mbis_monA_q}, dimer MBIS charges:\n{q}, difference:\n{mbis_monA_q - q}"
assert np.isclose(mbis_monA_mu, mu).all(), f"MBIS monA dipoles:\n{mbis_monA_mu}, dimer MBIS dipoles:\n{mu}, difference:\n{mbis_monA_mu - mu}"
assert np.isclose(mbis_monA_theta, theta).all(), f"MBIS monA quadrupoles:\n{mbis_monA_theta}, dimer MBIS quadrupoles:\n{theta}, difference:\n{mbis_monA_theta - theta}"
