import psi4
from pympfit import (
    generate_mpfit_charge_parameter,
    Psi4MBISGenerator,
    MBISSettings,
    MoleculeMBISRecord,
    MPFITSVDSolver,
)
from pympfit.mbis.multipole_transform import (
    spherical_to_cartesian_multipoles,
    flat_to_cartesian_multipoles,
)
from openff.units import unit
from openff.recharge.utilities.molecule import extract_conformers
from openff.toolkit import Molecule
import numpy as np
import qcelemental as qcel
from qcelemental import constants
from pprint import pprint as pp

from pympfit.mbis.evaluate_cartesian_multipoles import eval_qcel_dimer


# Create molecule
molecule = Molecule.from_smiles("O")
molecule.generate_conformers(n_conformers=1)
[conformer] = extract_conformers(molecule)
mol_dict = molecule.to_dict()
# conformer is np.array([[X, Y, Z], [X, Y, Z], ...])
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

# Generate MBIS multipoles - use Cartesian format for testing
settings = MBISSettings(
    max_radial_moment=3,
    max_moment=2,
    limit=2,
    method="hf",
    basis="aug-cc-pvdz",
    multipole_format="cartesian",
)
coords, multipoles = Psi4MBISGenerator.generate(
    molecule, conformer, settings, n_threads=12, memory=32 * unit.gigabyte
)

conformer_2 = conformer.copy()
conformer_2[:, 0] += shift * unit.angstrom
conformer_2[:, 1] += shift * unit.angstrom
coords_2, multipoles_2 = Psi4MBISGenerator.generate(
    molecule, conformer_2, settings, n_threads=12, memory=32 * unit.gigabyte
)

# Create record and fit charges
record = MoleculeMBISRecord.from_molecule(molecule, coords, multipoles, settings)
charges = generate_mpfit_charge_parameter([record], MPFITSVDSolver())

print("MBIS multipoles: ", multipoles)
print("MBIS charges: ", multipoles[:, 0])

print("fitted charges: ", charges)

# Difference between MBIS and fitted charges
print("MBIS charges vs. MBIS fitted charges:\n", multipoles[:, 0] - charges.value)


# Convert multipoles to Cartesian for evaluation
# Check if the multipoles are in spherical or Cartesian format
if settings.multipole_format == "cartesian":
    charges, dipoles, quadrupoles, octupoles = flat_to_cartesian_multipoles(
        multipoles, max_moment=2
    )
else:
    charges, dipoles, quadrupoles, octupoles = spherical_to_cartesian_multipoles(
        multipoles, max_moment=2
    )
q = charges
mu = dipoles
theta = quadrupoles

print("Charges: ", q)
print("Dipoles: ", mu)
print("Quadrupoles: ", theta)

# Convert second conformer multipoles to Cartesian
if settings.multipole_format == "cartesian":
    charges_2, dipoles_2, quadrupoles_2, octupoles_2 = flat_to_cartesian_multipoles(
        multipoles_2, max_moment=2
    )
else:
    charges_2, dipoles_2, quadrupoles_2, octupoles_2 = (
        spherical_to_cartesian_multipoles(multipoles_2, max_moment=2)
    )
q2 = charges_2
mu2 = dipoles_2
theta2 = quadrupoles_2

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
assert np.allclose(mbis_monA_q, q, rtol=1e-5, atol=1e-5), (
    f"MBIS monA charges:\n{mbis_monA_q}, dimer MBIS charges:\n{q}, difference:\n{mbis_monA_q - q}"
)
assert np.allclose(mbis_monA_mu, mu, rtol=1e-5, atol=1e-5), (
    f"MBIS monA dipoles:\n{mbis_monA_mu}, dimer MBIS dipoles:\n{mu}, difference:\n{mbis_monA_mu - mu}"
)
assert np.allclose(mbis_monA_theta, theta, rtol=1e-5, atol=1e-5), (
    f"MBIS monA quadrupoles:\n{mbis_monA_theta}, dimer MBIS quadrupoles:\n{theta}, difference:\n{mbis_monA_theta - theta}"
)
