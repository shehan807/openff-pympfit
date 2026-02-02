"""Quickstart: unconstrained MPFIT charge fitting for ethanol.

Requires Psi4. Run with:
    python examples/tutorials/quickstart.py
"""

import time

from openff.recharge.utilities.molecule import extract_conformers
from openff.toolkit import Molecule
from openff.units.elements import SYMBOLS

from pympfit import (
    GDMASettings,
    MoleculeGDMARecord,
    MPFITSVDSolver,
    Psi4GDMAGenerator,
    generate_mpfit_charge_parameter,
)

# Settings
settings = GDMASettings(
    method="pbe0",
    basis="def2-SVP",
    limit=4,
    switch=4.0,
    radius=["C", 0.53, "O", 0.53, "H", 0.53],
    mpfit_inner_radius=6.78,
    mpfit_outer_radius=12.45,
    mpfit_atom_radius=3.0,
)

# Conformer and multipoles
molecule = Molecule.from_smiles("CCO")
molecule.generate_conformers(n_conformers=1)
[conformer] = extract_conformers(molecule)

print(f"Molecule: {molecule.to_smiles()} ({molecule.n_atoms} atoms)")

t0 = time.perf_counter()
coords, multipoles = Psi4GDMAGenerator.generate(
    molecule,
    conformer,
    settings,
    minimize=True,
)
print(f"GDMA generation: {time.perf_counter() - t0:.2f}s")
print()

# Fit charges (SVD)
record = MoleculeGDMARecord.from_molecule(molecule, coords, multipoles, settings)
solver = MPFITSVDSolver(svd_threshold=1e-4)
parameter = generate_mpfit_charge_parameter([record], solver)

print("Fitted charges:")
for i, atom in enumerate(molecule.atoms):
    element = SYMBOLS[atom.atomic_number]
    print(f"  {element}{i + 1:>2d}: {parameter.value[i]:+.4f}")
print(f"  Total: {sum(parameter.value):+.4f}")
