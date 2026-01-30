"""Quickstart Tutorial: Fitting partial charges with PyMPFIT.

Requires Psi4. Run with:
    python examples/tutorials/quickstart.py
"""

import time

import numpy as np
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

# ---------------------------------------------------------------------------
# 1. QM and GDMA Settings
# ---------------------------------------------------------------------------
# GDMASettings controls both the Psi4 QM calculation and the GDMA/MPFIT
# integration parameters.

settings = GDMASettings(
    method="pbe0",              # DFT functional (or "hf", "mp2", etc.)
    basis="def2-SVP",           # Basis set for the QM calculation
    limit=4,                    # Max multipole rank (4 = hexadecapole)
    switch=4.0,                 # Exponent sum threshold for grid-based DMA
    radius=[                    # Per-element DMA radii (angstrom)
        "C", 0.53,
        "O", 0.53,
        "H", 0.53,
    ],
    mpfit_inner_radius=6.78,    # Inner integration radius (bohr)
    mpfit_outer_radius=12.45,   # Outer integration radius (bohr)
    mpfit_atom_radius=3.0,      # Atom inclusion radius for per-site fitting (bohr)
)

print("GDMA Settings:")
print(f"  Method: {settings.method}")
print(f"  Basis:  {settings.basis}")
print(f"  Limit:  {settings.limit}")
print()

# ---------------------------------------------------------------------------
# 2. Generate Conformer
# ---------------------------------------------------------------------------
# Create an ethanol molecule and generate a single conformer.

molecule = Molecule.from_smiles("CCO")
molecule.generate_conformers(n_conformers=1)
[conformer] = extract_conformers(molecule)

print(f"Molecule: {molecule.to_smiles()} ({molecule.n_atoms} atoms)")
print()

# ---------------------------------------------------------------------------
# 3. Generate Multipoles
# ---------------------------------------------------------------------------
# Run Psi4 to compute the wavefunction and GDMA multipole moments.
# minimize=True optimizes the geometry at the same level of theory first.

t0 = time.perf_counter()
coords, multipoles = Psi4GDMAGenerator.generate(
    molecule, conformer, settings, minimize=True
)
elapsed = time.perf_counter() - t0

print(f"Multipoles shape: {multipoles.shape}")
print(f"GDMA generation time: {elapsed:.2f}s")
print()

# ---------------------------------------------------------------------------
# 4. Fit Charges
# ---------------------------------------------------------------------------
# Create a GDMA record and solve for partial charges using SVD.

record = MoleculeGDMARecord.from_molecule(molecule, coords, multipoles, settings)
solver = MPFITSVDSolver(svd_threshold=1e-4)
parameter = generate_mpfit_charge_parameter([record], solver)

print("Fitted charges:")
for i, atom in enumerate(molecule.atoms):
    element = SYMBOLS[atom.atomic_number]
    print(f"  {element}{i + 1:>2d}: {parameter.value[i]:+.4f}")
print(f"  Total: {sum(parameter.value):+.4f}")
