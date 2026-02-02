"""Constrained fitting: single-molecule charge fitting with symmetry.

Fits ethanol with constrained MPFIT so that symmetry-equivalent atoms
(e.g., methyl hydrogens) receive identical charges.

Requires Psi4. Run with:
    python examples/tutorials/constrained_fitting.py
"""

from openff.recharge.utilities.molecule import extract_conformers
from openff.toolkit import Molecule
from openff.units.elements import SYMBOLS

from pympfit import (
    ConstrainedSciPySolver,
    GDMASettings,
    MoleculeGDMARecord,
    Psi4GDMAGenerator,
    generate_constrained_mpfit_charge_parameter,
    generate_global_atom_type_labels,
)

# molecule, conformer, and multipoles ---
molecule = Molecule.from_smiles("CCO")
molecule.generate_conformers(n_conformers=1)
[conformer] = extract_conformers(molecule)
settings = GDMASettings()
coords, multipoles = Psi4GDMAGenerator.generate(
    molecule, conformer, settings, minimize=True,
)
record = MoleculeGDMARecord.from_molecule(molecule, coords, multipoles, settings)

# atom type labels from symmetry detection ---
labels = generate_global_atom_type_labels(
    [molecule],
    equivalize_between_methyl_carbons=True,
    equivalize_between_methyl_hydrogens=True,
    equivalize_between_other_heavy_atoms=True,
    equivalize_between_other_hydrogen_atoms=True,
)
print("Atom type labels:")
for i, atom in enumerate(molecule.atoms):
    element = SYMBOLS[atom.atomic_number]
    print(f"  {element}{i + 1:>2d}: {labels[0][i]}")
print()

# constrained fit ---
solver = ConstrainedSciPySolver(conchg=10.0)
[parameter] = generate_constrained_mpfit_charge_parameter(
    [record], [molecule], solver=solver,
)

print("Fitted charges:")
for i, atom in enumerate(molecule.atoms):
    element = SYMBOLS[atom.atomic_number]
    print(f"  {element}{i + 1:>2d} ({labels[0][i]:>3s}): {parameter.value[i]:+.4f}")
print(f"  Total: {sum(parameter.value):+.4f}")
