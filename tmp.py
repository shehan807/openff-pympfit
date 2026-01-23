from openff.toolkit import Molecule
from openff.recharge.utilities.molecule import extract_conformers
from openff_pympfit import (
    generate_mpfit_charge_parameter,
    GDMASettings,
    Psi4GDMAGenerator,
    MoleculeGDMARecord,
    MPFITSVDSolver,
)

# Create molecule and generate conformer
molecule = Molecule.from_smiles("CCO")
molecule.generate_conformers(n_conformers=1)
[conformer] = extract_conformers(molecule)

# Generate GDMA multipoles (requires Psi4)
settings = GDMASettings()
coords, multipoles = Psi4GDMAGenerator.generate(molecule, conformer, settings)
record = MoleculeGDMARecord.from_molecule(molecule, coords, multipoles, settings)

# Fit charges
charges = generate_mpfit_charge_parameter([record], MPFITSVDSolver())
print(f"Charges: {charges.value}")
