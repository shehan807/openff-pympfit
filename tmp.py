from openff.toolkit import Molecule
from openff.recharge.utilities.molecule import extract_conformers
from openff_pympfit import (
    generate_mpfit_charge_parameter,
    GDMASettings,
    Psi4GDMAGenerator,
    MoleculeGDMARecord,
    MPFITSVDSolver,
    MBISSettings,
    Psi4MBISGenerator,
    # MoleculeMBISRecord,
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

# Generate MBIS multipoles (requires Psi4)
settings = MBISSettings()
coords, multipoles = Psi4MBISGenerator.generate(molecule, conformer, settings, compute_mp=True)
record = MoleculeGDMARecord.from_molecule(molecule, coords, multipoles, settings)

# Fit charges
charges = generate_mpfit_charge_parameter([record], MPFITSVDSolver())
print(f"Charges: {charges.value}")
