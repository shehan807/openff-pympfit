import numpy
from openff.toolkit.topology import Molecule

# From openff-recharge (dependency)
from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeGenerator,
)
from openff.recharge.utilities.molecule import extract_conformers

# From openff-pympfit
from openff_pympfit.mpfit import generate_mpfit_charge_parameter
from openff_pympfit.mpfit.solvers import MPFITSVDSolver
from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
from openff_pympfit.gdma.storage import MoleculeGDMARecord


def main():
    qc_data_settings = GDMASettings(
        #method="pbe0",
        #basis="def2-SVP",
    )

    molecule: Molecule = Molecule.from_mapped_smiles(
        "[C:1]([H:5])([H:6])([H:8])[C:2]([H:7])([H:9])[O:3][H:4]"
    )
    molecule.generate_conformers(n_conformers=1)

    [input_conformer] = extract_conformers(molecule)

    conformer, mp = Psi4GDMAGenerator.generate(
        molecule, input_conformer, qc_data_settings, minimize=True
    )
    
    qc_data_record = MoleculeGDMARecord.from_molecule(
       molecule, conformer, mp, qc_data_settings
    )
    print("Multipole moments shape:", mp.shape)
    mpfit_solver = MPFITSVDSolver(svd_threshold=1.0e-4)

    mpfit_charge_parameter = generate_mpfit_charge_parameter(
       [qc_data_record], mpfit_solver
    )
    # TODO: Fix tolerance issue with conda openff-recharge 0.5.3 vs fork
    # mpfit_charges = LibraryChargeGenerator.generate(
    #    molecule, LibraryChargeCollection(parameters=[mpfit_charge_parameter])
    # )

    print(f"MPFIT SMILES         : {mpfit_charge_parameter.smiles}")
    print(f"MPFIT VALUES (UNIQUE): {numpy.round(mpfit_charge_parameter.value, 4)}")
    print("")

    # Print each atom with its corresponding charge
    print("Atom-by-atom charges:")
    from openff.units.elements import SYMBOLS

    for i, atom in enumerate(molecule.atoms):
        element = SYMBOLS[atom.atomic_number]
        print(f"Atom {i} ({element}): {mpfit_charge_parameter.value[i]:.4f}")


if __name__ == "__main__":
    main()
