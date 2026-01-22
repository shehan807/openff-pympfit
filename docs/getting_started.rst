Getting Started
===============

Installation
------------

**Prerequisites:**

- Python 3.9+
- Psi4 (for GDMA calculations)
- openff-recharge

**Install from source:**

.. code-block:: bash

    git clone https://github.com/shehan807/openff-pympfit.git
    cd openff-pympfit
    pip install -e .

**Note:** GDMA functionality requires Psi4 and PyGDMA:

.. code-block:: bash

    pip install psi4 pygdma

Quick Start
-----------

The MPFIT algorithm derives partial atomic charges by fitting to distributed multipole
analysis (GDMA) data computed from quantum chemistry calculations.

.. code-block:: python

    from openff.toolkit import Molecule
    from openff.recharge.utilities.molecule import extract_conformers
    from openff_pympfit import (
        generate_mpfit_charge_parameter,
        GDMASettings,
        Psi4GDMAGenerator,
        MoleculeGDMARecord,
        MPFITSVDSolver,
    )

    # Load molecule
    molecule = Molecule.from_smiles("CCO")
    molecule.generate_conformers(n_conformers=1)

    # Setup GDMA calculation
    settings = GDMASettings(
        basis="aug-cc-pvdz",
        method="hf",
        limit=2,
    )

    # Generate GDMA data (requires Psi4)
    [conformer] = extract_conformers(molecule)
    coords, multipoles = Psi4GDMAGenerator.generate(molecule, conformer, settings)
    record = MoleculeGDMARecord.from_molecule(molecule, coords, multipoles, settings)

    # Fit charges
    charges = generate_mpfit_charge_parameter([record], MPFITSVDSolver())
    print(f"Charges: {charges.value}")
