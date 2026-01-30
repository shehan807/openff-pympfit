User Guide
==========

This guide covers the main components and workflows in OpenFF PyMPFIT.

GDMA Settings
-------------

The ``GDMASettings`` class configures the quantum chemistry calculation for
distributed multipole analysis.

.. code-block:: python

    from pympfit import GDMASettings

    settings = GDMASettings(
        basis="aug-cc-pvdz",      # Basis set
        method="hf",              # QM method (hf, pbe0, mp2, etc.)
        limit=2,                  # Max multipole rank (2 = up to quadrupole)
        radius=[0.65, 0.65],      # DMA radius parameters
        switch=4.0,               # Switching function parameter
    )

Working with Multiple Conformers
--------------------------------

MPFIT can average over multiple conformers for more robust charge fitting.

.. code-block:: python

    from openff.toolkit import Molecule
    from openff.recharge.utilities.molecule import extract_conformers
    from pympfit import (
        generate_mpfit_charge_parameter,
        GDMASettings,
        Psi4GDMAGenerator,
        MoleculeGDMARecord,
    )

    molecule = Molecule.from_smiles("CCO")
    molecule.generate_conformers(n_conformers=5)

    settings = GDMASettings(basis="aug-cc-pvdz", method="hf", limit=2)

    # Generate GDMA data for each conformer
    records = []
    for conformer in extract_conformers(molecule):
        coords, multipoles = Psi4GDMAGenerator.generate(molecule, conformer, settings)
        records.append(
            MoleculeGDMARecord.from_molecule(molecule, coords, multipoles, settings)
        )

    # Fit charges using all conformers
    library_charge = generate_mpfit_charge_parameter(records=records)

Storing GDMA Records
--------------------

Use ``MoleculeGDMAStore`` to persist GDMA data to a SQLite database.

.. code-block:: python

    from pympfit.gdma.storage import MoleculeGDMAStore

    # Create or open a store
    store = MoleculeGDMAStore("gdma_data.sqlite")

    # Store records
    for record in records:
        store.store(record)

    # Retrieve records by SMILES
    retrieved = store.retrieve(smiles="[H:1][C:2]([H:3])([H:4])[C:5]([H:6])([H:7])[O:8][H:9]")

Solver Options
--------------

MPFIT uses SVD (Singular Value Decomposition) to solve the per-site fitting problem.

.. code-block:: python

    from pympfit import MPFITSVDSolver

    # Default solver with default threshold
    solver = MPFITSVDSolver()

    # Or customize the SVD threshold for numerical stability
    solver = MPFITSVDSolver(svd_threshold=1e-4)

    library_charge = generate_mpfit_charge_parameter(
        records=records,
        solver=solver,
    )
