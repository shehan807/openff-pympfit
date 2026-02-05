import numpy as np
import pytest
from openff.toolkit import Molecule
from openff.units import unit

from pympfit import GDMASettings, MoleculeGDMARecord
from pympfit.gdma.psi4 import Psi4GDMAGenerator


@pytest.fixture
def default_gdma_settings() -> GDMASettings:
    """Default GDMA settings used by test_gdma.py input generation tests."""
    return GDMASettings(
        limit=4,
        basis="def2-SVP",
        method="pbe0",
        switch=4.0,
        radius=["C", 0.53, "N", 0.53, "H", 0.53],
    )


@pytest.fixture
def sto3g_gdma_settings() -> GDMASettings:
    """STO-3G GDMA settings shared across tests."""
    return GDMASettings(
        method="scf",
        basis="sto-3g",
        switch=0.0,
        radius=["C", 0.53, "O", 0.53, "N", 0.53, "H", 0.53, "F", 0.53],
    )


@pytest.fixture
def meoh_gdma_sto3g(sto3g_gdma_settings) -> MoleculeGDMARecord:
    """Generate GDMA data for methanol using STO-3G basis."""
    gdma_settings = sto3g_gdma_settings

    mol = Molecule.from_mapped_smiles("[C:1]([O:2][H:6])([H:3])([H:4])[H:5]")

    input_conformer = unit.Quantity(
        np.array(
            [
                [-0.3507534772694063, -0.005072983373261072, -0.01802813259570673],
                [0.9643704239531461, -0.362402385308100760, -0.26011999900333500],
                [-0.5984566510608367, 0.061649323412126030, 1.05237343039367470],
                [-0.9862545465164105, -0.815675279603002200, -0.46260343550818340],
                [-0.6578568673402391, 0.926898309244869800, -0.50107705552408850],
                [1.6289511182337486, 0.194603015627369500, 0.18945519223763901],
            ],
        ),
        unit.angstrom,
    )

    conformer, multipoles = Psi4GDMAGenerator.generate(
        mol,
        input_conformer,
        gdma_settings,
        minimize=False,
    )
    return MoleculeGDMARecord.from_molecule(mol, conformer, multipoles, gdma_settings)


@pytest.fixture
def make_methanol_record(meoh_gdma_sto3g):
    """Factory fixture that creates methanol records with custom mpfit_atom_radius."""
    base_record = meoh_gdma_sto3g
    molecule = Molecule.from_mapped_smiles(base_record.tagged_smiles)
    conformer = base_record.conformer
    multipoles = base_record.multipoles

    def _make(mpfit_atom_radius: float) -> MoleculeGDMARecord:
        settings = GDMASettings(
            limit=4,
            mpfit_atom_radius=mpfit_atom_radius,
            mpfit_inner_radius=0.001,
            mpfit_outer_radius=10.0,
        )
        return MoleculeGDMARecord.from_molecule(
            molecule, conformer, multipoles, settings
        )

    return _make
