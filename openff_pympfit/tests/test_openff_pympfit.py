from pathlib import Path

import numpy as np
import pytest
from openff.toolkit import Molecule
from openff.units import unit

from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator
from openff_pympfit.gdma.storage import MoleculeGDMARecord
from openff_pympfit.mpfit import generate_mpfit_charge_parameter
from openff_pympfit.mpfit.solvers import MPFITSVDSolver

DATA_DIR = Path(__file__).parent / "data" / "esp"
BOHR_TO_ANGSTROM = unit.convert(1.0, unit.bohr, unit.angstrom)


@pytest.mark.parametrize(
    "record_class, generator, solver",
    [
        (MoleculeGDMARecord, Psi4GDMAGenerator, MPFITSVDSolver()),
    ],
)
@pytest.mark.parametrize(
    "molecule_name, smiles",
    [
        ("formaldehyde", "[C:1]([H:3])([H:4])=[O:2]"),
        ("methyl_fluoride", "[C:1]([H:3])([H:4])([H:5])[F:2]"),
        ("formic_acid", "[H:4][C:1](=[O:2])[O:3][H:5]"),
        ("methylamine", "[C:1]([H:3])([H:4])([H:5])[N:2]([H:6])[H:7]"),
        ("acetaldehyde", "[C:1]([H:4])([H:5])([H:6])[C:2](=[O:3])[H:7]"),
        ("water", "[O:1]([H:2])[H:3]"),
        (
            "benzene",
            "[c:1]1([H:7])[c:2]([H:8])[c:3]([H:9])[c:4]([H:10])[c:5]([H:11])[c:6]1[H:12]",
        ),
        ("co2", "[O:1]=[C:2]=[O:3]"),
    ],
)
def test_pympfit(
    molecule_name, smiles, sto3g_gdma_settings, record_class, generator, solver
):
    """Test that multipole method and fitting reproduces QM ESP."""

    grid = np.load(DATA_DIR / f"{molecule_name}_grid.npy")
    ref_esp = np.load(DATA_DIR / f"{molecule_name}_esp.npy").flatten()
    conformer = np.load(DATA_DIR / f"{molecule_name}_conformer.npy")

    molecule = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
    formal_charge = molecule.total_charge.m_as(unit.elementary_charge)
    n_atoms = molecule.n_atoms

    gdma_conformer, multipoles = generator.generate(
        molecule,
        conformer * unit.angstrom,
        sto3g_gdma_settings,
        minimize=False,
    )
    expected_components = (sto3g_gdma_settings.limit + 1) ** 2

    record = record_class.from_molecule(
        molecule, gdma_conformer, multipoles, sto3g_gdma_settings
    )

    parameter = generate_mpfit_charge_parameter([record], solver)
    charges = np.array(parameter.value)

    # Point-charge ESP via Coulomb's law
    coord = gdma_conformer.m_as(unit.angstrom)
    diff = grid[:, np.newaxis, :] - coord[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    calc_esp = np.sum(charges[np.newaxis, :] / distances, axis=1) * BOHR_TO_ANGSTROM

    esp_diff = ref_esp - calc_esp
    rmse = np.sqrt(np.mean(esp_diff**2))

    assert multipoles.shape == (n_atoms, expected_components), (
        f"multipoles shape {multipoles.shape}, "
        f"expected ({n_atoms}, {expected_components})"
    )
    assert len(charges) == n_atoms
    assert np.isclose(
        np.sum(charges), formal_charge, atol=0.05
    ), f"sum(charges) = {np.sum(charges):.4f}, expected {formal_charge}"
    assert rmse < 1e-2, f"RMSE = {rmse:.6e} exceeds 1e-2 tolerance"
