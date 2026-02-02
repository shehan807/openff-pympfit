from pathlib import Path

import numpy as np
import pytest
from openff.toolkit import Molecule
from openff.units import unit

from pympfit.gdma.psi4 import Psi4GDMAGenerator
from pympfit.gdma.storage import MoleculeGDMARecord, MoleculeGDMAStore
from pympfit.mpfit import (
    generate_constrained_mpfit_charge_parameter,
    generate_global_atom_type_labels,
    generate_mpfit_charge_parameter,
)
from pympfit.mpfit.solvers import (
    ConstrainedMPFITSolver,
    ConstrainedSciPySolver,
    MPFITSVDSolver,
)

DATA_DIR = Path(__file__).parent / "data" / "esp"
GDMA_DIR = Path(__file__).parent / "data" / "gdma"
BOHR_TO_ANGSTROM = unit.convert(1.0, unit.bohr, unit.angstrom)


@pytest.mark.parametrize(
    "record_class, generator, solver",
    [
        (MoleculeGDMARecord, Psi4GDMAGenerator, MPFITSVDSolver()),
        (MoleculeGDMARecord, Psi4GDMAGenerator, ConstrainedSciPySolver()),
    ],
)
@pytest.mark.parametrize(
    "molecule_name, smiles, gdma_record_exists",
    [
        ("formaldehyde", "[C:1]([H:3])([H:4])=[O:2]", False),
        ("methyl_fluoride", "[C:1]([H:3])([H:4])([H:5])[F:2]", False),
        ("formic_acid", "[H:4][C:1](=[O:2])[O:3][H:5]", False),
        ("methylamine", "[C:1]([H:3])([H:4])([H:5])[N:2]([H:6])[H:7]", False),
        ("acetaldehyde", "[C:1]([H:4])([H:5])([H:6])[C:2](=[O:3])[H:7]", False),
        ("water", "[O:1]([H:2])[H:3]", False),
        (
            "benzene",
            "[c:1]1([H:7])[c:2]([H:8])[c:3]([H:9])[c:4]([H:10])[c:5]([H:11])[c:6]1[H:12]",
            False,
        ),
        ("co2", "[O:1]=[C:2]=[O:3]", False),
        ("mmim", "CN1C=C[N+](=C1)C", True),
        ("emim", "CCN1C=C[N+](=C1)C", True),
        ("bmim", "CCCCN1C=C[N+](=C1)C", True),
        ("c6mim", "CCCCCCN1C=C[N+](=C1)C", True),
        ("n4444", "CCCC[N+](CCCC)(CCCC)CCCC", True),
        ("p4444", "CCCC[P+](CCCC)(CCCC)CCCC", True),
    ],
)
def test_pympfit_single(
    molecule_name,
    smiles,
    gdma_record_exists,
    sto3g_gdma_settings,
    record_class,
    generator,
    solver,
):
    """Test that multipole method and fitting reproduces QM ESP."""

    grid = np.load(DATA_DIR / f"{molecule_name}_grid.npy")
    ref_esp = np.load(DATA_DIR / f"{molecule_name}_esp.npy").flatten()
    conformer = np.load(DATA_DIR / f"{molecule_name}_conformer.npy")

    if gdma_record_exists:
        molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    else:
        molecule = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)

    formal_charge = molecule.total_charge.m_as(unit.elementary_charge)
    n_atoms = molecule.n_atoms

    if gdma_record_exists:
        gdma_db_path = GDMA_DIR / "ionic_liquids.sqlite"
        store = MoleculeGDMAStore(str(gdma_db_path))
        records = store.retrieve(smiles=smiles)
        assert (
            len(records) > 0
        ), f"No GDMA records found for {molecule_name} in {gdma_db_path.name}"
        record = records[0]
        gdma_conformer = record.conformer_quantity
        multipoles = record.multipoles_quantity
    else:
        gdma_conformer, multipoles = generator.generate(
            molecule,
            conformer * unit.angstrom,
            sto3g_gdma_settings,
            minimize=False,
        )
        record = record_class.from_molecule(
            molecule, gdma_conformer, multipoles, sto3g_gdma_settings
        )

    if isinstance(solver, ConstrainedMPFITSolver):
        [parameter] = generate_constrained_mpfit_charge_parameter(
            [record],
            [molecule],
            solver=solver,
        )
    else:
        parameter = generate_mpfit_charge_parameter([record], solver)
    charges = np.array(parameter.value)

    # Point-charge ESP via Coulomb's law
    coord = gdma_conformer.m_as(unit.angstrom)
    diff = grid[:, np.newaxis, :] - coord[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    calc_esp = np.sum(charges[np.newaxis, :] / distances, axis=1) * BOHR_TO_ANGSTROM

    esp_diff = ref_esp - calc_esp
    rmse = np.sqrt(np.mean(esp_diff**2))

    if gdma_record_exists:
        expected_components = (record.gdma_settings.limit + 1) ** 2
    else:
        expected_components = (sto3g_gdma_settings.limit + 1) ** 2
    assert multipoles.shape == (n_atoms, expected_components), (
        f"multipoles shape {multipoles.shape}, "
        f"expected ({n_atoms}, {expected_components})"
    )
    assert len(charges) == n_atoms
    assert np.isclose(
        np.sum(charges), formal_charge, atol=0.05
    ), f"sum(charges) = {np.sum(charges):.4f}, expected {formal_charge}"
    assert rmse < 1e-2, f"RMSE = {rmse:.6e} exceeds 1e-2 tolerance"


@pytest.mark.slow
@pytest.mark.parametrize(
    "solver",
    [ConstrainedSciPySolver()],
)
@pytest.mark.parametrize(
    "molecule_names, smiles_list",
    [
        (
            ["mmim", "emim"],
            ["CN1C=C[N+](=C1)C", "CCN1C=C[N+](=C1)C"],
        ),
        (
            ["mmim", "emim", "bmim"],
            ["CN1C=C[N+](=C1)C", "CCN1C=C[N+](=C1)C", "CCCCN1C=C[N+](=C1)C"],
        ),
    ],
)
def test_pympfit_multi(molecule_names, smiles_list, solver):
    """Test constrained fitting across multiple molecules with shared charges."""
    store = MoleculeGDMAStore(str(GDMA_DIR / "ionic_liquids.sqlite"))

    molecules, records = [], []
    for smi in smiles_list:
        molecules.append(Molecule.from_smiles(smi, allow_undefined_stereo=True))
        recs = store.retrieve(smiles=smi)
        assert len(recs) > 0, f"No GDMA records for {smi}"
        records.append(recs[0])

    parameters = generate_constrained_mpfit_charge_parameter(
        records,
        molecules,
        solver=solver,
    )
    assert len(parameters) == len(molecules)

    # Per-molecule: charge conservation and ESP accuracy
    for i, (mol, param, name) in enumerate(
        zip(molecules, parameters, molecule_names, strict=False)
    ):
        charges = np.array(param.value)
        formal_q = mol.total_charge.m_as(unit.elementary_charge)
        assert len(charges) == mol.n_atoms
        assert np.isclose(
            np.sum(charges), formal_q, atol=0.05
        ), f"{name}: sum(charges)={np.sum(charges):.4f}, expected {formal_q}"

        grid = np.load(DATA_DIR / f"{name}_grid.npy")
        ref_esp = np.load(DATA_DIR / f"{name}_esp.npy").flatten()
        coord = records[i].conformer_quantity.m_as(unit.angstrom)
        diff = grid[:, np.newaxis, :] - coord[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        calc_esp = np.sum(charges[np.newaxis, :] / distances, axis=1) * BOHR_TO_ANGSTROM
        rmse = np.sqrt(np.mean((ref_esp - calc_esp) ** 2))
        # Multi-molecule fits sacrifice some per-molecule ESP accuracy for
        # cross-molecule charge transferability, but should be similar magnitude.
        assert rmse < 5e-2, f"{name}: RMSE={rmse:.6e} exceeds 5e-2"

    # Cross-molecule: atoms sharing a label must have equal charges
    labels = generate_global_atom_type_labels(molecules)
    flat_labels = [lbl for mol_labels in labels for lbl in mol_labels]
    flat_charges = np.concatenate([np.array(p.value) for p in parameters])

    from collections import defaultdict

    label_to_charges = defaultdict(list)
    for lbl, q in zip(flat_labels, flat_charges, strict=False):
        label_to_charges[lbl].append(q)
    for lbl, qs in label_to_charges.items():
        if len(qs) > 1:
            assert np.allclose(
                qs, qs[0], atol=1e-4
            ), f"label {lbl}: charges {qs} not equal"
