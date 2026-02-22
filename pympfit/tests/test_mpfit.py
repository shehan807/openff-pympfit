"""Tests for the high-level MPFIT charge generation API.

Analog: openff-recharge fork/_tests/charges/resp/test_resp.py
"""

import importlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from openff.toolkit import Molecule
from openff.units import unit

from pympfit import GDMASettings
from pympfit.mpfit._mpfit import (
    _generate_dummy_values,
    generate_global_atom_type_labels,
    generate_mpfit_charge_parameter,
    molecule_to_mpfit_library_charge,
)
from pympfit.mpfit.solvers import MPFITSVDSolver


@pytest.mark.parametrize(
    "smiles, expected_values",
    [
        ("[Cl:1][H:2]", [0.0, 0.0]),
        ("[O-:1][H:2]", [-0.5, -0.5]),
        ("[N+:1]([H:2])([H:2])([H:2])([H:2])", [0.2, 0.2]),
        ("[N+:1]([H:2])([H:3])([H:4])([H:5])", [0.2, 0.2, 0.2, 0.2, 0.2]),
        (
            "[H:1][c:9]1[c:10]([c:13]([c:16]2[c:15]([c:11]1[H:3])[c:12]([c:14]"
            "([c:17]([n+:20]2[C:19]([H:8])([H:8])[H:8])[C:18]([H:7])([H:7])[H:7])"
            "[H:6])[H:4])[H:5])[H:2]",
            [1.0 / 24.0] * 20,
        ),
    ],
)
def test_generate_dummy_values(smiles, expected_values):
    """Test that dummy values conserve charge."""
    actual_values = _generate_dummy_values(smiles)
    assert actual_values == expected_values

    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    total_charge = molecule.total_charge.m_as(unit.elementary_charge)
    sum_charge = sum(
        actual_values[i - 1] for i in molecule.properties["atom_map"].values()
    )
    assert np.isclose(total_charge, sum_charge)


@pytest.mark.filterwarnings(
    "ignore::openff.toolkit.utils.exceptions.AtomMappingWarning"
)
@pytest.mark.parametrize(
    "input_smiles, " "expected_groupings",
    [
        (
            "[C:1]([H:2])([H:3])([H:4])[H:5]",
            [(0,), (1,), (2,), (3,), (4,)],
        ),
        (
            "[C:1]([H:3])([H:4])([H:5])[C:2]([H:6])([H:7])([H:8])",
            [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)],
        ),
        (
            "[C:1]([H:7])([H:8])([O:3][H:5])[C:2]([H:9])([H:10])([O:4][H:6])",
            [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)],
        ),
    ],
)
def test_molecule_to_mpfit_library_charge(
    input_smiles,
    expected_groupings,
):
    """Test that molecule_to_mpfit_library_charge creates correct atom groupings."""
    input_molecule = Molecule.from_mapped_smiles(input_smiles)

    parameter = molecule_to_mpfit_library_charge(input_molecule)

    output_molecule = Molecule.from_smiles(parameter.smiles)

    _, output_to_input_index = Molecule.are_isomorphic(
        output_molecule, input_molecule, return_atom_map=True
    )

    actual_groupings_dict = defaultdict(list)

    for atom_index, map_index in output_molecule.properties["atom_map"].items():
        actual_groupings_dict[map_index].append(output_to_input_index[atom_index])

    actual_groupings = [
        tuple(sorted(group)) for group in actual_groupings_dict.values()
    ]

    assert len(actual_groupings) == len(expected_groupings)
    assert set(actual_groupings) == set(expected_groupings)


@pytest.mark.parametrize("n_copies", [1, 2, 5])
def test_generate_mpfit_charge_parameter(meoh_gdma_sto3g, n_copies: int):
    try:
        importlib.import_module("openeye.oechem")

        expected_smiles = "[H:1][O:2][C:3]([H:4])([H:5])[H:6]"
        expected_charges = [0.33829, -0.53397, -0.04925, -0.02403, -0.04912, 0.31808]

    except ModuleNotFoundError:
        expected_smiles = "[H:1][O:2][C:3]([H:4])([H:5])[H:6]"
        expected_charges = [0.33829, -0.53397, -0.04925, -0.02403, -0.04912, 0.31808]

    solver = MPFITSVDSolver()

    parameter = generate_mpfit_charge_parameter([meoh_gdma_sto3g] * n_copies, solver)

    assert parameter.smiles == expected_smiles

    assert len(parameter.value) == len(expected_charges)
    assert np.allclose(parameter.value, expected_charges, atol=1e-4)


class TestGenerateGlobalAtomTypeLabels:
    """Test atom type label generation for symmetry and cross-molecule sharing."""

    @pytest.mark.parametrize(
        "smiles, symmetric_groups",
        [
            ("[O:1]([H:2])[H:3]", [(1, 2)]),
            ("[O:1]=[C:2]=[O:3]", [(0, 2)]),
            (
                "[c:1]1([H:7])[c:2]([H:8])[c:3]([H:9])"
                "[c:4]([H:10])[c:5]([H:11])[c:6]1[H:12]",
                [(0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11)],
            ),
        ],
    )
    def test_single_molecule_symmetry(self, smiles, symmetric_groups):
        mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
        [labels] = generate_global_atom_type_labels([mol])
        assert len(labels) == mol.n_atoms
        for group in symmetric_groups:
            assert len({labels[i] for i in group}) == 1, (
                f"atoms {group} should share a label, got "
                f"{[labels[i] for i in group]}"
            )

    def test_cross_molecule_label_sharing(self):
        mmim = Molecule.from_smiles("CN1C=C[N+](=C1)C", allow_undefined_stereo=True)
        emim = Molecule.from_smiles("CCN1C=C[N+](=C1)C", allow_undefined_stereo=True)
        labels = generate_global_atom_type_labels([mmim, emim])

        shared = set(labels[0]) & set(labels[1])
        assert len(shared) > 0, "no labels shared between mmim and emim"

        # Cross-molecule sharing should reduce total unique label count
        total_unique = len(set(labels[0] + labels[1]))
        sum_per_mol = len(set(labels[0])) + len(set(labels[1]))
        assert total_unique < sum_per_mol


def test_generate_mpfit_charge_parameter_with_vsite(meoh_gdma_sto3g):
    """Test MPFIT with virtual sites returns both atom and vsite charges."""
    from openff.recharge.charges.vsite import (
        BondChargeSiteParameter,
        VirtualSiteCollection,
    )

    # Define a bond charge site on the C-O bond of methanol
    vsite_collection = VirtualSiteCollection(
        parameters=[
            BondChargeSiteParameter(
                smirks="[#6:1]-[#8:2]",
                name="EP",
                distance=0.5,  # Angstrom
                charge_increments=(0.0, 0.0),
                sigma=0.0,  # LJ params not used in MPFIT
                epsilon=0.0,
                match="all-permutations",
            )
        ]
    )

    solver = MPFITSVDSolver()
    result = generate_mpfit_charge_parameter(
        [meoh_gdma_sto3g], solver, vsite_collection=vsite_collection
    )

    assert isinstance(result, tuple)
    parameter, vsite_charges = result

    assert len(parameter.value) == 6  # methanol has 6 atoms
    assert vsite_charges is not None
    assert len(vsite_charges) == 1  # one vsite from the SMIRKS match

    total_charge = sum(parameter.value) + sum(vsite_charges)
    assert np.isclose(total_charge, 0.0, atol=1e-6)


DATA_DIR_FL = Path(__file__).parent / "data" / "esp"
GDMA_DIR_FL = Path(__file__).parent / "data" / "gdma"


def _build_record_at_limit(name, smiles, from_db, limit):
    """Run Psi4GDMAGenerator at the given limit and return a MoleculeGDMARecord.

    For DB molecules, the conformer is extracted from the pre-computed DB record
    (but GDMA is re-run at the requested limit).
    """
    from pympfit.gdma.psi4 import Psi4GDMAGenerator
    from pympfit.gdma.storage import MoleculeGDMARecord, MoleculeGDMAStore

    settings = GDMASettings(
        method="scf",
        basis="sto-3g",
        limit=limit,
        switch=0.0,
        radius=["C", 0.53, "O", 0.53, "N", 0.53, "H", 0.53, "F", 0.53],
    )

    if from_db:
        store = MoleculeGDMAStore(str(GDMA_DIR_FL / "ionic_liquids.sqlite"))
        db_rec = store.retrieve(smiles=smiles)[0]
        mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        conformer = db_rec.conformer_quantity
    else:
        mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
        conformer = np.load(DATA_DIR_FL / f"{name}_conformer.npy") * unit.angstrom

    conf_out, mults = Psi4GDMAGenerator.generate(
        mol,
        conformer,
        settings,
        minimize=False,
    )
    return MoleculeGDMARecord.from_molecule(mol, conf_out, mults, settings)


@pytest.mark.parametrize(
    "molecule_name, smiles, from_db",
    [
        ("formaldehyde", "[C:1]([H:3])([H:4])=[O:2]", False),
        ("methyl_fluoride", "[C:1]([H:3])([H:4])([H:5])[F:2]", False),
        ("formic_acid", "[H:4][C:1](=[O:2])[O:3][H:5]", False),
        ("methylamine", "[C:1]([H:3])([H:4])([H:5])[N:2]([H:6])[H:7]", False),
        ("acetaldehyde", "[C:1]([H:4])([H:5])([H:6])[C:2](=[O:3])[H:7]", False),
        ("water", "[O:1]([H:2])[H:3]", False),
        (
            "benzene",
            "[c:1]1([H:7])[c:2]([H:8])[c:3]([H:9])"
            "[c:4]([H:10])[c:5]([H:11])[c:6]1[H:12]",
            False,
        ),
        ("co2", "[O:1]=[C:2]=[O:3]", False),
        ("mmim", "CN1C=C[N+](=C1)C", True),
        ("emim", "CCN1C=C[N+](=C1)C", True),
        ("bmim", "CCCCN1C=C[N+](=C1)C", True),
        ("c6mim", "CCCCCCN1C=C[N+](=C1)C", True),
    ],
)
@pytest.mark.parametrize("limit", [0, 1, 2, 3, 4])
def test_fit_limit(molecule_name, smiles, from_db, limit):
    """Verify that GDMA@limit=L + MPFIT(fit_limit=None) produces the same
    charges as GDMA@limit=8 + MPFIT(fit_limit=L).

    Also checks that fit_limit > GDMA limit raises ValueError.
    """
    from pympfit.optimize import MPFITObjective

    solver = MPFITSVDSolver()

    # Direct
    record_direct = _build_record_at_limit(molecule_name, smiles, from_db, limit)
    charges_direct = np.array(
        generate_mpfit_charge_parameter([record_direct], solver, fit_limit=None).value
    )
    arrays_direct = MPFITObjective.extract_arrays(record_direct, fit_limit=None)

    # Truncated
    record_high = _build_record_at_limit(molecule_name, smiles, from_db, limit=8)
    charges_truncated = np.array(
        generate_mpfit_charge_parameter([record_high], solver, fit_limit=limit).value
    )
    arrays_truncated = MPFITObjective.extract_arrays(record_high, fit_limit=limit)

    np.testing.assert_array_equal(
        arrays_truncated["multipoles"],
        arrays_direct["multipoles"],
    )
    assert arrays_truncated["maxl"] == limit
    assert arrays_direct["maxl"] == limit

    np.testing.assert_allclose(charges_truncated, charges_direct, atol=1e-12)

    with pytest.raises(ValueError, match="cannot exceed"):
        generate_mpfit_charge_parameter([record_direct], solver, fit_limit=limit + 1)
