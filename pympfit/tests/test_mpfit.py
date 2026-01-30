"""Tests for the high-level MPFIT charge generation API.

Analog: openff-recharge fork/_tests/charges/resp/test_resp.py
"""

import importlib
from collections import defaultdict

import numpy as np
import pytest
from openff.toolkit import Molecule
from openff.units import unit

from pympfit.mpfit._mpfit import (
    _generate_dummy_values,
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
        expected_charges = [0.33727, -0.53375, -0.04897, -0.02379, -0.04885, 0.31809]

    except ModuleNotFoundError:
        expected_smiles = "[H:1][O:2][C:3]([H:4])([H:5])[H:6]"
        expected_charges = [0.33727, -0.53375, -0.04897, -0.02379, -0.04885, 0.31809]

    solver = MPFITSVDSolver()

    parameter = generate_mpfit_charge_parameter([meoh_gdma_sto3g] * n_copies, solver)

    assert parameter.smiles == expected_smiles

    assert len(parameter.value) == len(expected_charges)
    assert np.allclose(parameter.value, expected_charges, atol=1e-4)
