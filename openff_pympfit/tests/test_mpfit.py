"""Tests for the high-level MPFIT charge generation API.

Analog: openff-recharge fork/_tests/charges/resp/test_resp.py
"""

import numpy as np
import pytest
from openff.toolkit import Molecule
from openff.units import unit

from openff_pympfit.mpfit._mpfit import _generate_dummy_values


@pytest.mark.filterwarnings(
    "ignore::openff.toolkit.utils.exceptions.AtomMappingWarning"
)
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
