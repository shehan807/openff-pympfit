"""Pytest fixtures for openff-pympfit tests."""

import numpy as np
import pytest

# =============================================================================
# Molecule Fixtures
# =============================================================================


@pytest.fixture
def water_molecule():
    """Create a water molecule for testing."""
    from openff.toolkit import Molecule

    molecule = Molecule.from_smiles("O")
    molecule.generate_conformers(n_conformers=1)
    return molecule


@pytest.fixture
def ethanol_molecule():
    """Create an ethanol molecule for testing."""
    from openff.toolkit import Molecule

    molecule = Molecule.from_smiles("CCO")
    molecule.generate_conformers(n_conformers=1)
    return molecule


# =============================================================================
# GDMA Settings Fixtures
# =============================================================================


@pytest.fixture
def default_gdma_settings():
    """Create default GDMASettings."""
    from openff_pympfit import GDMASettings

    return GDMASettings()


# =============================================================================
# Mock Multipole Data Fixtures
# Ref: fork/_tests/charges/resp/test_resp.py::mock_esp_records
# Pattern: Create mock data with known values for validation
# =============================================================================


@pytest.fixture
def mock_multipoles_water():
    """Mock multipoles for water (3 atoms, limit=4 -> 25 components).

    # TODO: Set monopoles to sum to 0 (neutral)
    # multipoles[0, 0] = -0.8  # O
    # multipoles[1, 0] = 0.4   # H
    # multipoles[2, 0] = 0.4   # H
    """
    n_atoms = 3
    n_components = 25
    rng = np.random.default_rng()
    return rng.standard_normal((n_atoms, n_components)) * 0.01


@pytest.fixture
def mock_multipoles_ethanol():
    """Mock multipoles for ethanol (9 atoms)."""
    n_atoms = 9
    n_components = 25
    rng = np.random.default_rng()
    return rng.standard_normal((n_atoms, n_components)) * 0.01


@pytest.fixture
def mock_conformer_water():
    """Mock coordinates for water in angstroms."""
    from openff.units import unit

    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # O
            [0.96, 0.0, 0.0],  # H
            [-0.24, 0.93, 0.0],  # H
        ]
    )
    return coords * unit.angstrom


@pytest.fixture
def mock_conformer_ethanol(ethanol_molecule):
    """Get conformer from ethanol molecule."""
    from openff.recharge.utilities.molecule import extract_conformers

    [conformer] = extract_conformers(ethanol_molecule)
    return conformer


# =============================================================================
# GDMA Record Fixtures
# =============================================================================


@pytest.fixture
def mock_gdma_record_water(
    water_molecule, mock_conformer_water, mock_multipoles_water, default_gdma_settings
):
    """Create a mock MoleculeGDMARecord for water."""
    from openff_pympfit import MoleculeGDMARecord

    return MoleculeGDMARecord.from_molecule(
        water_molecule,
        mock_conformer_water,
        mock_multipoles_water,
        default_gdma_settings,
    )


# =============================================================================
# Storage Fixtures
# =============================================================================


@pytest.fixture
def gdma_store(tmp_path):
    """Create a temporary MoleculeGDMAStore."""
    from openff_pympfit.gdma.storage import MoleculeGDMAStore

    db_path = tmp_path / "test_gdma.sqlite"
    return MoleculeGDMAStore(str(db_path))


# =============================================================================
# Solver Test Fixtures
# Ref: fork/_tests/charges/resp/test_solvers.py::TestIterativeSolver
# Pattern: Known solution systems for verifying solver correctness
# =============================================================================


@pytest.fixture
def simple_linear_system():
    """Simple 2x2 system with known solution for solver testing.

    # A @ x = b where x = [3.0, -3.0]
    # design_matrix = [[1.0, 0.5], [0.5, 1.0]]
    # constraint: sum(x) = 0
    """
    # TODO: Implement with known solution
    pytest.skip("TODO: Implement")


@pytest.fixture
def solver_test_matrices():
    """Matrices for jacobian/loss verification.

    # Ref: fork/_tests/charges/resp/test_solvers.py::test_jacobian
    # Pattern: Verify analytical jacobian vs finite difference
    """
    # TODO: Implement
    pytest.skip("TODO: Implement")


# =============================================================================
# Reference Data for Physics Accuracy Tests
# Ref: fork/_tests/charges/resp/test_resp.py::test_generate_resp_charge_parameter
# Pattern: Compare fitted charges against known reference values (atol=1e-4)
# =============================================================================


@pytest.fixture
def reference_charges_methanol():
    """Reference MPFIT charges for methanol (regression testing).

    # TODO: Populate with validated MPFIT charges
    # Ref: RESP methanol charges = [0.0285, 0.3148, 0.0822, -0.4825]
    # Pattern: assert np.allclose(actual, expected, atol=1e-4)
    """
    return {
        "smiles": "CO",
        "n_atoms": 6,
        "total_charge": 0.0,
        "atol": 1e-4,
        # 'expected_charges': [...],  # TODO: Add validated values
    }
