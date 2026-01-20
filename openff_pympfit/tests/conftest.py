"""Pytest fixtures for openff-pympfit tests.

Following openff-recharge patterns from:
- _tests/conftest.py (root fixtures)
- _tests/esp/test_storage.py (storage fixtures)
"""

import pytest
import numpy as np


# =============================================================================
# Molecule Fixtures
# =============================================================================

@pytest.fixture
def ethanol_molecule():
    """Create an ethanol molecule for testing.

    Returns
    -------
    Molecule
        OpenFF Molecule object for ethanol with atom mapping.
    """
    pytest.skip("TODO: Implement fixture")


@pytest.fixture
def water_molecule():
    """Create a water molecule for testing.

    Returns
    -------
    Molecule
        OpenFF Molecule object for water with atom mapping.
    """
    pytest.skip("TODO: Implement fixture")


@pytest.fixture
def methane_molecule():
    """Create a methane molecule for testing.

    Returns
    -------
    Molecule
        OpenFF Molecule object for methane with atom mapping.
    """
    pytest.skip("TODO: Implement fixture")


# =============================================================================
# GDMA Settings Fixtures
# =============================================================================

@pytest.fixture
def default_gdma_settings():
    """Create default GDMASettings for testing.

    Returns
    -------
    GDMASettings
        GDMASettings with default values.
    """
    pytest.skip("TODO: Implement fixture")


@pytest.fixture
def custom_gdma_settings():
    """Create custom GDMASettings for testing.

    Returns
    -------
    GDMASettings
        GDMASettings with non-default values.
    """
    pytest.skip("TODO: Implement fixture")


# =============================================================================
# Mock Multipole Data Fixtures
# =============================================================================

@pytest.fixture
def mock_multipoles_water():
    """Create mock multipole moments for water.

    Returns
    -------
    numpy.ndarray
        Mock multipole array with shape (n_atoms, n_multipole_components).
    """
    pytest.skip("TODO: Implement fixture")


@pytest.fixture
def mock_multipoles_ethanol():
    """Create mock multipole moments for ethanol.

    Returns
    -------
    numpy.ndarray
        Mock multipole array with shape (n_atoms, n_multipole_components).
    """
    pytest.skip("TODO: Implement fixture")


# =============================================================================
# GDMA Record Fixtures
# =============================================================================

@pytest.fixture
def mock_gdma_record_water(water_molecule, mock_multipoles_water, default_gdma_settings):
    """Create a mock MoleculeGDMARecord for water.

    Returns
    -------
    MoleculeGDMARecord
        GDMA record with mock data for water.
    """
    pytest.skip("TODO: Implement fixture")


@pytest.fixture
def mock_gdma_record_ethanol(ethanol_molecule, mock_multipoles_ethanol, default_gdma_settings):
    """Create a mock MoleculeGDMARecord for ethanol.

    Returns
    -------
    MoleculeGDMARecord
        GDMA record with mock data for ethanol.
    """
    pytest.skip("TODO: Implement fixture")


# =============================================================================
# Solver Test Fixtures
# =============================================================================

@pytest.fixture
def simple_design_matrix():
    """Create a simple design matrix for solver testing.

    Returns
    -------
    numpy.ndarray
        Simple, well-conditioned design matrix.
    """
    pytest.skip("TODO: Implement fixture")


@pytest.fixture
def simple_reference_values():
    """Create simple reference values for solver testing.

    Returns
    -------
    numpy.ndarray
        Reference multipole values.
    """
    pytest.skip("TODO: Implement fixture")


@pytest.fixture
def charge_constraint_matrix():
    """Create charge constraint matrix (sum to total charge).

    Returns
    -------
    numpy.ndarray
        Constraint matrix for charge conservation.
    """
    pytest.skip("TODO: Implement fixture")


# =============================================================================
# Storage Fixtures
# =============================================================================

@pytest.fixture
def gdma_store(tmp_path):
    """Create a temporary MoleculeGDMAStore for testing.

    Parameters
    ----------
    tmp_path
        Pytest tmp_path fixture for temporary directory.

    Returns
    -------
    MoleculeGDMAStore
        Store backed by temporary SQLite database.
    """
    pytest.skip("TODO: Implement fixture")


@pytest.fixture
def populated_gdma_store(gdma_store, mock_gdma_record_water, mock_gdma_record_ethanol):
    """Create a MoleculeGDMAStore populated with test records.

    Returns
    -------
    MoleculeGDMAStore
        Store containing water and ethanol GDMA records.
    """
    pytest.skip("TODO: Implement fixture")
