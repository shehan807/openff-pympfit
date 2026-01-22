import numpy as np
import pytest
from openff.recharge.utilities.molecule import smiles_to_molecule
from openff.units import unit

from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.storage import MoleculeGDMARecord, MoleculeGDMAStore
from openff_pympfit.gdma.storage.db import DB_VERSION, DBGDMASettings, DBInformation
from openff_pympfit.gdma.storage.exceptions import IncompatibleDBVersion


class TestMoleculeGDMARecord:
    """Test MoleculeGDMARecord creation and properties."""

    @pytest.fixture
    def mock_record(self):
        return MoleculeGDMARecord(
            tagged_smiles="[Ar:1]",
            gdma_settings=GDMASettings(),
            conformer=np.array([[0.0, 5.0, 0.0]]) * unit.nanometers,
            multipoles=np.array([[1.0, 2.0, 3.0] + [0.0] * 22]),
        )

    def test_validate_quantity(self, mock_record):
        assert np.allclose(mock_record.conformer, np.array([[0.0, 50.0, 0.0]]))
        expected_multipoles = np.array([[1.0, 2.0, 3.0] + [0.0] * 22])
        assert np.allclose(mock_record.multipoles, expected_multipoles)

    def test_conformer_quantity(self, mock_record):
        assert np.allclose(
            mock_record.conformer_quantity,
            np.array([[0.0, 5.0, 0.0]]) * unit.nanometers,
        )

    def test_multipoles_quantity(self, mock_record):
        expected_multipoles = np.array([[1.0, 2.0, 3.0] + [0.0] * 22])
        assert np.allclose(
            mock_record.multipoles_quantity,
            expected_multipoles * unit.AU,
        )


def test_db_version(tmp_path):
    """Tests that a version is correctly added to a new gdma store."""

    gdma_store = MoleculeGDMAStore(f"{tmp_path}.sqlite")

    with gdma_store._get_session() as db:
        db_info = db.query(DBInformation).first()
        assert db_info is not None
        assert db_info.version == DB_VERSION
    assert gdma_store.db_version == DB_VERSION


def test_provenance(tmp_path):
    """Tests that a store's provenance can be set and retrieved."""

    gdma_store = MoleculeGDMAStore(f"{tmp_path}.sqlite")

    assert gdma_store.general_provenance == {}
    assert gdma_store.software_provenance == {}

    general_provenance = {"author": "Author 1"}
    software_provenance = {"psi4": "1.9.0"}

    gdma_store.set_provenance(general_provenance, software_provenance)

    assert gdma_store.general_provenance == general_provenance
    assert gdma_store.software_provenance == software_provenance


def test_db_invalid_version(tmp_path):
    """Tests that the correct exception is raised when loading a store
    with an unsupported version."""

    gdma_store = MoleculeGDMAStore(f"{tmp_path}.sqlite")

    # Manually set the DB version to an old/incompatible version
    with gdma_store._get_session() as db:
        db_info = db.query(DBInformation).first()
        db_info.version = DB_VERSION - 1

    with pytest.raises(IncompatibleDBVersion) as error_info:
        MoleculeGDMAStore(f"{tmp_path}.sqlite")

    assert error_info.value.found_version == DB_VERSION - 1
    assert error_info.value.expected_version == DB_VERSION


def test_record_from_molecule():
    """Tests that a MoleculeGDMARecord can be correctly created from a Molecule."""

    molecule = smiles_to_molecule("C")
    conformer = np.array([[1.0, 0.0, 0.0]]) * unit.angstrom
    multipoles = np.array([[1.0, 2.0, 3.0] + [0.0] * 22]) * unit.AU
    gdma_settings = GDMASettings()

    record = MoleculeGDMARecord.from_molecule(
        molecule=molecule,
        conformer=conformer,
        multipoles=multipoles,
        gdma_settings=gdma_settings,
    )

    # SMILES order may vary depending on toolkit
    expected_smiles = (
        "[H:2][C:1]([H:3])([H:4])[H:5]",
        "[C:1]([H:2])([H:3])([H:4])[H:5]",
    )
    assert record.tagged_smiles in expected_smiles

    assert np.allclose(record.conformer, conformer.magnitude)
    assert np.allclose(record.multipoles, multipoles.magnitude)
    assert record.gdma_settings == gdma_settings


def test_tagged_to_canonical_smiles():
    """Tests that tagged SMILES are correctly converted to canonical SMILES."""
    assert (
        MoleculeGDMAStore._tagged_to_canonical_smiles("[H:2][C:1]([H:3])([H:4])[H:5]")
        == "C"
    )


def test_store(tmp_path):
    """Tests that records can be stored in a store."""

    gdma_store = MoleculeGDMAStore(f"{tmp_path}.sqlite")

    gdma_store.store(
        MoleculeGDMARecord(
            tagged_smiles="[Ar:1]",
            conformer=np.array([[0.0, 0.0, 0.0]]),
            multipoles=np.array([[1.0, 2.0, 3.0] + [0.0] * 22]),
            gdma_settings=GDMASettings(),
        ),
        MoleculeGDMARecord(
            tagged_smiles="[Ar:1]",
            conformer=np.array([[1.0, 0.0, 0.0]]),
            multipoles=np.array([[1.0, 2.0, 3.0] + [0.0] * 22]),
            gdma_settings=GDMASettings(basis="aug-cc-pVDZ"),
        ),
    )

    assert gdma_store.list() == ["[Ar]"]


def test_unique_gdma_settings(tmp_path):
    """Tests that GDMA settings are stored uniquely in the DB."""

    gdma_store = MoleculeGDMAStore(f"{tmp_path}.sqlite")
    gdma_settings = GDMASettings()

    # Store duplicate settings in the same session
    with gdma_store._get_session() as db:
        db.add(DBGDMASettings.unique(db, gdma_settings))
        db.add(DBGDMASettings.unique(db, gdma_settings))

    with gdma_store._get_session() as db:
        assert db.query(DBGDMASettings.id).count() == 1

    # Store a duplicate setting in a new session
    with gdma_store._get_session() as db:
        db.add(DBGDMASettings.unique(db, gdma_settings))

    with gdma_store._get_session() as db:
        assert db.query(DBGDMASettings.id).count() == 1

    # Store a non-duplicate set of settings
    gdma_settings.method = "dft"

    with gdma_store._get_session() as db:
        db.add(DBGDMASettings.unique(db, gdma_settings))

    with gdma_store._get_session() as db:
        assert db.query(DBGDMASettings.id).count() == 2


def test_gdma_settings_round_trip():
    """Test the round trip to/from the DB representation of a GDMASettings object."""

    original_settings = GDMASettings()

    db_settings = DBGDMASettings._instance_to_db(original_settings)
    recreated_settings = DBGDMASettings.db_to_instance(db_settings)

    assert original_settings.basis == recreated_settings.basis
    assert original_settings.method == recreated_settings.method
    assert original_settings.limit == recreated_settings.limit
    assert original_settings.multipole_units == recreated_settings.multipole_units
    assert np.isclose(original_settings.switch, recreated_settings.switch)
    assert original_settings.radius == recreated_settings.radius


def test_retrieve(tmp_path):
    """Tests that records can be retrieved from a store with various filters."""

    gdma_store = MoleculeGDMAStore(f"{tmp_path}.sqlite")
    molecule_c = smiles_to_molecule("C")
    molecule_co = smiles_to_molecule("CO")

    conformer_c = np.array([[index, 0.0, 0.0] for index in range(5)]) * unit.angstrom
    multipoles_c = np.zeros((5, 25)) * unit.AU

    conformer_co = np.array([[index, 0.0, 0.0] for index in range(6)]) * unit.angstrom
    multipoles_co = np.zeros((6, 25)) * unit.AU

    gdma_store.store(
        MoleculeGDMARecord.from_molecule(
            molecule_c,
            conformer=conformer_c,
            multipoles=multipoles_c,
            gdma_settings=GDMASettings(basis="6-31g*", method="scf"),
        ),
        MoleculeGDMARecord.from_molecule(
            molecule_c,
            conformer=conformer_c,
            multipoles=multipoles_c,
            gdma_settings=GDMASettings(basis="6-31g**", method="hf"),
        ),
        MoleculeGDMARecord.from_molecule(
            molecule_co,
            conformer=conformer_co,
            multipoles=multipoles_co,
            gdma_settings=GDMASettings(basis="6-31g*", method="hf"),
        ),
    )

    assert len(gdma_store.retrieve()) == 3

    records = gdma_store.retrieve(smiles="CO")
    assert len(records) == 1
    expected_smiles_co = (
        "[H:3][C:1]([H:4])([H:5])[O:2][H:6]",
        "[C:1]([O:2][H:6])([H:3])([H:4])[H:5]",
    )
    assert records[0].tagged_smiles in expected_smiles_co

    records = gdma_store.retrieve(basis="6-31g*")
    assert len(records) == 2
    assert records[0].gdma_settings.basis == "6-31g*"
    assert records[1].gdma_settings.basis == "6-31g*"

    records = gdma_store.retrieve(smiles="C", basis="6-31g*")
    assert len(records) == 1
    assert records[0].gdma_settings.basis == "6-31g*"
    expected_smiles_c = (
        "[H:2][C:1]([H:3])([H:4])[H:5]",
        "[C:1]([H:2])([H:3])([H:4])[H:5]",
    )
    assert records[0].tagged_smiles in expected_smiles_c

    records = gdma_store.retrieve(method="hf")
    assert len(records) == 2
    assert records[0].gdma_settings.method == "hf"
    assert records[1].gdma_settings.method == "hf"
