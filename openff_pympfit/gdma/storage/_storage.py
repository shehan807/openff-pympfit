"""Store and retrieve calculated GDMA data in unified data collections."""

import functools
import warnings
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager

from openff.toolkit import Molecule, Quantity
from openff.toolkit.utils.exceptions import AtomMappingWarning
from openff.units import unit
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from openff_pympfit._annotations import MP, Coordinates
from openff_pympfit.gdma import GDMASettings
from openff_pympfit.gdma.storage.db import (
    DB_VERSION,
    DBBase,
    DBConformerRecord,
    DBGDMASettings,
    DBGeneralProvenance,
    DBInformation,
    DBMoleculeRecord,
    DBSoftwareProvenance,
)
from openff_pympfit.gdma.storage.exceptions import IncompatibleDBVersion

unit.define("AU = [] = au = atomic_unit")


class MoleculeGDMARecord(BaseModel):
    """Record containing GDMA results for a molecule conformer.

    Includes molecule information, conformer coordinates, GDMA settings
    provenance, and multipole values for each atom.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tagged_smiles: str = Field(
        ...,
        description="The tagged SMILES patterns (SMARTS) which encodes both the "
        "molecule stored in this record, a map between the atoms and the molecule and "
        "their coordinates.",
    )

    conformer: Coordinates = Field(
        ...,
        description="The coordinates [Angstrom] of this conformer with "
        "shape=(n_atoms, 3).",
    )

    multipoles: MP = Field(
        ...,
        description="The multipole moments [AU] for each atom in the molecule.",
    )

    gdma_settings: GDMASettings = Field(
        ..., description="The settings used to generate the GDMA stored in this record."
    )

    @property
    def conformer_quantity(self) -> Quantity:
        return Quantity(self.conformer, "angstrom")

    @property
    def multipoles_quantity(self) -> Quantity:
        return Quantity(self.multipoles, "AU")

    @classmethod
    def from_molecule(
        cls,
        molecule: Molecule,
        conformer: Quantity,
        multipoles: Quantity,
        gdma_settings: GDMASettings,
    ) -> "MoleculeGDMARecord":
        """Create a new ``MoleculeGDMARecord`` from an existing molecule.

        Takes care of creating the InChI and SMARTS representations.

        Parameters
        ----------
        molecule
            The molecule to store in the record.
        conformer
            The coordinates [Angstrom] of this conformer with shape=(n_atoms, 3).
        multipoles
            The multipole moments [AU] for each atom in the molecule.
        gdma_settings
            The settings used to generate the GDMA stored in this record.

        Returns
        -------
            The created record.
        """
        tagged_smiles = molecule.to_smiles(
            isomeric=True, explicit_hydrogens=True, mapped=True
        )

        return MoleculeGDMARecord(
            tagged_smiles=tagged_smiles,
            conformer=conformer,
            multipoles=multipoles,
            gdma_settings=gdma_settings,
        )


class MoleculeGDMAStore:
    """Store and retrieve GDMA results for molecules in multiple conformers.

    This class currently can only store the data in a SQLite database.
    """

    @property
    def db_version(self) -> int:
        with self._get_session() as db:
            db_info = db.query(DBInformation).first()

            return db_info.version

    @property
    def general_provenance(self) -> dict[str, str]:
        with self._get_session() as db:
            db_info = db.query(DBInformation).first()

            return {
                provenance.key: provenance.value
                for provenance in db_info.general_provenance
            }

    @property
    def software_provenance(self) -> dict[str, str]:
        with self._get_session() as db:
            db_info = db.query(DBInformation).first()

            return {
                provenance.key: provenance.value
                for provenance in db_info.software_provenance
            }

    def __init__(
        self,
        database_path: str = "gdma-store.sqlite",
        cache_size: None | int = None,
    ) -> None:
        """Initialize the GDMA store.

        Parameters
        ----------
        database_path
            The path to the SQLite database to store to and retrieve data from.
        cache_size
            The size in pages (20000 pages (~20MB)) of the cache size of the db
        """
        self._database_url = f"sqlite:///{database_path}"

        self._engine = create_engine(self._database_url, echo=False)
        DBBase.metadata.create_all(self._engine)

        if cache_size:

            @event.listens_for(self._engine, "connect")
            def set_sqlite_pragma(
                dbapi_connection: object, _connection_record: object
            ) -> None:
                cursor = dbapi_connection.cursor()
                cursor.execute(
                    f"PRAGMA cache_size = -{cache_size}"
                )  # 20000 pages (~20MB), adjust based on your needs
                cursor.execute(
                    "PRAGMA synchronous = OFF"
                )  # Improves speed but less safe
                cursor.execute(
                    "PRAGMA journal_mode = MEMORY"
                )  # Use in-memory journaling
                cursor.close()

        self._session_maker = sessionmaker(
            autocommit=False, autoflush=False, bind=self._engine
        )

        # Validate the DB version if present, or add one if not.
        with self._get_session() as db:
            db_info = db.query(DBInformation).first()

            if not db_info:
                db_info = DBInformation(version=DB_VERSION)
                db.add(db_info)

            if db_info.version != DB_VERSION:
                raise IncompatibleDBVersion(db_info.version, DB_VERSION)

    def set_provenance(
        self,
        general_provenance: dict[str, str],
        software_provenance: dict[str, str],
    ) -> None:
        """Set the stores provenance information.

        Parameters
        ----------
        general_provenance
            A dictionary storing provenance about the store such as the author,
            which QCArchive data set it was generated from, when it was generated
            etc.
        software_provenance
            A dictionary storing the provenance of the software and packages used
            to generate the data in the store.
        """
        with self._get_session() as db:
            db_info: DBInformation = db.query(DBInformation).first()
            db_info.general_provenance = [
                DBGeneralProvenance(key=key, value=value)
                for key, value in general_provenance.items()
            ]
            db_info.software_provenance = [
                DBSoftwareProvenance(key=key, value=value)
                for key, value in software_provenance.items()
            ]

    @contextmanager
    def _get_session(self) -> AbstractContextManager[Session]:
        session = self._session_maker()

        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    @classmethod
    def _db_records_to_model(
        cls, db_records: list[DBMoleculeRecord]
    ) -> list[MoleculeGDMARecord]:
        """Map a set of database records into their corresponding data models.

        Parameters
        ----------
        db_records
            The records to map.

        Returns
        -------
            The mapped data models.
        """
        # noinspection PyTypeChecker
        return [
            MoleculeGDMARecord(
                tagged_smiles=db_conformer.tagged_smiles,
                conformer=db_conformer.coordinates,
                multipoles=db_conformer.multipoles,
                gdma_settings=DBGDMASettings.db_to_instance(db_conformer.gdma_settings),
            )
            for db_record in db_records
            for db_conformer in db_record.conformers
        ]

    @classmethod
    def _store_smiles_records(
        cls, db: Session, smiles: str, records: list[MoleculeGDMARecord]
    ) -> DBMoleculeRecord:
        """Store a set of records which all store information for the same molecule.

        Parameters
        ----------
        db
            The current database session.
        smiles
            The smiles representation of the molecule.
        records
            The records to store.
        """
        existing_db_molecule = (
            db.query(DBMoleculeRecord).filter(DBMoleculeRecord.smiles == smiles).first()
        )

        if existing_db_molecule is not None:
            db_record = existing_db_molecule
        else:
            db_record = DBMoleculeRecord(smiles=smiles)

        # noinspection PyTypeChecker
        # noinspection PyUnresolvedReferences
        db_record.conformers.extend(
            DBConformerRecord(
                tagged_smiles=record.tagged_smiles,
                coordinates=record.conformer,
                multipoles=record.multipoles,
                gdma_settings=DBGDMASettings.unique(db, record.gdma_settings),
            )
            for record in records
        )

        if existing_db_molecule is None:
            db.add(db_record)

        return db_record

    @classmethod
    @functools.lru_cache(10000)
    def _tagged_to_canonical_smiles(cls, tagged_smiles: str) -> str:
        """Convert a smiles pattern with atom indices to canonical smiles.

        Parameters
        ----------
        tagged_smiles
            The tagged smiles pattern to convert.

        Returns
        -------
            The canonical smiles pattern.
        """
        from openff.toolkit import Molecule

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=AtomMappingWarning)
            return Molecule.from_smiles(
                tagged_smiles, allow_undefined_stereo=True
            ).to_smiles(isomeric=False, explicit_hydrogens=False, mapped=False)

    def store(self, *records: MoleculeGDMARecord) -> None:
        """Store the GDMA calculated for a given molecule in the data store.

        Parameters
        ----------
        records
            The records to store.

        Returns
        -------
            The records as they appear in the store.
        """
        # Validate and re-partition the records by their smiles patterns.
        records_by_smiles: dict[str, list[MoleculeGDMARecord]] = defaultdict(list)

        for record in records:
            validated_record = MoleculeGDMARecord(**record.model_dump())
            smiles = self._tagged_to_canonical_smiles(validated_record.tagged_smiles)

            records_by_smiles[smiles].append(validated_record)

        # Store the records.
        with self._get_session() as db:
            for smiles, smiles_records in records_by_smiles.items():
                self._store_smiles_records(db, smiles, smiles_records)

    def retrieve(
        self,
        smiles: str | None = None,
        basis: str | None = None,
        method: str | None = None,
    ) -> list[MoleculeGDMARecord]:
        """Retrieve records stored in this data store.

        Optionally filters according to a set of criteria.
        """
        with self._get_session() as db:
            db_records = db.query(DBMoleculeRecord)

            if smiles is not None:
                smiles = self._tagged_to_canonical_smiles(smiles)
                db_records = db_records.filter(DBMoleculeRecord.smiles == smiles)

            if basis is not None or method is not None:
                db_records = db_records.join(DBConformerRecord)

                if basis is not None or method is not None:
                    db_records = db_records.join(
                        DBGDMASettings, DBConformerRecord.gdma_settings
                    )

                    if basis is not None:
                        db_records = db_records.filter(DBGDMASettings.basis == basis)
                    if method is not None:
                        db_records = db_records.filter(DBGDMASettings.method == method)

            db_records = db_records.all()

            records = self._db_records_to_model(db_records)

            if basis:
                records = [
                    record for record in records if record.gdma_settings.basis == basis
                ]
            if method:
                records = [
                    record
                    for record in records
                    if record.gdma_settings.method == method
                ]

            return records

    def list(self) -> list[str]:
        """List the molecules which exist in and may be retrieved from the store."""
        with self._get_session() as db:
            return [smiles for (smiles,) in db.query(DBMoleculeRecord.smiles).all()]
