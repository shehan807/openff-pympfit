"""This module contains classes which are able to store and retrieve
calculated distributed multipole analysis (GDMA) data in unified data collections.
"""

import warnings
import functools
from collections import defaultdict
from contextlib import contextmanager
from typing import ContextManager

from openff.toolkit import Quantity, Molecule
from openff.recharge._pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from openff.toolkit.utils.exceptions import AtomMappingWarning
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
from openff.recharge._annotations import (
    MP,
    Coordinates,
)


class MoleculeGDMARecord(BaseModel):
    """A record which contains information about the molecule that the distributed
    multipole analysis was performed for (including the exact conformer coordinates),
    provenance about how the GDMA was calculated, and the values of the multipoles
    for each atom."""

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
        """Creates a new ``MoleculeGDMARecord`` from an existing molecule
        object, taking care of creating the InChI and SMARTS representations.

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
    """A class used to store the distributed multipole analysis (GDMA) results
    for multiple molecules in multiple conformers, as well as to retrieve and
    query this stored data.

    This class currently can only store the data in a SQLite data base.
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
    ):
        """

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
            def set_sqlite_pragma(dbapi_connection, connection_record):
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
    ):
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
    def _get_session(self) -> ContextManager[Session]:
        session = self._session_maker()

        try:
            yield session
            session.commit()
        except BaseException as e:
            session.rollback()
            raise e
        finally:
            session.close()

    @classmethod
    def _db_records_to_model(
        cls, db_records: list[DBMoleculeRecord]
    ) -> list[MoleculeGDMARecord]:
        """Maps a set of database records into their corresponding
        data models.

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
                gdma_settings=DBGDMASettings.db_to_instance(
                    db_conformer.gdma_settings
                ),
            )
            for db_record in db_records
            for db_conformer in db_record.conformers
        ]

    @classmethod
    def _store_smiles_records(
        cls, db: Session, smiles: str, records: list[MoleculeGDMARecord]
    ) -> DBMoleculeRecord:
        """Stores a set of records which all store information for the same
        molecule.

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
        """Converts a smiles pattern which contains atom indices into
        a canonical smiles pattern without indices.

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
            smiles = Molecule.from_smiles(
                tagged_smiles, allow_undefined_stereo=True
            ).to_smiles(isomeric=False, explicit_hydrogens=False, mapped=False)

        return smiles

    def store(self, *records: MoleculeGDMARecord):
        """Store the distributed multipole analysis (GDMA) calculated for
        a given molecule in the data store.

        Parameters
        ----------
        records
            The records to store.

        Returns
        -------
            The records as they appear in the store.
        """

        # Validate an re-partition the records by their smiles patterns.
        records_by_smiles: dict[str, list[MoleculeGDMARecord]] = defaultdict(list)

        for record in records:
            record = MoleculeGDMARecord(**record.dict())
            smiles = self._tagged_to_canonical_smiles(record.tagged_smiles)

            records_by_smiles[smiles].append(record)

        # Store the records.
        with self._get_session() as db:
            for smiles in records_by_smiles:
                self._store_smiles_records(db, smiles, records_by_smiles[smiles])

    def retrieve(
        self,
        smiles: str | None = None,
        basis: str | None = None,
        method: str | None = None,
    ) -> list[MoleculeGDMARecord]:
        """Retrieve records stored in this data store, optionally
        according to a set of filters."""

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
                    record for record in records if record.gdma_settings.method == method
                ]

            return records

    def list(self) -> list[str]:
        """Lists the molecules which exist in and may be retrieved from the
        store."""

        with self._get_session() as db:
            return [smiles for (smiles,) in db.query(DBMoleculeRecord.smiles).all()]
