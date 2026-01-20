"""Utilities for storing GDMA data in a SQLite database"""

import abc
import math
from typing import TypeVar

from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    PickleType,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Query, Session, relationship, declarative_base

from openff_pympfit.gdma import GDMASettings

DBBase = declarative_base()

_InstanceType = TypeVar("_InstanceType")
_DBInstanceType = TypeVar("_DBInstanceType")

DB_VERSION = 1
_DB_FLOAT_PRECISION = 100000.0


def _float_to_db_int(value: float) -> int:
    return int(math.floor(value * _DB_FLOAT_PRECISION))


def _db_int_to_float(value: int) -> float:
    return value / _DB_FLOAT_PRECISION


class _UniqueMixin:
    """A base class for records which should be unique in the
    database."""

    @classmethod
    @abc.abstractmethod
    def _hash(cls, instance: _InstanceType) -> int:
        """Returns the hash of the instance that this record represents."""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def _query(cls, db: Session, instance: _InstanceType) -> Query:
        """Returns a query which should find existing copies of an instance."""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def _instance_to_db(cls, instance: _InstanceType) -> _DBInstanceType:
        """Map an instance into a database version of itself."""
        raise NotImplementedError()

    @classmethod
    def unique(cls, db: Session, instance: _InstanceType) -> _DBInstanceType:
        """Creates a new database object from the specified instance if it
        does not already exist on the database, otherwise the existing
        instance is returned.
        """

        cache = getattr(db, "_unique_cache", None)

        if cache is None:
            db._unique_cache = cache = {}

        key = (cls, cls._hash(instance))

        if key in cache:
            return cache[key]

        with db.no_autoflush:
            existing_instance = cls._query(db, instance).first()

            if not existing_instance:
                existing_instance = cls._instance_to_db(instance)
                db.add(existing_instance)

        cache[key] = existing_instance
        return existing_instance


class DBGDMASettings(_UniqueMixin, DBBase):
    __tablename__ = "gdma_settings"
    __table_args__ = (UniqueConstraint("basis", "method"),)

    id = Column(Integer, primary_key=True, index=True)

    basis = Column(String, index=True, nullable=False)
    method = Column(String, index=True, nullable=False)

    limit = Column(Integer, nullable=False)
    multipole_units = Column(String, nullable=False)
    switch = Column(Integer, nullable=False)
    
    # Radius will be stored as a string representation of the list
    radius = Column(String, nullable=False)

    @classmethod
    def _hash(cls, instance: GDMASettings) -> int:
        return hash(
            (
                instance.basis, 
                instance.method, 
                instance.limit,
                instance.multipole_units,
                _float_to_db_int(instance.switch),
                str(instance.radius)
            )
        )

    @classmethod
    def _query(cls, db: Session, instance: GDMASettings) -> Query:
        switch = _float_to_db_int(instance.switch)
        radius = str(instance.radius)
        
        return (
            db.query(DBGDMASettings)
            .filter(DBGDMASettings.basis == instance.basis)
            .filter(DBGDMASettings.method == instance.method)
            .filter(DBGDMASettings.limit == instance.limit)
            .filter(DBGDMASettings.multipole_units == instance.multipole_units)
            .filter(DBGDMASettings.switch == switch)
            .filter(DBGDMASettings.radius == radius)
        )

    @classmethod
    def _instance_to_db(cls, instance: GDMASettings) -> "DBGDMASettings":
        return DBGDMASettings(
            basis=instance.basis,
            method=instance.method,
            limit=instance.limit,
            multipole_units=instance.multipole_units,
            switch=_float_to_db_int(instance.switch),
            radius=str(instance.radius)
        )

    @classmethod
    def db_to_instance(cls, db_instance: "DBGDMASettings") -> GDMASettings:
        import ast
        
        # Convert the radius string back to a list
        radius_list = ast.literal_eval(db_instance.radius)
        
        # noinspection PyTypeChecker
        return GDMASettings(
            basis=db_instance.basis,
            method=db_instance.method,
            limit=db_instance.limit,
            multipole_units=db_instance.multipole_units,
            switch=_db_int_to_float(db_instance.switch),
            radius=radius_list
        )


class DBConformerRecord(DBBase):
    __tablename__ = "conformers"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(String, ForeignKey("molecules.smiles"), nullable=False)

    tagged_smiles = Column(String, nullable=False)

    coordinates = Column(PickleType, nullable=False)
    multipoles = Column(PickleType, nullable=False)

    gdma_settings = relationship("DBGDMASettings", uselist=False)
    gdma_settings_id = Column(Integer, ForeignKey("gdma_settings.id"), nullable=False)


class DBMoleculeRecord(DBBase):
    __tablename__ = "molecules"

    smiles = Column(String, primary_key=True, index=True)
    conformers = relationship("DBConformerRecord")


class DBGeneralProvenance(DBBase):
    __tablename__ = "general_provenance"

    key = Column(String, primary_key=True, index=True, unique=True)
    value = Column(String, nullable=False)

    parent_id = Column(Integer, ForeignKey("db_info.version"))


class DBSoftwareProvenance(DBBase):
    __tablename__ = "software_provenance"

    key = Column(String, primary_key=True, index=True, unique=True)
    value = Column(String, nullable=False)

    parent_id = Column(Integer, ForeignKey("db_info.version"))


class DBInformation(DBBase):
    """A class which keeps track of the current database
    settings.
    """

    __tablename__ = "db_info"

    version = Column(Integer, primary_key=True)

    general_provenance = relationship(
        "DBGeneralProvenance", cascade="all, delete-orphan"
    )
    software_provenance = relationship(
        "DBSoftwareProvenance", cascade="all, delete-orphan"
    )
