"""Utilities for storing MBIS data in a SQLite database."""

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
from sqlalchemy.orm import Query, Session, declarative_base, relationship

from pympfit.mbis import MBISSettings

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
    """Base class for records which should be unique in the database."""

    @classmethod
    @abc.abstractmethod
    def _hash(cls, instance: _InstanceType) -> int:
        """Return the hash of the instance that this record represents."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def _query(cls, db: Session, instance: _InstanceType) -> Query:
        """Return a query which should find existing copies of an instance."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def _instance_to_db(cls, instance: _InstanceType) -> _DBInstanceType:
        """Map an instance into a database version of itself."""
        raise NotImplementedError

    @classmethod
    def unique(cls, db: Session, instance: _InstanceType) -> _DBInstanceType:
        """Create a new database object from the instance if it doesn't exist.

        If the instance already exists on the database, the existing
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


class DBMBISSettings(_UniqueMixin, DBBase):
    """Database representation of MBISSettings."""

    __tablename__ = "mbis_settings"
    __table_args__ = (UniqueConstraint("basis", "method"),)

    id = Column(Integer, primary_key=True, index=True)

    basis = Column(String, index=True, nullable=False)
    method = Column(String, index=True, nullable=False)

    limit = Column(Integer, nullable=False)
    e_convergence = Column(Integer, nullable=False)
    d_convergence = Column(Integer, nullable=False)
    dft_radial_points = Column(Integer, nullable=False)
    dft_spherical_points = Column(Integer, nullable=False)
    max_radial_moment = Column(Integer, nullable=False)
    mbis_d_convergence = Column(Integer, nullable=False)
    mbis_radial_points = Column(Integer, nullable=False)
    mbis_spherical_points = Column(Integer, nullable=False)
    guess = Column(String, nullable=False)
    multipole_units = Column(String, nullable=False)

    # MPFIT specific parameters stored as integers
    mpfit_inner_radius = Column(Integer, nullable=False)
    mpfit_outer_radius = Column(Integer, nullable=False)
    mpfit_atom_radius = Column(Integer, nullable=False)

    @classmethod
    def _hash(cls, instance: MBISSettings) -> int:
        return hash(
            (
                instance.basis,
                instance.method,
                instance.limit,
                instance.e_convergence,
                instance.d_convergence,
                instance.dft_radial_points,
                instance.dft_spherical_points,
                instance.max_radial_moment,
                instance.mbis_d_convergence,
                instance.mbis_radial_points,
                instance.mbis_spherical_points,
                instance.guess,
                instance.multipole_units,
                _float_to_db_int(instance.mpfit_inner_radius),
                _float_to_db_int(instance.mpfit_outer_radius),
                _float_to_db_int(instance.mpfit_atom_radius),
            )
        )

    @classmethod
    def _query(cls, db: Session, instance: MBISSettings) -> Query:
        return (
            db.query(DBMBISSettings)
            .filter(DBMBISSettings.basis == instance.basis)
            .filter(DBMBISSettings.method == instance.method)
            .filter(DBMBISSettings.limit == instance.limit)
            .filter(DBMBISSettings.e_convergence == instance.e_convergence)
            .filter(DBMBISSettings.d_convergence == instance.d_convergence)
            .filter(DBMBISSettings.dft_radial_points == instance.dft_radial_points)
            .filter(
                DBMBISSettings.dft_spherical_points == instance.dft_spherical_points
            )
            .filter(DBMBISSettings.max_radial_moment == instance.max_radial_moment)
            .filter(DBMBISSettings.mbis_d_convergence == instance.mbis_d_convergence)
            .filter(DBMBISSettings.mbis_radial_points == instance.mbis_radial_points)
            .filter(
                DBMBISSettings.mbis_spherical_points == instance.mbis_spherical_points
            )
            .filter(DBMBISSettings.guess == instance.guess)
            .filter(DBMBISSettings.multipole_units == instance.multipole_units)
            .filter(
                DBMBISSettings.mpfit_inner_radius
                == _float_to_db_int(instance.mpfit_inner_radius)
            )
            .filter(
                DBMBISSettings.mpfit_outer_radius
                == _float_to_db_int(instance.mpfit_outer_radius)
            )
            .filter(
                DBMBISSettings.mpfit_atom_radius
                == _float_to_db_int(instance.mpfit_atom_radius)
            )
        )

    @classmethod
    def _instance_to_db(cls, instance: MBISSettings) -> "DBMBISSettings":
        return DBMBISSettings(
            basis=instance.basis,
            method=instance.method,
            limit=instance.limit,
            e_convergence=instance.e_convergence,
            d_convergence=instance.d_convergence,
            dft_radial_points=instance.dft_radial_points,
            dft_spherical_points=instance.dft_spherical_points,
            max_radial_moment=instance.max_radial_moment,
            mbis_d_convergence=instance.mbis_d_convergence,
            mbis_radial_points=instance.mbis_radial_points,
            mbis_spherical_points=instance.mbis_spherical_points,
            guess=instance.guess,
            multipole_units=instance.multipole_units,
            mpfit_inner_radius=_float_to_db_int(instance.mpfit_inner_radius),
            mpfit_outer_radius=_float_to_db_int(instance.mpfit_outer_radius),
            mpfit_atom_radius=_float_to_db_int(instance.mpfit_atom_radius),
        )

    @classmethod
    def db_to_instance(cls, db_instance: "DBMBISSettings") -> MBISSettings:
        """Convert a database record to a MBISSettings instance."""
        # noinspection PyTypeChecker
        return MBISSettings(
            basis=db_instance.basis,
            method=db_instance.method,
            limit=db_instance.limit,
            e_convergence=db_instance.e_convergence,
            d_convergence=db_instance.d_convergence,
            dft_radial_points=db_instance.dft_radial_points,
            dft_spherical_points=db_instance.dft_spherical_points,
            max_radial_moment=db_instance.max_radial_moment,
            mbis_d_convergence=db_instance.mbis_d_convergence,
            mbis_radial_points=db_instance.mbis_radial_points,
            mbis_spherical_points=db_instance.mbis_spherical_points,
            guess=db_instance.guess,
            multipole_units=db_instance.multipole_units,
            mpfit_inner_radius=_db_int_to_float(db_instance.mpfit_inner_radius),
            mpfit_outer_radius=_db_int_to_float(db_instance.mpfit_outer_radius),
            mpfit_atom_radius=_db_int_to_float(db_instance.mpfit_atom_radius),
        )


class DBConformerRecord(DBBase):
    """Database representation of a conformer record."""

    __tablename__ = "conformers"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(String, ForeignKey("molecules.smiles"), nullable=False)

    tagged_smiles = Column(String, nullable=False)

    coordinates = Column(PickleType, nullable=False)
    multipoles = Column(PickleType, nullable=False)

    mbis_settings = relationship("DBMBISSettings", uselist=False)
    mbis_settings_id = Column(Integer, ForeignKey("mbis_settings.id"), nullable=False)


class DBMoleculeRecord(DBBase):
    """Database representation of a molecule record."""

    __tablename__ = "molecules"

    smiles = Column(String, primary_key=True, index=True)
    conformers = relationship("DBConformerRecord")


class DBGeneralProvenance(DBBase):
    """Database representation of general provenance information."""

    __tablename__ = "general_provenance"

    key = Column(String, primary_key=True, index=True, unique=True)
    value = Column(String, nullable=False)

    parent_id = Column(Integer, ForeignKey("db_info.version"))


class DBSoftwareProvenance(DBBase):
    """Database representation of software provenance information."""

    __tablename__ = "software_provenance"

    key = Column(String, primary_key=True, index=True, unique=True)
    value = Column(String, nullable=False)

    parent_id = Column(Integer, ForeignKey("db_info.version"))


class DBInformation(DBBase):
    """Track current database settings and version."""

    __tablename__ = "db_info"

    version = Column(Integer, primary_key=True)

    general_provenance = relationship(
        "DBGeneralProvenance", cascade="all, delete-orphan"
    )
    software_provenance = relationship(
        "DBSoftwareProvenance", cascade="all, delete-orphan"
    )
