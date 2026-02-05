import abc
import os
from enum import Enum
from typing import TYPE_CHECKING, Literal

from openff.units import Quantity, unit
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from openff.toolkit import Molecule


class MultipoleFormat(str, Enum):
    """Enumeration for multipole representation formats."""

    SPHERICAL = "spherical"
    CARTESIAN = "cartesian"


class MBISSettings(BaseModel):
    """Settings for MBIS calculation and related MPFIT operations."""

    basis: str = Field(
        "def2-SVP", description="The basis set to use in the MBIS calculation."
    )
    method: str = Field(
        "pbe0", description="The method to use in the MBIS calculation."
    )

    limit: int = Field(
        3,
        description=(
            "The order of multipole expansion on each site. Currently limited to "
            "the same order for all sites; for more advanced usage a user-provided "
            "MBIS data file should be provided."
        ),
    )
    e_convergence: int = Field(
        8, description="Energy convergence criterion for the MBIS calculation."
    )
    d_convergence: int = Field(
        8, description="Density convergence criterion for the MBIS calculation."
    )
    dft_radial_points: int = Field(
        99, description="Number of radial points for DFT integration."
    )
    dft_spherical_points: int = Field(
        590, description="Number of spherical points for DFT integration."
    )
    max_radial_moment: int = Field(
        4,
        description=(
            "Maximum radial moment to compute for each atom. "
            "n=1 to 4 supported: 1=charges, 2=dipoles, 3=quadrupoles, 4=octupoles."
        ),
    )
    mbis_d_convergence: int = Field(
        9, description="Density convergence criterion specifically for MBIS."
    )
    mbis_radial_points: int = Field(
        99, description="Number of radial points for MBIS integration."
    )
    mbis_spherical_points: int = Field(
        590, description="Number of spherical points for MBIS integration."
    )
    guess: str = Field("sad", description="The initial guess method for the SCF.")

    multipole_units: str = Field(
        "AU", description="Whether to print MBIS results in atomic units or SI."
    )

    multipole_format: Literal["spherical", "cartesian"] = Field(
        "spherical",
        description=(
            "Format for multipole representation. 'spherical' converts MBIS "
            "Cartesian multipoles to spherical harmonics (required for MPFIT). "
            "'cartesian' keeps the native MBIS Cartesian representation."
        ),
    )

    # MPFIT specific parameters - Agrees with GDMA defaults
    mpfit_inner_radius: float = Field(
        6.78, description="Inner radius (r1) for MPFIT integration in Bohr."
    )
    mpfit_outer_radius: float = Field(
        12.45, description="Outer radius (r2) for MPFIT integration in Bohr."
    )
    mpfit_atom_radius: float = Field(
        3.0,
        description=(
            "Default atomic radius (rvdw) for determining which atoms to include "
            "in MPFIT calculations in Bohr."
        ),
    )


class MBISGenerator(abc.ABC):
    """Base class for generating electrostatic potential of a molecule on a grid."""

    @classmethod
    @abc.abstractmethod
    def _generate(
        cls,
        molecule: "Molecule",
        conformer: Quantity,
        settings: MBISSettings,
        directory: str,
        minimize: bool,
        compute_mp: bool,
        n_threads: int,
        memory: Quantity = 500 * unit.mebibytes,
    ) -> tuple[Quantity, Quantity | None, Quantity | None]:
        """Implement the public ``generate`` function returning MBIS for conformer.

        Parameters
        ----------
        molecule
            The molecule to generate the MBIS for.
        conformer
            The conformer of the molecule to generate the MBIS for.
        settings
            The settings to use when generating the MBIS data.
        directory
            The directory to run the calculation in. If none is specified,
            a temporary directory will be created and used.
        minimize
            Whether to energy minimize the conformer prior to computation using
            the same level of theory that will be used for MBIS.
        compute_mp
            Whether to compute the multipole moments.
        n_threads
            Number of threads to use for the calculation.
        memory
            The memory to make available for computation.

        Returns
        -------
            The final conformer [A] which will be identical to ``conformer`` if
            ``minimize=False`` and the computed multipole moments.
        """
        raise NotImplementedError

    @classmethod
    def generate(
        cls,
        molecule: "Molecule",
        conformer: Quantity,
        settings: MBISSettings,
        directory: str = None,
        minimize: bool = False,
        compute_mp: bool = True,
        n_threads: int = 1,
        memory: Quantity = 32 * unit.gigabytes,
    ) -> tuple[Quantity, Quantity]:
        """Generate the MBIS multipole moments for a molecule.

        Parameters
        ----------
        molecule
            The molecule to generate the MBIS data for.
        conformer
            The molecule conformer to analyze.
        settings
            The settings to use when generating the MBIS data.
        directory
            The directory to run the calculation in. If none is specified,
            a temporary directory will be created and used.
        minimize
            Whether to energy minimize the conformer prior to computation using
            the same level of theory that will be used for MBIS.
        compute_mp
            Whether to compute the multipole moments.
        n_threads
            Number of threads to use for the calculation.
        memory
            The memory to make available for computation.
            Default is 500 MiB, as is the default in Psi4
            (see psicode.org/psi4manual/master/psithoninput.html#memory-specification).

        Returns
        -------
            The final conformer [A] which will be identical to ``conformer`` if
            ``minimize=False``, and the computed multipole moments.
        """
        if directory is not None and len(directory) > 0:
            os.makedirs(directory, exist_ok=True)

        conformer, mp = cls._generate(
            molecule,
            conformer,
            settings,
            directory,
            minimize,
            compute_mp,
            n_threads,
            memory=memory,
        )

        return conformer, mp
