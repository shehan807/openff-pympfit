import abc
import os
from typing import TYPE_CHECKING, Literal

from openff.units import Quantity, unit
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from openff.toolkit import Molecule


class GDMASettings(BaseModel):
    """Settings for GDMA calculation and related MPFIT operations."""

    basis: str = Field(
        "def2-SVP", description="The basis set to use in the GDMA calculation."
    )
    method: str = Field(
        "pbe0", description="The method to use in the GDMA calculation."
    )

    limit: int = Field(
        4,
        description=(
            "The order of multipole expansion on each site. Currently limited to "
            "the same order for all sites; for more advanced usage a user-provided "
            "GDMA data file should be provided."
        ),
    )

    multipole_units: Literal["AU"] = Field(
        "AU",
        description=(
            "Units for multipole moments. Only 'AU' (atomic units) is supported. "
            "While GDMA documentation mentions SI units, Psi4's interface only "
            "accepts AU."
        ),
    )

    radius: list = Field(
        ["C", 0.53, "N", 0.53, "H", 0.53],
        description=(
            "The radii to be used, overriding the defaults. Specified as an array "
            "[ n1, r1, n2, r2, ... ] where n1,n2,n3... are atom type strings and "
            "r1,r2,r3 are radii in Angstrom."
        ),
    )

    switch: float = Field(
        4.0,
        description=(
            "The value to switch between the older standard DMA and the new "
            "grid-based approach. Pairs of primitives whose exponents sum is above "
            "this value will be treated using standard DMA. Set to 0 to force all "
            "pairs to be treated with standard DMA."
        ),
    )

    # MPFIT specific parameters
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


class GDMAGenerator(abc.ABC):
    """Base class for generating electrostatic potential of a molecule on a grid."""

    @classmethod
    @abc.abstractmethod
    def _generate(
        cls,
        molecule: "Molecule",
        conformer: Quantity,
        settings: GDMASettings,
        directory: str,
        minimize: bool,
        compute_mp: bool,
        n_threads: int,
        memory: Quantity = 500 * unit.mebibytes,
    ) -> tuple[Quantity, Quantity | None, Quantity | None]:
        """Implement the public ``generate`` function returning GDMA for conformer.

        Parameters
        ----------
        molecule
            The molecule to generate the GDMA for.
        conformer
            The conformer of the molecule to generate the GDMA for.
        settings
            The settings to use when generating the GDMA data.
        directory
            The directory to run the calculation in. If none is specified,
            a temporary directory will be created and used.
        minimize
            Whether to energy minimize the conformer prior to computation using
            the same level of theory that will be used for GDMA.
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
        settings: GDMASettings,
        directory: str = None,
        minimize: bool = False,
        compute_mp: bool = True,
        n_threads: int = 1,
        memory: Quantity = 500 * unit.mebibytes,
    ) -> tuple[Quantity, Quantity]:
        """Generate the GDMA multipole moments for a molecule.

        Parameters
        ----------
        molecule
            The molecule to generate the GDMA data for.
        conformer
            The molecule conformer to analyze.
        settings
            The settings to use when generating the GDMA data.
        directory
            The directory to run the calculation in. If none is specified,
            a temporary directory will be created and used.
        minimize
            Whether to energy minimize the conformer prior to computation using
            the same level of theory that will be used for GDMA.
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
