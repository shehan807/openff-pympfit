"""Compute GDMA and multipole data using Psi4."""

import os
import subprocess
from typing import TYPE_CHECKING

import jinja2
import numpy as np
from openff.recharge.esp.exceptions import Psi4Error
from openff.units import Quantity, unit
from openff.units.elements import SYMBOLS
from openff.utilities import get_data_file_path, temporary_cd

from openff_pympfit.gdma import GDMAGenerator, GDMASettings

if TYPE_CHECKING:
    from openff.toolkit import Molecule


class Psi4GDMAGenerator(GDMAGenerator):
    """Compute the multipole moments of a molecule using Psi4."""

    @classmethod
    def _generate_input(
        cls,
        molecule: "Molecule",
        conformer: Quantity,
        settings: GDMASettings,
        minimize: bool,
        compute_mp: bool,
        memory: Quantity = 500 * unit.mebibytes,
    ) -> str:
        """Generate the input files for Psi4.

        Parameters
        ----------
        molecule
            The molecule to generate the GDMA for.
        conformer
            The conformer of the molecule to generate the GDMA for.
        settings
            The settings to use when generating the GDMA.
        minimize
            Whether to energy minimize the conformer prior to computing the GDMA using
            the same level of theory that the GDMA will be computed at.
        compute_esp
            Whether to compute the multipoles.
        compute_field
            Whether to compute the field at each grid point.
        memory
            The memory to make available to Psi4 for computation

        Returns
        -------
            The contents of the input file.
        """
        # Compute the total formal charge on the molecule.
        # Trust that it's in units of elementary charge.
        formal_charge = sum(atom.formal_charge for atom in molecule.atoms).m

        # Compute the spin multiplicity
        total_atomic_number = sum(atom.atomic_number for atom in molecule.atoms)
        spin_multiplicity = 1 if (formal_charge + total_atomic_number) % 2 == 0 else 2

        # Store the atoms and coordinates in a jinja friendly dict.
        conformer = conformer.to(unit.angstrom).m

        atoms = [
            {
                "element": SYMBOLS[atom.atomic_number],
                "x": conformer[index, 0],
                "y": conformer[index, 1],
                "z": conformer[index, 2],
            }
            for index, atom in enumerate(molecule.atoms)
        ]

        # Format the jinja template
        template_path = get_data_file_path(
            os.path.join("psi4", "input.dat"), "openff_pympfit"
        )

        with open(template_path) as file:
            template = jinja2.Template(file.read())

        properties = []

        if compute_mp:
            properties.append("MULTIPOLE_MOMENT")

        template_inputs = {
            "charge": formal_charge,
            "spin": spin_multiplicity,
            "atoms": atoms,
            "basis": settings.basis,
            "method": settings.method,
            "limit": settings.limit,
            "multipole_units": settings.multipole_units,
            "radius": settings.radius,
            "switch": settings.switch,
            "minimize": minimize,
            "compute_mp": compute_mp,
            "properties": str(properties),
            "memory": f"{memory:~P}",
        }

        # Remove the white space after the for loop
        return template.render(template_inputs).replace("  \n}", "}")

    @classmethod
    def _generate(
        cls,
        molecule: "Molecule",
        conformer: Quantity,
        settings: GDMASettings,
        _directory: str,
        minimize: bool,
        compute_mp: bool,
        n_threads: int,
        memory: Quantity = 500 * unit.mebibytes,
    ) -> tuple[Quantity, Quantity | None, Quantity | None]:
        # Perform the calculation in a temporary directory
        with temporary_cd("./"):  # directory):
            # Store the input file.
            input_contents = cls._generate_input(
                molecule,
                conformer,
                settings,
                minimize,
                compute_mp,
                memory=memory,
            )

            with open("input.dat", "w") as file:
                file.write(input_contents)

            # Attempt to run the calculation
            psi4_process = subprocess.Popen(
                ["psi4", "--nthread", str(n_threads), "input.dat", "output.dat"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            std_output, std_error = psi4_process.communicate()
            exit_code = psi4_process.returncode

            if exit_code != 0:
                raise Psi4Error(std_output.decode(), std_error.decode())

            mp = None
            if compute_mp:
                mp = np.load("dma_distributed.npy")

            with open("final-geometry.xyz") as file:
                output_lines = file.read().splitlines(keepends=False)

            final_coordinates = (
                np.array(
                    [
                        [
                            float(coordinate)
                            for coordinate in coordinate_line.split()[1:]
                        ]
                        for coordinate_line in output_lines[2:]
                        if len(coordinate_line) > 0
                    ]
                )
                * unit.angstrom
            )

        return final_coordinates, mp
