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

from openff_pympfit.mbis import MBISGenerator, MBISSettings

if TYPE_CHECKING:
    from openff.toolkit import Molecule


class Psi4MBISGenerator(MBISGenerator):
    """Compute the multipole moments of a molecule using Psi4."""

    @classmethod
    def _generate_input(
        cls,
        molecule: "Molecule",
        conformer: Quantity,
        settings: MBISSettings,
        minimize: bool,
        compute_mp: bool,
        memory: Quantity = 500 * unit.mebibytes,
    ) -> str:
        """Generate the input files for Psi4.

        Parameters
        ----------
        molecule
            The molecule to generate the MBIS for.
        conformer
            The conformer of the molecule to generate the MBIS for.
        settings
            The settings to use when generating the MBIS.
        minimize
            Whether to energy minimize the conformer prior to computing the MBIS using
            the same level of theory that the MBIS will be computed at.
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
            os.path.join("psi4", "mbis.dat"), "openff_pympfit"
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
            "minimize": minimize,
            "compute_mp": compute_mp,
            "properties": str(properties),
            "memory": f"{memory:~P}",
            "e_convergence": settings.e_convergence,
            "d_convergence": settings.d_convergence,
            "guess": settings.guess,
            "dft_radial_points": settings.dft_radial_points,
            "dft_spherical_points": settings.dft_spherical_points,
            "mbis_d_convergence": settings.mbis_d_convergence,
            "mbis_radial_points": settings.mbis_radial_points,
            "mbis_spherical_points": settings.mbis_spherical_points,
            "max_radial_moment": settings.max_radial_moment,
        }

        # Remove the white space after the for loop
        return template.render(template_inputs).replace("  \n}", "}")

    @classmethod
    def _generate(
        cls,
        molecule: "Molecule",
        conformer: Quantity,
        settings: MBISSettings,
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
                # Load MBIS multipoles and combine into a single array
                # The array format should match GDMA: (n_atoms, n_components)
                # where n_components = (max_radial_moment + 1)^2
                mbis_charges = np.load("mbis_charges.npy")
                n_atoms = len(mbis_charges)
                max_moment = settings.max_radial_moment

                # Calculate total components: (max_moment + 1)^2 for spherical harmonics
                # l=0: 1, l=1: 3, l=2: 5, l=3: 7, l=4: 9 -> total = 25 for max_moment=4
                n_components = (max_moment + 1) ** 2
                mp = np.zeros((n_atoms, n_components))

                # Monopole (charge) - 1 component (l=0)
                mp[:, 0] = mbis_charges.flatten()

                # Dipoles - 3 components (l=1, indices 1-3)
                if max_moment >= 1 and os.path.exists("mbis_dipoles.npy"):
                    mbis_dipoles = np.load("mbis_dipoles.npy")
                    # Dipoles are (n_atoms, 3) - x, y, z components
                    mp[:, 1:4] = mbis_dipoles

                # Quadrupoles - 5 components (l=2, indices 4-8)
                if max_moment >= 2 and os.path.exists("mbis_quadrupoles.npy"):
                    mbis_quadrupoles = np.load("mbis_quadrupoles.npy")
                    # Quadrupoles are (n_atoms, 3, 3) symmetric tensors
                    # Extract unique components: xx, xy, xz, yy, yz (traceless form)
                    # The zz component is determined by tracelessness: zz = -(xx + yy)
                    for i in range(n_atoms):
                        q = mbis_quadrupoles[i]
                        # Store in order: xx, xy, xz, yy, yz
                        mp[i, 4] = q[0, 0]  # xx
                        mp[i, 5] = q[0, 1]  # xy
                        mp[i, 6] = q[0, 2]  # xz
                        mp[i, 7] = q[1, 1]  # yy
                        mp[i, 8] = q[1, 2]  # yz

                # Octupoles - 7 components (l=3, indices 9-15)
                if max_moment >= 3 and os.path.exists("mbis_octupoles.npy"):
                    mbis_octupoles = np.load("mbis_octupoles.npy")
                    # Octupoles are (n_atoms, 3, 3, 3) symmetric tensors
                    # Extract 7 unique components for traceless form
                    for i in range(n_atoms):
                        o = mbis_octupoles[i]
                        # Store in order: xxx, xxy, xxz, xyy, xyz, xzz, yyy
                        mp[i, 9] = o[0, 0, 0]  # xxx
                        mp[i, 10] = o[0, 0, 1]  # xxy
                        mp[i, 11] = o[0, 0, 2]  # xxz
                        mp[i, 12] = o[0, 1, 1]  # xyy
                        mp[i, 13] = o[0, 1, 2]  # xyz
                        mp[i, 14] = o[0, 2, 2]  # xzz
                        mp[i, 15] = o[1, 1, 1]  # yyy

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
