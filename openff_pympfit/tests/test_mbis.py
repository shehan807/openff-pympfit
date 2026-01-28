import numpy as np
import pytest
from openff.recharge.utilities.molecule import smiles_to_molecule
from openff.units import unit
from openff_pympfit import MBISSettings


@pytest.fixture
def default_mbis_settings():
    """Default MBIS settings used by test_mbis.py input generation tests."""
    return MBISSettings()


class TestPsi4MBISGenerator:
    """Test Psi4MBISGenerator and verify correct input generated from jinja template."""

    @pytest.mark.parametrize(
        "compute_mp, expected_mbis_section",
        [
            (
                True,
                [
                    "d_convergence 8",
                    "e_convergence 8",
                    "dft_radial_points 99",
                    "dft_spherical_points 590",
                    "guess sad",
                    "mbis_d_convergence 9",
                    "mbis_radial_points 99",
                    "mbis_spherical_points 590",
                    "max_radial_moment 4",
                    "}",
                    "",
                    "# Calculate the wavefunction",
                    "energy, wfn = energy('pbe0', return_wfn=True)",
                    "# Run OEPROP",
                    "oeprop(",
                    "    wfn,",
                    "    'mbis_charges',",
                    ")",
                    "print('OEPROP')",
                    "",
                    "# Save final geometry",
                    "mol.save_xyz_file('final-geometry.xyz', 1)",
                    "",
                    "import numpy as np",
                    "mbis_charges = wfn.variable('MBIS CHARGES')",
                    "np.save('mbis_charges.npy', mbis_charges)",
                    "# Need to ensure max_radial_mooment is > 1",
                    "mbis_dipoles = wfn.variable('MBIS DIPOLES')",
                    "np.save('mbis_dipoles.npy', mbis_dipoles)",
                    "mbis_quadrupoles = wfn.variable('MBIS QUADRUPOLES')",
                    "np.save('mbis_quadrupoles.npy', mbis_quadrupoles)",
                    "mbis_octupoles = wfn.variable('MBIS OCTUPOLES')",
                    "np.save('mbis_octupoles.npy', mbis_octupoles)",
                ],
            ),
            (
                False,
                [
                    "}",
                    "",
                    "# Calculate the wavefunction",
                    "energy, wfn = energy('pbe0', return_wfn=True)",
                    "",
                    "# Save final geometry",
                    "mol.save_xyz_file('final-geometry.xyz', 1)",
                    "",
                    "import numpy as np",
                ],
            ),
        ],
    )
    def test_generate_mbis_input_base(
        self, default_mbis_settings, compute_mp, expected_mbis_section
    ):
        """Test that correct input is generated from the jinja template."""
        pytest.importorskip("psi4")
        from openff_pympfit.mbis.psi4 import Psi4MBISGenerator

        # Create a closed shell molecule
        molecule = smiles_to_molecule("[Cl-]")
        conformer = np.array([[0.0, 0.0, 0.0]]) * unit.angstrom

        input_contents = Psi4MBISGenerator._generate_input(
            molecule,
            conformer,
            default_mbis_settings,
            minimize=False,
            compute_mp=compute_mp,
        )

        expected_output = "\n".join(
            [
                "memory 500 MiB",
                "",
                "molecule mol {",
                "  noreorient",
                "  nocom",
                "  -1 1",
                "  Cl  0.000000000  0.000000000  0.000000000",
                "}",
                "",
                "set {",
                "  basis def2-SVP",
                *expected_mbis_section,
            ]
        )

        assert expected_output == input_contents

    @pytest.mark.parametrize(
        "mbis_settings_kwargs, expected_mbis_settings, expected_method",
        [
            # Default settings
            (
                {},
                [
                    "  basis def2-SVP",
                    "d_convergence 8",
                    "e_convergence 8",
                    "dft_radial_points 99",
                    "dft_spherical_points 590",
                    "guess sad",
                    "mbis_d_convergence 9",
                    "mbis_radial_points 99",
                    "mbis_spherical_points 590",
                    "max_radial_moment 4",
                ],
                "pbe0",
            ),
            # Custom settings
            (
                {
                    "basis": "6-31G*",
                    "method": "hf",
                    "e_convergence": 10,
                    "d_convergence": 10,
                    #  purely for testing
                    "dft_radial_points": 75,
                    #  purely for testing
                    "dft_spherical_points": 302,
                    #  purely for testing
                    "max_radial_moment": 2,
                    "mbis_d_convergence": 8,
                    "mbis_radial_points": 75,
                    "mbis_spherical_points": 302,
                    "mpfit_inner_radius": 10.0,
                    "mpfit_outer_radius": 15.0,
                    "mpfit_atom_radius": 3.5,
                },
                [
                    "  basis 6-31G*",
                    "d_convergence 10",
                    "e_convergence 10",
                    #  purely for testing
                    "dft_radial_points 75",
                    #  purely for testing
                    "dft_spherical_points 302",
                    "guess sad",
                    "mbis_d_convergence 8",
                    "mbis_radial_points 75",
                    "mbis_spherical_points 302",
                    "max_radial_moment 2",
                ],
                "hf",
            ),
        ],
    )
    def test_generate_input_mbis_settings(
        self, mbis_settings_kwargs, expected_mbis_settings, expected_method
    ):
        """Test that MBIS settings are correctly applied to the template."""
        pytest.importorskip("psi4")
        from openff_pympfit import MBISSettings
        from openff_pympfit.mbis.psi4 import Psi4MBISGenerator

        settings = MBISSettings(**mbis_settings_kwargs)

        molecule = smiles_to_molecule("[Cl-]")
        conformer = np.array([[0.0, 0.0, 0.0]]) * unit.angstrom

        input_contents = Psi4MBISGenerator._generate_input(
            molecule, conformer, settings, minimize=False, compute_mp=True
        )

        # Check that the expected settings appear in the input
        for expected_line in expected_mbis_settings:
            assert expected_line in input_contents, (
                f"Expected '{expected_line}' not found in:\n{input_contents}"
            )

        # Check the method is correct
        assert f"energy('{expected_method}'" in input_contents

    @pytest.mark.parametrize("minimize, n_threads", [(True, 1), (False, 1), (False, 2)])
    def test_generate(self, minimize, n_threads):
        """Perform a test run of Psi4 MBIS."""
        pytest.importorskip("psi4")
        from openff_pympfit import MBISSettings
        from openff_pympfit.mbis.psi4 import Psi4MBISGenerator

        # Define the settings to use
        settings = MBISSettings()

        molecule = smiles_to_molecule("C")
        input_conformer = (
            np.array(
                [
                    [-0.0000658, -0.0000061, 0.0000215],
                    [-0.0566733, 1.0873573, -0.0859463],
                    [0.6194599, -0.3971111, -0.8071615],
                    [-1.0042799, -0.4236047, -0.0695677],
                    [0.4415590, -0.2666354, 0.9626540],
                ]
            )
            * unit.angstrom
        )

        output_conformer, mp = Psi4MBISGenerator.generate(
            molecule,
            input_conformer,
            settings,
            minimize=minimize,
            n_threads=n_threads,
        )

        n_atoms = 5  # methane: 1 C + 4 H
        n_components = (settings.max_radial_moment + 1) ** 2  # 25 for limit=4

        assert mp.shape == (n_atoms, n_components)
        assert output_conformer.shape == input_conformer.shape
        assert np.allclose(output_conformer, input_conformer) != minimize

    def test_generate_no_properties(self):
        """Test that multipoles are None when compute_mp=False."""
        pytest.importorskip("psi4")
        from openff_pympfit import MBISSettings
        from openff_pympfit.mbis.psi4 import Psi4MBISGenerator

        settings = MBISSettings()

        molecule = smiles_to_molecule("C")
        input_conformer = (
            np.array(
                [
                    [-0.0000658, -0.0000061, 0.0000215],
                    [-0.0566733, 1.0873573, -0.0859463],
                    [0.6194599, -0.3971111, -0.8071615],
                    [-1.0042799, -0.4236047, -0.0695677],
                    [0.4415590, -0.2666354, 0.9626540],
                ]
            )
            * unit.angstrom
        )

        _, mp = Psi4MBISGenerator.generate(
            molecule,
            input_conformer,
            settings,
            minimize=False,
            compute_mp=False,
        )

        assert mp is None


if __name__ == "__main__":
    pytest.main([__file__])
