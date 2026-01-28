import numpy as np
import pytest
from openff.recharge.utilities.molecule import smiles_to_molecule
from openff.units import unit


class TestPsi4MBISGenerator:
    """Test Psi4MBISGenerator and verify correct input generated from jinja template."""

    @pytest.mark.parametrize(
        "compute_mp, expected_mbis_section",
        [
            (
                True,
                [
                    "  # MBIS options",
                    "  mbis_limit    4",
                    "  mbis_multipole_units AU",
                    "  mbis_radius   ['C', 0.53, 'N', 0.53, 'H', 0.53]",
                    "  mbis_switch   4.0",
                    "  ",
                    "}",
                    "",
                    "# Calculate the wavefunction",
                    "energy, wfn = energy('pbe0', return_wfn=True)",
                    "# Run MBIS",
                    "mbis(wfn)",
                    "",
                    "# Save final geometry",
                    "mol.save_xyz_file('final-geometry.xyz', 1)",
                    "# Get MBIS results",
                    'dma_distributed = variable("DMA DISTRIBUTED MULTIPOLES")',
                    'dma_total = variable("DMA TOTAL MULTIPOLES")',
                    "",
                    "import numpy as np",
                    "",
                    "# Convert Matrix objects to NumPy arrays",
                    "dma_distributed_array = dma_distributed.to_array()",
                    "dma_total_array = dma_total.to_array()",
                    "",
                    "# Save arrays to disk",
                    "np.save('dma_distributed.npy', dma_distributed_array)",
                    "np.save('dma_total.npy', dma_total_array)",
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
                    "  # MBIS options",
                    "  mbis_limit    4",
                    "  mbis_multipole_units AU",
                    "  mbis_radius   ['C', 0.53, 'N', 0.53, 'H', 0.53]",
                    "  mbis_switch   4.0",
                    "  ",
                ],
                "pbe0",
            ),
            # Coarse settings
            (
                {
                    "basis": "6-31G*",
                    "method": "hf",
                    "limit": 2,
                    "multipole_units": "Bohr",
                    "radius": ["C", 0.53, "N", 0.53, "H", 0.53, "Cl", 0.53],
                    "switch": 2.0,
                    "mpfit_inner_radius": 10.0,
                    "mpfit_outer_radius": 15.0,
                    "mpfit_atom_radius": 3.5,
                },
                [
                    "  basis 6-31G*",
                    "  # MBIS options",
                    "  mbis_limit    2",
                    "  mbis_multipole_units Bohr",
                    "  mbis_radius   ['C', 0.53, 'N', 0.53, 'H', 0.53, 'Cl', 0.53]",
                    "  mbis_switch   2.0",
                    "  ",
                ],
                "hf",
            ),
            # Fine settings
            (
                {
                    "basis": "aug-cc-pVTZ",
                    "method": "mp2",
                    "limit": 6,
                    "multipole_units": "AU",
                    "radius": ["C", 0.65, "N", 0.65, "H", 0.35, "O", 0.60, "Cl", 0.75],
                    "switch": 6.0,
                    "mpfit_inner_radius": 5.0,
                    "mpfit_outer_radius": 20.0,
                    "mpfit_atom_radius": 2.5,
                },
                [
                    "  basis aug-cc-pVTZ",
                    "  # MBIS options",
                    "  mbis_limit    6",
                    "  mbis_multipole_units AU",
                    "  mbis_radius   ['C', 0.65, 'N', 0.65, 'H', 0.35, 'O', 0.6, 'Cl', 0.75]",  # noqa: E501
                    "  mbis_switch   6.0",
                    "  ",
                ],
                "mp2",
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

        # Create a closed shell molecule
        molecule = smiles_to_molecule("[Cl-]")
        conformer = np.array([[0.0, 0.0, 0.0]]) * unit.angstrom

        input_contents = Psi4MBISGenerator._generate_input(
            molecule, conformer, settings, minimize=False, compute_mp=True
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
                *expected_mbis_settings,
                "}",
                "",
                "# Calculate the wavefunction",
                f"energy, wfn = energy('{expected_method}', return_wfn=True)",
                "# Run MBIS",
                "mbis(wfn)",
                "",
                "# Save final geometry",
                "mol.save_xyz_file('final-geometry.xyz', 1)",
                "# Get MBIS results",
                'dma_distributed = variable("DMA DISTRIBUTED MULTIPOLES")',
                'dma_total = variable("DMA TOTAL MULTIPOLES")',
                "",
                "import numpy as np",
                "",
                "# Convert Matrix objects to NumPy arrays",
                "dma_distributed_array = dma_distributed.to_array()",
                "dma_total_array = dma_total.to_array()",
                "",
                "# Save arrays to disk",
                "np.save('dma_distributed.npy', dma_distributed_array)",
                "np.save('dma_total.npy', dma_total_array)",
            ]
        )

        assert expected_output == input_contents

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
        n_components = (settings.limit + 1) ** 2  # 25 for limit=4

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

        output_conformer, mp = Psi4MBISGenerator.generate(
            molecule,
            input_conformer,
            settings,
            minimize=False,
            compute_mp=False,
        )

        assert mp is None

if __name__ == "__main__":
    pytest.main([__file__])
