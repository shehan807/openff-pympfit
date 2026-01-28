import numpy as np
import pytest
from openff.recharge.utilities.molecule import smiles_to_molecule
from openff.units import unit


class TestPsi4GDMAGenerator:
    """Test Psi4GDMAGenerator and verify correct input generated from jinja template."""

    @pytest.mark.parametrize(
        "compute_mp, expected_gdma_section",
        [
            (
                True,
                [
                    "  # GDMA options",
                    "  gdma_limit    4",
                    "  gdma_multipole_units AU",
                    "  gdma_radius   ['C', 0.53, 'N', 0.53, 'H', 0.53]",
                    "  gdma_switch   4.0",
                    "  ",
                    "}",
                    "",
                    "# Calculate the wavefunction",
                    "energy, wfn = energy('pbe0', return_wfn=True)",
                    "# Run GDMA",
                    "gdma(wfn)",
                    "",
                    "# Save final geometry",
                    "mol.save_xyz_file('final-geometry.xyz', 1)",
                    "# Get GDMA results",
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
    def test_generate_gdma_input_base(
        self, default_gdma_settings, compute_mp, expected_gdma_section
    ):
        """Test that correct input is generated from the jinja template."""
        pytest.importorskip("psi4")
        from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator

        # Create a closed shell molecule
        molecule = smiles_to_molecule("[Cl-]")
        conformer = np.array([[0.0, 0.0, 0.0]]) * unit.angstrom

        input_contents = Psi4GDMAGenerator._generate_input(
            molecule,
            conformer,
            default_gdma_settings,
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
                *expected_gdma_section,
            ]
        )

        assert expected_output == input_contents

    @pytest.mark.parametrize(
        "gdma_settings_kwargs, expected_gdma_settings, expected_method",
        [
            # Default settings
            (
                {},
                [
                    "  basis def2-SVP",
                    "  # GDMA options",
                    "  gdma_limit    4",
                    "  gdma_multipole_units AU",
                    "  gdma_radius   ['C', 0.53, 'N', 0.53, 'H', 0.53]",
                    "  gdma_switch   4.0",
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
                    "radius": ["C", 0.53, "N", 0.53, "H", 0.53, "Cl", 0.53],
                    "switch": 2.0,
                    "mpfit_inner_radius": 10.0,
                    "mpfit_outer_radius": 15.0,
                    "mpfit_atom_radius": 3.5,
                },
                [
                    "  basis 6-31G*",
                    "  # GDMA options",
                    "  gdma_limit    2",
                    "  gdma_multipole_units AU",
                    "  gdma_radius   ['C', 0.53, 'N', 0.53, 'H', 0.53, 'Cl', 0.53]",
                    "  gdma_switch   2.0",
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
                    "radius": ["C", 0.65, "N", 0.65, "H", 0.35, "O", 0.60, "Cl", 0.75],
                    "switch": 6.0,
                    "mpfit_inner_radius": 5.0,
                    "mpfit_outer_radius": 20.0,
                    "mpfit_atom_radius": 2.5,
                },
                [
                    "  basis aug-cc-pVTZ",
                    "  # GDMA options",
                    "  gdma_limit    6",
                    "  gdma_multipole_units AU",
                    "  gdma_radius   ['C', 0.65, 'N', 0.65, 'H', 0.35, 'O', 0.6, 'Cl', 0.75]",  # noqa: E501
                    "  gdma_switch   6.0",
                    "  ",
                ],
                "mp2",
            ),
        ],
    )
    def test_generate_input_gdma_settings(
        self, gdma_settings_kwargs, expected_gdma_settings, expected_method
    ):
        """Test that GDMA settings are correctly applied to the template."""
        pytest.importorskip("psi4")
        from openff_pympfit import GDMASettings
        from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator

        settings = GDMASettings(**gdma_settings_kwargs)

        # Create a closed shell molecule
        molecule = smiles_to_molecule("[Cl-]")
        conformer = np.array([[0.0, 0.0, 0.0]]) * unit.angstrom

        input_contents = Psi4GDMAGenerator._generate_input(
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
                *expected_gdma_settings,
                "}",
                "",
                "# Calculate the wavefunction",
                f"energy, wfn = energy('{expected_method}', return_wfn=True)",
                "# Run GDMA",
                "gdma(wfn)",
                "",
                "# Save final geometry",
                "mol.save_xyz_file('final-geometry.xyz', 1)",
                "# Get GDMA results",
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
        """Perform a test run of Psi4 GDMA."""
        pytest.importorskip("psi4")
        from openff_pympfit import GDMASettings
        from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator

        # Define the settings to use
        settings = GDMASettings()

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

        output_conformer, mp = Psi4GDMAGenerator.generate(
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
        from openff_pympfit import GDMASettings
        from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator

        settings = GDMASettings()

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

        output_conformer, mp = Psi4GDMAGenerator.generate(
            molecule,
            input_conformer,
            settings,
            minimize=False,
            compute_mp=False,
        )
        assert mp is None
