import numpy as np
import pytest
from openff.recharge.utilities.molecule import smiles_to_molecule
from openff.units import unit


class TestPsi4GDMAGenerator:
    """Test Psi4GDMAGenerator and verify correct input generated from jinja template."""

    def test_generate_input_gdma(self, default_gdma_settings):
        pytest.importorskip("psi4")
        from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator

        # Create closed shell molecule as simple test case
        molecule = smiles_to_molecule("[Cl-]")
        conformer = np.array([[0.0, 0.0, 0.0]]) * unit.angstrom

        input_contents = Psi4GDMAGenerator._generate_input(
            molecule, conformer, default_gdma_settings, minimize=False, compute_mp=True
        )

        expected_output = "\n".join(  # noqa: FLY002
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
                "  ",
                "  # GDMA options",
                "  gdma_limit    4",
                "  gdma_multipole_units AU",
                "  gdma_radius   ['C', 0.53, 'N', 0.53, 'H', 0.53]",
                "  gdma_switch   4.0",
                "  ",
                "  }",
                "",
                "# Calculate the wavefunction",
                "energy, wfn = energy('pbe0', return_wfn=True)",
                "",
                "# Run GDMA",
                "gdma(wfn)",
                "",
                "# Save final geometry",
                "mol.save_xyz_file('final-geometry.xyz', 1)",
                "",
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

    def test_input_contains_molecule_coordinates(
        self, ethanol_molecule, mock_conformer_ethanol, default_gdma_settings
    ):
        """Test that generated input includes molecule geometry.

        Ref: fork/_tests/esp/test_psi4.py - checks for atom coords
        Pattern: Verify key sections present (molecule block, basis, gdma options).
        """
        pytest.importorskip("psi4")
        from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator

        input_contents = Psi4GDMAGenerator._generate_input(
            ethanol_molecule,
            mock_conformer_ethanol,
            default_gdma_settings,
            minimize=False,
            compute_mp=True,
        )

        # Verify molecule block
        assert "molecule mol {" in input_contents
        assert "0 1" in input_contents  # neutral, singlet
        assert "C " in input_contents  # carbon atoms
        assert "O " in input_contents  # oxygen atom
        assert "H " in input_contents  # hydrogen atoms

        # Verify settings are applied
        assert f"basis {default_gdma_settings.basis}" in input_contents
        assert f"gdma_limit    {default_gdma_settings.limit}" in input_contents
        assert "gdma(wfn)" in input_contents

    @pytest.mark.parametrize("minimize", [True, False])
    def test_generate_input_minimize_option(self, default_gdma_settings, minimize):
        """Test that minimize option adds optimize() call.

        Ref: fork/_tests/esp/test_psi4.py::test_generate_input_base
        Pattern: Parametrized test for different generation options.
        """
        pytest.importorskip("psi4")
        from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator

        molecule = smiles_to_molecule("[Cl-]")
        conformer = np.array([[0.0, 0.0, 0.0]]) * unit.angstrom

        input_contents = Psi4GDMAGenerator._generate_input(
            molecule,
            conformer,
            default_gdma_settings,
            minimize=minimize,
            compute_mp=True,
        )

        if minimize:
            assert f"optimize('{default_gdma_settings.method}')" in input_contents
        else:
            assert "optimize(" not in input_contents

        # Always should have energy calculation
        assert (
            f"energy('{default_gdma_settings.method}', return_wfn=True)"
            in input_contents
        )

    def test_generate_input_gdma_settings(self):
        """Test that different GDMASettings values affect template output.

        Ref: fork/_tests/esp/test_psi4.py::test_generate_input_dft_settings
        Pattern: Test that settings are correctly applied to template.
        """
        pytest.importorskip("psi4")
        from openff_pympfit import GDMASettings
        from openff_pympfit.gdma.psi4 import Psi4GDMAGenerator

        custom_settings = GDMASettings(
            basis="aug-cc-pVTZ",
            method="hf",
            limit=2,
            switch=0.0,
        )

        molecule = smiles_to_molecule("[Cl-]")
        conformer = np.array([[0.0, 0.0, 0.0]]) * unit.angstrom

        input_contents = Psi4GDMAGenerator._generate_input(
            molecule, conformer, custom_settings, minimize=False, compute_mp=True
        )

        # Verify custom settings are applied
        assert "basis aug-cc-pVTZ" in input_contents
        assert "energy('hf', return_wfn=True)" in input_contents
        assert "gdma_limit    2" in input_contents
        assert "gdma_switch   0.0" in input_contents
