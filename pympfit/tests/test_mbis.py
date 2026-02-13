import numpy as np
import pytest
from openff.recharge.utilities.molecule import smiles_to_molecule
from openff.units import unit
from pympfit import MBISSettings


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
        from pympfit.mbis.psi4 import Psi4MBISGenerator

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
        from pympfit import MBISSettings
        from pympfit.mbis.psi4 import Psi4MBISGenerator

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
        from pympfit import MBISSettings
        from pympfit.mbis.psi4 import Psi4MBISGenerator

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
        # Number of components depends on max_moment (what we load) not max_radial_moment (what Psi4 computes)
        # For spherical format: sum of (2*l + 1) for l=0 to max_moment-1
        # max_moment=3 (default): 1 + 3 + 5 = 9 components
        n_components = sum(2 * l + 1 for l in range(settings.max_moment))

        assert mp.shape == (n_atoms, n_components)
        assert output_conformer.shape == input_conformer.shape
        assert np.allclose(output_conformer, input_conformer) != minimize

    def test_generate_no_properties(self):
        """Test that multipoles are None when compute_mp=False."""
        pytest.importorskip("psi4")
        from pympfit import MBISSettings
        from pympfit.mbis.psi4 import Psi4MBISGenerator

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


class TestMultipoleTransform:
    """Test Cartesian to spherical harmonic multipole transformations."""

    def test_cartesian_to_spherical_dipole(self):
        """Test dipole transformation: Cartesian (x,y,z) -> Spherical (Q10,Q11c,Q11s)."""
        from pympfit.mbis.multipole_transform import (
            cartesian_to_spherical_dipole,
        )

        # Test case: single atom with known dipole
        # Cartesian: [x, y, z] = [1.0, 2.0, 3.0]
        # Expected spherical: [Q10, Q11c, Q11s] = [z, x, y] = [3.0, 1.0, 2.0]
        mu_cart = np.array([[1.0, 2.0, 3.0]])
        mu_sph = cartesian_to_spherical_dipole(mu_cart)

        expected = np.array([[3.0, 1.0, 2.0]])
        np.testing.assert_allclose(mu_sph, expected, rtol=1e-10)

    def test_cartesian_to_spherical_dipole_multi_atom(self):
        """Test dipole transformation for multiple atoms."""
        from pympfit.mbis.multipole_transform import (
            cartesian_to_spherical_dipole,
        )

        mu_cart = np.array(
            [
                [1.0, 0.0, 0.0],  # x-only dipole
                [0.0, 1.0, 0.0],  # y-only dipole
                [0.0, 0.0, 1.0],  # z-only dipole
            ]
        )
        mu_sph = cartesian_to_spherical_dipole(mu_cart)

        # [Q10, Q11c, Q11s] = [z, x, y]
        expected = np.array(
            [
                [0.0, 1.0, 0.0],  # Q10=0, Q11c=1, Q11s=0
                [0.0, 0.0, 1.0],  # Q10=0, Q11c=0, Q11s=1
                [1.0, 0.0, 0.0],  # Q10=1, Q11c=0, Q11s=0
            ]
        )
        np.testing.assert_allclose(mu_sph, expected, rtol=1e-10)

    def test_cartesian_to_spherical_quadrupole(self):
        """Test quadrupole transformation consistency.

        Verifies that Θzz = Q20 directly.
        """
        from pympfit.mbis.multipole_transform import (
            cartesian_to_spherical_quadrupole,
        )

        # Create a simple traceless quadrupole
        # Θzz = 2, Θxx = -1, Θyy = -1 (traceless)
        theta = np.zeros((1, 3, 3))
        theta[0, 0, 0] = -1.0  # xx
        theta[0, 1, 1] = -1.0  # yy
        theta[0, 2, 2] = 2.0  # zz
        theta[0, 0, 1] = theta[0, 1, 0] = 0.5  # xy
        theta[0, 0, 2] = theta[0, 2, 0] = 0.3  # xz
        theta[0, 1, 2] = theta[0, 2, 1] = 0.2  # yz

        q_sph = cartesian_to_spherical_quadrupole(theta)

        # Q20 = Θzz = 2.0
        assert np.isclose(q_sph[0, 0], 2.0, rtol=1e-10)

        # Check Q22c = (Θxx - Θyy) / √3 = (-1 - (-1)) / √3 = 0
        assert np.isclose(q_sph[0, 3], 0.0, rtol=1e-10)

    def test_cartesian_to_spherical_octupole(self):
        """Test octupole transformation consistency.

        Verifies that Ωzzz = Q30 directly.
        """
        from pympfit.mbis.multipole_transform import (
            cartesian_to_spherical_octupole,
        )

        # Create a simple octupole with Ωzzz = 5.0
        omega = np.zeros((1, 3, 3, 3))
        omega[0, 2, 2, 2] = 5.0  # zzz

        o_sph = cartesian_to_spherical_octupole(omega)

        # Q30 = Ωzzz = 5.0
        assert np.isclose(o_sph[0, 0], 5.0, rtol=1e-10)

    def test_cartesian_to_spherical_multipoles_combined(self):
        """Test combined multipole transformation produces correct shape."""
        from pympfit.mbis.multipole_transform import (
            cartesian_to_spherical_multipoles,
        )

        n_atoms = 3
        charges = np.array([0.5, -0.25, -0.25])
        dipoles = np.random.randn(n_atoms, 3)
        quadrupoles = np.random.randn(n_atoms, 3, 3)
        octupoles = np.random.randn(n_atoms, 3, 3, 3)

        mp = cartesian_to_spherical_multipoles(
            charges=charges,
            dipoles=dipoles,
            quadrupoles=quadrupoles,
            octupoles=octupoles,
            max_moment=4,
        )

        # Shape should be (n_atoms, max_moment^2) = (3, 16)
        assert mp.shape == (n_atoms, 16)

        # Charges should be preserved in first column
        np.testing.assert_allclose(mp[:, 0], charges, rtol=1e-10)

    def test_cartesian_to_spherical_to_cartesian_multipoles(self):
        """Test round-trip conversion preserves data.

        Note: Quadrupoles and octupoles must be traceless for the round-trip
        to work, as spherical harmonics only encode the traceless components.
        """
        from pympfit.mbis.multipole_transform import (
            cartesian_to_spherical_multipoles,
            spherical_to_cartesian_multipoles,
        )

        rng = np.random.default_rng(42)
        n_atoms = 3
        charges = np.array([0.5, -0.25, -0.25])
        dipoles = rng.standard_normal((n_atoms, 3))

        # Create symmetric traceless quadrupoles
        quadrupoles = np.zeros((n_atoms, 3, 3))
        for i in range(n_atoms):
            q = rng.standard_normal((3, 3))
            q = (q + q.T) / 2  # symmetrize
            trace = np.trace(q) / 3.0
            q -= trace * np.eye(3)  # make traceless
            quadrupoles[i] = q

        # Create symmetric traceless octupoles
        # For octupoles, the tracelessness condition is more complex:
        # sum over any contracted index pair gives zero
        octupoles = np.zeros((n_atoms, 3, 3, 3))
        for i in range(n_atoms):
            # Build a traceless symmetric octupole by construction
            # Use the 7 independent components and their relationships
            # Start with unique components and make symmetric
            vals = rng.standard_normal(10)  # 10 unique symmetric components
            # xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
            xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz = vals

            # For traceless: contract on first two indices = 0
            # O_iik = 0 for all k => xxx + xyy + xzz = 0, xxy + yyy + yzz = 0,
            # xxz + yyz + zzz = 0
            # Solve: xzz = -xxx - xyy, yzz = -xxy - yyy, zzz = -xxz - yyz
            xzz = -xxx - xyy
            yzz = -xxy - yyy
            zzz = -xxz - yyz

            o = octupoles[i]
            o[0, 0, 0] = xxx
            o[2, 2, 2] = zzz
            o[1, 1, 1] = yyy

            # xxy and permutations
            o[0, 0, 1] = o[0, 1, 0] = o[1, 0, 0] = xxy
            # xxz and permutations
            o[0, 0, 2] = o[0, 2, 0] = o[2, 0, 0] = xxz
            # xyy and permutations
            o[0, 1, 1] = o[1, 0, 1] = o[1, 1, 0] = xyy
            # xyz and permutations
            o[0, 1, 2] = o[0, 2, 1] = o[1, 0, 2] = xyz
            o[1, 2, 0] = o[2, 0, 1] = o[2, 1, 0] = xyz
            # xzz and permutations
            o[0, 2, 2] = o[2, 0, 2] = o[2, 2, 0] = xzz
            # yyz and permutations
            o[1, 1, 2] = o[1, 2, 1] = o[2, 1, 1] = yyz
            # yzz and permutations
            o[1, 2, 2] = o[2, 1, 2] = o[2, 2, 1] = yzz

        # Convert to spherical
        mp = cartesian_to_spherical_multipoles(
            charges=charges,
            dipoles=dipoles,
            quadrupoles=quadrupoles,
            octupoles=octupoles,
            max_moment=4,
        )

        # Shape should be (n_atoms, max_moment^2) = (3, 16)
        assert mp.shape == (n_atoms, 16)

        # Convert back to Cartesian
        (
            charges_rt,
            dipoles_rt,
            quadrupoles_rt,
            octupoles_rt,
        ) = spherical_to_cartesian_multipoles(mp, max_moment=4)

        # Verify round-trip preserves data
        np.testing.assert_allclose(charges_rt, charges, rtol=1e-10)
        assert dipoles_rt is not None
        assert quadrupoles_rt is not None
        assert octupoles_rt is not None
        np.testing.assert_allclose(dipoles_rt, dipoles, rtol=1e-10)
        np.testing.assert_allclose(quadrupoles_rt, quadrupoles, rtol=1e-10)
        np.testing.assert_allclose(octupoles_rt, octupoles, rtol=1e-10)

    def test_cartesian_multipoles_to_flat(self):
        """Test flattened Cartesian format produces correct shape."""
        from pympfit.mbis.multipole_transform import (
            cartesian_multipoles_to_flat,
        )

        n_atoms = 2
        charges = np.array([1.0, -1.0])
        dipoles = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        quadrupoles = np.random.randn(n_atoms, 3, 3)
        octupoles = np.random.randn(n_atoms, 3, 3, 3)

        mp = cartesian_multipoles_to_flat(
            charges=charges,
            dipoles=dipoles,
            quadrupoles=quadrupoles,
            octupoles=octupoles,
            max_moment=4,
        )

        # Shape: 1 (charge) + 3 (dipole) + 6 (quadrupole) + 10 (octupole) = 20
        assert mp.shape == (n_atoms, 20)

        # Charges should be in first column
        np.testing.assert_allclose(mp[:, 0], charges, rtol=1e-10)

        # Dipoles should be in columns 1-3 (x, y, z order)
        np.testing.assert_allclose(mp[:, 1:4], dipoles, rtol=1e-10)


class TestMBISMultipoleEvaluation:
    """Test MBIS multipole generation and electrostatic energy evaluation."""

    def test_mbis_dimer_multipoles_match_monomers(self):
        """Test that MBIS multipoles from dimer match monomer calculations.

        This test verifies that:
        1. Multipoles generated with max_moment parameter are correctly sized
        2. Multipole energies can be evaluated using Cartesian format
        3. Dimer MBIS multipoles match monomer MBIS multipoles within tolerance
        4. The electrostatic interaction energy is reasonable
        """
        pytest.importorskip("psi4")
        pytest.importorskip("qcelemental")

        import psi4
        import qcelemental as qcel
        from openff.recharge.utilities.molecule import extract_conformers
        from openff.toolkit import Molecule
        from openff.units import unit
        from qcelemental import constants

        from pympfit import (
            MBISSettings,
            MoleculeMBISRecord,
            MPFITSVDSolver,
            Psi4MBISGenerator,
            generate_mpfit_charge_parameter,
        )
        from pympfit.mbis.evaluate_cartesian_multipoles import (
            evaluate_dimer_interaction_energy,
        )
        from pympfit.mbis.multipole_transform import flat_to_cartesian_multipoles

        # Create water molecule
        molecule = Molecule.from_smiles("O")
        molecule.generate_conformers(n_conformers=1)
        [conformer] = extract_conformers(molecule)

        # Create dimer by shifting second molecule
        shift = 2.0  # Angstrom
        mol_dict = molecule.to_dict()
        mol_str = "0 1\n"
        mol_str += "\n".join(
            f"{atom['atomic_number']} "
            f"{conformer[i, 0].to(unit.angstrom).magnitude} "
            f"{conformer[i, 1].to(unit.angstrom).magnitude} "
            f"{conformer[i, 2].to(unit.angstrom).magnitude}"
            for i, atom in enumerate(mol_dict["atoms"])
        )
        mol_str += "\n--\n0 1\n"
        mol_str += "\n".join(
            f"{atom['atomic_number']} "
            f"{conformer[i, 0].to(unit.angstrom).magnitude + shift} "
            f"{conformer[i, 1].to(unit.angstrom).magnitude + shift} "
            f"{conformer[i, 2].to(unit.angstrom).magnitude}"
            for i, atom in enumerate(mol_dict["atoms"])
        )
        mol_str += "\nunits angstrom"
        qcel_mol = qcel.models.Molecule.from_data(mol_str)

        # Generate MBIS multipoles with Cartesian format
        settings = MBISSettings(
            max_radial_moment=3,
            max_moment=3,
            limit=3,
            method="hf",
            basis="aug-cc-pvdz",
            multipole_format="cartesian",
        )

        # Generate multipoles for first conformer
        coords, multipoles = Psi4MBISGenerator.generate(
            molecule, conformer, settings, n_threads=1, memory=2 * unit.gigabyte
        )

        # Verify shape: 3 atoms, with charge (1) + dipole (3) + quadrupole (6) = 10 components
        assert multipoles.shape == (3, 10)

        # Generate multipoles for shifted conformer
        conformer_2 = conformer.copy()
        conformer_2[:, 0] += shift * unit.angstrom
        conformer_2[:, 1] += shift * unit.angstrom
        coords_2, multipoles_2 = Psi4MBISGenerator.generate(
            molecule, conformer_2, settings, n_threads=1, memory=2 * unit.gigabyte
        )

        # Create record and verify charge fitting works
        record = MoleculeMBISRecord.from_molecule(
            molecule, coords, multipoles, settings
        )
        charges = generate_mpfit_charge_parameter([record], MPFITSVDSolver())
        assert len(charges.value) == 3

        # Convert multipoles to Cartesian tensors
        charges_a, dipoles_a, quadrupoles_a, _ = flat_to_cartesian_multipoles(
            multipoles, max_moment=3
        )
        charges_b, dipoles_b, quadrupoles_b, _ = flat_to_cartesian_multipoles(
            multipoles_2, max_moment=3
        )

        # Evaluate electrostatic interaction energy
        e_elst = evaluate_dimer_interaction_energy(
            qcel_mol,
            charges_a,
            dipoles_a,
            quadrupoles_a,
            charges_b,
            dipoles_b,
            quadrupoles_b,
        )

        # Energy should be negative (attractive) and reasonable magnitude
        assert e_elst < 0.0, f"Expected negative interaction energy, got {e_elst:.4f}"
        assert abs(e_elst) < 20.0, f"Interaction energy {e_elst:.4f} seems too large"

        # Run SAPT0 for comparison
        psi4.core.be_quiet()
        psi4.set_num_threads(1)
        psi4.set_memory("2 GB")
        psi4.set_options(
            {
                "basis": "aug-cc-pVDZ",
                "scf_type": "df",
                "freeze_core": True,
                "guess": "sad",
            }
        )
        psi4.geometry(mol_str)
        psi4.energy("sapt0")
        qcvars = psi4.core.variables()
        sapt0_elst = qcvars["SAPT0 ELST ENERGY"] * constants.hartree2kcalmol

        # MBIS multipole energy should be within ~50% of SAPT0
        # (exact match not expected due to different physical approximations)
        assert abs(e_elst - sapt0_elst) / abs(sapt0_elst) < 0.5

        # Calculate monomer MBIS multipoles directly with psi4
        psi4.set_options(
            {
                "basis": "aug-cc-pVDZ",
                "scf_type": "df",
                "freeze_core": True,
                "guess": "sad",
                "mbis_radial_points": 99,
                "mbis_spherical_points": 590,
                "mbis_d_convergence": 9,
                "max_radial_moment": 3,
            }
        )

        # Monomer A
        psi4.geometry(qcel_mol.get_fragment(0).to_string("psi4"))
        _, wfn = psi4.energy("hf", return_wfn=True)
        psi4.oeprop(wfn, "mbis_charges")
        wfn_vars = wfn.variables()
        mbis_mon_a_q = wfn_vars["MBIS CHARGES"].flatten()
        mbis_mon_a_mu = wfn_vars["MBIS DIPOLES"].reshape(-1, 3)
        mbis_mon_a_theta = wfn_vars["MBIS QUADRUPOLES"].reshape(-1, 3, 3)

        # Make quadrupoles traceless (Psi4 doesn't return them traceless)
        for i in range(mbis_mon_a_theta.shape[0]):
            trace = np.trace(mbis_mon_a_theta[i])
            mbis_mon_a_theta[i, 0, 0] -= trace / 3.0
            mbis_mon_a_theta[i, 1, 1] -= trace / 3.0
            mbis_mon_a_theta[i, 2, 2] -= trace / 3.0

        # Monomer B
        psi4.geometry(qcel_mol.get_fragment(1).to_string("psi4"))
        _, wfn = psi4.energy("hf", return_wfn=True)
        psi4.oeprop(wfn, "mbis_charges")
        wfn_vars = wfn.variables()
        mbis_mon_b_q = wfn_vars["MBIS CHARGES"].flatten()
        mbis_mon_b_mu = wfn_vars["MBIS DIPOLES"].reshape(-1, 3)
        mbis_mon_b_theta = wfn_vars["MBIS QUADRUPOLES"].reshape(-1, 3, 3)

        # Make quadrupoles traceless (Psi4 doesn't return them traceless)
        for i in range(mbis_mon_b_theta.shape[0]):
            trace = np.trace(mbis_mon_b_theta[i])
            mbis_mon_b_theta[i, 0, 0] -= trace / 3.0
            mbis_mon_b_theta[i, 1, 1] -= trace / 3.0
            mbis_mon_b_theta[i, 2, 2] -= trace / 3.0

        # Verify monomer multipoles match dimer multipoles
        np.testing.assert_allclose(
            mbis_mon_a_q,
            charges_a,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Monomer A charges don't match dimer charges",
        )
        np.testing.assert_allclose(
            mbis_mon_a_mu,
            dipoles_a,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Monomer A dipoles don't match dimer dipoles",
        )
        np.testing.assert_allclose(
            mbis_mon_a_theta,
            quadrupoles_a,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Monomer A quadrupoles don't match dimer quadrupoles",
        )

        # Evaluate monomer-based interaction energy
        e_elst_monomers = evaluate_dimer_interaction_energy(
            qcel_mol,
            mbis_mon_a_q,
            mbis_mon_a_mu,
            mbis_mon_a_theta,
            mbis_mon_b_q,
            mbis_mon_b_mu,
            mbis_mon_b_theta,
        )

        # Monomer and dimer energies should match closely
        np.testing.assert_allclose(
            e_elst_monomers,
            e_elst,
            rtol=1e-3,
            atol=1e-3,
            err_msg=f"Monomer energy {e_elst_monomers:.4f} doesn't match dimer energy {e_elst:.4f}",
        )


if __name__ == "__main__":
    pytest.main([__file__])
