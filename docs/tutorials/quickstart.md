# Quick Start

This tutorial walks through fitting partial charges for ethanol using PyMPFIT
with GDMA and MBIS. The full runnable script is at
`examples/tutorials/quickstart.py`.

# GDMA Multipoles
## 1. QM and GDMA Settings

`GDMASettings` controls both the Psi4 QM calculation and the
[GDMA](https://psicode.org/psi4manual/master/gdma.html) parameters.

```python
from pympfit import GDMASettings

settings = GDMASettings(
    method="pbe0",
    basis="def2-SVP",
    limit=4,
    switch=4.0,
    radius=[
        "C", 0.53,
        "O", 0.53,
        "H", 0.53,
    ],
    mpfit_inner_radius=6.78,
    mpfit_outer_radius=12.45,
    mpfit_atom_radius=3.0,
)
```

## 2. Generate a Conformer

Create an ethanol molecule and generate a single conformer.

```python
from openff.toolkit import Molecule
from openff.recharge.utilities.molecule import extract_conformers

molecule = Molecule.from_smiles("CCO")
molecule.generate_conformers(n_conformers=1)
[conformer] = extract_conformers(molecule)
```

## 3. Generate Multipoles

Run Psi4 to compute the wavefunction and GDMA multipole moments.
Setting `minimize=True` optimizes the geometry at the same level of theory first.

```python
import time
from pympfit import Psi4GDMAGenerator

t0 = time.perf_counter()
coords, multipoles = Psi4GDMAGenerator.generate(
    molecule, conformer, settings, minimize=True
)
elapsed = time.perf_counter() - t0

print(f"Multipoles shape: {multipoles.shape}")
print(f"GDMA generation time: {elapsed:.2f}s")
```

```text
Multipoles shape: (9, 25)
GDMA generation time: 25.22s
```

## 4. Fit Charges

Create a GDMA record and solve for partial charges using SVD.

```python
from pympfit import MoleculeGDMARecord, MPFITSVDSolver, generate_mpfit_charge_parameter
from openff.units.elements import SYMBOLS

record = MoleculeGDMARecord.from_molecule(molecule, coords, multipoles, settings)
solver = MPFITSVDSolver(svd_threshold=1e-4)
parameter = generate_mpfit_charge_parameter([record], solver)

for i, atom in enumerate(molecule.atoms):
    element = SYMBOLS[atom.atomic_number]
    print(f"  {element}{i + 1:>2d}: {parameter.value[i]:+.4f}")
print(f"  Total: {sum(parameter.value):+.4f}")
```

```text
Fitted charges:
  C 1: +0.3348
  C 2: +0.4628
  O 3: -0.5387
  H 4: -0.1124
  H 5: -0.1180
  H 6: -0.1018
  H 7: -0.1065
  H 8: -0.1180
  H 9: +0.2977
  Total: -0.0000
```

# MBIS Multipoles

MBIS (Minimal Basis Iterative Stockholder) is an alternative charge
method that can be used instead of GDMA.

## 5. MBIS Settings

`MBISSettings` controls the Psi4 QM calculation through MBIS-specific parameters.

```python
from pympfit import MBISSettings

mbis_settings = MBISSettings(
    method="pbe0",
    basis="def2-SVP",
    max_moment=3,  # 1=charges, 2=+dipoles, 3=+quadrupoles, 4=+octupoles
    max_radial_moment=4,  # Must be >= max_moment
    limit=3,  # Multipole expansion order for MPFIT (should match max_moment)
    multipole_format="spherical",  # "spherical" or "cartesian"
    mpfit_inner_radius=6.78,
    mpfit_outer_radius=12.45,
    mpfit_atom_radius=3.0,
)
```

## 6. Generate MBIS Multipoles

Run Psi4 to compute MBIS multipole moments. The process is similar to GDMA.

```python
from pympfit import Psi4MBISGenerator

t0 = time.perf_counter()
coords, multipoles = Psi4MBISGenerator.generate(
    molecule, conformer, mbis_settings, minimize=True
)
elapsed = time.perf_counter() - t0

print(f"Multipoles shape: {multipoles.shape}")
print(f"MBIS generation time: {elapsed:.2f}s")
```

```text
Multipoles shape: (9, 9)
MBIS generation time: 28.45s
```

Note: MBIS with `max_moment=3` produces 9 components per atom (charges, dipoles,
and quadrupoles)

## 7. Fit Charges from MBIS

Create an MBIS record and fit partial charges using the same solver.

```python
from pympfit import MoleculeMBISRecord

mbis_record = MoleculeMBISRecord.from_molecule(
    molecule, coords, multipoles, mbis_settings
)
solver = MPFITSVDSolver(svd_threshold=1e-4)
parameter = generate_mpfit_charge_parameter([mbis_record], solver)

print("Fitted charges vs. MBIS charges:")
for i, atom in enumerate(molecule.atoms):
    element = SYMBOLS[atom.atomic_number]
    print(f"  {element}{i + 1:>2d}: {parameter.value[i]:+.4f} (MBIS: {multipoles[i, 0]:+.4f})")
print(f"  Total: {sum(parameter.value):+.4f}")


```

```text
Fitted charges vs. MBIS charges:
  C 1: -0.1208 (MBIS: -0.4504)
  C 2: -0.0499 (MBIS: +0.1712)
  O 3: -0.5360 (MBIS: -0.6016)
  H 4: +0.0413 (MBIS: +0.1308)
  H 5: +0.0292 (MBIS: +0.1111)
  H 6: +0.0651 (MBIS: +0.1291)
  H 7: +0.1159 (MBIS: +0.0693)
  H 8: +0.0628 (MBIS: +0.0289)
  H 9: +0.3924 (MBIS: +0.4115)
  Total: +0.0001
```

**Note:** MBIS and GDMA produce different charges because they use different
charge models.
