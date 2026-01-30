# Quick Start

This tutorial walks through fitting partial charges for ethanol using PyMPFIT.
The full runnable script is at `examples/tutorials/quickstart.py`.

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
