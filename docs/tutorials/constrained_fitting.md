# Constrained Fitting

This tutorial fits ethanol with constrained MPFIT so that
symmetry-equivalent atoms (e.g., methyl hydrogens) receive identical
charges. The full runnable script is at
`examples/tutorials/constrained_fitting.py`.

The [Quick Start](quickstart.md) tutorial covers the unconstrained SVD
path. Here we add atom-type equivalence constraints and a charge
conservation penalty so the fitted charges respect molecular symmetry.

## 1. Molecule, Conformer, and Multipoles

The setup is identical to the quick start — create a molecule, generate a
conformer, run Psi4/GDMA, and store the result as a record.

```python
from openff.recharge.utilities.molecule import extract_conformers
from openff.toolkit import Molecule

from pympfit import GDMASettings, MoleculeGDMARecord, Psi4GDMAGenerator

molecule = Molecule.from_smiles("CCO")
molecule.generate_conformers(n_conformers=1)
[conformer] = extract_conformers(molecule)
settings = GDMASettings()
coords, multipoles = Psi4GDMAGenerator.generate(
    molecule, conformer, settings, minimize=True,
)
record = MoleculeGDMARecord.from_molecule(molecule, coords, multipoles, settings)
```

## 2. Atom Type Labels

The OpenFF Toolkit provides atom symmetry functions to spot equivalent
atom types within the *same* molecule. `generate_global_atom_type_labels`
wraps this detection and returns a label per atom — atoms with the same
label will be constrained to carry the same total charge.

The `equivalize_between_*` flags control which atom classes are allowed
to share labels. Setting all four to `True` (the default) means every
topologically equivalent pair gets the same label:

```python
from pympfit import generate_global_atom_type_labels

labels = generate_global_atom_type_labels(
    [molecule],
    equivalize_between_methyl_carbons=True,
    equivalize_between_methyl_hydrogens=True,
    equivalize_between_other_heavy_atoms=True,
    equivalize_between_other_hydrogen_atoms=True,
)
```

For ethanol this produces three hydrogen groups and three unique heavy
atoms:

```text
Atom type labels:
  C 1: C1
  C 2: C2
  O 3: O1
  H 4: H1
  H 5: H1
  H 6: H1
  H 7: H2
  H 8: H2
  H 9: H3
```

H4–H6 (methyl hydrogens) share label `H1`, H7–H8 (methylene hydrogens)
share `H2`, and the hydroxyl hydrogen H9 is unique (`H3`). The optimizer
will force atoms within each group to have the same total charge.

## 3. Constrained Fit

Pass the record and molecule to
`generate_constrained_mpfit_charge_parameter` with a
`ConstrainedSciPySolver`. The `conchg` parameter controls the strength of
the charge conservation penalty — larger values enforce stricter total
charge conservation at the cost of a slightly worse multipole fit.

```python
from pympfit import ConstrainedSciPySolver, generate_constrained_mpfit_charge_parameter

solver = ConstrainedSciPySolver(conchg=10.0)
[parameter] = generate_constrained_mpfit_charge_parameter(
    [record], [molecule], solver=solver,
)
```

```text
Fitted charges:
  C 1 ( C1): +0.0386
  C 2 ( C2): +0.1576
  O 3 ( O1): -0.5213
  H 4 ( H1): -0.0041
  H 5 ( H1): -0.0041
  H 6 ( H1): -0.0041
  H 7 ( H2): +0.0066
  H 8 ( H2): +0.0066
  H 9 ( H3): +0.3243
  Total: -0.0000
```

All H1 atoms carry exactly the same charge, as do the H2 atoms. The
total charge sums to zero, enforced by the conservation penalty.

Compare with the unconstrained SVD result from the [Quick Start](quickstart.md)
where H4, H5, and H6 each had slightly different values. The constrained
fit trades a small amount of multipole accuracy for physically meaningful
symmetry.
