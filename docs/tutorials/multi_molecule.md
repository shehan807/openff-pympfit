# Multi-Molecule Constrained Parameterization

Transferability across multiple molecules that share common atom types is
convenient for classical force field partial charge parameterization as
well as machine learning interatomic potential (MLIP) feature generation.
To this end, this tutorial highlights an automated atom typing scheme
using
[Morgan fingerprints](https://www.rdkit.org/docs/cppapi/namespaceRDKit_1_1MorganFingerprints.html)
to constrain shared chemical environments across molecules to carry
identical charges. The full runnable script is at
`examples/tutorials/multi_constrained_fitting.py`.

## 1a. Molecule, Conformer, and Record Generation

First, as before, we select four ionic liquid cations for this example:
MMIM, EMIM, BMIM, and C6MIM. As in the previous tutorials, we define
the SMILES, generate conformers, and create the GDMA records. Here, it
is useful to save the `.sqlite` record file for later use.

```python
import pathlib

from openff.recharge.utilities.molecule import extract_conformers
from openff.toolkit import Molecule

from pympfit import GDMASettings, MoleculeGDMARecord, Psi4GDMAGenerator
from pympfit.gdma.storage import MoleculeGDMAStore

smiles_map = {
    "MMIM": "CN1C=C[N+](=C1)C",
    "EMIM": "CCN1C=C[N+](=C1)C",
    "BMIM": "CCCCN1C=C[N+](=C1)C",
    "C6MIM": "CCCCCCN1C=C[N+](=C1)C",
}

molecules = [Molecule.from_smiles(smi) for smi in smiles_map.values()]
names = list(smiles_map.keys())

settings = GDMASettings()
records = []
for mol in molecules:
    mol.generate_conformers(n_conformers=1)
    [conf] = extract_conformers(mol)
    coords, multipoles = Psi4GDMAGenerator.generate(
        mol, conf, settings, minimize=True,
    )
    records.append(
        MoleculeGDMARecord.from_molecule(mol, coords, multipoles, settings)
    )
```

## 1b. [Optional] Import GDMA SQLite Records

Instead of calculating expensive Psi4/GDMA multipoles each time, one can
also import the `.sqlite` data from a previous run or as provided in this
repo at `pympfit/pympfit/tests/data/gdma/ionic_liquids.sqlite`:

```python
SQLITE_FILE = pathlib.Path("pympfit/tests/data/gdma/ionic_liquids.sqlite")

store = MoleculeGDMAStore(str(SQLITE_FILE))
records = []
for smi in smiles_map.values():
    recs = store.retrieve(smiles=smi)
    records.append(recs[0])
```

## 2. Generate Atom Type Labels

Morgan fingerprints are molecular fingerprints — each fingerprint bit
(provided as a unique hash) corresponds to a fragment of the molecule
based on a `radius` parameter, measured in number of bonds beyond the
selected atom.

`generate_global_atom_type_labels` combines these fingerprints with
within-molecule symmetry detection to assign labels. Atoms with the same
label across different molecules will be constrained to carry the same
charge.

```python
from pympfit import generate_global_atom_type_labels

labels = generate_global_atom_type_labels(
    molecules,
    radius=2,
    equivalize_between_methyl_carbons=True,
    equivalize_between_methyl_hydrogens=True,
    equivalize_between_other_heavy_atoms=True,
    equivalize_between_other_hydrogen_atoms=True,
)
```

The shared atom types can be visualized using the tutorial utility:

```python
from util import _generate_rdkit_imgs

_generate_rdkit_imgs(molecules, labels, names, "figures/", prefix="multi_molecule_")
```

````{list-table}
:header-rows: 0
:widths: 50 50

* - ```{image} /_static/tutorials/multi_molecule_mmim_shared_labels.svg
    ```
  - ```{image} /_static/tutorials/multi_molecule_emim_shared_labels.svg
    ```
* - ```{image} /_static/tutorials/multi_molecule_bmim_shared_labels.svg
    ```
  - ```{image} /_static/tutorials/multi_molecule_c6mim_shared_labels.svg
    ```
````

The generated atom labels highlight the transferable atom types across
the molecules. Atoms with the same color share the same label and will
be constrained to carry identical charges. Unhighlighted atoms are
molecule-specific.

## 3. Obtain Charges

Pass all records and molecules together. The solver fits a single
reduced parameter vector across all four molecules simultaneously,
enforcing that shared atom types carry identical total charges. Each
molecule's total charge is constrained to +1.0 (cation) via the
`conchg` penalty.

```python
from pympfit import ConstrainedSciPySolver, generate_constrained_mpfit_charge_parameter

solver = ConstrainedSciPySolver(conchg=10.0)
parameters = generate_constrained_mpfit_charge_parameter(
    records, molecules, solver=solver,
)
```

:::{dropdown} MMIM charges
:open:

```{code-block} text
C 1 (  C1): +0.6183
N 2 (  N1): +0.0871
C 3 (  C2): +0.0216
C 4 (  C3): +0.0687
N 5 (  N2): +0.0750
C 6 (  C4): +0.1778
C 7 (  C5): +0.0499
H 8 (  H1): -0.1942
H 9 (  H1): -0.1942
H10 (  H1): -0.1942
H11 (  H2): +0.0488
H12 (  H3): +0.0949
H13 (  H4): +0.1592
H14 (  H5): +0.0605
H15 (  H5): +0.0605
H16 (  H5): +0.0605
Total: +1.0000
```

:::

:::{dropdown} EMIM charges

```{code-block} text
C 1 (  C6): +0.0181
C 2 (  C7): +0.1877
N 3 (  N3): +0.1253
C 4 (  C8): +0.0503
C 5 (  C3): +0.0687
N 6 (  N2): +0.0750
C 7 (  C9): +0.0231
C 8 (  C5): +0.0499
H 9 (  H6): -0.0118
H10 (  H6): -0.0118
H11 (  H6): -0.0118
H12 (  H7): -0.0236
H13 (  H7): -0.0236
H14 (  H2): +0.0488
H15 (  H3): +0.0949
H16 (  H4): +0.1592
H17 (  H5): +0.0605
H18 (  H5): +0.0605
H19 (  H5): +0.0605
Total: +1.0000
```

:::

:::{dropdown} BMIM charges

```{code-block} text
C 1 ( C10): +0.1110
C 2 ( C11): +0.0932
C 3 ( C12): +0.1596
C 4 ( C13): +0.3213
N 5 (  N4): -0.0026
C 6 (  C8): +0.0503
C 7 (  C3): +0.0687
N 8 (  N2): +0.0750
C 9 (  C9): +0.0231
C10 (  C5): +0.0499
H11 (  H6): -0.0118
H12 (  H6): -0.0118
H13 (  H6): -0.0118
H14 (  H8): -0.0569
H15 (  H8): -0.0569
H16 (  H9): -0.0989
H17 (  H9): -0.0989
H18 ( H10): -0.0434
H19 ( H10): -0.0434
H20 (  H2): +0.0488
H21 (  H3): +0.0949
H22 (  H4): +0.1592
H23 (  H5): +0.0605
H24 (  H5): +0.0605
H25 (  H5): +0.0605
Total: +1.0000
```

:::

:::{dropdown} C6MIM charges

```{code-block} text
C 1 ( C10): +0.1110
C 2 ( C11): +0.0932
C 3 ( C14): +0.5221
C 4 ( C15): +0.3511
C 5 ( C16): -0.3182
C 6 ( C13): +0.3213
N 7 (  N4): -0.0026
C 8 (  C8): +0.0503
C 9 (  C3): +0.0687
N10 (  N2): +0.0750
C11 (  C9): +0.0231
C12 (  C5): +0.0499
H13 (  H6): -0.0118
H14 (  H6): -0.0118
H15 (  H6): -0.0118
H16 (  H8): -0.0569
H17 (  H8): -0.0569
H18 (  H9): -0.0989
H19 (  H9): -0.0989
H20 (  H9): -0.0989
H21 (  H9): -0.0989
H22 (  H9): -0.0989
H23 (  H9): -0.0989
H24 ( H10): -0.0434
H25 ( H10): -0.0434
H26 (  H2): +0.0488
H27 (  H3): +0.0949
H28 (  H4): +0.1592
H29 (  H5): +0.0605
H30 (  H5): +0.0605
H31 (  H5): +0.0605
Total: +1.0000
```

:::

Each molecule's total charge is +1.0, enforced by the conservation
penalty. Shared atom types carry identical charges across molecules —
for example, all N2 atoms have charge +0.0750, all H5 atoms have
+0.0605, and all C3 atoms have +0.0687, regardless of which molecule
they appear in.
