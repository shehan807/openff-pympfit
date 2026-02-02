"""Multi-molecule constrained fitting: transferable charges across imidazolium ILs.

Fits MMIM, EMIM, BMIM, and C6MIM cations simultaneously such that shared
chemical environments (e.g., imidazolium ring) carry identical charges
across all four molecules.

Requires Psi4 if no sqlite file is available. Run with:
    python examples/tutorials/multi_constrained_fitting.py
"""

import pathlib
import time

from openff.recharge.utilities.molecule import extract_conformers
from openff.toolkit import Molecule
from openff.units.elements import SYMBOLS

from pympfit import (
    ConstrainedSciPySolver,
    GDMASettings,
    MoleculeGDMARecord,
    Psi4GDMAGenerator,
    generate_constrained_mpfit_charge_parameter,
    generate_global_atom_type_labels,
)
from pympfit.gdma.storage import MoleculeGDMAStore

# pre-computed sqlite record file
SQLITE_FILE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "pympfit"
    / "tests"
    / "data"
    / "gdma"
    / "ionic_liquids.sqlite"
)

smiles_map = {
    "MMIM": "CN1C=C[N+](=C1)C",
    "EMIM": "CCN1C=C[N+](=C1)C",
    "BMIM": "CCCCN1C=C[N+](=C1)C",
    "C6MIM": "CCCCCCN1C=C[N+](=C1)C",
}

molecules = []
names = list(smiles_map.keys())
for name, smi in smiles_map.items():
    molecules.append(Molecule.from_smiles(smi))

# atom type labels
labels = generate_global_atom_type_labels(
    molecules,
    radius=2,
    equivalize_between_methyl_carbons=True,
    equivalize_between_methyl_hydrogens=True,
    equivalize_between_other_heavy_atoms=True,
    equivalize_between_other_hydrogen_atoms=True,
)

# GDMA records
records = []
if SQLITE_FILE.exists():
    store = MoleculeGDMAStore(str(SQLITE_FILE))
    for name, smi in smiles_map.items():
        recs = store.retrieve(smiles=smi)
        assert len(recs) > 0, f"No record for {name}"
        records.append(recs[0])
    print(f"Loaded records from {SQLITE_FILE.name}")
else:
    settings = GDMASettings()
    for name, mol in zip(names, molecules):
        mol.generate_conformers(n_conformers=1)
        [conf] = extract_conformers(mol)
        t0 = time.perf_counter()
        coords, multipoles = Psi4GDMAGenerator.generate(
            mol, conf, settings, minimize=True,
        )
        print(f"{name} GDMA: {time.perf_counter() - t0:.2f}s")
        records.append(
            MoleculeGDMARecord.from_molecule(mol, coords, multipoles, settings)
        )
print()

# constrained fit
solver = ConstrainedSciPySolver(conchg=10.0)
parameters = generate_constrained_mpfit_charge_parameter(
    records, molecules, solver=solver,
)

for name, mol, param, lab in zip(names, molecules, parameters, labels):
    print(f"--- {name} ---")
    for i, atom in enumerate(mol.atoms):
        element = SYMBOLS[atom.atomic_number]
        print(f"  {element}{i + 1:>2d} ({lab[i]:>4s}): {param.value[i]:+.4f}")
    print(f"  Total: {sum(param.value):+.4f}")
    print()
