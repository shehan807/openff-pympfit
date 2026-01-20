# OpenFF PyMPFIT

<!-- Logo placeholder -->
<!-- ![OpenFF PyMPFIT Logo](docs/_static/logo.png) -->

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/shehan807/openff_pympfit/workflows/CI/badge.svg)](https://github.com/shehan807/openff_pympfit/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/shehan807/openff_pympfit/branch/main/graph/badge.svg)](https://codecov.io/gh/shehan807/openff_pympfit)
[![Documentation Status](https://readthedocs.org/projects/openff-pympfit/badge/?version=latest)](https://openff-pympfit.readthedocs.io/en/latest/?badge=latest)

OpenFF-PyMPFIT is a free, open-source software for performing partial atomic charge fitting using the Gaussian distributed multipole analysis (GDMA). Features include:

* Built on open-source libraries: [OpenFF Recharge](https://github.com/openforcefield/openff-recharge), [OpenFF Toolkit](https://github.com/openforcefield/openff-toolkit), and [NumPy](https://numpy.org/)
* Direct interface to [Psi4](https://psicode.org/) / [GDMA](https://github.com/psi4/gdma) and from wavefunctions stored within [QCFractal](https://github.com/MolSSI/QCFractal) (i.e., [QCArchive](https://qcarchive.molssi.org/))
* Generating multipole moments for multi-conformer molecules
* An SQLite database backend for efficient high-throughput scaling
* Bayesian methods ([Pyro](https://pyro.ai/)) for flexible virtual site fitting

## Installation

**Install from conda-forge (recommended):**

```bash
conda install -c conda-forge openff-pympfit
```

**Note:** GDMA functionality requires Psi4 and PyGDMA:

```bash
conda install -c conda-forge psi4 pygdma
```

**Install from source:**

```bash
git clone https://github.com/shehan807/openff-pympfit.git
cd openff-pympfit
pip install -e .
```

## Quick Example

```python
from openff.toolkit import Molecule
from openff.recharge.utilities.molecule import extract_conformers
from openff_pympfit import (
    generate_mpfit_charge_parameter,
    GDMASettings,
    Psi4GDMAGenerator,
    MoleculeGDMARecord,
    MPFITSVDSolver,
)

# Create molecule and generate conformer
molecule = Molecule.from_smiles("CCO")
molecule.generate_conformers(n_conformers=1)
[conformer] = extract_conformers(molecule)

# Generate GDMA multipoles (requires Psi4)
settings = GDMASettings()
coords, multipoles = Psi4GDMAGenerator.generate(molecule, conformer, settings)
record = MoleculeGDMARecord.from_molecule(molecule, coords, multipoles, settings)

# Fit charges
charges = generate_mpfit_charge_parameter([record], MPFITSVDSolver())
print(f"Charges: {charges.value}")
```

## License

The main package is released under the [MIT license](LICENSE).

## Copyright

Copyright (c) 2026, Shehan M. Parmar

### Acknowledgements

Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.
