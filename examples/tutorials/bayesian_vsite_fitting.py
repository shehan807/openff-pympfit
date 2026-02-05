"""Bayesian virtual site fitting with NUTS MCMC.

Jointly optimizes vsite charge increment and distance for chloromethane
using Pyro's NUTS sampler with differentiable MPFIT predictions.

Requires: pyro-ppl, torch, sphericart-torch, arviz, matplotlib
Run with: python examples/tutorials/bayesian_vsite_fitting.py
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from openff.recharge.charges.vsite import BondChargeSiteParameter, VirtualSiteCollection
from openff.recharge.utilities.molecule import extract_conformers
from openff.toolkit import Molecule
from pyro.infer import MCMC, NUTS

from pympfit import GDMASettings, MoleculeGDMARecord, Psi4GDMAGenerator
from pympfit.optimize import MPFITObjective

# molecule, conformer, and multipoles ---
molecule = Molecule.from_smiles("CCl")
molecule.generate_conformers(n_conformers=1)
[conformer] = extract_conformers(molecule)
settings = GDMASettings()
coords, multipoles = Psi4GDMAGenerator.generate(
    molecule, conformer, settings, minimize=True
)
record = MoleculeGDMARecord.from_molecule(molecule, coords, multipoles, settings)

# Define vsite on Cl-C bond
vsite_collection = VirtualSiteCollection(
    parameters=[
        BondChargeSiteParameter(
            smirks="[#17:1]-[#6:2]",
            name="EP",
            distance=0.0,
            charge_increments=(0.0, 0.0),
            sigma=0.0,
            epsilon=0.0,
            match="all-permutations",
        )
    ]
)

# Build objective with trainable vsite parameters
[objective_term] = list(
    MPFITObjective.compute_objective_terms(
        gdma_records=[record],
        vsite_collection=vsite_collection,
        _vsite_charge_parameter_keys=[("[#17:1]-[#6:2]", "BondCharge", "EP", 0)],
        _vsite_coordinate_parameter_keys=[
            ("[#17:1]-[#6:2]", "BondCharge", "EP", "distance")
        ],
    )
)

# Convert targets to torch
n_atoms = molecule.n_atoms
targets = [
    torch.from_numpy(r.astype(np.float64).flatten())
    for r in objective_term.reference_values
]


def model():
    # Priors
    free_charges = pyro.sample(
        "free_charges",
        dist.Normal(torch.zeros(n_atoms - 1, 1, dtype=torch.float64), 0.5),
    )
    charge_inc = pyro.sample(
        "charge_inc", dist.Normal(torch.zeros(1, 1, dtype=torch.float64), 0.2)
    )
    distance = pyro.sample(
        "distance", dist.Normal(torch.tensor([[1.5]], dtype=torch.float64), 0.5)
    )
    sigma = pyro.sample(
        "sigma", dist.HalfCauchy(torch.tensor([[0.1]], dtype=torch.float64))
    )

    preds = objective_term.predict_from_free_charges(free_charges, charge_inc, distance)

    # Likelihood
    for i, (pred, target) in enumerate(zip(preds, targets, strict=False)):
        pyro.sample(f"obs_{i}", dist.Normal(pred.flatten(), sigma), obs=target)


# Run MCMC
print("\nRunning NUTS (200 warmup, 500 samples)...")
mcmc = MCMC(NUTS(model), num_samples=500, warmup_steps=200, num_chains=1)
mcmc.run()

# Results
samples = mcmc.get_samples()

# Atom charges
free = samples["free_charges"].numpy()
last = -free.sum(axis=1, keepdims=True)
atom_q = np.concatenate([free, last], axis=1)
print("\nAtom charges:")
for i, atom in enumerate(molecule.atoms):
    mean, std = atom_q[:, i, 0].mean(), atom_q[:, i, 0].std()
    print(f"  {atom.symbol}{i+1}: {mean:+.4f} +/- {std:.4f}")

# Vsite parameters
charge_inc = samples["charge_inc"].numpy()[:, 0, 0]
distance = samples["distance"].numpy()[:, 0, 0]
print(f"\nVsite charge increment: {charge_inc.mean():+.4f} +/- {charge_inc.std():.4f}")
print(f"Vsite distance: {distance.mean():.3f} ± {distance.std():.3f} Å")

# Visualization with ArviZ
idata = az.from_pyro(mcmc, log_likelihood=False)

idata.posterior["vsite_charge_increment"] = idata.posterior["charge_inc"].squeeze()
idata.posterior["vsite_distance_A"] = idata.posterior["distance"].squeeze()

az.plot_trace(idata, var_names=["vsite_charge_increment", "vsite_distance_A"])
plt.tight_layout()
plt.savefig("bayesian_vsite_trace.png", dpi=150)
print("\nTrace plot saved to bayesian_vsite_trace.png")
