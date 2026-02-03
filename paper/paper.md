---
title: 'PyMPFIT: Multipole Moment-based Partial Charge Assignment'
tags:
  - Python
authors:
  - name: Shehan M. Parmar
    orcid: 0000-0002-2033-0862
    corresponding: true
    affiliation: 1
  - name: Jesse McDaniel
    orcid: 0000-0002-9211-1108
    affiliation: 1
affiliations:
  - name: Georgia Institute of Technology, College of Chemistry and Biochemistry, Atlanta, GA, United States
    index: 1
date: 3 February 2026
bibliography: paper.bib
---

# Summary

# Statement of need

Modern *in silico* materials discovery pipelines depend on mature, high-throughput (HT) software infrastructure. Central to parameterizing HT classical molecular dynamics (MD) force fields [CITE] or more recently, featurizing machine learning interatomic potentials (MLIPs) [CITE], is accurately representing the molecular electrostatic potential (ESP). Short- and long-range electrostatics are consequential in computational property prediction across domains, including drugs [CITE], battery electrolytes [CITE], and fuels [CITE].

For decades, point charge models have matured as a widely accepted, coarse-grained representation for molecular electrostatics [CITE; CITE; CITE], in which a set of partial charges located at atomic nuclei are fit via least-squares regression to reproduce the quantum mechanical (QM) ESP...discuss common ESP-based point charge methods like MSK, CHELPG, RESP [@zhao_pyresp_2022]...

To this end, PyMPFIT address the need for multipole moment-aware point charge models..introduce the fundamental need of (1) support for multipole moment based fitting codes built for high throughput, transferrability, and user friendliness. cite the various papers that use GDMA/MPFIT yet the support for software is slim or bespoke (AMOEBA, CHARMM, etc.); moreover, However, a persistent challenge in deploying
HTMD pipelines lies in parameterizing classical force fields, particularly in assigning accurate and
transferable partial atomic charges.

# State of the field

- discuss all the ESP-based codes and support, like chelpg, openff-recharge, psi4resp, etc. also briefly mention cheaper population analyses like mulliken charges, hiersfield, etc.

- discuss software support for MPFIT/GDMA and cite discrepancy between use cases (many) and support (limited)

# Software design

PyMPFIT adopts a modular architecture designed for high-throughput workflows. The software is organized into four layers: (1) `gdma`, providing an abstraction layer for computing multipole moments by calling the Psi4/GDMA Python API; (2) `storage`, implementing an SQLITE/SQLAlchemy backend for persistent multipole moment data across various molecules and multiple conformers; (3) `mpfit`, the multipole fitting driver with support for (a) singular value decomposition and (b) nonlinear, constrained optimization for transferrable partial charges; (4) `optimize`, providing modular objective functions, including Bayesian virtual site fitting.

The user-facing API exposes three primary functions:

# Research impact statement

# AI usage disclosure

# Acknowledgements

# References
