Getting Started
===============

Installation
------------

PyMPFIT depends on `OpenFF <https://openforcefield.org/>`_ packages which are distributed via conda-forge.

.. code-block:: bash

    conda create -n pympfit python=3.12 openff-recharge openff-utilities psi4 libint=2.9 pygdma -c conda-forge -y
    conda activate pympfit

.. note::
   The ``libint=2.9`` pin is a temporary workaround for a psi4/libint2 compatibility issue.
   This will be resolved in psi4 v1.10 build 3.
    conda activate pympfit
    pip install pympfit

*Works with conda, mamba, or micromamba.*

Tutorial
--------

See the :doc:`Quick Start tutorial </tutorials/quickstart>` for a step-by-step guide.
