Getting Started
===============

Installation
------------

PyMPFIT depends on `OpenFF <https://openforcefield.org/>`_ packages which are distributed via conda-forge.

.. code-block:: bash

    conda create -n pympfit python=3.12 openff-recharge openff-utilities psi4 pygdma -c conda-forge -y
    conda activate pympfit
    pip install pympfit

*Works with conda, mamba, or micromamba.*

Tutorial
--------

See the :doc:`Quick Start tutorial </tutorials/quickstart>` for a step-by-step guide.
