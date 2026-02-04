============
Installation
============

PyMPFIT depends on `OpenFF <https://openforcefield.org/>`_ packages which are distributed via conda-forge.

.. code-block:: bash

   conda create -n pympfit python=3.12 openff-recharge openff-utilities psi4 pygdma -c conda-forge -y
   conda activate pympfit
   pip install pympfit

*Works with conda, mamba, or micromamba.*

Bayesian Virtual Site Fitting
-----------------------------

For Bayesian optimization of virtual site parameters:

.. code-block:: bash

   pip install pyro-ppl arviz matplotlib sphericart-torch

Development Installation
------------------------

For development or to get the latest changes:

.. code-block:: bash

   conda create -n pympfit-dev python=3.12 openff-recharge openff-utilities psi4 pygdma -c conda-forge -y
   conda activate pympfit-dev
   git clone https://github.com/shehan807/pympfit.git
   cd pympfit
   pip install -e ".[test]"
