============
Installation
============

From pip
--------

.. code-block:: bash

   pip install pympfit

From source
-----------

For development or to get the latest changes:

.. code-block:: bash

   git clone https://github.com/openforcefield/pympfit.git
   cd pympfit

   # Create conda environment with all dependencies
   conda env create -f devtools/conda-envs/test_env_rdkit.yaml
   conda activate pympfit-test-rdkit

   # Install in editable mode
   pip install -e .

Psi4 and GDMA
-------------

GDMA functionality requires `Psi4 <https://psicode.org/>`_ and
`PyGDMA <https://github.com/psi4/gdma>`_, which must be installed via conda:

.. code-block:: bash

   conda install -c psi4 psi4 pygdma

This is required for both pip and source installations.
