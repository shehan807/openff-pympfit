.. toctree::
   :caption: Getting Started
   :hidden:

   Introduction <self>
   installation
   tutorials/index

.. toctree::
   :caption: User Guide
   :hidden:

   user_guide/theory

.. toctree::
   :caption: Reference
   :hidden:

   API <api>

.. Comment out developer guide for now
.. .. toctree::
..    :caption: Development
..    :hidden:
..
..    developer_guide

============
Introduction
============

**PyMPFIT** is a Python implementation of the Multipole Fitting (MPFIT)
algorithm for deriving partial atomic charges from Gaussian distributed multipole
moments. **PyMPFIT**, originally a fork of
`openff-recharge <https://github.com/openforcefield/openff-recharge>`_, is built
upon open-source libraries, including
`OpenFF Recharge <https://github.com/openforcefield/openff-recharge>`_ and
`OpenFF Toolkit <https://github.com/openforcefield/openff-toolkit>`_.

Features
--------

The framework currently supports:

* **Generating multi-conformer QC DMA and multipole moment data**

  - Directly by interfacing with the `Psi4 <https://psicode.org/>`_ /
    `GDMA <https://github.com/psi4/gdma>`_ quantum chemical code
  - From wavefunctions stored within a
    `QCFractal <https://github.com/MolSSI/QCFractal>`_ instance, including the
    `QCArchive <https://qcarchive.molssi.org/>`_

* **Flexible, Bayesian virtual site fitting** powered by
  `Pyro <https://pyro.ai/>`_ with planned support for
  `NumPyro <https://num.pyro.ai/>`_

* A SMARTS port for direct partial charge output

* An `SQLite <https://www.sqlite.org/>`_ database backend for efficient
  high-throughput scaling
