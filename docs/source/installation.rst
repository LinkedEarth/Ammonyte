Installation
============

System Requirements
-------------------

Ammonyte requires Python 3.8 or above. We recommend using Python 3.12.

Setting up an Environment
--------------------------

We recommend using `Anaconda <https://www.anaconda.com/>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ to manage your Python environment. To create a dedicated environment for Ammonyte:

.. code-block:: bash

   conda create -n ammonyte_env python=3.12
   conda activate ammonyte_env

Some dependencies require the conda-forge channel. Install cartopy before proceeding:

.. code-block:: bash

   conda install -c conda-forge cartopy

Installing Ammonyte
-------------------

**Stable release (recommended)**

Install the latest stable release from PyPI:

.. code-block:: bash

   pip install ammonyte

**Development version**

To install the latest development version directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/LinkedEarth/Ammonyte.git

Note that the development version may contain bugs or incomplete features.

Dependencies
------------

Ammonyte requires the following packages, which are installed automatically via pip:

- `pyleoclim <https://github.com/LinkedEarth/Pyleoclim_util>`_
- numpy
- scipy
- scikit-learn
- PyRQA
- ruptures

Testing the Installation
------------------------

To verify that Ammonyte has been installed correctly, run the following in Python:

.. code-block:: python

   import ammonyte as amt
   print(amt.__version__)

To run the full test suite:

.. code-block:: bash

   pytest ammonyte/tests/
