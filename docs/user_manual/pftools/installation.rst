.. _pftools_installation:

Installation
============

``pftools`` can be installed with the following command:

.. code-block::

    pip install pftools[all]

The ``[all]`` argument will download the dependencies necessary for running ParFlow
and fully employing the other tools within this package. ``[all]`` encompasses the
subsets of dependencies, including:

- ``[pfsol]``: installs the ``imageio`` package for handling image processing to assist some workflows to build ParFlow solid (.pfsol) files.
- ``[io]``: installs the ``numpy``, ``xarray``, and ``dask`` packages for handling reading and storing of ParFlow binary (.pfb) data.
- ``[fastio]``: installs the ``numba`` package for translating certain I/O operations into fast machine code.

If you would like to set up a virtual environment to install ``pftools``, execute the following commands:

.. code-block::

    python3 -m venv py-env
    source py-env/bin/activate
    pip install pftools[all]
