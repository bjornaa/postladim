Postladim documentation
=======================

Postladim is a simple python package for analyzing particle distributions. It is
designed for `sparse`(ragged array)` output from the `LADiM
<https://github.com/bjornaa/ladim2>`_ particle tracking model. The user
interface is to a large degree modelled after the `xarray
<https://xarray.pydata.org/en/stable/>`_ package.

The alternative `dense` (full array) output from LADiM is not considered as this
can be handled directly by xarray.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   usage.rst
   format.rst
   variables.rst
   api.rst

.. only: html

  Indices and tables
  ==================

  * :ref:`genindex`
  * :ref:`modindex`
  * :ref:`search`
