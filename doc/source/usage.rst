Basic Usage
-----------

The basic class is ``ParticleFile`` giving access to the content
of a LADiM output file.

.. code-block:: python

    from postladim import ParticleFile
    pf = ParticleFile("output.nc")

The ``ds`` method shows the underlying xarray Dataset, which is useful for a quick
overview of the content and for more advanced data processing.

The time at a given time step n is given by:

.. code-block:: python

  pf.time[n]

The output may be a bit verbose, ``pf.time(n)`` is syntactic sugar for the more concise``pf.time[n].values``.

The number of particles at time step n is given by:

.. code-block:: python

  pf.count[n]

For use in indexing ``pf.count`` is an integer array instead of an xarray DataArray.
The DataArray is avaiable as ``pf.ds.particle_count``.

Following pandas and xarray, an instance variable, like X,  is given both by
attribute ``pf.X`` and item ``pf['X']`` notation. The NetCDF inspired notation
``pf.variables['X']`` is obsolete.

The most basic operation for an instance variable is to get the values at time step n as
a xarray DataArray:

.. code-block:: python

  pf.X[n]

An alternative notation is ``pf.X.isel(time=n)``. The time stamp can be used instead of
the time index:

.. code-block:: python

  pf.X.sel(time='2020-02-05 12')

The format is optimized for particle distributions at a given time. Trajectories and
other time series for a given particle may take longer time to extract. For the particle
with identifier `pid=p`, the X-coordinate of the trajectory is given
by:

.. code-block:: python

  pf.X.sel(pid=p)

If many trajectories are needed, it may be useful to turn the dataset into a full (i.e.
non-sparse) 2D DataArray, indexed by time and particle identifier.

.. code-block:: python

  pf.X.full()

Note that for long simulations with particles of limited life span, this array may
become much larger than the ParticleFile.

