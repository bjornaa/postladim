Basic Usage
-----------

The most fundamental task is to open and examine a LADiM output file.
This is done with the class ``ParticleFile``.

To open the file use:

.. code-block:: python

    from postladim import ParticleFile
    pf = ParticleFile("output.nc")

or better with a context manager

.. code-block:: python

    from postladim import ParticleFile
    with ParticleFile("output.nc") as pf:
        # Do something with the data in pf

To get an overview of the content, simply print `pf` in a jupyter notebook or an ipython session.

To access all particle positions at timestep n use:

.. code-block:: python

    X, Y = pf.position(n)

Here X and Y becomes xarray DataArrays.

Following pandas and xarray, an instance variable, like X,  is given both by
attribute notation ``pf.X`` and item notation ``pf["X"]``. The NetCDF inspired
notation ``pf.variables["X"]`` is obsolete.

To get all values of an instance variable `X` at the same timestep use one of:

.. code-block: python

   X = pf.X[n]
   X = pf.X.isel(time=n)

The time at a given time step n is given by:

.. code-block:: python

  pf.time[n]

The output may be a bit verbose. To just get the time formatted as an iso-string
use "formatted time", ``pf.ftime(n)``, which is shorthand for
``pf.time[n].values``.

The number of particles at time step n is given by:

.. code-block:: python

  pf.count[n]

For use in indexing operations ``pf.count`` is an numpy integer array instead
of an xarray DataArray.

For time, value based indexing is available py the ``sel`` method:

.. code-block:: python

    X = pf.X.sel(time="2022-10-19 12")

The sparse format is optimized for particle distributions at a given time. Trajectories and
other time series for a given particle may take longer time to extract. For the particle
with identifier `pid=p`, the X-coordinate of the trajectory is given
by:

.. code-block:: python

  X = pf.X.sel(pid=p)

For the trajectory as a curve, one can use the shorthand:

.. code-block:: python

  X, Y = pf.trajectory(p)

If many trajectories are needed, it may be useful to turn the dataset into a full (i.e.
dense) 2D DataArray, indexed by time and particle identifier.

.. code-block:: python

  pf.X.full()

Note that for long simulations with particles of limited life span, this array may
become much larger than the ParticleFile.

