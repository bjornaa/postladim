Postladim API
-------------

ParticleSet
............

.. class:: ParticleSet(ds: xarray.Dataset)

   Class for LADiM result files. Documentation is incomplete.

  .. attribute:: ds

     The underlying xarray Dataset.

  .. attribute:: num_times

     Number of time frames in the data set.

  .. attribute:: num_particles

    The number of distinct particles

  .. attribute:: count

     Integer numpy ndarray of particle counts at each time step

  .. attribute:: start

     Integer ndarray of start indices at each time step

  .. attribute:: end

     Integer ndarray of end indices, short hand for  ps.start + ps.count

  .. attribute:: instance_variables

     List of particle instance variables

  .. attribute:: particle_variables

     List of particle variables.

  .. attribute:: time

     xarray DataArray of the time steps

  .. method:: position(n)

     Returns a tuple with position (X, Y) of particle-distribution at n-th time time,
     ``pf.position(n) = (ps.X[n], ps.Y[n])``

  .. method:: trajectory(p)

     Returns a tuple of X and Y coordinates of the particle with identifier pid=p,
     ``trajectory(p) = (ps.X.sel(pid=p), ps.Y.sel(pid=p))``.

  .. attribute:: variables

     Deprecated, dictionary of variables, ``ps.variables['X'] = ps['X'] = ps.X``.

  .. method:: ftime(n)

     ISO-formatted time, ``ps.ftime(n) = ps.time[n].values``


ParticleFile
............

.. class:: ParticleFile(filename)

    Subclasss of ParticleSet for content of a particle file

    .. attribute:: path
        
    The filename

    ParticleFile can be used as a context manager.


InstanceVariable
................

.. class:: InstanceVariable

  .. attribute:: da

    The underlying xarray DataArray

  .. attribute:: count

    Number of particles at given time

  .. attribute:: time

    The time steps

  .. attribute:: num_particles

    The number of distinct particles

  .. method:: isel(time=n)

    Select by time index, ``V.isel(time=n) = V[n]``.

  .. method:: sel(time=t, pid=p)

    Select by time value and/or pid value.

    ``V.sel(time='2019-09-19 12')`` selects particles at that time value.
    ``V.sel(pid=23)`` selects the time history of particle with that pid value.
    ``V.sel(time='2019-09-19 12', pid=23)`` selects the unique particle instance.

  .. method:: full()

     Return the full (non-sparse) 2D DataArray. May become vary large.



The InstanceVariable class support item notation:

.. code-block:: python

  V[n] = V.isel(time=n)
  V[n, p] = V.isel(time=n).sel(pid=p)

Note that ``V[n, p]`` may be different from ``V[n][p]`` if particles have been removed.

The `length` of an instance variable is the number of time steps,
``len(V) == len(V.time)``.
