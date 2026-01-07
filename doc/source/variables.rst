Particle tracking variables
---------------------------

There are two types of variables considered. 

Instance variables
..................

`Instance variables` are connected with a `particle instance`, a particle at a
given time. Examples are position coordinates and time-changing properties such
as age, weight, e.t.c. Even if the particle identifier, the ``pid``, is constant
over time it is considered an instance variable. It can be viewed as a
projection from the ``particle_instance`` dimension to the ``particle`` dimension.

Postladim has an ``InstanceVariable`` class. Usually it is instantiated from an
open ``ParticleFile``, say ``pf``, by ``X = pf.X = pf["X"]``. It tries to
emulate an xarray DataArray with dimensions ``time`` and ``pid``. The values at
time step ``n`` are extracted as an xarray DataArray by item notation ``X[n]``
or alternatively by the ``sel`` method ``X.isel(time=n)``. The ``sel`` method
can also be used to extract the time history of an individual particle
``X.sel(pid=p)``. The underlying DataArray of an InstanceVariable ``X`` can be
accessed as ``X.da``.

Particle variables
..................

`Particle variables` are connected to the particles, not specific points in time.
Examples are release time, release location, size of non-living substances e.t.c.
They are simply xarray DataArrays indexed by the particle identifier `pid`.