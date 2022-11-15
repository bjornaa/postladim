Sparse LADiM data format
------------------------

The `sparse` output layout may also be called `ragged array representation`.
This is the default output format formatted LADiM. For particle tracking, the
CF-standard defines a format for "Ragged array representation of trajectories".
This is not suitable for our purpose since we are more interested in the
geographical distribution of particles at a given time than the individual
trajectories. Chris Baker at NOAA has an interesting discussion on this topic
and a `suggestion <https://github.com/NOAA-ORR-ERD/nc_particles/blob/master/
nc_particle_standard.md>`_. The LADiM `sparse` format is closely related to
this suggestion.

The dimensions are ``time``, ``particle`` and
``particle_instance``. The particle dimension indexes the individual particles,
while the particle_instance indexes the `instances` i.e. a particle at a given
time. 

Particle count
.............. 

The variable ``particle_count`` with dimension time gives the number of
particles at each time step and is used to address the data values.
The start of each segment along the ``particle_instance`` dimension
is found by the cumulative sum of the particle counts.

.. code-block:: python

    start = np.cumsum(particle_count)

With python indexing (zero-based), the particles at the ``n``-th time step are found from
index ``start[n]`` to ``start[n] + particle_count[n]``.

Particle identifier
...................

The particle identifier, ``pid`` should always be present in a sparse particle
file. It is an integer that identifies the particle. If a particle is released
later than another, it has a higher ``pid`` value. The value is not reused if a
particle is terminated. The sparse data format is optimized for extracting the
particle distribution at a given time, however using the ``pid`` it is possible
(but less efficient) to extract the individual trajectories of a particle.
