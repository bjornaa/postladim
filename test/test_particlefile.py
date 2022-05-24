#from inspect import Attribute
import os
import pytest

import numpy as np
from netCDF4 import Dataset
from postladim import ParticleFile


@pytest.fixture(scope="module")
def particle_file():
    """Create a small particle file"""
    #
    #  0   -   -
    #  1  11   -
    #  2   -  22
    #  -   -  23
    #
    pfile = "test_tmp.nc"
    X = np.array(
        [[0, np.nan, np.nan], [1, 11, np.nan], [2, np.nan, 22], [np.nan, np.nan, 23]]
    )
    Y = np.array(
        [[2, np.nan, np.nan], [3, 8, np.nan], [4, np.nan, 9], [np.nan, np.nan, 10]]
    )
    num_times, num_particles = X.shape
    pid = np.multiply.outer(num_times * [1], np.arange(num_particles))
    pid[np.isnan(X)] = -999  # Use a negative integer for undefined
    time = 3600 * np.arange(num_times)  # hourly timesteps
    count = (np.ones(np.shape(X)) - np.isnan(X)).sum(axis=1)
    with Dataset(pfile, mode="w") as nc:
        # Dimensions
        nc.createDimension("particle", num_particles)
        nc.createDimension("particle_instance", None)
        nc.createDimension("time", num_times)
        # Variables
        v = nc.createVariable("time", "f8", ("time",))
        v.units = "seconds since 2022-01-01 00:00:00"
        v = nc.createVariable("particle_count", "i", ("time",))
        v = nc.createVariable("start_time", "f8", ("particle",))
        v.units = "seconds since 2022-01-01 00:00:00"
        v = nc.createVariable("location_id", "i", ("particle",))
        v = nc.createVariable("pid", "i", ("particle_instance",))
        v = nc.createVariable("X", "f4", ("particle_instance",))
        v = nc.createVariable("Y", "f4", ("particle_instance",))
        # Data
        nc.variables["time"][:] = time
        nc.variables["particle_count"][:] = count
        nc.variables["start_time"][:] = time[:num_particles]
        nc.variables["location_id"][:] = [10000, 10001, 10002]
        nc.variables["pid"][:] = [v for v in pid.flat if v >= 0]
        nc.variables["X"][:] = [v for v in X.flat if not np.isnan(v)]
        nc.variables["Y"][:] = [v for v in Y.flat if not np.isnan(v)]

    yield pfile

    # tear down
    os.remove(pfile)


@pytest.fixture()
def non_particle():
    """Make an empty netcdf file"""
    pfile = "test_non_particle_tmp.nc"

    with Dataset(pfile, mode="w") as nc:
        nc.createDimension('lon', 4)
        v = nc.createVariable('lon', 'f4', ('lon',))
        v[:] = [1, 2, 3, 4]

    yield pfile

    os.remove(pfile)


def test_open_fail(non_particle):
    # non-existing file
    with pytest.raises(FileNotFoundError):
        ParticleFile("no_such_file.nc")
    # file is not a netcdf file
    with pytest.raises(ValueError):
        ParticleFile("test_particlefile.py")
    # Netcdf file that is not a particlefile
    with pytest.raises(SystemExit):
        ParticleFile(non_particle)
        # ParticleFile("/home/bjorn/ladim2/examples/data/ocean_avg_0014.nc")


def test_numbers(particle_file):
    """Alignment of time frames in the particle file."""
    with ParticleFile(particle_file) as pf:
        assert pf.num_times == 4
        assert all(pf.start == [0, 1, 3, 5])
        assert list(pf.count) == [1, 2, 2, 1]
        assert list(pf.end) == [1, 3, 5, 6]
        assert len(pf) == 4
        assert pf.num_particles == 3


def test_time(particle_file):
    """Time handled correctly"""
    with ParticleFile(particle_file) as pf:
        assert pf.time[2] == np.datetime64("2022-01-01 02")
        assert pf.time[3] == np.datetime64("2022-01-01 03")


def test_variable_type(particle_file):
    """Check that the variables belong to correct type"""
    with ParticleFile(particle_file) as pf:
        assert pf.instance_variables == ["pid", "X", "Y"]
        assert pf.particle_variables == ["start_time", "location_id"]


def test_pid(particle_file):
    """The pid is correct"""
    with ParticleFile(particle_file) as pf:
        assert pf.pid.isel(time=0) == 0
        assert pf["pid"][0] == 0
        assert pf.pid[0] == 0
        assert all(pf.pid[1] == [0, 1])
        assert list(pf.pid[2]) == [0, 2]
        assert pf.pid[3] == 2


def test_pid2(particle_file):
    """The pid from an instance variable"""
    with ParticleFile(particle_file) as pf:
        X = pf.X
        assert all(X.pid == pf.ds.pid)
        for i in range(4):
            assert all(X[i].pid == pf.pid[i])
        # X.pid[3] is not the same as X[3].pid
        assert not all(X.pid[3] == X[3].pid)


def test_position(particle_file):
    with ParticleFile(particle_file) as pf:
        X, Y = pf.position(time=1)
        assert all(X == pf.X[1])
        assert all(Y == pf.Y[1])
        X, Y = pf.position(2)
        assert all(X == pf.X[2])
        assert all(Y == pf.Y[2])


def test_getX(particle_file):
    with ParticleFile(particle_file) as pf:
        X = pf["X"]
        assert pf.X == X
        assert pf.variables["X"] == X # Obsolete netcdf-inspired notation
        assert X.isel(time=0) == 0
        assert X[0] == 0
        assert all(X[1] == [1, 11])
        assert all(X[2] == [2, 22])
        assert X[3] == 23


def test_trajectory(particle_file):
    with ParticleFile(particle_file) as pf:
        X, Y = pf.trajectory(2)
        assert all(X == [22, 23])
        assert all(Y == [9, 10])
        traj = pf.trajectory(0)
        assert len(traj) == 3
        assert all(traj.time == pf.time[:-1])
        assert all(traj.X == pf.X.sel(pid=0))
        assert all(traj.Y == pf.Y.sel(pid=0))


def test_particle_variable(particle_file):
    """Two particle variables, start_time and location_id"""
    with ParticleFile(particle_file) as pf:
        assert pf.start_time[0] == np.datetime64("2022-01-01")
        assert pf["start_time"][1] == np.datetime64("2022-01-01 01")
        assert all(pf.location_id == np.array([10000, 10001, 10002]))
        assert all(pf.location_id == pf["location_id"][:])
