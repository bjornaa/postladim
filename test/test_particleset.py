"""Check the ParticleSet class"""


import numpy as np
import pytest
import xarray as xr
from postladim import ParticleSet


@pytest.fixture(scope="module")
def particle_set():
    """Define a small ParticleSet"""

    #  pid:         X:
    #  0  -  -      0   -   -
    #  0  1  -      1  11   -
    #  0  -  2      2   -  22
    #  -  -  3      -   -  23
    #

    pid = np.array([[0, -999, -999], [0, 1, -999], [0, -999, 2], [-999, -999, 2]])
    X = np.array([[0, -999, -999], [1, 11, -999], [2, -999, 22], [-999, -999, 23]])
    Y = np.array([[2, -999, -999], [3, 8, -999], [4, -999, 9], [-999, -999, 10]])
    num_times, num_particles = pid.shape

    time = 3600 * np.arange(num_times)  # hourly timesteps
    count = np.sum(pid >= 0, axis=1)

    pid1 = [v for v in pid.flat if v >= 0]  # Flattened pid

    ds = xr.Dataset(
        data_vars=dict(
            particle_count=("time", count),
            start_time=(
                "particle",
                np.datetime64("2022-01-01", "ns")
                + [np.timedelta64(v, "h") for v in range(num_particles)],
            ),
            location_id=("particle", [10001, 10002, 10003]),
            pid=("particle_instance", pid1),
            X=("particle_instance", [v for v in X.flat if v >= 0]),
            Y=("particle_instance", [v for v in Y.flat if v >= 0]),
        ),
        coords=dict(
            time=np.datetime64("2022-01-01", "ns")
            + [np.timedelta64(v, "s") for v in time],
            particle=np.unique(pid1),
        ),
    )

    yield ParticleSet(ds)

    # tear down
    ds.close()


def test_numbers(particle_set):
    """Check the dimensions of the particle set"""
    ps = particle_set
    # assert len(ps) == 4
    assert ps.num_times == 4
    assert ps.num_particles == 3
    assert ps.num_instances == 6
    assert all(ps.count == [1, 2, 2, 1])
    assert all(ps.start == [0, 1, 3, 5])
    assert all(ps.end == [1, 3, 5, 6])


def test_time(particle_set):
    """Check time representation"""
    ps = particle_set
    assert ps.time[2] == np.datetime64("2022-01-01 02", "ns")
    assert ps.time[2].values == np.datetime64("2022-01-01 02", "ns")
    assert ps.time[3] == np.datetime64("2022-01-01 03", "ns")


def test_ftime(particle_set):
    """Check simplified time representation"""
    ps = particle_set
    assert ps.ftime(3) == "2022-01-01T03:00:00"
    assert ps.ftime(3) == str(ps.time[3].values.astype("M8[s]"))
    assert ps.ftime(3, "h") == "2022-01-01T03"


def test_variable_type(particle_set):
    """Check variable classification"""
    ps = particle_set
    assert set(ps.instance_variables) == {"pid", "X", "Y"}
    assert set(ps.particle_variables) == {"particle", "start_time", "location_id"}


def test_pid(particle_set):
    """Check the pid"""
    ps = particle_set
    assert ps.pid.isel(time=0) == 0
    assert ps["pid"][0] == 0
    assert ps.pid[0] == 0
    assert all(ps.pid[1] == [0, 1])
    assert all(ps.pid[2] == [0, 2])
    assert ps.pid[3] == 2


def test_position(particle_set):
    """Check the position method"""
    ps = particle_set
    distr = ps.position(time=1)
    assert all(distr.X == ps.X[1])
    assert all(distr.Y == ps.Y[1])
    X, Y = ps.position(2)
    assert all(X == ps.X[2])
    assert all(Y == ps.Y[2])


def test_trajectory(particle_set):
    """Check the trajectory extraction"""
    ps = particle_set
    traj = ps.trajectory(pid=0)
    assert all(traj.X == [0, 1, 2])
    assert all(traj.Y == [2, 3, 4])
    X, Y = ps.trajectory(2)
    assert all(X == [22, 23])
    assert all(Y == [9, 10])


def test_getX(particle_set):
    """Test various ways to access a variable"""
    ps = particle_set
    X = ps["X"]
    assert ps.X == X
    assert ps.variables["X"] == X  # Obsolete netcdf-inspired notation


def test_particle_variable(particle_set):
    """Check the particle variables"""
    ps = particle_set
    assert ps.start_time[0] == np.datetime64("2022-01-01", "ns")
    assert ps["start_time"][1] == np.datetime64("2022-01-01 01", "ns")
    assert all(ps.location_id == np.array([10001, 10002, 10003]))
    assert all(ps.location_id == ps["location_id"][:])


def test_isel_time(particle_set):
    """Select a snapshot"""
    ps = particle_set
    ps2 = ps.isel(time=2)
    # Dimensions
    assert ps2.num_times == 1
    assert ps2.num_instances == 2
    assert ps2.num_particles == 2
    # Time coordinate
    assert ps2.time == ps.time[2]
    assert ps2.count == ps.count[2]
    # Instance variables
    assert all(ps2.pid == np.array([0, 2]))
    assert all(ps2.X == np.array([2, 22]))
    assert all(ps2.X == ps.X.isel(time=2))
    # Particle variables
    assert all(ps2.location_id == np.array([10001, 10003]))


def test_isel_time_slice(particle_set):
    """Select a time slice"""
    ps = particle_set
    I = slice(0, 2)
    ps2 = ps.isel(time=I)
    # Dimensions
    assert ps2.num_times == 2
    assert ps2.num_instances == 3
    assert ps2.num_particles == 2
    # Time
    assert all(ps2.time == ps.time.isel(time=I))
    # Instance variables
    assert all(ps2.pid == np.array([0, 0, 1]))
    # assert all(ps2.X == ps.X.isel(time=I))
    assert all(ps2.X == np.array([0, 1, 11]))
    # Particle variables
    assert all(ps2.location_id == np.array([10001, 10002]))


def test_isel_time_slice_wrong(particle_set):
    ps = particle_set
    I = slice(0, 3, 2)
    with pytest.raises(IndexError):
        ps.isel(time=I)


def test_sel_time(particle_set):
    ps = particle_set
    time = "2022-01-01 02"
    ps2 = ps.sel(time=time)
    # Dimensions
    assert ps2.num_times == 1
    assert ps2.num_instances == 2
    assert ps2.num_particles == 2
    # Time coordinate
    assert ps2.time == np.datetime64(time, "ns")
    assert ps2.time == ps.time.sel(time=time)
    # Instance variables
    assert all(ps2.pid == ps.pid.sel(time=time))
    assert all(ps2.X == ps.X.sel(time=time))
    # Particle variables
    assert all(ps2.location_id == np.array([10001, 10003]))
    assert all(ps2.particle == np.array([0, 2]))


def test_sel_pid(particle_set):
    ps = particle_set
    pid = 2
    ps2 = ps.sel(pid=pid)
    # Dimensions
    assert ps2.num_times == 2
    assert ps2.num_instances == 2
    assert ps2.num_particles == 1
    # Time
    assert ps2.time[0] == np.datetime64("2022-01-01 02", "ns")
    assert all(ps2.count == np.array([1, 1]))
    # Instance variables
    assert all(ps2.pid == ps.pid.sel(pid=pid))
    assert all(ps2.X == ps.X.sel(pid=pid))
    # Particle variables
    assert ps2.particle == np.array(pid)
    assert ps2.particle[0] == pid
    assert ps2.location_id == np.array([10003])
    assert ps2.location_id == np.array(ps.location_id[pid])
    assert ps2.location_id[0] == ps.location_id[pid]


def test_sel_both(particle_set):
    ps = particle_set
    time = "2022-01-01 02"
    pid = 2
    ps2 = ps.sel(time=time, pid=pid)
    assert ps2.num_times == 1
    assert ps2.num_instances == 1
    assert ps2.num_particles == 1
    assert ps2.particle[0] == pid
    assert ps2.pid[0] == pid
    # The below does not work, right side is a DataArray,
    # assert all(ps2.X == ps.X.sel(time=time, pid=pid))
    assert ps2.X.da == ps.X.sel(time=time, pid=pid)
    assert all(ps2.X == np.array([22]))
    # Opposite order
    ps3 = ps.sel(pid=pid).sel(time=time)
    assert ps2 == ps3


def test_item_isel(particle_set):
    """Testing item notation for time selection"""
    ps = particle_set
    # The below does not work, why?
    assert ps[2] == ps.isel(time=2)
    assert ps[:2] == ps.isel(time=slice(0, 2))


def test_particlefile(particle_set):
    """Test writing and reading NetCDF"""
    ps = particle_set
    ps.to_netcdf("test.nc")
    # pf = ParticleFile("test.nc")
    # os.unlink("test.nc")
    # assert pf == ps
