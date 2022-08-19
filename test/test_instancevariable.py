"""Test suite for the InstanceVariable class in postladim"""

# Bjørn Ådlandsvik <bjorn@hi.no>
# Institute of Marine Research

import numpy as np
import xarray as xr
import pytest

from postladim.variable import InstanceVariable


@pytest.fixture(scope="module")
def X():
    """Create a small InstanceVariable,

    4 times, 3 particles, 6 particle instances

      0   -   -
      1  11   -
      2   -  22
      -   -  23

    """
    data = np.array([0, 1, 11, 2, 22, 23])
    count = np.array([1, 2, 2, 1])
    pid = np.array([0, 0, 1, 0, 2, 2])
    # Hourly time values
    time0 = np.datetime64("2022-05-16")
    time = time0 + np.timedelta64(1, "h") * range(4)

    # Make xarray DataArrays
    data = xr.DataArray(data, dims=("particle_instance"))
    pid = xr.DataArray(pid, dims=("particle_instance"))
    time = xr.DataArray(time, coords=[("time", time)])
    return InstanceVariable(data=data, pid=pid, time=time, count=count)


def test_creation(X):
    """Check that the instance variable is created"""
    assert type(X) == InstanceVariable


def test_data(X):
    """Check that the data are represented correctly"""
    assert X[0] == 0
    assert all(X[1] == [1, 11])
    assert all(X[2] == [2, 22])
    assert X[3] == 23
    assert all(X.values == [0, 1, 11, 2, 22, 23])


def test_pid(X):
    """Check the particle identifier"""
    assert X[0].pid == 0
    assert all(X[1].pid == [0, 1])
    assert all(X[2].pid == [0, 2])
    assert X[3].pid == 2
    assert X.pid[3] == 0  # Note: pid and item does not commute
    assert all(X.pid == [0, 0, 1, 0, 2, 2])


def test_unique_pids(X):
    """Check that the particled are identified correctly"""
    assert all(X.particles == [0, 1, 2])
    assert all(np.unique(X.pid) == [0, 1, 2])


def test_time(X):
    """Check time representation"""
    assert X[0].time == np.datetime64("2022-05-16")
    assert X[1].time == np.datetime64("2022-05-16 01")
    assert X[2].time == np.datetime64("2022-05-16 02")
    assert X[3].time == np.datetime64("2022-05-16 03")
    assert X.time[2] == X[2].time  # Time and item commutes


def test_count(X):
    """Check particle count and related attributes"""
    assert all(X.count == [1, 2, 2, 1])
    assert all(X.start == [0, 1, 3, 5])
    assert all(X.end == [1, 3, 5, 6])


def test_num(X):
    """Check shape"""
    assert X.num_particles == 3
    assert X.num_times == 4
    assert len(X) == 4  # len == num_times
    assert len(X.da) == 6  # Number of particle instances


def test_time_select(X):
    """xarray-like data selection by time"""
    assert all(X.isel(time=2) == X[2])
    assert all(X.sel(time=np.datetime64("2022-05-16 02")) == X[2])
    assert all(X.sel(time="2022-05-16 02") == X[2])


def test_select_by_nonexisting_time(X):
    with pytest.raises(IndexError):
        X.isel(time=4)
    with pytest.raises(KeyError):
        X.sel(time="2020-02-02 02")


def test_item(X):
    """Test the item notation with single index"""
    assert X[0] == 0
    assert all(X[1] == [1, 11])
    assert all(X[2] == [2, 22])
    assert X[3] == 23


def test_time_slice(X):
    """Test item notation with slice"""
    V = X[1:3]
    assert type(V) == InstanceVariable
    assert len(V) == 2
    assert all(V[0] == X[1])
    assert all(V[1] == X[2])


def test_item2(X):
    """Test item notation with two variables"""
    assert X[1, 1] == 11
    assert X[2, 0] == 2
    assert np.isnan(X[2, 1])
    assert X[2, 2] == 22  # X[2][2] gives IndexError


def test_item_fail(X):
    with pytest.raises(IndexError):
        X[4]


def test_time_slice_fail(X):
    """step > 1 is not alllowed (yet?)"""
    with pytest.raises(IndexError):
        X[1::2]


def test_select_by_pid(X):
    """Select items by pid"""
    assert all(X.sel(pid=0) == [0, 1, 2])
    times = ["2022-05-16", "2022-05-16 01", "2022-05-16 02", "2022-05-16T03"]
    assert all(X.sel(pid=0).time == [np.datetime64(t) for t in times[:3]])
    assert X.sel(pid=1).time == np.datetime64("2022-05-16 01")
    assert all(X.sel(pid=2) == [22, 23])
    assert all(X.sel(pid=2).time == [np.datetime64(t) for t in times[2:]])


def test_dtype(X):
    assert np.issubdtype(X.dtype, np.number)
    assert X.dtype == np.int64


def test_select_by_nonexisting_pid(X):
    with pytest.raises(KeyError):
        X.sel(pid=3)


def test_to_dense(X):
    A = X.to_dense()
    expected = [
        [0, np.nan, np.nan],
        [1, 11, np.nan],
        [2, np.nan, 22],
        [np.nan, np.nan, 23],
    ]
    # all = fails because of NaNs, use np.testing instead
    np.testing.assert_array_equal(A, expected)
    assert type(A) == xr.DataArray
    assert set(A.dims) == {"time", "pid"}
