# import os
# import datetime

import numpy as np
import xarray as xr
import pytest

from postladim.variable import InstanceVariable

# Should not have to repeat this from test_particlefile.py
@pytest.fixture(scope="module")
def X():
    # set up a small InstanceVariable, 4 times, 3 particles, 6 particle instances
    #
    #  0   -   -
    #  1  11   -
    #  2   -  22
    #  -   -  23
    #
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


# --- InstanceVariable tests ---


def test_creation(X):
    assert type(X) == InstanceVariable


def test_data(X):
    assert X[0] == 0
    assert all(X[1] == [1, 11])
    assert all(X[2] == [2, 22])
    assert X[3] == 23


def test_pid(X):
    assert X[0].pid == 0
    assert all(X[1].pid == [0, 1])
    assert all(X[2].pid == [0, 2])
    assert X[3].pid == 2
    # Note: X.pid[n] != X[n].pid
    assert all(X.pid == [0, 0, 1, 0, 2, 2])


def test_time(X):
    assert X[0].time == np.datetime64("2022-05-16")
    assert X[1].time == np.datetime64("2022-05-16 01")
    assert X[2].time == np.datetime64("2022-05-16 02")
    assert X[3].time == np.datetime64("2022-05-16 03")
    # Time and item commutes
    assert X.time[2] == X[2].time


def test_count(X):
    assert all(X.count == [1, 2, 2, 1])


def test_values(X):
    assert all(X.values == [0, 1, 11, 2, 22, 23])


def test_start(X):
    assert all(X.start == [0, 1, 3, 5])


def test_end(X):
    assert all(X.end == [1, 3, 5, 6])


def test_num(X):
    assert X.num_particles == 3
    assert X.num_times == 4
    assert len(X) == 4  # len == num_times


def test_time_select(X):
    """xarray-like data selection by time"""
    assert all(X.isel(time=2) == X[2])
    assert all(X.sel(time=np.datetime64("2022-05-16 02")) == X[2])
    assert all(X.sel(time="2022-05-16 02") == X[2])


def test_time_slice(X):
    V = X[1:3]
    assert type(V) == InstanceVariable
    assert len(V) == 2
    assert all(V[0] == X[1])
    assert all(V[1] == X[2])


def test_time_slice_fail(X):
    """step > 1 is not alllowed (yet?)"""
    with pytest.raises(IndexError):
        X[1::2]


def test_select_by_pid(X):
    all(X.sel(pid=0) == [0, 1, 2])
    all(X.sel(pid=0).time == ["2022-05-1600", "2022-05-16 01", "2022-05-16 02"])
    all(X.sel(pid=1) == [11])
    all(X.sel(pid=1).time == ["2022-05-16T01"])
    all(X.sel(pid=2) == [22, 23])
    all(X.sel(pid=2).time == ["2022-05-16T02:00:00 2022-05-16T03:00:00"])


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
    assert set(A.dims) == {'time', 'pid'}
