"""Classes for LADiM variables

Defines two kinds of LADiM variables

- InstanceVariables:
    Depend on the particle and time, for instance position, temperature, ...
- ParticleVariables:
    Do not depend on time, for instance release position, size of a 'dead' particle, ...

Note that pid, the particle identifier is an InstanceVariable, projecting from a
particle instance to the particle itself.

"""

import datetime
from typing import Any, Union, Optional
import numpy as np
import xarray as xr  

Timetype = Union[str, np.datetime64, datetime.datetime]
Array = Union[np.ndarray, xr.DataArray]
Variable = Union["InstanceVariable", "ParticleVariable"]

class InstanceVariable:
    """Time dependent LADiM variable

    An InstanceVariable is a variable from LADiM depending on the particle instance
    i.e. the particle and time. Examples are position variables, environmental
    conditions like temperature or biological variables like size, weight, age.
    Note that the particle identifier, pid, is an InstanceVariable.

    The implementation tries to emulate a xarray DataArray.

    """

    def __init__(
        self,
        data: xr.DataArray,
        pid: xr.DataArray,
        time: xr.DataArray,
        count: np.ndarray,
    ) -> None:
        self.da = data
        self.pid = pid
        self.time = time
        self.count = count
        self.end = self.count.cumsum()
        self.start = self.end - self.count
        self.num_times = len(self.time)
        self.particles = np.unique(self.pid)
        self.num_particles = len(self.particles)  # Number of distinct particles

    @property
    def values(self) -> np.ndarray:
        # Same as self.da.values
        return np.array(self.da)

    def _sel_time_index(self, n: int) -> xr.DataArray:
        """Select by time index, return xarray."""
        start = self.start[n]
        end = self.end[n]
        V = self.da[start:end]
        V = V.assign_coords(time=self.time[n])
        V = V.assign_coords(pid=self.pid[start:end])
        V = V.swap_dims({"particle_instance": "pid"})
        return V

    def _sel_time_slice_index(self, tslice: slice) -> "InstanceVariable":
        """Take a time slice based on time indices"""
        n = self.num_times
        istart, istop, step = tslice.indices(n)
        if step != 1:
            raise IndexError("step > 1 is not allowed")
        start = self.start[istart]
        end = self.end[istop - 1]
        return InstanceVariable(
            data=self.da[start:end],
            pid=self.pid[start:end],
            time=self.time[tslice],
            count=self.count[tslice],
        )

    def _sel_time_value(self, time_val: Timetype) -> xr.DataArray:
        idx = self.time.get_index("time").get_loc(time_val)
        return self._sel_time_index(idx)

    def _sel_pid_value2(self, pid: int) -> xr.DataArray:
        """Selection based on single pid value"""
        # Make it 100 times faster using the pf.pid.da.values
        # self.da.values[pf.pid.da.values==pid]
        # Problem: get the times. This is missing a time step
        data = self.da.values[self.pid.da.values==pid]
        I, = np.nonzero(pf.X.pid.values==10000);
        t0, t1 = np.searchsorted(self.start, [I[0], I[-1]], side='right')
        # times = self.time[t0:t1], try t0-1:t0+1
        V = xr.DataArray(data, coords={"time": self.time[t0:t1]}, dims=("time",))
        V["pid"] = pid
        return V

    def _sel_pid_value(self, pid: int) -> xr.DataArray:
        """Selection based on single pid value"""
        data = []
        times = []
        for t_idx in range(self.num_times):
            try:
                data.append(self._sel_time_index(t_idx).sel(pid=pid))
                times.append(t_idx)
            except KeyError:
                pass
        if not data:
            raise KeyError(f"No such pid = {pid}")
        V = xr.DataArray(data, coords={"time": self.time[times]}, dims=("time",))
        V["pid"] = pid
        return V

    # def isel(self, *, time: Optional[int] = None) -> xr.DataArray:
    #     if time is not None:
    #         return self._sel_time_index(time)
    #     else:
    #         raise ValueError("Need one argument")
    def isel(self, *, time: int) -> xr.DataArray:
        return self._sel_time_index(time)

    def sel(
        self, *, pid: Optional[int] = None, time: Optional[Timetype] = None
    ) -> xr.DataArray:
        """Select from InstanceVariable by value of pid or time or both"""
        if pid is not None and time is None:
            return self._sel_pid_value(pid)
        if time is not None and pid is None:
            return self._sel_time_value(time)
        if time is not None and pid is not None:
            return self._sel_time_value(time).sel(pid=pid)
        # No arguments
        raise ValueError("Need 1 or 2 arguments")

    # TODO: Do something like dask if the array gets to big
    def to_dense(self) -> xr.DataArray:
        """Return a full (dense) DataArray"""
        data = np.empty((self.num_times, self.num_particles))
        data[:, :] = np.nan
        for n in range(self.num_times):
            data[n, self.pid[self.start[n] : self.end[n]]] = self._sel_time_index(n)
        # coords = dict(time=self.time, pid=self.particles)
        coords = [("time", self.time.values), ("pid", self.particles)]
        V = xr.DataArray(data=data, coords=coords, dims=("time", "pid"))
        return V

    def full(self) -> xr.DataArray:
        """Deprecated, use to_dense instead"""
        return self.to_dense()

    # More complicated typing
    # def __getitem__(self, index: Union[int, slice]) -> xr.DataArray:
    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[xr.DataArray, "InstanceVariable"]:
        if isinstance(index, int):  # index = time_idx
            return self._sel_time_index(index)
        if isinstance(index, slice):
            return self._sel_time_slice_index(index)
        else:  # index = time_idx, pid
            time_idx, pid = index
            if 0 <= pid < self.num_particles:
                try:
                    v = self._sel_time_index(time_idx).sel(pid=pid)
                except KeyError:
                    # This only works for float types
                    v = np.nan
            else:
                raise IndexError(f"pid={pid} is out of bound={self.num_particles}")
            return v

    def __array__(self) -> np.ndarray:
        return np.array(self.da)

    def __repr__(self) -> str:
        s = "<postladim.InstanceVariable>\n"
        s += f"num_times: {self.num_times}, particle_instance: {len(self.da)}\n"
        s += arraystr(self.da)
        return s

    def __len__(self) -> int:
        return len(self.time)


# --------------------------------------------

# Need this?, just use the DataArray
class ParticleVariable:
    """Particle variable, time-independent"""

    # def __init__(self, particlefile: "ParticleFile", varname: str) -> None:
    def __init__(self, data: xr.DataArray) -> None:
        self.da = data

    def __getitem__(self, p: int) -> Any:
        """Get the value of particle with pid = p"""
        return self.da[p]

    def __array__(self) -> np.ndarray:
        return np.array(self.da)

    def __repr__(self) -> str:
        s = "<postladim.ParticleVariable>\n"
        s += f"particle: {len(self.da)}\n"
        s += arraystr(self.da)
        return s

    def __len__(self) -> int:
        return len(self.da)


def itemstr(v: Array) -> str:
    """Pretty print array item"""

    # Date
    if str(v.dtype).startswith("datetime64"):
        return str(v.__array__()).rstrip("0.:T")

    # Number
    return f"{v:g}"


def arraystr(A: Array) -> str:
    """Pretty print array"""
    B = np.asarray(A).ravel()
    if len(B) <= 3:
        return " ".join([itemstr(v) for v in B])
    return " ".join([itemstr(B[0]), itemstr(B[1]), "...", itemstr(B[-1])])
