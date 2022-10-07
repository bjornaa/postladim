"""Main classes for postprocessing LADiM output

This module contains classes:
  - ParticleFile
  - Trajectory
  - Time (obsolete)

"""


from collections import namedtuple
import datetime
from types import TracebackType
from typing import List, Dict, Union, Optional, Literal, Tuple

import numpy as np
import xarray as xr

from .variable import InstanceVariable, ParticleVariable, Variable, arraystr

Timetype = Union[str, np.datetime64, datetime.datetime]
Array = Union[np.ndarray, xr.DataArray]

# --------------------------------------------

Position = namedtuple("Position", "X Y")
Trajectory = namedtuple("Trajectory", "X Y")


"""
    num_times : integer
        Number of time steps
    num_particles: integer
        Total number of particles involved
    count : 1D integer array, length = num_times
        Number of particles per time step
    start: 1D integer array, length = num_times
        Start index for particles per time step
    end: 1D integer array, length = num_times
        End index for particles per tile step
    time: 1D xarray DataArray, length = num_times
        Date and time of time step
    instance_variables:
        List of instance variables
    particle_variables:
        List of particle variables
    variables:
        Combined list of variables

"""


class ParticleSet:

    # ds must have particle_count, time, pid

    def __init__(self, ds: xr.Dataset):

        if "particle" in ds.dims:
            self.ds = ds
        else:
            self.ds = ds.expand_dims(particle=np.unique(ds.pid).size)

        self.count = np.atleast_1d(ds.particle_count.values)
        self.end = self.count.cumsum()
        self.start = self.end - self.count
        self.num_times = len(self.count)
        self.num_instances = self.count.sum()

        self.time = ds.time
        if self.time.shape == ():
            self.time = self.time.expand_dims("time")
        self.num_particles = self.ds.particle.size

        # Extract instance and particle variables from the netCDF file
        self.instance_variables: List[str] = []
        self.particle_variables: List[str] = []
        self.variables: Dict[str, Variable] = {}
        for var in ds.variables:
            if "particle_instance" in ds[var].dims:
                self.instance_variables.append(var)
                self.variables[var] = InstanceVariable(
                    ds[var], ds.pid, ds.time, self.count
                )
            elif "particle" in ds[var].dims:
                self.particle_variables.append(var)
                self.variables[var] = ParticleVariable(ds[var])

    def __getattr__(self, var: str) -> Variable:
        return self.variables[var]

    # Missing: Add global attributes
    def __repr__(self):
        s = "<postladim.ParticleSet>\n"
        s += f"num_times: {self.num_times}, "
        s += f"num_particles: {self.num_particles}, "
        s += f"num_instances: {self.num_instances}\n"
        s += f"time: {arraystr(self.time)}\n"
        s += f"count: {arraystr(self.count)}\n"
        s += "Instance variables:\n"
        for var in self.instance_variables:
            s += f"  {var:16s} {arraystr(self[var].da)}\n"
        s += "Particle variables:\n"
        for var in self.particle_variables:
            s += f"  {var:16s} {arraystr(self[var].da)}\n"
        s += "Attributes:\n"
        for a, v in self.ds.attrs.items():
            s += f"  {a:16s} {v}\n"
        return s

    def ftime(self, n: int, unit: Literal["s", "m", "h"] = "s") -> str:
        """Return a nicely formated version of xarray time"""
        return str(self.time[n].values.astype(f"M8[{unit}]"))

    def position(
        self, time: int, system: Optional[Literal["xy", "lonlat"]] = None
    ) -> Position:
        """Extract the positions of all particles at given time step"""
        if system is None and "X" in self.instance_variables:
            system = "xy"
        if system == "xy":
            return Position(self.X[time], self.Y[time])
        return Position(self.lon[time], self.lat[time])

    def trajectory(
        self, pid: int, system: Optional[Literal["xy", "lonlat"]] = None
    ) -> Trajectory:
        if system is None and "X" in self.instance_variables:
            system = "xy"
        if system == "xy":
            return Trajectory(self.X.sel(pid=pid), self.Y.sel(pid=pid))
        return Trajectory(self.lon.sel(pid=pid), self.lat.sel(pid=pid))

    def _sel_time_index(self, index: Union[int, slice]) -> "ParticleSet":
        if isinstance(index, int):
            start = self.start[index]
            end = self.end[index]
            ds = self.ds.isel(time=[index], particle_instance=slice(start, end))
        elif isinstance(index, slice):
            istart, istop, step = index.indices(self.num_times)
            if step != 1:
                raise IndexError("step > 1 is not allowed")
            start = self.start[istart]
            end = self.end[istop - 1]
            ds = self.ds.isel(time=index, particle_instance=slice(start, end))
        ds = ds.sel(particle=np.unique(ds.pid))
        return ParticleSet(ds)

    def _sel_time_value(self, time_val: Timetype) -> "ParticleSet":
        try:
            idx = self.time.get_index("time").get_loc(time_val)
        except KeyError as e:
            raise KeyError(f"No data for time = {time_val}") from e
        return self._sel_time_index(idx)

    def _sel_pid_value(self, pid: int) -> "ParticleSet":
        """Selection based on single pid value"""
        (I,) = np.nonzero(self.pid.values == pid)
        if len(I) == 0:
            raise KeyError(f"No data for pid = {pid}")
        # Limit actual time
        t0, t1 = np.searchsorted(self.start, [I[0], I[-1]], side="right")
        t0 -= 1
        ds = self.ds.isel(time=slice(t0, t1), particle_instance=I).sel(particle=[pid])
        ds["particle_count"] = xr.DataArray(np.ones(t1 - t0, dtype=int), dims=["time"])
        return ParticleSet(ds)

    def isel(self, *, time: int) -> "ParticleSet":
        """Selection by time step number (time index)"""
        return self._sel_time_index(time)

    def sel(
        self, *, pid: Optional[int] = None, time: Optional[Timetype] = None
    ) -> "ParticleSet":
        """Select by value of pid or time or both"""
        if pid is not None and time is None:
            return self._sel_pid_value(pid)
        if time is not None and pid is None:
            return self._sel_time_value(time)
        if time is not None and pid is not None:
            return self._sel_time_value(time).sel(pid=pid)
        # Neither time or pid
        raise TypeError("Need time or pid argument")

    def __getitem__(
        self, index: Union[str, int, slice]
    ) -> Union["Variable", "ParticleSet"]:
        if isinstance(index, str):
            return self.variables[index]
        if isinstance(index, int) or isinstance(index, slice):
            return self._sel_time_index(index)
        raise TypeError

    def to_netcdf(self, path: str, **args) -> None:
        self.ds.to_netcdf(path, **args)

    def close(self):
        self.ds.close()

    def __eq__(self, other: "ParticleSet") -> bool:
        return self.ds.identical(other.ds)


class ParticleFile(ParticleSet):
    """Convenience class, for backwards compatibility"""

    def __init__(self, path: str):
        ds = xr.load_dataset(path)
        super().__init__(ds)

    # Make ParticleFile a context manager
    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Union[type, None],
        exc_val: Union[BaseException, None],
        exc_tb: Union[TracebackType, None],
    ) -> None:
        self.close()
