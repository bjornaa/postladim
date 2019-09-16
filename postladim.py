from collections import namedtuple
import datetime
from typing import Any, List, Dict, Union, Optional
import numpy as np     # type: ignore
import xarray as xr    # type: ignore

Timetype = Union[str, np.datetime64, datetime.datetime]


class InstanceVariable:
    def __init__(
        self,
        data: xr.DataArray,
        pid: xr.DataArray,
        ptime: xr.DataArray,
        pcount: np.ndarray,
    ) -> None:
        self.da = data
        self.pid = pid
        self.time = ptime
        self.count = pcount
        self.end = self.count.cumsum()
        self.start = self.end - self.count
        self.num_times = len(self.time)
        self.particles = np.unique(self.pid)
        self.num_particles = len(self.particles)  # Number of distinct particles

    def _sel_time_index(self, n: int) -> xr.DataArray:
        """Select by time index, return xarray."""
        start = self.start[n]
        end = self.end[n]
        V = self.da[start:end]
        V = V.assign_coords(time=self.time[n])
        V = V.assign_coords(pid=self.pid[start:end])
        V = V.swap_dims({"particle_instance": "pid"})
        return V

    # # def _sel_time_idx2(self, n):
    #     # Glemmer strukturen og bygger opp på ny
    #     start = int(self.pf.start[n])
    #     end = int(self.pf.end[n])
    #     coords = {"time": self.pf.ds.time[n], "pid": self.pf.ds.pid[start:end].values}
    #     dims = ("pid",)
    #     return xr.DataArray(self.da[start:end].values, dims=dims, coords=coords)

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
            ptime=self.time[tslice],
            pcount=self.count[tslice],
        )

    def _sel_time_value(self, time_val: Timetype) -> xr.DataArray:
        idx = self.time.get_index("time").get_loc(time_val)
        return self._sel_time_index(idx)

    def _sel_pid_value(self, pid: int) -> xr.DataArray:
        """Selection based on single pid value"""
        # Burde få en pid-koordinat også
        data = []
        times = []
        for t_idx in range(self.num_times):
            try:
                data.append(self._sel_time_index(t_idx).sel(pid=pid))
                times.append(t_idx)
            except KeyError:
                pass
        # Bedre, på forhånd sjekk om pid > maximum
        if not data:
            raise KeyError(f"No such pid = {pid}")
        V = xr.DataArray(data, coords={"time": self.time[times]}, dims=("time",))
        V["pid"] = pid
        return V

    def isel(self, *, time: Optional[int] = None) -> xr.DataArray:
        if time is not None:
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

    # Do something like dask if the array gets to big
    def full(self) -> xr.DataArray:
        """Return a full DataArray"""
        data = np.empty((self.num_times, self.num_particles))
        data[:, :] = np.nan
        for n in range(self.num_times):
            data[n, self.pid[self.start[n] : self.end[n]]] = self._sel_time_index(n)
        coords = dict(time=self.time, pid=self.particles)
        V = xr.DataArray(data=data, coords=coords, dims=("time", "pid"))
        return V

    # More complicated typing
    def __getitem__(self, index: Union[int, slice]) -> xr.DataArray:
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
                    # Også håndtere v != floatpf.
                    v = np.nan
            else:
                raise IndexError(f"pid={pid} is out of bound={self.num_particles}")
            return v

    def __len__(self) -> int:
        return len(self.time)


# --------------------------------------------


class ParticleVariable:
    """Particle variable, time-independent"""

    # def __init__(self, particlefile: "ParticleFile", varname: str) -> None:
    def __init__(self, data: xr.DataArray) -> None:
        self.da = data

    def __getitem__(self, p: int) -> Any:
        """Get the value of particle with pid = p
        """
        return self.da[p]


# --------------------------------------------


Position = namedtuple("Position", "X Y")


class Trajectory(namedtuple("Trajectory", "X Y")):
    """Single particle trajectory"""

    @property
    def time(self) -> np.datetime64:
        return self.X.time

    def __len__(self) -> int:
        return len(self.X.time)


# ---------------------------------------------


class Time:
    """Callable version of time DataArray

    For backwards compability, obsolete
    """

    def __init__(self, ptime):
        self._time = ptime

    def __call__(self, n: int) -> np.datetime64:
        """Prettier version of self[n]"""
        return self._time[n].values.astype("M8[s]")

    def __getitem__(self, arg):
        return self._time[arg]

    def __repr__(self) -> str:
        return repr(self._time)

    def __str__(self) -> str:
        return repr(self._time)

    def __len__(self) -> int:
        return len(self._time)


# --------------------------------------


class ParticleFile:
    def __init__(self, filename: str) -> None:
        ds = xr.open_dataset(filename)
        self.ds = ds
        # End and start of segment with particles at a given time
        self.count = ds.particle_count.values
        self.end = self.count.cumsum()
        self.start = self.end - self.count
        self.num_times = len(self.count)
        self.time = Time(ds.time)
        self.num_particles = int(ds.pid.max()) + 1  # Number of particles

        # Extract instance and particle variables from the netCDF file
        self.instance_variables: List["InstanceVariable"] = []
        self.particle_variables: List["ParticleVariable"] = []
        self.variables: Dict[str, Union["InstanceVariable", "ParticleVariable"]] = {}
        for var in list(self.ds.variables):
            if "particle_instance" in self.ds[var].dims:
                self.instance_variables.append(var)
                self.variables[var] = InstanceVariable(
                    self.ds[var], self.ds.pid, self.ds.time, self.count
                )
            elif "particle" in self.ds[var].dims:
                self.particle_variables.append(var)
                self.variables[var] = ParticleVariable(self.ds[var])

    # For backwards compability
    # should it be a DataSet
    def position(self, n: int) -> Position:
        return Position(self.X[n], self.Y[n])

    # For backwards compability
    # Could define ParticleDataset (from file)
    # This could slice and take trajectories og that
    # Could improve speed by computing X and Y at same time
    def trajectory(self, pid: int) -> Trajectory:
        X = self["X"].sel(pid=pid)
        Y = self["Y"].sel(pid=pid)
        return Trajectory(X, Y)

    # Obsolete
    def particle_count(self, n: int) -> int:
        return self.count[n]

    def __len__(self) -> int:
        return len(self.time)

    def __getattr__(self, var: str) -> Union[InstanceVariable, ParticleVariable]:
        return self.variables[var]

    def __getitem__(self, var: str) -> Union[InstanceVariable, ParticleVariable]:
        return self.variables[var]

    def close(self) -> None:
        self.ds.close()

    # Make ParticleFile a context manager
    def __enter__(self):
        return self

    def __exit__(self, atype, value, traceback):
        self.close()
