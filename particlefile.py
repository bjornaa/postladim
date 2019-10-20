from collections import namedtuple
import datetime
from typing import Any, List, Dict, Union, Optional
import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from .variable import InstanceVariable, ParticleVariable, arraystr


Timetype = Union[str, np.datetime64, datetime.datetime]
Array = Union[np.ndarray, xr.DataArray]


# --------------------------------------------


Position = namedtuple("Position", "X Y")


class Trajectory:
    """Single particle trajectory"""

    def __init__(self, X: xr.DataArray, Y: xr.DataArray):
        self._data = X, Y

    # For unpacking: X, Y = pf.trajectory(pid=4)
    def __getitem__(self, n):
        return self._data[n]

    @property
    def X(self) -> xr.DataArray:
        return self._data[0]

    @property
    def Y(self) -> xr.DataArray:
        return self._data[1]

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
        self.da = ptime

    def __call__(self, n: int) -> np.datetime64:
        """Prettier version of self[n]"""
        return self.da[n].values.astype("M8[s]")

    def __getitem__(self, arg):
        return self.da[arg]

    def __repr__(self) -> str:
        return arraystr(self.da)

    def __len__(self) -> int:
        return len(self.da)


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

    # For convenience
    def position(self, time: int) -> Position:
        return Position(self.X[time], self.Y[time])

    # For backwards compability
    # Could define ParticleDataset (from file)
    # This could slice and take trajectories og that
    # Could improve speed by computing X and Y at same time
    def trajectory(self, pid: int) -> Trajectory:
        # mypy barks on these two
        # X = self["X"].sel(pid=pid)
        # Y = self["Y"].sel(pid=pid)
        X = InstanceVariable(self.ds["X"], self.ds.pid, self.ds.time, self.count).sel(
            pid=pid
        )
        Y = InstanceVariable(self.ds["Y"], self.ds.pid, self.ds.time, self.count).sel(
            pid=pid
        )
        return Trajectory(X, Y)

    # Obsolete
    def particle_count(self, time: int) -> int:
        return self.count[time]

    def __len__(self) -> int:
        return len(self.time)

    def __getattr__(self, var: str) -> Union[InstanceVariable, ParticleVariable]:
        return self.variables[var]

    def __getitem__(self, var: str) -> Union[InstanceVariable, ParticleVariable]:
        return self.variables[var]

    # Missing: Add global attributes
    def __repr__(self):
        s = "<postladim.ParticleFile>\n"
        s += f"num_times: {self.num_times}, num_particles: {self.num_particles}\n"
        s += f"time: {arraystr(self.time.da)}\n"
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

    def close(self) -> None:
        self.ds.close()

    # Make ParticleFile a context manager
    def __enter__(self):
        return self

    def __exit__(self, atype, value, traceback):
        self.close()


# ----------------------
# Utility functions
# ---------------------
