"""
Atoms: Stateful Processing Pipeline Framework
=============================================

Essential for processing large datasets in chunks while maintaining
mathematical continuity at chunk boundaries.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

import numpy as np
from scipy import signal

if TYPE_CHECKING:
    from ..api.dasframe import DASFrame

logger = logging.getLogger(__name__)


class Atom(ABC):
    """Base class for atomic processing units with state."""

    @abstractmethod
    def __call__(self, data: "DASFrame") -> "DASFrame":
        """Process a data chunk and update internal state."""
        pass

    def reset(self) -> None:
        """Reset internal state to initial values."""
        pass


class Partial(Atom):
    """Wraps a stateless function or DASFrame method as an Atom."""

    def __init__(self, func_or_name: Union[Callable, str], *args: Any, **kwargs: Any):
        self.func_or_name = func_or_name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, data: "DASFrame") -> "DASFrame":
        if isinstance(self.func_or_name, str):
            method = getattr(data, self.func_or_name)
            return method(*self.args, **self.kwargs)
        else:
            return self.func_or_name(data, *self.args, **self.kwargs)


class SosFilt(Atom):
    """Stateful Second-Order Sections (SOS) filter."""

    def __init__(self, sos: np.ndarray, axis: Union[int, str] = "time"):
        self.sos = sos
        self.axis = axis
        self._zi: Optional[np.ndarray] = None

    def __call__(self, data: "DASFrame") -> "DASFrame":
        from ..api.dasframe import DASFrame

        arr = data.collect()
        ax_idx = 0 if self.axis == "time" or self.axis == 0 else 1

        if self._zi is None:
            zi_base = signal.sosfilt_zi(self.sos)
            n_channels = arr.shape[1] if ax_idx == 0 else arr.shape[0]
            self._zi = np.tile(zi_base[:, :, np.newaxis], (1, 1, n_channels))

        filtered, self._zi = signal.sosfilt(self.sos, arr, axis=ax_idx, zi=self._zi)

        meta = data._metadata.copy()
        meta.pop("fs", None)
        meta.pop("dx", None)
        return DASFrame(filtered, fs=data.fs, dx=data._dx, **meta)

    def reset(self) -> None:
        self._zi = None


class Sequential(Atom):
    """Composes multiple Atoms into a sequential pipeline."""

    def __init__(self, atoms: List[Atom]):
        self.atoms = atoms

    def __call__(self, data: "DASFrame") -> "DASFrame":
        for atom in self.atoms:
            data = atom(data)
        return data

    def reset(self) -> None:
        for atom in self.atoms:
            atom.reset()


class Parallel(Atom):
    """Executes multiple Atoms in parallel on the same input data."""

    def __init__(self, atoms: List[Atom]):
        self.atoms = atoms

    def __call__(self, data: "DASFrame") -> Any:
        return [atom(data) for atom in self.atoms]

    def reset(self) -> None:
        for atom in self.atoms:
            atom.reset()
