"""Unit conversion helpers for DASFrame."""

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .dasframe import DASFrame


def to_standard_units(frame: "DASFrame") -> "DASFrame":
    """Convert frame data to project-standard SI units when metadata is available."""
    inv = frame.inventory
    if not inv or not inv.acquisition:
        warnings.warn("缺少 Inventory/Acquisition 信息，无法自动转换单位")
        return frame

    current_unit = get_unit(frame)
    if current_unit == "unknown":
        return frame

    if current_unit in ["rad", "radians", "rad/s"]:
        wavelength = 1550.0
        if inv.interrogator and inv.interrogator.wavelength:
            wavelength = inv.interrogator.wavelength

        gl = 10.0
        if inv.fiber and inv.fiber.gauge_length:
            gl = inv.fiber.gauge_length

        n_refractive = 1.46
        g_factor = 0.78
        scale = (wavelength * 1e-9) / (4 * np.pi * n_refractive * gl * g_factor)

        target_unit = "strain" if current_unit != "rad/s" else "strain_rate"
        return update_unit_metadata(frame.scale(scale), target_unit)

    return frame


def update_unit_metadata(frame: "DASFrame", unit_name: str) -> "DASFrame":
    """Update unit metadata in-place and return frame for chaining."""
    new_meta = frame._metadata.copy()
    if "inventory" in new_meta:
        new_meta["inventory"].acquisition.data_unit = unit_name
    new_meta["units"] = unit_name
    frame._metadata = new_meta
    return frame


def get_unit(frame: "DASFrame") -> Any:
    """Return current unit string from inventory or fallback metadata."""
    inv = frame.inventory
    if inv and inv.acquisition:
        return inv.acquisition.data_unit
    return frame._metadata.get("units", "unknown")


def convert_units(frame: "DASFrame", target_unit: str) -> "DASFrame":
    """Convert explicit units using Pint."""
    from ..units import ureg

    current_unit = get_unit(frame)
    if current_unit == "unknown":
        raise ValueError("Current units are unknown, cannot convert.")

    q = ureg.Quantity(frame.collect(), current_unit)
    converted = q.to(target_unit).magnitude
    return update_unit_metadata(frame.__class__(converted, fs=frame._fs, dx=frame._dx), target_unit)
