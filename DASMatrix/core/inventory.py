"""
DASMatrix Inventory System
==========================

This module defines the metadata structure for DAS datasets, following
PRODML v2.1 and DAS-RCN standards.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field


class FiberGeometry(BaseModel):
    """Geometry information of the sensing fiber."""

    # Configuration to allow arbitrary types if needed,
    # though mostly standard types here
    model_config = ConfigDict(extra="ignore")

    coordinates: Optional[List[Tuple[float, float, float]]] = Field(
        None, description="List of (lon, lat, elevation) tuples"
    )
    gauge_length: float = Field(..., description="Gauge length in meters")
    channel_spacing: float = Field(..., description="Spacing between channels in meters")
    total_length: Optional[float] = Field(None, description="Total fiber length in meters")
    start_distance: float = Field(0.0, description="Distance of the first channel from the interrogator (m)")


class Interrogator(BaseModel):
    """DAS Interrogator Unit (IU) information."""

    model_config = ConfigDict(extra="ignore")

    manufacturer: Optional[str] = None
    model: str = Field(..., description="Model name of the interrogator")
    serial_number: Optional[str] = None
    sampling_rate: float = Field(..., description="Temporal sampling rate in Hz")
    pulse_width: Optional[float] = Field(None, description="Pulse width in ns")
    pulse_rate: Optional[float] = Field(None, description="Pulse repetition rate in Hz")
    wavelength: Optional[float] = Field(None, description="Laser wavelength in nm")


class Acquisition(BaseModel):
    """Data acquisition parameters."""

    model_config = ConfigDict(extra="ignore")

    start_time: datetime = Field(..., description="Start time of the recording (UTC)")
    end_time: Optional[datetime] = None
    n_channels: int = Field(..., description="Number of spatial channels")
    n_samples: Optional[int] = Field(None, description="Number of time samples")
    data_unit: str = Field(
        "strain_rate",
        description="Physical unit of the data (e.g., strain, strain_rate, velocity)",
    )
    spatial_reference: str = Field("measured_depth", description="Reference for channel positions")


class ProcessingStep(BaseModel):
    """Record of a processing operation applied to the data."""

    model_config = ConfigDict(extra="ignore")

    operation: str = Field(..., description="Name of the operation (e.g., bandpass, detrend)")
    parameters: Dict[str, Any] = Field(..., description="Parameters used for the operation")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Time of processing",
    )
    software: str = "DASMatrix"
    version: Optional[str] = None


class DASInventory(BaseModel):
    """
    Complete metadata inventory for a DAS dataset.

    Acts as a container for all metadata attributes, compatible with
    PRODML v2.1 and DAS-RCN standards.
    """

    model_config = ConfigDict(extra="ignore")

    # Basic Info
    project_name: str = Field("Unknown", description="Name of the project")
    experiment_name: Optional[str] = None
    description: Optional[str] = None

    # Components
    fiber: Optional[FiberGeometry] = None
    interrogator: Optional[Interrogator] = None
    acquisition: Acquisition

    # History
    processing_history: List[ProcessingStep] = Field(default_factory=list)

    # Custom attributes for flexibility
    custom_attrs: Dict[str, Any] = Field(default_factory=dict)

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """Serialize inventory to JSON string or file."""
        json_str = self.model_dump_json(indent=indent)
        if path:
            Path(path).write_text(json_str, encoding="utf-8")
        return json_str

    @classmethod
    def from_json(cls, path_or_str: Union[str, Path]) -> "DASInventory":
        """Load inventory from JSON string or file."""
        if isinstance(path_or_str, Path) or (
            isinstance(path_or_str, str)
            and len(path_or_str) < 256
            and (path_or_str.endswith(".json") or Path(path_or_str).exists())
        ):
            # Treat as path if it looks like one or exists
            try:
                content = Path(path_or_str).read_text(encoding="utf-8")
            except OSError:
                # Fallback: maybe it's a JSON string that happens to be short?
                # Unlikely but possible. For now assume it's a file path
                # if shorter than 256 char
                content = str(path_or_str)
        else:
            content = str(path_or_str)

        return cls.model_validate_json(content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    def add_processing_step(self, operation: str, **params) -> None:
        """Record a processing step."""
        from DASMatrix import __version__

        step = ProcessingStep(
            operation=operation,
            parameters=params,
            timestamp=datetime.now(timezone.utc),
            version=__version__,
        )
        self.processing_history.append(step)

    def __repr__(self) -> str:
        """Brief string representation."""
        info = [f"DASInventory(project='{self.project_name}')"]
        if self.acquisition:
            info.append(f"  Shape: {self.acquisition.n_channels} ch x {self.acquisition.n_samples or '?'} samples")
            info.append(f"  Time: {self.acquisition.start_time}")
        if self.interrogator:
            info.append(
                f"  Interrogator: {self.interrogator.manufacturer} "
                f"{self.interrogator.model} @ {self.interrogator.sampling_rate} Hz"
            )
        return "\n".join(info)
