"""
DASMatrix Event Management System
=================================

High-performance event catalog based on Polars.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class DASEvent(BaseModel):
    """Single event detected in DAS data."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique event ID")
    start_time: datetime = Field(..., description="Event start time (UTC)")
    end_time: Optional[datetime] = None
    min_channel: int = Field(..., description="Start channel index")
    max_channel: int = Field(..., description="End channel index")
    confidence: float = Field(0.0, description="Detection confidence (0-1)")
    event_type: str = Field("unknown", description="Type of event (e.g., 'vehicle', 'leak', 'quake')")
    attrs: Dict[str, Any] = Field(default_factory=dict, description="Additional attributes")


class EventCatalog:
    """High-performance event catalog powered by Polars."""

    def __init__(self, events: Optional[Union[List[DASEvent], pl.DataFrame]] = None):
        if events is None:
            self.df = pl.DataFrame(
                schema={
                    "id": pl.String,
                    "start_time": pl.Datetime,
                    "end_time": pl.Datetime,
                    "min_channel": pl.Int64,
                    "max_channel": pl.Int64,
                    "confidence": pl.Float64,
                    "event_type": pl.String,
                    "attrs": pl.Object,
                }
            )
        elif isinstance(events, list):
            if events and not isinstance(events[0], DASEvent):
                raise TypeError("events must be List[DASEvent]")
            data = [cast(DASEvent, e).model_dump() for e in events]
            self.df = pl.DataFrame(data)
        elif isinstance(events, pl.DataFrame):
            self.df = events
        else:
            raise TypeError("events must be None, List[DASEvent], or pl.DataFrame")

    def add(self, event: DASEvent) -> None:
        """Add a single event to the catalog."""
        row = pl.DataFrame([event.model_dump()])
        self.df = pl.concat([self.df, row], how="vertical")

    def filter(self, expr: pl.Expr) -> "EventCatalog":
        """Filter events using Polars expressions."""
        return EventCatalog(self.df.filter(expr))

    def to_csv(self, path: Union[str, Path]) -> None:
        """Save catalog to CSV."""
        df_save = self.df.with_columns(pl.col("attrs").cast(pl.String))
        df_save.write_csv(path)

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> "EventCatalog":
        """Load catalog from CSV."""
        df = pl.read_csv(path, try_parse_dates=True)
        return cls(df)

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"EventCatalog({len(self)} events)\n{self.df.head()}"
