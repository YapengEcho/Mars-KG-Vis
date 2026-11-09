"""Shared response and catalog models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SlideType = Literal["SL", "SP", "FL", "CL"]
DataMode = Literal["demo", "catalog"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CoordinateSystem(StrictModel):
    body: Literal["Mars"]
    longitude: Literal["east-positive, -180..180"]
    latitude: Literal["planetocentric, degrees"]


class Source(StrictModel):
    title: str = Field(min_length=1)
    url: str | None

    @field_validator("url")
    @classmethod
    def check_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("https://"):
            raise ValueError("Source URL must use HTTPS or be null")
        return value


class Slide(StrictModel):
    id: str = Field(min_length=1)
    sourceId: int | None = Field(ge=0)
    name: str = Field(min_length=1)
    type: SlideType
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    areaKm2: float | None = Field(ge=0, allow_inf_nan=False)
    summary: str
    imageUrl: str | None
    source: Source
    isDemo: bool

    @field_validator("imageUrl")
    @classmethod
    def check_image_url(cls, value: str | None) -> str | None:
        if value is not None and (
            value.startswith("//")
            or not (value.startswith("/") or value.startswith("https://"))
        ):
            raise ValueError("Image URL must use HTTPS, a same-origin path, or null")
        return value


class Catalog(StrictModel):
    version: str = Field(min_length=1)
    mode: DataMode
    coordinateSystem: CoordinateSystem
    records: list[Slide] = Field(min_length=1)

    @model_validator(mode="after")
    def check_records(self) -> "Catalog":
        ids = [record.id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Catalog record IDs must be unique")
        expected_demo = self.mode == "demo"
        if any(record.isDemo != expected_demo for record in self.records):
            raise ValueError("Record isDemo must agree with catalog mode")
        return self


class SlidePage(StrictModel):
    items: list[Slide]
    total: int
    page: int
    pageSize: int
    dataMode: DataMode
    version: str


class Statistics(StrictModel):
    total: int
    byType: dict[SlideType, int]
    dataMode: DataMode
    version: str


class Capability(StrictModel):
    available: bool
    reason: str


class AgentCapability(StrictModel):
    available: bool
    model: str
    reason: str | None = None
class Capabilities(StrictModel):
    dataMode: DataMode
    version: str
    classifier: Capability
    detection: Capability
    agent: AgentCapability


class Health(StrictModel):
    status: Literal["ok"]
    service: Literal["mars-api"]
    dataMode: DataMode
    version: str
