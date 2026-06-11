"""Shared base for v1 response models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ORMModel(BaseModel):
    """Allow Pydantic to read attributes from SQLAlchemy ORM rows.

    All v1 schemas inherit from this so routers can do
    ``SchemaOut.model_validate(orm_obj)`` directly.
    """

    model_config = ConfigDict(from_attributes=True)
