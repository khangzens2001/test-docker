from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ValidationErrorDetail(BaseModel):
    loc: List[Any] = Field(..., description="Location of field errors")
    msg: str = Field(..., description="Details on validation failures")
    type: str = Field(..., description="Underlying validator issue type")


class ProblemDetails(BaseModel):
    type: str = Field(..., description="RFC 7807 problem type URI")
    title: str = Field(..., description="Short overview of issue")
    status: int = Field(..., description="HTTP status code")
    detail: str = Field(..., description="Occurrence details")
    instance: str = Field(..., description="Target route URI")
    invalid_params: Optional[List[Any]] = Field(
        None, description="RFC 7807 invalid parameters list"
    )
    error_ref: Optional[str] = Field(None, description="Bare UUIDv4 correlation ID")
    errors: Optional[List[ValidationErrorDetail]] = Field(
        None, description="Pydantic validation details"
    )

