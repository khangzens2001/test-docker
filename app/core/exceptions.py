from typing import Any, List, Optional
from fastapi import status


class BaseDomainException(Exception):
    """Base domain exception for server-side VIO errors."""

    def __init__(
        self,
        detail: str = "Domain exception occurred",
        status_code: int = status.HTTP_400_BAD_REQUEST,
        type_uri: str = "https://api.lidarscan.com/errors/domain-error",
        title: str = "Domain Error",
        invalid_params: Optional[List[Any]] = None,
    ):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
        self.type_uri = type_uri
        self.title = title
        self.invalid_params = invalid_params


class VioInitializationError(BaseDomainException):
    """Raised when VIO initialization fails."""

    def __init__(
        self,
        detail: str = "VIO initialization failed",
        invalid_params: Optional[List[Any]] = None,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        type_uri: str = "https://api.lidarscan.com/errors/vio-initialization-failed",
        title: str = "VIO Initialization Error",
    ):
        super().__init__(
            detail=detail,
            status_code=status_code,
            type_uri=type_uri,
            title=title,
            invalid_params=invalid_params,
        )


class SensorDataIngestionError(BaseDomainException):
    """Raised when sensor data ingestion fails or required telemetry streams are missing."""

    def __init__(
        self,
        detail: str = "Sensor data ingestion failed",
        invalid_params: Optional[List[Any]] = None,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        type_uri: str = "https://api.lidarscan.com/errors/sensor-data-ingestion-failed",
        title: str = "Sensor Data Ingestion Error",
    ):
        super().__init__(
            detail=detail,
            status_code=status_code,
            type_uri=type_uri,
            title=title,
            invalid_params=invalid_params,
        )


class TrackingLostError(BaseDomainException):
    """Raised when VIO tracking is lost during processing."""

    def __init__(
        self,
        detail: str = "VIO tracking lost",
        invalid_params: Optional[List[Any]] = None,
        status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY,
        type_uri: str = "https://api.lidarscan.com/errors/tracking-lost",
        title: str = "Tracking Lost Error",
    ):
        super().__init__(
            detail=detail,
            status_code=status_code,
            type_uri=type_uri,
            title=title,
            invalid_params=invalid_params,
        )


class MaterialEstimationError(BaseDomainException):
    """Raised when wallpaper/tatami material estimation hits a physical limit."""

    def __init__(
        self,
        detail: str = "Material estimation failed",
        invalid_params: Optional[List[Any]] = None,
        status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY,
        type_uri: str = "https://api.lidarscan.com/errors/material-estimation",
        title: str = "Material Estimation Error",
    ):
        super().__init__(
            detail=detail,
            status_code=status_code,
            type_uri=type_uri,
            title=title,
            invalid_params=invalid_params,
        )


class TatamiShavingLimitExceededError(MaterialEstimationError):
    """Raised when combined |δ| exceeds the 30 mm hard limit."""

    def __init__(
        self,
        detail: str = "Tatami shaving exceeds 30 mm hard limit",
        invalid_params: Optional[List[Any]] = None,
        status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY,
        type_uri: str = "https://api.lidarscan.com/errors/tatami-shaving-limit-exceeded",
        title: str = "Tatami Shaving Limit Exceeded",
    ):
        super().__init__(
            detail=detail,
            status_code=status_code,
            type_uri=type_uri,
            title=title,
            invalid_params=invalid_params,
        )


class TatamiSkewAngleExceededError(MaterialEstimationError):
    """Raised when wall skew |α| is at least 15°."""

    def __init__(
        self,
        detail: str = "Tatami wall skew exceeds 15° limit",
        invalid_params: Optional[List[Any]] = None,
        status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY,
        type_uri: str = "https://api.lidarscan.com/errors/tatami-skew-angle-exceeded",
        title: str = "Tatami Skew Angle Exceeded",
    ):
        super().__init__(
            detail=detail,
            status_code=status_code,
            type_uri=type_uri,
            title=title,
            invalid_params=invalid_params,
        )

