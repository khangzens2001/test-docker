import os

# DevOps/SRE: Set single-threading env vars programmatically at import time
# to prevent excessive thread creation by NumPy, OpenBLAS, MKL, PyTorch, Arrow, and ONNX Runtime.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["ARROW_IO_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["ORT_NUM_THREADS"] = "1"

from typing import List, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Lidar Room Scan Server"
    API_V1_STR: str = "/api/v1"

    DEBUG: bool = False
    ENV_MODE: str = "local"

    CORS_ORIGINS: List[str] = Field(default_factory=list)
    # Trusted proxy hosts for ProxyHeadersMiddleware (empty = do not trust
    # X-Forwarded-For at all). Populate with your load balancer IP(s) in
    # production (see Task 6 notes).
    TRUSTED_HOSTS: List[str] = Field(default_factory=list)

    DATA_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
    )
    UPLOAD_DIR: str = ""
    SESSION_DIR: str = ""

    DATABASE_URL: str = ""

    UPLOAD_MAX_SIZE: int = 300 * 1024 * 1024
    EXTRACT_MAX_SIZE: int = 1500 * 1024 * 1024
    MAX_COMPRESSION_RATIO: float = 20.0
    MIN_DISK_SPACE_GB: float = 5.0
    MAX_ZIP_ENTRIES: int = 10_000

    REDIS_URL: Optional[str] = None
    CELERY_BROKER_URL: Optional[str] = None

    ONNX_INTRA_THREADS: int = 2
    ONNX_INTER_THREADS: int = 2
    ONNX_VRAM_LIMIT_BYTES: int = 2 * 1024 * 1024 * 1024
    CUDA_DEVICE_ID: int = 0
    MODEL_WEIGHTS_SHA: str = "5b69a73a98435e5c3e222197b579497546377a30665e89f8ee3c288634211fef"
    ENABLE_TOF_PIPELINE: bool = True
    ENABLE_SERVER_VIO: bool = True
    REQUIRE_CUDA: bool = False
    MODEL_WEIGHTS_PATH: str = "weights/depthor.onnx"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @model_validator(mode="after")
    def _fill_paths_and_validate(self) -> "Settings":
        if not self.UPLOAD_DIR:
            self.UPLOAD_DIR = os.path.join(self.DATA_DIR, "uploads")
        if not self.SESSION_DIR:
            self.SESSION_DIR = os.path.join(self.DATA_DIR, "sessions")
        if not self.DATABASE_URL:
            self.DATABASE_URL = (
                f"sqlite+aiosqlite:///{os.path.join(self.DATA_DIR, 'server.db')}"
            )
        if not os.path.isabs(self.MODEL_WEIGHTS_PATH):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            abs_weights = os.path.join(project_root, self.MODEL_WEIGHTS_PATH)
            if os.path.exists(abs_weights):
                self.MODEL_WEIGHTS_PATH = abs_weights

        if not self.CELERY_BROKER_URL:
            if self.REDIS_URL:
                self.CELERY_BROKER_URL = self.REDIS_URL
            else:
                self.CELERY_BROKER_URL = "redis://localhost:6379/0"
        return self


settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.SESSION_DIR, exist_ok=True)
