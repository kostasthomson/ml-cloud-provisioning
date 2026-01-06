"""
Application configuration and settings.
"""
from pydantic_settings import BaseSettings
from typing import Optional, Literal


class FastApiConfiguration(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    app_name: str = "Smart Task Allocator"
    app_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # or "text"

    # Model Settings
    model_type: str = ""
    default_task_duration: float = 3600.0  # seconds
    allocator_type: Literal["heuristic", "nn", "energy_regression", "rl"] = "rl"

    # Performance
    enable_cors: bool = True
    cors_origins: list = ["*"]

    class Config:
        env_prefix = "ALLOCATOR_"
        case_sensitive = False


fast_api_configuration = FastApiConfiguration()
