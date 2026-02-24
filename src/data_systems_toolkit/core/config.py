"""
Configuration management for the Data Systems Toolkit.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass
class Config:
 """Configuration class for the Data Systems Toolkit."""

 # DuckDB Configuration
 duckdb_path: str = "/data/analytics.duckdb"

 # Marquez/Lineage Configuration
 marquez_host: str = "localhost"
 marquez_port: int = 5000

 # API Keys (for Phase 1B+)
 openai_api_key: Optional[str] = None

 # Synthetic Data Settings
 synthetic_seed: int = 42
 synthetic_default_rows: int = 1000

 # Development Settings
 debug: bool = True
 log_level: str = "INFO"
 environment: str = "development"

 # Paths
 data_path: str = "/data"
 samples_path: str = "/workspace/samples"
 outputs_path: str = "/workspace/outputs"

 def __post_init__(self):
 """Validate configuration after initialization."""
 self._ensure_paths_exist

 def _ensure_paths_exist(self) -> None:
 """Ensure all required paths exist."""
 for path_attr in ["data_path", "samples_path", "outputs_path"]:
 path = Path(getattr(self, path_attr))
 # Only create if path is writable (skip for /data in non-Docker)
 try:
 path.mkdir(parents=True, exist_ok=True)
 except PermissionError:
 pass # Skip paths we can't create

 @classmethod
 def from_env(cls, env_file: Optional[str] = None) -> "Config":
 """Load configuration from environment variables."""
 if env_file:
 load_dotenv(env_file)
 else:
 load_dotenv

 return cls(
 duckdb_path=os.getenv("DUCKDB_PATH", cls.duckdb_path),
 marquez_host=os.getenv("MARQUEZ_HOST", cls.marquez_host),
 marquez_port=int(os.getenv("MARQUEZ_PORT", str(cls.marquez_port))),
 openai_api_key=os.getenv("OPENAI_API_KEY"),
 synthetic_seed=int(os.getenv("SYNTHETIC_SEED", str(cls.synthetic_seed))),
 synthetic_default_rows=int(
 os.getenv("SYNTHETIC_DEFAULT_ROWS", str(cls.synthetic_default_rows))
 ),
 debug=os.getenv("DEBUG", "true").lower == "true",
 log_level=os.getenv("LOG_LEVEL", cls.log_level),
 environment=os.getenv("ENVIRONMENT", cls.environment),
 data_path=os.getenv("DATA_PATH", cls.data_path),
 samples_path=os.getenv("SAMPLES_PATH", cls.samples_path),
 outputs_path=os.getenv("OUTPUTS_PATH", cls.outputs_path),
 )

 def to_dict(self) -> Dict[str, Any]:
 """Convert configuration to dictionary."""
 return {
 field.name: getattr(self, field.name)
 for field in self.__dataclass_fields__.values
 }


def load_config(env_file: Optional[str] = None) -> Config:
 """Load configuration from environment variables.

 Args:
 env_file: Optional path to .env file

 Returns:
 Config: Configuration instance
 """
 return Config.from_env(env_file)
