"""Configuration management for the project."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration class for managing project settings."""
    
    def __init__(self, config_path: str = None):
        self.project_root = Path(__file__).parent.parent.parent
        self.config_path = config_path or self.project_root / "configs" / "base_config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return self.project_root / "data"
    
    @property
    def raw_data_dir(self) -> Path:
        """Get raw data directory path."""
        return self.data_dir / "raw"
    
    @property
    def processed_data_dir(self) -> Path:
        """Get processed data directory path."""
        return self.data_dir / "processed"
    
    @property
    def results_dir(self) -> Path:
        """Get results directory path."""
        return self.project_root / "results"
    
    @property
    def models_dir(self) -> Path:
        """Get models directory path."""
        return self.results_dir / "models"
