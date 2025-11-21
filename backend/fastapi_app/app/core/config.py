"""
Configuration loader for the Waze Biobío ML system.

This module loads configuration from config.yaml and provides
easy access to all system parameters.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Configuration manager for the application."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config.yaml. If None, looks for config.yaml
                        in the project root.
        """
        if config_path is None:
            # Try to find config.yaml in project root
            current_dir = Path(__file__).resolve()
            # Navigate up to project root (from backend/fastapi_app/app/core/)
            project_root = current_dir.parent.parent.parent.parent
            config_path = project_root / "config.yaml"

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please create config.yaml from config.example.yaml"
            )

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., 'routing.penalty_radius_m')
            default: Default value if key is not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    @property
    def routing(self) -> Dict[str, Any]:
        """Get routing configuration."""
        return self._config.get('routing', {})

    @property
    def recommendations(self) -> Dict[str, Any]:
        """Get recommendations configuration."""
        return self._config.get('recommendations', {})

    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config.get('data', {})

    @property
    def backend(self) -> Dict[str, Any]:
        """Get backend configuration."""
        return self._config.get('backend', {})

    @property
    def frontend(self) -> Dict[str, Any]:
        """Get frontend configuration."""
        return self._config.get('frontend', {})

    @property
    def geospatial(self) -> Dict[str, Any]:
        """Get geospatial configuration."""
        return self._config.get('geospatial', {})


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config() -> None:
    """Reload configuration from file."""
    global _config
    _config = None
    get_config()
