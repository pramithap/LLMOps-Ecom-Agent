# utils/config_loader.py
# Utility module for loading the YAML configuration file.
# Resolves the config file path relative to the project root,
# regardless of where the script is executed from (CWD-independent).

from pathlib import Path
import os
import yaml

def _project_root() -> Path:
    # Navigate from this file's location up 2 levels to reach prod_assistant/ root
    # e.g. prod_assistant/utils/config_loader.py -> parents[1] == prod_assistant/
    return Path(__file__).resolve().parents[1]

def load_config(config_path: str | None = None) -> dict:
    """
    Load and return the YAML config as a dictionary.
    Resolution priority:
    1. Explicit `config_path` argument (if provided)
    2. CONFIG_PATH environment variable (if set)
    3. Default: <project_root>/config/config.yaml
    """
    # Check if a config path is set via environment variable
    env_path = os.getenv("CONFIG_PATH")

    # Fall back through the priority chain to determine the final path
    if config_path is None:
        config_path = env_path or str(_project_root() / "config" / "config.yml")

    path = Path(config_path)

    # If a relative path was given, resolve it against the project root
    if not path.is_absolute():
        path = _project_root() / path

    # Fail fast if the config file doesn't exist
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Read and parse the YAML file; return empty dict if file is blank
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}