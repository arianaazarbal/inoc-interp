"""Utility functions for JSON and YAML handling."""

import json
import os
from typing import Any

import yaml


def dump_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_yaml(path: str) -> dict:
    """Load a YAML file and return its contents."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
