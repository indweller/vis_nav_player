"""The repository's root: data, caches and results live under it, wherever a script is run from."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
