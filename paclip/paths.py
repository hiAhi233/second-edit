"""Resolve project assets without depending on a command file's location."""
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("PACLIP_ROOT", Path(__file__).resolve().parents[1])).resolve()
