"""Public, process-isolated exHad decay interface."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the checkout: cpp/, data/, .runtime/
DATA = ROOT / 'data'

from .api import Generator
from .models import model_info

__all__ = ['Generator', 'model_info']
