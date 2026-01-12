"""
API package for Identity Circuit Factory.
"""

from .endpoints import router
from .generator_endpoints import router as generator_router
from .models import *
from .server import create_app, run_server

__version__ = "1.0.0"
__all__ = ["create_app", "models", "endpoints"]
