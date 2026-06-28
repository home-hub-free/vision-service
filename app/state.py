"""Shared singletons — the live world-model + stores, shared by workers and routes.

Kept in one tiny module so the FastAPI routes and the camera worker threads read/write
the SAME OccupancyTracker / Gallery / EventIndex without an import cycle.
"""
from __future__ import annotations

from typing import Dict

from .gallery import Gallery
from .index_db import EventIndex
from .occupancy import OccupancyTracker

tracker = OccupancyTracker()
gallery = Gallery()
index = EventIndex()

# cam_id -> CameraWorker (typed loosely to avoid importing camera.py here).
workers: Dict[str, object] = {}
