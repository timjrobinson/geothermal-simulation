"""Ingestion pipeline — minimal M1 'write + register' slice (doc 03 §7).

The full doc-03 pipeline is ``upload → store-raw → detect → parse → normalize → write →
register`` over pluggable adapters (doc 03 §1, §7). M1 needs only the terminal
**write + register** step (doc 03 §7 steps 6–7) for a single, already-modeled
synthetic volume: write the doc-02 PropertyModel Zarr group (doc 02 §10.2) and insert
the catalog rows — ``project`` + local-mode ``spatial_frame`` (doc 01 §2), ``dataset``
(``kind=propertyModel``), ``property_model``, and the mandatory ``provenance`` edge
(``process="synthesize"``, doc 02 §7) — atomically, so the dataset only becomes visible
once registration commits (doc 03 §7 step 7).
"""

from .seed import seed_m1_project

__all__ = ["seed_m1_project"]
