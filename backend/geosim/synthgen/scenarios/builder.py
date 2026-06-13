"""``build_scenario`` — compile the earth, run all forwards, write the folder (doc 05 §5).

This is the scenario *driver*. Given a :class:`~.registry.ScenarioSpec` it:

1. compiles the declarative earth into a :class:`~geosim.synthgen.truth.TruthEarth`
   (:func:`~geosim.synthgen.compiler.compile_scene`, doc 05 §2);
2. runs **every** registered T0 forward (:data:`~geosim.synthgen.forward.FORWARD_MODELS`)
   onto it, each into ``measured/`` with method-independent seeded sub-streams
   (``numpy.random.SeedSequence`` so a run is byte-reproducible, doc 05 §1, §8);
3. writes the ``truth/`` scoring oracle (:func:`~geosim.synthgen.truth.write_truth_bundle`,
   doc 05 §5) — ground-truth zarr + ``features.geojson``, NEVER ingested;
4. emits the self-contained scenario folder (doc 05 §5): ``scene.jsonc``,
   ``acquisition.jsonc``, ``frame.json`` (the scenario :class:`~geosim.spatial.SpatialFrame`,
   doc 01 §2), and ``manifest.json`` (seed, library versions, per-file SHA-256 checksums +
   the synthetic provenance stamped on each measured artifact, doc 05 §5).

Forwards receive their ``out_dir`` (= ``measured/``) via ``Acquisition.params['out_dir']``
(the convention every forward in :mod:`geosim.synthgen.forward` reads). A forward that
raises is recorded in the manifest under ``errors`` and skipped, so one flaky method never
fails the whole build.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata as _md
import json
import platform
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from geosim.spatial import Aabb, DepthRange, FrameMode, GeorefStatus, SpatialFrame
from geosim.storage import sha256_bytes

from ..compiler import compile_scene
from ..forward import Artifact, all_forwards
from ..scene import SceneSpec
from ..truth import TruthEarth, write_truth_bundle
from .registry import ScenarioSpec, get_scenario

__all__ = ["build_scenario", "BuildResult"]


@dataclass(frozen=True)
class BuildResult:
    """The outcome of a scenario build (doc 05 §5)."""

    scenario_id: str
    out_dir: Path
    truth: TruthEarth
    artifacts: list[Artifact] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- frame


def _frame_from_scene(spec: SceneSpec) -> SpatialFrame:
    """Build the scenario :class:`SpatialFrame` from the scene frame (doc 01 §2, doc 05 §5).

    Local mode by default (``surfaceModel: synthetic:<id>``); ROI + depth range copied from
    the truth-grid frame so a project can ingest the whole scenario into one frame.
    """
    f = spec.frame
    mode = (
        FrameMode.GEOREFERENCED if f.mode == "georeferenced" else FrameMode.LOCAL
    )
    return SpatialFrame(
        mode=mode,
        roi=Aabb(xmin=f.xmin, xmax=f.xmax, ymin=f.ymin, ymax=f.ymax),
        depth_range=DepthRange(zmin=f.zmin, zmax=f.zmax),
        surface_model=f"synthetic:{spec.id}",
        georef_status=GeorefStatus.ASSUMED_LOCAL,
    )


def _frame_to_dict(frame: SpatialFrame) -> dict:
    """JSON-serialisable view of a :class:`SpatialFrame` (doc 01 §2 catalog metadata)."""
    return {
        "mode": frame.mode.value,
        "axisConvention": frame.axis_convention,
        "lengthUnit": frame.length_unit,
        "horizontalCrs": frame.horizontal_crs,
        "verticalDatum": frame.vertical_datum,
        "anchor": (
            None
            if frame.anchor is None
            else {
                "easting": frame.anchor.easting,
                "northing": frame.anchor.northing,
                "elevation": frame.anchor.elevation,
            }
        ),
        "rotationDeg": frame.rotation_deg,
        "roi": dataclasses.asdict(frame.roi),
        "depthRange": dataclasses.asdict(frame.depth_range),
        "surfaceModel": frame.surface_model,
        "georefStatus": frame.georef_status.value,
    }


# --------------------------------------------------------------------------- manifest


_LIBS = ("numpy", "scipy", "rasterio", "segyio", "lasio", "obspy", "pandas", "zarr")


def _library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _LIBS:
        try:
            out[name] = _md.version(name)
        except Exception:  # pragma: no cover - missing optional dep
            out[name] = "unknown"
    return out


def _checksum(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _file_records(out_dir: Path, artifacts: list[Artifact]) -> list[dict]:
    """Per-artifact manifest records: relative path, format, checksum, provenance (§5)."""
    records: list[dict] = []
    for art in artifacts:
        try:
            rel = art.path.relative_to(out_dir)
        except ValueError:  # pragma: no cover - defensive
            rel = art.path
        records.append(
            {
                "path": str(rel),
                "format": art.fmt,
                "method": art.method,
                "submethod": art.submethod,
                "synthetic": True,
                "sha256": _checksum(art.path) if art.path.exists() else None,
                "provenance": art.provenance.to_dict(),
            }
        )
    return records


# --------------------------------------------------------------------------- driver


def build_scenario(
    scenario: str | ScenarioSpec,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> BuildResult:
    """Build ``scenario`` into a self-contained scenario folder at ``out_dir`` (doc 05 §5).

    Compiles the earth, runs all T0 forwards into ``measured/``, writes the ``truth/``
    scoring oracle + ``scene.jsonc`` / ``acquisition.jsonc`` / ``frame.json`` /
    ``manifest.json``. Deterministic from ``(scene, seed)`` (doc 05 §1): the root seed is
    the scene seed, fanned out into per-method sub-streams. Returns a :class:`BuildResult`.
    """
    spec = scenario if isinstance(scenario, ScenarioSpec) else get_scenario(scenario)
    scene = spec.scene
    out = Path(out_dir)
    measured = out / "measured"
    truth_dir = out / "truth"
    out.mkdir(parents=True, exist_ok=True)
    measured.mkdir(parents=True, exist_ok=True)

    # 1. compile the earth (doc 05 §2).
    earth = compile_scene(scene)

    # 2. run every T0 forward into measured/ with seeded per-method sub-streams (§1, §8).
    forwards = all_forwards()
    root_ss = np.random.SeedSequence(scene.seed)
    child_seeds = root_ss.spawn(len(forwards))
    artifacts: list[Artifact] = []
    errors: dict[str, str] = {}
    for fwd, child in zip(forwards, child_seeds, strict=True):
        rng = np.random.default_rng(child)
        acq = replace(
            spec.acquisition, params={**spec.acquisition.params, "out_dir": measured}
        )
        key = f"{fwd.method}/{fwd.submethod}" if fwd.submethod else fwd.method
        try:
            artifacts.extend(fwd.simulate(earth, acq, rng))
        except Exception as e:  # one flaky method must not fail the whole build
            errors[key] = f"{type(e).__name__}: {e}"

    # 3. write the truth/ scoring oracle — NEVER ingested (doc 05 §5, decision #6).
    write_truth_bundle(earth, truth_dir, overwrite=overwrite)

    # 4. the self-contained scenario folder scaffolding (doc 05 §5).
    _write_text(out / "scene.jsonc", _source_jsonc(spec, "scene.jsonc"))
    _write_text(out / "acquisition.jsonc", _source_jsonc(spec, "acquisition.jsonc"))

    frame = _frame_from_scene(scene)
    _write_text(out / "frame.json", json.dumps(_frame_to_dict(frame), indent=2))

    manifest = {
        "sceneId": scene.id,
        "title": spec.title,
        "description": spec.description,
        "seed": scene.seed,
        "synthetic": True,
        "generator": "geosim.synthgen",
        "rockPhysics": scene.rock_physics,
        "platform": platform.platform(),
        "pythonVersion": platform.python_version(),
        "libraryVersions": _library_versions(),
        "truthGrid": {
            "shape": list(earth.shape),
            "origin_zyx": list(earth.origin),
            "spacing_zyx": list(earth.spacing),
        },
        "frame": _frame_to_dict(frame),
        "measured": _file_records(out, artifacts),
        "truth": _truth_records(out, truth_dir),
        "errors": errors,
    }
    _write_text(out / "manifest.json", json.dumps(manifest, indent=2))

    return BuildResult(
        scenario_id=scene.id,
        out_dir=out,
        truth=earth,
        artifacts=artifacts,
        errors=errors,
    )


def _truth_records(out_dir: Path, truth_dir: Path) -> list[dict]:
    """Manifest records for the truth bundle entries (zarr dirs + features, doc 05 §5)."""
    records: list[dict] = []
    if not truth_dir.exists():
        return records
    for entry in sorted(truth_dir.iterdir()):
        rec: dict = {
            "path": str(entry.relative_to(out_dir)),
            "kind": "zarr" if entry.suffix == ".zarr" else entry.suffix.lstrip(".") or "dir",
            "ingested": False,  # truth is NEVER ingested (decision #6)
        }
        if entry.is_file():
            rec["sha256"] = _checksum(entry)
        records.append(rec)
    return records


def _source_jsonc(spec: ScenarioSpec, name: str) -> str:
    """Return the authored JSONC text for a scenario file, if it ships one (doc 05 §5).

    Scenario modules in this package author ``<id>/scene.jsonc`` + ``acquisition.jsonc``;
    we copy the original text so provenance keeps the human-authored comments. Falls back
    to a serialised view if the source file is absent (e.g. a programmatically-built spec).
    """
    from . import great_basin_v1, unit_cube_v1

    for mod in (unit_cube_v1, great_basin_v1):
        d = getattr(mod, "DIR", None)
        if d is not None and getattr(mod, "SCENARIO", None) is not None:
            if mod.SCENARIO.id == spec.id and (d / name).exists():
                return (d / name).read_text(encoding="utf-8")
    # fallback: minimal JSON view (programmatic spec without authored JSONC)
    if name == "scene.jsonc":
        return json.dumps({"id": spec.scene.id, "seed": spec.scene.seed}, indent=2)
    return json.dumps({}, indent=2)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
