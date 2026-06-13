"""CLI entry for the synthetic data generator (doc 05 §8).

Usage (doc 05 §8)::

    python -m geosim.synthgen build <scenario> --out scenarios/<id>/
    python -m geosim.synthgen list

``build`` compiles the named scenario's earth, runs every T0 forward, and writes the
self-contained scenario folder (doc 05 §5) via
:func:`geosim.synthgen.scenarios.build_scenario`. ``list`` prints the registered scenarios.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .scenarios import build_scenario, get_scenario, list_scenarios


def _cmd_list(_args: argparse.Namespace) -> int:
    for sid in list_scenarios():
        spec = get_scenario(sid)
        print(f"{sid}\t{spec.title}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else Path("scenarios") / args.scenario
    result = build_scenario(args.scenario, out, overwrite=args.overwrite)
    print(f"built {result.scenario_id} -> {result.out_dir}")
    print(f"  measured artifacts: {len(result.artifacts)}")
    if result.errors:
        print(f"  forward errors: {result.errors}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m geosim.synthgen", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="build a scenario folder (doc 05 §5)")
    pb.add_argument("scenario", help="scenario id (see `list`)")
    pb.add_argument("--out", default=None, help="output dir (default scenarios/<id>/)")
    pb.add_argument("--overwrite", action="store_true", help="overwrite existing truth zarr")
    pb.set_defaults(func=_cmd_build)

    pl = sub.add_parser("list", help="list registered scenarios (doc 05 §7)")
    pl.set_defaults(func=_cmd_list)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
