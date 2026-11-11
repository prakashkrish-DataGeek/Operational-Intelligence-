"""Command line: kasauti fetch | run | reproduce | manifest."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from . import __version__, data, report


def _load_agent(spec: str):
    mod, _, cls = spec.partition(":")
    sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(mod), cls)()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kasauti", description="Touchstone benchmark for operational AI on Volve data.")
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("--data", type=Path, default=None, help="directory holding the five Volve files (default ./data/raw)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download the pinned Volve files and verify checksums")
    sub.add_parser("manifest", help="print sources and checksums")
    r = sub.add_parser("run", help="run all tracks and write results")
    r.add_argument("--out", type=Path, default=Path("results"))
    r.add_argument("--agent", help="extra agent to score, as module:Class")
    c = sub.add_parser("reproduce", help="re-run and compare with published results")
    c.add_argument("--against", type=Path, default=Path("results/results.json"))
    args = ap.parse_args(argv)

    try:
        if args.cmd == "fetch":
            data.fetch(args.data)
            print("all files present and verified")
        elif args.cmd == "manifest":
            print(data.manifest_json())
        elif args.cmd == "run":
            res = report.run_all(args.data, _load_agent(args.agent) if args.agent else None)
            report.write(res, args.out)
            print(f"wrote {args.out}/results.json, RESULTS.md and figures")
        elif args.cmd == "reproduce":
            old = json.loads(args.against.read_text())
            new = json.loads(json.dumps(report.run_all(args.data)))
            diffs = report.compare(new, old)
            if new["environment"] != old["environment"]:
                print("note: library versions differ from the published run")
                print("  published:", old["environment"])
                print("  here:     ", new["environment"])
            if diffs:
                print(f"{len(diffs)} differences from published results:")
                for d in diffs[:40]:
                    print("  " + d)
                return 1
            print("reproduced: every published number matches")
    except data.DataError as e:
        print(f"data error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
