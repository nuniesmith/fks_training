from __future__ import annotations
import argparse
from runtime import list_apps, run_app  # type: ignore
import apps  # noqa: F401  (ensure registrations)

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fks-shared", description="FKS Shared (worker phase) CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list-apps", help="List registered apps")
    rp = sub.add_parser("run", help="Run an app by name")
    rp.add_argument("--app", required=True, help="Application name (e.g. worker)")
    return p

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "list-apps":
        for name in list_apps():
            print(name)
        return 0
    if args.cmd == "run":
        run_app(args.app)
        return 0
    parser.print_help()
    return 1

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
