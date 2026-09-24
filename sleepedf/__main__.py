"""Local-only command line entry point."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from . import __version__
from .contracts import code_identity, inventory_id, json_text, read_json, validate_split


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def environment_report() -> dict:
    versions = {}
    for name in ("numpy", "pyedflib", "xlrd", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    missing = [name for name in ("numpy", "pyedflib", "xlrd") if versions[name] is None]
    gpu = {"status": "not_detected", "devices": []}
    command = shutil.which("nvidia-smi")
    if command:
        try:
            run = subprocess.run([command, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5, check=False)
            gpu = {"status": "visible" if run.returncode == 0 else "query_failed",
                   "devices": run.stdout.strip().splitlines() if run.returncode == 0 else []}
        except (OSError, subprocess.TimeoutExpired):
            gpu["status"] = "query_failed"
    errors = [{"code": "missing_dependency", "path": "",
               "message": "Audit/smoke dependencies missing: " + ", ".join(missing) +
                          "; run uv sync --frozen --extra audit --group dev"}] if missing else []
    if sys.version_info[:2] != (3, 11):
        errors.append({"code": "unsupported_python", "path": "",
                       "message": "Use Python 3.11 for this locked project environment"})
    return {"python": platform.python_version(), "platform": platform.system(),
            "architecture": platform.machine(), "isolated_environment": sys.prefix != sys.base_prefix,
            "python_supported": sys.version_info[:2] == (3, 11), "dependencies": versions,
            "header_audit_ready": not missing, "missing_audit_dependencies": missing,
            "disk_free_bytes": shutil.disk_usage(Path.cwd()).free, "gpu": gpu,
            "setup_command": "uv sync --frozen --extra audit --group dev",
            "code_identity": code_identity(Path(__file__).resolve().parents[1]),
            "errors": errors, "warnings": []}


def write_report(path: Path, report: dict, data_root: Path | None) -> None:
    path = path.resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("Report output must have a .json extension")
    roots = [data_root] if data_root is not None else []
    configured = os.environ.get("SLEEPEDF_DATA_ROOT")
    if configured:
        roots.append(Path(configured))
    if any(path.is_relative_to(root.resolve()) for root in roots) or any(
        parent.name == "sleep-edf-database-expanded-1.0.0" for parent in (path, *path.parents)
    ):
        raise ValueError("Report output cannot be inside immutable dataset inputs")
    if path.exists():
        old = read_json(path)
        if (not isinstance(old, dict) or old.get("producer") != "sleepedf" or
            old.get("report_kind") != report["report_kind"] or
            old.get("schema_version") != report["schema_version"]):
            raise ValueError("Refusing to overwrite a file that is not a report of this kind")
    text = json_text(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=".sleepedf-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = JsonArgumentParser(description="Local Sleep-EDF readiness tools; no training or network calls")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    env = commands.add_parser("env", help="Check runtime and optional dependencies")
    env.add_argument("--output", type=Path)
    for name in ("audit", "smoke"):
        command = commands.add_parser(name)
        command.add_argument("--data-root", type=Path, default=None)
        command.add_argument("--output", type=Path)
        if name == "audit":
            command.add_argument("--files-only", action="store_true", help="Skip EDF header reads explicitly")
            command.add_argument("--verify-checksums", action="store_true")
            command.add_argument("--max-hash-bytes", type=int)
        else:
            command.add_argument("--recording", action="append", required=True)
            command.add_argument("--seconds", type=float, default=60)
    split = commands.add_parser("validate-split", help="Validate an existing participant split; never generate one")
    split.add_argument("--split", type=Path, required=True)
    split.add_argument("--inventory", type=Path, required=True)
    split.add_argument("--output", type=Path)
    raw_args = sys.argv[1:] if argv is None else argv
    if "--help" in raw_args or "-h" in raw_args:
        command_name = next((arg for arg in raw_args if arg in commands.choices), None)
        help_parser = commands.choices.get(command_name, parser)
        print(json_text({"help": help_parser.format_help(), "tool_version": __version__}), end="")
        return 0
    if "--version" in raw_args:
        print(json_text({"tool_version": __version__}), end="")
        return 0
    root = None
    try:
        args = parser.parse_args(raw_args)
        if args.command in ("audit", "smoke"):
            from .audit import audit_dataset
            raw_root = args.data_root or os.environ.get("SLEEPEDF_DATA_ROOT")
            if not raw_root:
                raise ValueError("Provide --data-root or set SLEEPEDF_DATA_ROOT")
            root = Path(raw_root).resolve()
            if not root.is_dir():
                raise ValueError("Dataset root is not an existing directory")
            if args.command == "audit":
                if args.max_hash_bytes is not None and (args.max_hash_bytes < 0 or not args.verify_checksums):
                    raise ValueError("--max-hash-bytes requires --verify-checksums and a nonnegative bound")
                report = audit_dataset(root, headers=not args.files_only,
                                       verify_checksums=args.verify_checksums,
                                       max_hash_bytes=args.max_hash_bytes)
            else:
                from .smoke import smoke_recordings
                if len(args.recording) > 2 or len(set(args.recording)) != len(args.recording):
                    raise ValueError("Select at most two distinct recordings")
                inventory = audit_dataset(root, headers=False)
                by_id = {r["recording_id"]: r for r in inventory["recordings"]}
                if any(r not in by_id for r in args.recording):
                    raise ValueError("Recording ID was not found in inventory")
                report = smoke_recordings(root, [by_id[r] for r in args.recording], args.seconds)
        elif args.command == "env":
            report = environment_report()
        else:
            inventory = read_json(args.inventory)
            if isinstance(inventory, dict) and isinstance(inventory.get("scope"), dict):
                raw_root = inventory["scope"].get("data_root")
                if isinstance(raw_root, str) and raw_root:
                    root = Path(raw_root)
            report = validate_split(read_json(args.split), inventory)
            report.update(errors=[], warnings=[])
        report.update(schema_version="1.0", tool_version=__version__, report_kind=args.command,
                      producer="sleepedf")
        if args.command == "audit":
            report["manifest_id"] = inventory_id(report)
        if args.output:
            write_report(args.output, report, root)
            summary = report.get("summary", {"recordings": len(report.get("recordings", []))})
            print(json_text({"report": str(args.output), "summary": summary,
                             "errors": len(report.get("errors", [])), "warnings": len(report.get("warnings", []))}), end="")
        else:
            print(json_text(report), end="")
        if any(e.get("code") == "missing_dependency" for e in report.get("errors", [])):
            return 2
        return 1 if report.get("errors") else 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(json_text({"error": str(exc), "report_kind": getattr(locals().get("args"), "command", None),
                         "tool_version": __version__}), file=sys.stderr, end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
