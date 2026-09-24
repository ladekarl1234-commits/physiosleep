"""Local research artifacts and a single compute lease; no remote services."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import socket
import tempfile

from .contracts import content_id, json_text, read_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict, *, immutable: bool = False) -> None:
    path = Path(path).resolve()
    if any(p.name == "sleep-edf-database-expanded-1.0.0" for p in path.parents):
        raise ValueError("Cannot write inside original dataset")
    if immutable and path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable artifact already exists: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json_text(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def append_event(path: Path, event: dict) -> None:
    """Append one event; callers serialize writes with the compute/evaluator lease."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(event, recorded_at=utc_now())
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def compute_lease(root: Path, run_id: str):
    import psutil
    import shutil
    lease = root / "runs" / "compute.lock"
    lease.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 20 * 1024**3:
        raise RuntimeError("Fewer than 20 GiB disk free; preserve artifacts and pause")
    if psutil.virtual_memory().available < 4 * 1024**3:
        raise RuntimeError("Fewer than 4 GiB host memory available; pause")
    evidence = {"pid": os.getpid(), "process_start": psutil.Process().create_time(),
                "host": socket.gethostname(), "run_id": run_id, "started_at": utc_now()}
    if lease.exists():
        old = read_json(lease)
        try:
            active = (old.get("host") == evidence["host"] and
                      psutil.Process(old["pid"]).create_time() == old["process_start"])
        except (psutil.NoSuchProcess, KeyError):
            active = False
        if active or old.get("host") != evidence["host"]:
            raise RuntimeError("Compute lease is owned by another active/unknown process")
        append_event(root / "runs" / "scheduler.jsonl", {"event": "recover_stale_lease", "lease": old})
        lease.unlink()
    with lease.open("x", encoding="utf-8") as handle:
        handle.write(json_text(evidence))
    try:
        yield evidence
    finally:
        if lease.exists() and read_json(lease) == evidence:
            lease.unlink()


def artifact_id(value: dict, key: str) -> str:
    return content_id({k: v for k, v in value.items() if k != key})
