"""Run the frozen waveform experiment with a fresh process for each fit/predict.

Only process lifetime and resource supervision change. Original sources,
configuration, model identities and completed artifacts remain unchanged.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

from .contracts import content_id, json_text, read_json
from .research import atomic_json, file_sha256, utc_now

PREFIX = "WAVEFORM_ISOLATED_RESULT "
MAX_BYTES = 10 * 1024**3
MIN_AVAILABLE = 4 * 1024**3


def frozen_inputs(root: Path, run: Path) -> dict:
    root, run = root.resolve(), run.resolve()
    if run.parent != root / "runs":
        raise ValueError("Isolated waveform run must remain in the canonical run directory")
    config = read_json(run / "config.json")
    if run.name != f"waveform-s{config['seed']}-{content_id(config)[-10:]}":
        raise ValueError("Isolated waveform original config identity changed")
    sources = dict(config["implementation"]["source_sha256"], **config["controller_sources"])
    sources[config["implementation"]["runtime_lock_path"]] = config["implementation"]["runtime_lock_sha256"]
    if read_json(run / "source_snapshot/manifest.json") != {
            "capture_scope": "before fitting any fold", "files": sources}:
        raise ValueError("Isolated waveform original source inventory changed")
    for relative, expected in sources.items():
        source, snapshot = root / relative, run / "source_snapshot" / relative
        if (not source.resolve().is_relative_to(root) or
                not snapshot.resolve().is_relative_to(run / "source_snapshot") or
                file_sha256(source) != expected or file_sha256(snapshot) != expected):
            raise ValueError("Isolated waveform source/runtime snapshot differs")
    return config


def validate_request(request: dict) -> tuple[Path, Path, dict]:
    root, run = Path(request["root"]), Path(request["run_dir"])
    config = frozen_inputs(root, run)
    fold = request.get("fold_id")
    if (type(fold) is not int or not 0 <= fold < 5 or
            request.get("seed") != config["seed"] or
            request.get("original_config_hash") != content_id(config) or
            request.get("supervisor_sha256") != file_sha256(Path(__file__)) or
            request.get("action") not in ("fit", "predict") or
            request.get("checkpoint") != str(run / f"fold-{fold}" / "model.pt")):
        raise ValueError("Isolated waveform request differs from the frozen fold")
    if request["action"] == "predict":
        recording = request.get("recording_id")
        if (type(recording) is not str or not recording.isalnum() or
                request.get("output") != str(run / f"fold-{fold}" / "predictions" / (recording + ".npz"))):
            raise ValueError("Isolated waveform prediction output escapes its fold")
    return root, run, config


def parent_lease(root: Path, seed: int, request_path: Path):
    import psutil
    item = read_json(root / "runs/compute.lock")
    parent = psutil.Process(os.getppid())
    if item.get("pid") != parent.pid:
        expected = [sys.executable, "-m", "sleepedf.waveform_isolated", "--worker", str(request_path)]
        if Path(parent.exe()).resolve() != Path(sys.executable).resolve() or parent.cmdline() != expected:
            raise RuntimeError("Isolated waveform parent is not its exact Windows venv launcher")
        parent = parent.parent()
        if parent is None:
            raise RuntimeError("Isolated waveform launcher lost its lease-owning parent")
    if (item.get("host") != socket.gethostname() or item.get("pid") != parent.pid or
            item.get("process_start") != parent.create_time() or
            item.get("run_id") != f"waveform-controller-seed-{seed}"):
        raise RuntimeError("Isolated waveform worker requires its verified parent's live lease")
    return parent


def worker(request_path: Path) -> dict:
    from . import waveform
    import psutil
    request = read_json(request_path)
    root, _, _ = validate_request(request)
    parent = parent_lease(root, request["seed"], request_path)
    original_guard = waveform._check_host_memory

    def guard():
        parent_lease(root, request["seed"], request_path)
        own = psutil.Process().memory_info().rss
        combined = own + parent.memory_info().rss
        available = psutil.virtual_memory().available
        if combined > MAX_BYTES or available < MIN_AVAILABLE:
            raise RuntimeError(f"Waveform memory safeguard: combined_rss={combined}, available={available}")
        return original_guard()

    waveform._check_host_memory = guard
    guard()
    if request["action"] == "fit":
        return waveform.fit_fold(root, request["fold_id"], Path(request["cache_dir"]),
                                 Path(request["checkpoint"]), seed=request["seed"])
    return waveform.predict_fold(root, request["fold_id"], Path(request["checkpoint"]),
                                  request["recording_id"], Path(request["data_root"]),
                                  Path(request["output"]))


def cleanup_processes(owned: dict) -> None:
    """Reap only observed child identities before releasing the compute lease."""
    import psutil
    live = []
    for pid, started in owned.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == started:
                live.append(process)
        except psutil.NoSuchProcess:
            pass
    for process in reversed(live):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(live, timeout=3)
    for process in remaining:
        process.kill()
    _, remaining = psutil.wait_procs(remaining, timeout=3)
    if remaining:
        raise RuntimeError("Isolated waveform child cleanup did not complete")


def dispatch(root: Path, data_root: Path, cache_dir: Path, checkpoint: Path,
             fold_id: int, seed: int, action: str, *, recording_id=None, output=None) -> dict:
    import psutil
    run = checkpoint.resolve().parent.parent
    config = frozen_inputs(root, run)
    supervisor_hash = file_sha256(Path(__file__))
    orchestration = run / "orchestration" / supervisor_hash
    source = orchestration / "waveform_isolated.py"
    orchestration.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        with source.open("xb") as stream:
            stream.write(Path(__file__).read_bytes())
    if file_sha256(source) != supervisor_hash:
        raise ValueError("Isolated waveform supervisor snapshot changed")
    atomic_json(orchestration / "manifest.json", {
        "artifact_type": "process_lifetime_adapter", "original_config_hash": content_id(config),
        "supervisor_sha256": supervisor_hash,
        "change": "fresh child per fit or prediction; combined parent-child memory guard",
        "scientific_sources_changed": False}, immutable=True)
    attempt = orchestration / (action + "-" + uuid.uuid4().hex)
    attempt.mkdir()
    request = {"action": action, "root": str(root.resolve()), "run_dir": str(run),
               "seed": seed, "fold_id": fold_id, "checkpoint": str(checkpoint.resolve()),
               "cache_dir": str(cache_dir.resolve()), "data_root": str(data_root.resolve()),
               "original_config_hash": content_id(config), "supervisor_sha256": supervisor_hash,
               "recording_id": recording_id, "output": str(output.resolve()) if output else None}
    validate_request(request)
    request_path = attempt / "request.json"
    atomic_json(request_path, request, immutable=True)
    command = [sys.executable, "-m", "sleepedf.waveform_isolated", "--worker", str(request_path)]
    started = time.monotonic()
    peak = 0
    lowest_available = None
    error = None
    exit_code = None
    process = None
    owned = {}
    with (attempt / "stdout.log").open("wb") as stream:
        try:
            process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            print(f"isolated waveform {action} fold {fold_id} child {process.pid}", flush=True)
            while True:
                try:
                    parent = psutil.Process()
                    # Windows venv launchers may create the actual interpreter as a child.
                    children = parent.children(recursive=True)
                    for child in children:
                        owned[child.pid] = child.create_time()
                    combined = parent.memory_info().rss + sum(child.memory_info().rss for child in children)
                    available = psutil.virtual_memory().available
                    peak = max(peak, combined)
                    lowest_available = available if lowest_available is None else min(lowest_available, available)
                    if combined > MAX_BYTES or available < MIN_AVAILABLE:
                        raise RuntimeError(f"Waveform memory safeguard: combined_rss={combined}, available={available}")
                except psutil.NoSuchProcess:
                    pass
                if process.poll() is not None:
                    break
                time.sleep(1)
            exit_code = process.returncode
            survivors = []
            for pid, birth in owned.items():
                try:
                    child = psutil.Process(pid)
                    if child.create_time() == birth:
                        survivors.append(child)
                except psutil.NoSuchProcess:
                    pass
            _, survivors = psutil.wait_procs(survivors, timeout=1)
            if survivors:
                raise RuntimeError("Waveform launcher exited while an observed descendant remains active")
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            if process is not None:
                if process.poll() is None:
                    try:
                        child = psutil.Process(process.pid)
                        owned[child.pid] = child.create_time()
                        for descendant in child.children(recursive=True):
                            owned[descendant.pid] = descendant.create_time()
                    except psutil.NoSuchProcess:
                        pass
                cleanup_processes(owned)
                process.wait()
            raise
        finally:
            atomic_json(attempt / "execution.json", {
                "command": command, "recorded_at": utc_now(),
                "exit_code": process.returncode if process is not None else None,
                "error": error, "seconds": time.monotonic() - started,
                "peak_combined_rss_bytes": peak, "minimum_available_bytes": lowest_available,
                "request_sha256": file_sha256(request_path)}, immutable=True)
    text = (attempt / "stdout.log").read_text(encoding="utf-8", errors="replace")
    if exit_code != 0:
        raise RuntimeError(f"Isolated waveform {action} failed; inspect {attempt}")
    lines = [line[len(PREFIX):] for line in text.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise ValueError("Isolated waveform child has no unique result")
    import json
    result = json.loads(lines[0])
    json_text(result)
    return result


def run(root: Path, data_root: Path, cache_dir: Path, seed: int) -> dict:
    from . import waveform, waveform_experiment
    original_fit, original_predict = waveform.fit_fold, waveform.predict_fold
    waveform.fit_fold = lambda r, f, c, p, seed=seed: dispatch(r, data_root, c, p, f, seed, "fit")
    waveform.predict_fold = lambda r, f, p, record, data, output: dispatch(
        r, data, cache_dir, p, f, seed, "predict", recording_id=record, output=output)
    try:
        return waveform_experiment.run(root, data_root, cache_dir, seed)
    finally:
        waveform.fit_fold, waveform.predict_fold = original_fit, original_predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=Path("sleep-edf-database-expanded-1.0.0"))
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/waveform/cache"))
    parser.add_argument("--seed", type=int, choices=(17, 43, 101), default=17)
    args = parser.parse_args()
    if args.worker:
        import json
        print(PREFIX + json.dumps(worker(args.worker), sort_keys=True, allow_nan=False), flush=True)
    else:
        print(json_text(run(args.root.resolve(), args.data_root.resolve(), args.cache_dir.resolve(), args.seed)))


if __name__ == "__main__":
    main()
