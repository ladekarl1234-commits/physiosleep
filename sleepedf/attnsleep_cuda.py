"""Separate, development-only AttnSleep CUDA fit and OOF controller.

The isolated Python 3.7 worker owns all Torch/model operations. This parent
uses only verified development caches and enforces one bounded compute lease.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
import time
import uuid

import numpy as np

from . import attnsleep_baseline as native
from . import attn37_cuda_verify as cuda_verify
from .contracts import CLASS_ORDER, content_id, json_text, read_json
from .evaluation import evaluate_saved_records
from .predictions import load_prediction, save_prediction
from .protocol import development_records, load_development_truth
from .research import append_event, atomic_json, compute_lease, file_sha256

BASE = Path("runs/attnsleep-cuda-worker-verify")
PROBE_STAGES = ("continuous", "resume", "compare", "profile")
PREFIX = "ATTN37_CUDA_WORKER "
MAX_HOST = 10 * 1024**3
MIN_HOST_FREE = 4 * 1024**3
MAX_PROBE_SECONDS = 1800
MAX_FIT_SECONDS = 24 * 3600
SEEDS = native.SEEDS
SOURCE_FILES = ("sleepedf/attnsleep_cuda.py", "tools/attn37_cuda_worker.py",
                "tools/verify_attn37_native.py",
                "tools/attn37_worker.py", "sleepedf/attnsleep_baseline.py",
                "sleepedf/attn37_cuda_verify.py", "tools/verify_attn37_cuda.py",
                "sleepedf/protocol.py", "sleepedf/contracts.py", "sleepedf/research.py",
                "sleepedf/predictions.py", "sleepedf/evaluation.py",
                "sleepedf/splits.py", "sleepedf/dataset.py", "sleepedf/readers.py",
                "sleepedf/timing.py", "sleepedf/audit.py")
CPU_HELPERS = ("canonical_records", "train_samples", "batch", "checkpoint",
               "encoded_epoch", "log_recovery", "log_epoch", "publish_final")


def _strict_json(text: str) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate CUDA worker evidence key")
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite CUDA metadata")))
    json_text(value)
    if type(value) is not dict:
        raise ValueError("CUDA worker evidence must be an object")
    return value


def _parse(path: Path) -> dict:
    lines = [line[len(PREFIX):] for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
             if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise ValueError("CUDA worker did not emit one complete result")
    return _strict_json(lines[0])


def _snapshot(root: Path) -> dict:
    generic = cuda_verify.require_verification(root)
    runtime = cuda_verify.snapshot(root)
    files = {name: file_sha256(root / name) for name in SOURCE_FILES}
    return {"schema_version": "1.0", "artifact_type": "attnsleep_cuda_worker_verification_inputs",
            "scope": "synthetic_only_no_edf_truth_or_scientific_checkpoint",
            "generic_cuda_verification": generic, "cuda_runtime": runtime,
            "source_sha256": files, "batch_size": 128, "probe_batch_size": 3,
            "precision": "float32", "threads": 4, "max_host_bytes": MAX_HOST,
            "max_seconds": MAX_PROBE_SECONDS}


def _supervise(child: subprocess.Popen, started: float, seconds: int) -> dict:
    import psutil
    observed: dict[int, float] = {}
    peak = 0
    try:
        while True:
            try:
                parent = psutil.Process(child.pid)
                for process in [parent] + parent.children(recursive=True):
                    observed[process.pid] = process.create_time()
            except psutil.NoSuchProcess:
                pass
            live = []
            for pid, birth in observed.items():
                try:
                    process = psutil.Process(pid)
                    if process.create_time() == birth and process.is_running():
                        live.append(process)
                except psutil.NoSuchProcess:
                    pass
            rss = psutil.Process().memory_info().rss + sum(p.memory_info().rss for p in live)
            peak = max(peak, rss)
            if (rss > MAX_HOST or psutil.virtual_memory().available < MIN_HOST_FREE or
                    time.monotonic() - started > seconds):
                raise TimeoutError("AttnSleep CUDA worker exceeded host memory or invocation deadline")
            if child.poll() is not None and not live:
                child.wait()
                return {"peak_combined_rss_bytes": peak,
                        "elapsed_seconds": time.monotonic() - started}
            time.sleep(.25)
    finally:
        survivors = []
        for pid, birth in observed.items():
            try:
                process = psutil.Process(pid)
                if process.create_time() == birth and process.is_running():
                    survivors.append(process)
                    process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, remaining = psutil.wait_procs(survivors, timeout=3)
        for process in remaining:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(remaining, timeout=3)
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)


def _command(root: Path, action: str, *arguments: str) -> list[str]:
    return [str(root / ".venvs/attn37-cuda/python.exe"), "-I", "-B",
            str(root / "tools/attn37_cuda_worker.py"), action, *arguments]


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       NUMEXPR_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1")
    return environment


def _require_lease(root: Path, run_id: str) -> None:
    import psutil
    lease = read_json(root / "runs/compute.lock")
    process = psutil.Process()
    if (lease.get("pid") != process.pid or
            lease.get("process_start") != process.create_time() or
            lease.get("host") != socket.gethostname() or
            lease.get("run_id") != run_id):
        raise RuntimeError("AttnSleep CUDA requires the current parent's exact live compute lease")


def _launch(root: Path, action: str, log: Path, started: float, seconds: int,
            *arguments: str) -> tuple[dict, dict]:
    with log.open("xb") as stream:
        child = subprocess.Popen(_command(root, action, *arguments), cwd=root,
                                 env=_environment(), stdout=stream, stderr=subprocess.STDOUT)
        monitor = _supervise(child, started, seconds)
    if child.returncode != 0:
        raise RuntimeError("AttnSleep CUDA worker failed: " + str(log))
    return _parse(log), monitor


def _hash(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _trace(value: object) -> None:
    if (type(value) is not dict or set(value) != {"order", "loss", "lr"} or
            type(value["order"]) is not list or
            any(type(index) is not int for index in value["order"]) or
            sorted(value["order"]) != [0, 1, 2] or
            type(value["loss"]) not in (int, float) or not math.isfinite(value["loss"]) or
            value["loss"] < 0 or value["lr"] not in (.001, .0001)):
        raise ValueError("AttnSleep CUDA worker synthetic step trace is invalid")


def _check_probe(results: dict, attempt: Path, probe_id: str, hardware: dict) -> dict:
    if type(results) is not dict or set(results) != set(PROBE_STAGES):
        raise ValueError("AttnSleep CUDA worker probe stages are incomplete")
    fields = {"continuous": {"traces", "middle_sha256", "continuous_sha256"},
              "resume": {"traces", "checkpoint_sha256"},
              "compare": {"exact_resume", "traces"},
              "profile": {"batch_size", "steps", "warmup_seconds", "update_seconds",
                          "peak_reserved_bytes", "last_loss"}}
    common = {"artifact_type", "stage", "probe_id", "hardware"}
    for stage, row in results.items():
        if (type(row) is not dict or set(row) != common | fields[stage] or
                row["artifact_type"] != "attnsleep_cuda_worker_probe_stage" or
                row["stage"] != stage or row["probe_id"] != probe_id or
                type(row["hardware"]) is not dict or
                type(row["hardware"].get("free_mib")) is not int or
                not 512 <= row["hardware"]["free_mib"] <= hardware["total_mib"] or
                {key: value for key, value in row["hardware"].items() if key != "free_mib"} !=
                {key: value for key, value in hardware.items() if key != "free_mib"}):
            raise ValueError("AttnSleep CUDA worker probe identity or hardware differs")
    continuous, resume, compare, profile = (results[name] for name in PROBE_STAGES)
    for row in (continuous, resume, compare):
        if type(row["traces"]) is not list or len(row["traces"]) != 2:
            raise ValueError("AttnSleep CUDA worker must compare both training steps")
        for trace in row["traces"]:
            _trace(trace)
    if (continuous["traces"] != resume["traces"] or resume["traces"] != compare["traces"] or
            compare["exact_resume"] is not True or
            continuous["traces"][0]["lr"] != .001 or
            continuous["traces"][1]["lr"] != .0001):
        raise ValueError("AttnSleep CUDA worker fresh-process resume differs")
    for filename, digest in (("middle.pt", continuous["middle_sha256"]),
                             ("continuous.pt", continuous["continuous_sha256"]),
                             ("resumed.pt", resume["checkpoint_sha256"])):
        if not _hash(digest) or file_sha256(attempt / filename) != digest:
            raise ValueError("AttnSleep CUDA worker synthetic checkpoint changed")
    if (type(profile["batch_size"]) is not int or profile["batch_size"] != 128 or
            type(profile["steps"]) is not int or profile["steps"] != 3 or
            type(profile["update_seconds"]) is not list or len(profile["update_seconds"]) != 2 or
            any(type(value) not in (int, float) or not math.isfinite(value) or
                not 0 <= value <= MAX_PROBE_SECONDS for value in
                [profile["warmup_seconds"], *profile["update_seconds"]]) or
            type(profile["last_loss"]) not in (int, float) or
            not math.isfinite(profile["last_loss"]) or profile["last_loss"] < 0 or
            type(profile["peak_reserved_bytes"]) is not int or
            not 0 < profile["peak_reserved_bytes"] <= (hardware["total_mib"] - 512) * 1024**2):
        raise ValueError("AttnSleep CUDA worker production batch profile is invalid")
    return {"exact_worker_resume": True, "training_batch_size": 128,
            "last_loss": profile["last_loss"], "peak_reserved_bytes": profile["peak_reserved_bytes"]}


def _profile_timing(profile: dict, monitored_elapsed: object, attempt_elapsed: object) -> None:
    if (type(monitored_elapsed) not in (int, float) or
            type(attempt_elapsed) not in (int, float) or
            not math.isfinite(monitored_elapsed) or not math.isfinite(attempt_elapsed) or
            not 0 <= monitored_elapsed <= attempt_elapsed <= MAX_PROBE_SECONDS or
            profile["warmup_seconds"] + sum(profile["update_seconds"]) > monitored_elapsed + .001):
        raise ValueError("AttnSleep CUDA worker profile timings exceed monitored probe deadline")


def run_worker_preflight(root: Path) -> dict:
    root = root.resolve()
    with compute_lease(root, "attnsleep-cuda-worker-verify"):
        return _run_worker_preflight_locked(root)


def _run_worker_preflight_locked(root: Path) -> dict:
    _require_lease(root, "attnsleep-cuda-worker-verify")
    base = root / BASE
    attempt = base / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
                                   "-" + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    current = base / "current.json"
    atomic_json(current, {"status": "PENDING", "attempt_name": attempt.name})
    began = time.monotonic()
    before = results = summary = after = None
    logs = {}
    status, error = "FAILED", None
    try:
        before = _snapshot(root)
        probe_id = content_id(before)
        hardware = before["generic_cuda_verification"]["hardware"]
        results = {}
        for stage in PROBE_STAGES:
            request = {"artifact_type": "attnsleep_cuda_worker_probe_request",
                       "stage": stage, "probe_id": probe_id, "attempt_dir": str(attempt.resolve()),
                       "snapshot": before, "worker_sha256": before["source_sha256"]["tools/attn37_cuda_worker.py"],
                       "hardware": hardware}
            request_path = attempt / (stage + ".request.json")
            atomic_json(request_path, request, immutable=True)
            log = attempt / (stage + ".log")
            result, monitor = _launch(root, stage, log, began, MAX_PROBE_SECONDS,
                                      "--request", str(request_path.resolve()))
            results[stage] = result
            logs[stage] = {"path": str(log.resolve()), "sha256": file_sha256(log), **monitor}
        summary = _check_probe(results, attempt, probe_id, hardware)
        _profile_timing(results["profile"], logs["profile"]["elapsed_seconds"],
                        time.monotonic() - began)
        after = _snapshot(root)
        if after != before or time.monotonic() - began > MAX_PROBE_SECONDS:
            raise ValueError("AttnSleep CUDA worker inputs changed or probe deadline expired")
        status = "SUCCESS"
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    record = {"artifact_type": "attnsleep_cuda_worker_probe_attempt", "status": status,
              "before": before, "after": after, "results": results, "summary": summary,
              "logs": logs, "elapsed_seconds": time.monotonic() - began, "error": error}
    atomic_json(attempt / "result.json", record, immutable=True)
    if status != "SUCCESS":
        atomic_json(current, {"status": "FAILED", "attempt_name": attempt.name})
        raise RuntimeError("AttnSleep CUDA worker verification failed: " + str(attempt))
    atomic_json(current, {"status": "SUCCESS", "attempt_name": attempt.name,
                          "probe_id": probe_id, "result_sha256": file_sha256(attempt / "result.json")})
    return require_worker_verification(root)


def require_worker_verification(root: Path) -> dict:
    root = root.resolve()
    base = root / BASE
    current = read_json(base / "current.json")
    name = current.get("attempt_name")
    if (current.get("status") != "SUCCESS" or type(name) is not str or
            re.fullmatch(r"[0-9]{8}T[0-9]{12}Z-[0-9a-f]{8}", name) is None):
        raise ValueError("AttnSleep CUDA worker has no current successful synthetic probe")
    attempt = base / "attempts" / name
    if (not attempt.is_dir() or attempt != max((path for path in attempt.parent.iterdir()
                                               if path.is_dir()), default=None)):
        raise ValueError("AttnSleep CUDA worker probe is superseded")
    record_path = attempt / "result.json"
    if file_sha256(record_path) != current.get("result_sha256"):
        raise ValueError("AttnSleep CUDA worker probe receipt changed")
    record = read_json(record_path)
    frozen = _snapshot(root)
    identity = content_id(frozen)
    if (record.get("status") != "SUCCESS" or record.get("before") != frozen or
            record.get("after") != frozen or current.get("probe_id") != identity or
            type(record.get("elapsed_seconds")) not in (int, float) or
            not math.isfinite(record["elapsed_seconds"]) or
            not 0 <= record["elapsed_seconds"] <= MAX_PROBE_SECONDS or
            record.get("error") is not None):
        raise ValueError("AttnSleep CUDA worker synthetic probe is stale or incomplete")
    results, logs = record["results"], record["logs"]
    if type(logs) is not dict or set(logs) != set(PROBE_STAGES):
        raise ValueError("AttnSleep CUDA worker raw stage coverage differs")
    for stage in PROBE_STAGES:
        log = attempt / (stage + ".log")
        item = logs[stage]
        if (type(item) is not dict or set(item) != {"path", "sha256", "peak_combined_rss_bytes",
                                                "elapsed_seconds"} or
                item["path"] != str(log.resolve()) or file_sha256(log) != item["sha256"] or
                _parse(log) != results[stage] or
                type(item["peak_combined_rss_bytes"]) is not int or
                not 0 < item["peak_combined_rss_bytes"] <= MAX_HOST or
                type(item["elapsed_seconds"]) not in (int, float) or
                not math.isfinite(item["elapsed_seconds"]) or
                not 0 <= item["elapsed_seconds"] <= record["elapsed_seconds"]):
            raise ValueError("AttnSleep CUDA worker synthetic stage log or resource bound changed")
    summary = _check_probe(results, attempt, identity, frozen["generic_cuda_verification"]["hardware"])
    _profile_timing(results["profile"], logs["profile"]["elapsed_seconds"],
                    record["elapsed_seconds"])
    if summary != record["summary"]:
        raise ValueError("AttnSleep CUDA worker synthetic summary changed")
    return {"verification_id": identity, "generic_probe_id":
            frozen["generic_cuda_verification"]["probe_id"],
            "qualified_training_batch_size": 128, "synthetic_scope": frozen["scope"]}


def _manifest(root: Path, fold_id: int, cache: Path, labels: Path, seed: int,
              *, frozen_training: list[dict] | None = None) -> dict:
    if type(fold_id) is not int or not 0 <= fold_id < 5 or seed not in SEEDS:
        raise ValueError("AttnSleep CUDA requires one original D fold and registered seed")
    gpu = cuda_verify.require_verification(root)
    qualified = require_worker_verification(root)
    runtime = cuda_verify.snapshot(root)
    protocol, split, records = development_records(root)
    fold = split["folds"][fold_id]
    train_ids, val_ids = set(fold["train"]), set(fold["validation"])
    if (fold["fold_id"] != fold_id or len(train_ids) != 48 or len(val_ids) != 12 or
            train_ids & val_ids or train_ids | val_ids != set(split["participants"]["development"])):
        raise ValueError("AttnSleep CUDA requires original 48/12 D partition")
    train = [record for record in records if record["participant_id"] in train_ids]
    held = [record for record in records if record["participant_id"] in val_ids]
    if ({record["participant_id"] for record in train} != train_ids or
            {record["participant_id"] for record in held} != val_ids):
        raise ValueError("AttnSleep CUDA fold recordings are incomplete")
    if frozen_training is None:
        train_entries = [native._record_entry(record, cache, labels, training=True, split=split)
                         for record in train]
    else:
        if [entry["recording_id"] for entry in frozen_training] != [record["recording_id"] for record in train]:
            raise ValueError("AttnSleep CUDA frozen training record order changed")
        train_entries = [native._training_ancestry(record, cache, labels, entry)
                         for record, entry in zip(train, frozen_training)]
    identity = {"schema_version": "1.0", "artifact_type": "attnsleep_cuda_fit_manifest",
                "protocol_hash": protocol["protocol_hash"], "split_id": split["split_id"],
                "registry_hash": protocol["registry_hash"], "fold_id": fold_id, "seed": seed,
                "epochs": 100, "train_participants": sorted(train_ids),
                "validation_participants": sorted(val_ids), "train": train_entries,
                "validation": [native._record_entry(record, cache, labels, training=False)
                               for record in held],
                "channel": native.CHANNEL, "channel_index": 0, "unit": "uV",
                "rate_hz": 100, "samples_per_epoch": 3000, "class_order": list(CLASS_ORDER),
                "loss": "native_weighted_CrossEntropyLoss_train_only_class_weights",
                "optimizer": {"type": "Adam", "lr": .001, "weight_decay": .001,
                              "amsgrad": True, "lr_after_epoch_10": .0001},
                "selection_rule": "fixed_final_epoch_100", "batch_size_requested": 128,
                "backend": {"precision": "float32", "threads": 4,
                            "cudnn_deterministic": True, "cudnn_benchmark": False},
                "source": native._source(root), "cuda_runtime": runtime,
                "gpu_verification": gpu, "worker_verification_id": qualified["verification_id"],
                "adapter_sha256": file_sha256(Path(__file__)),
                "worker_sha256": file_sha256(root / "tools/attn37_cuda_worker.py"),
                "cpu_helper_sha256": file_sha256(root / "tools/attn37_worker.py"),
                "cpu_helper_allowlist": list(CPU_HELPERS)}
    identity["manifest_id"] = content_id(identity)
    return identity


def _inference_manifest(root: Path, fold_id: int, cache: Path, labels: Path,
                        run_dir: Path, seed: int) -> dict:
    saved = read_json(run_dir / "manifest.json")
    if saved.get("manifest_id") != content_id({k: v for k, v in saved.items() if k != "manifest_id"}):
        raise ValueError("AttnSleep CUDA saved fit identity differs")
    expected = _manifest(root, fold_id, cache, labels, seed, frozen_training=saved["train"])
    if saved != expected:
        raise ValueError("AttnSleep CUDA fit ancestry changed since training")
    return saved


def _worker(root: Path, action: str, manifest_path: Path, run_dir: Path,
            started: float, *arguments: str) -> dict:
    logs = run_dir / "worker_attempts"
    logs.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    log = logs / (action + "-" + name + ".log")
    result, monitor = _launch(root, action, log, started, MAX_FIT_SECONDS,
                              "--manifest", str(manifest_path.resolve()),
                              "--run-dir", str(run_dir.resolve()), *arguments)
    if result.get("manifest_id") != read_json(manifest_path)["manifest_id"]:
        raise ValueError("AttnSleep CUDA worker returned another fit identity")
    append_event(run_dir / "worker_attempts.jsonl",
                 {"event": "worker_completed", "action": action,
                  "log_path": str(log.resolve()), "log_sha256": file_sha256(log),
                  "peak_combined_rss_bytes": monitor["peak_combined_rss_bytes"],
                  "elapsed_seconds": monitor["elapsed_seconds"]})
    return result


def train_or_load(root: Path, fold_id: int, cache: Path, labels: Path,
                  run_dir: Path, seed: int, started: float | None = None) -> dict:
    """Called only inside the matching parent controller compute lease."""
    root, run_dir = root.resolve(), run_dir.resolve()
    _require_lease(root, f"attnsleep-cuda-controller-seed-{seed}")
    native._writable_path(run_dir)
    manifest = _manifest(root, fold_id, cache, labels, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    atomic_json(path, manifest, immutable=True)
    return _worker(root, "train", path, run_dir, started or time.monotonic())


def inspect(root: Path, fold_id: int, cache: Path, labels: Path,
            run_dir: Path, seed: int, started: float | None = None) -> dict:
    _require_lease(root.resolve(), f"attnsleep-cuda-controller-seed-{seed}")
    manifest = _inference_manifest(root, fold_id, cache, labels, run_dir, seed)
    result = _worker(root, "inspect", run_dir / "manifest.json", run_dir,
                     started or time.monotonic())
    if (result.get("completed_epochs") != 100 or result.get("batch_size") != 128 or
            result.get("checkpoint_sha256") != file_sha256(run_dir / "final.pt") or
            result.get("backend") != manifest["backend"]):
        raise ValueError("AttnSleep CUDA final checkpoint is incomplete")
    return result


def predict(root: Path, fold_id: int, cache: Path, labels: Path,
            run_dir: Path, record_id: str, output: Path, seed: int,
            started: float | None = None) -> dict:
    """Full-grid signal-only inference; no Hypnogram or held-out label input."""
    _require_lease(root.resolve(), f"attnsleep-cuda-controller-seed-{seed}")
    native._writable_path(output)
    manifest = _inference_manifest(root, fold_id, cache, labels, run_dir, seed)
    candidates = [entry for entry in manifest["validation"] if entry["recording_id"] == record_id]
    if len(candidates) != 1 or record_id in {entry["recording_id"] for entry in manifest["train"]}:
        raise ValueError("AttnSleep CUDA prediction requires one outer held-out D night")
    with tempfile.TemporaryDirectory(dir=run_dir, prefix="cuda-predict-") as temporary:
        raw_path = Path(temporary) / "raw.npz"
        result = _worker(root, "predict", run_dir / "manifest.json", run_dir,
                         started or time.monotonic(), "--recording-id", record_id,
                         "--output", str(raw_path))
        with np.load(raw_path, allow_pickle=False) as raw:
            hard, probabilities = raw["hard_label"], raw["probabilities"]
        entry = candidates[0]
        if (result.get("recording_id") != record_id or
                result.get("checkpoint_sha256") != file_sha256(run_dir / "final.pt") or
                result.get("raw_sha256") != file_sha256(raw_path) or
                hard.shape != (entry["n_epochs"],) or
                probabilities.shape != (entry["n_epochs"], 5) or
                not np.array_equal(hard, np.argmax(probabilities, axis=1))):
            raise ValueError("AttnSleep CUDA raw prediction coverage or lineage differs")
        provenance = {"candidate": "native_attnsleep_1_4_cuda101", "fold_id": fold_id,
                      "seed": seed, "training_participants": manifest["train_participants"],
                      "checkpoint_sha256": result["checkpoint_sha256"],
                      "fit_manifest_id": manifest["manifest_id"],
                      "source_psg_sha256": entry["source_psg_sha256"],
                      "input_channel": native.CHANNEL, "input_unit": "uV", "sample_rate_hz": 100,
                      "selection_rule": "fixed_final_epoch_100",
                      "gpu_probe_id": manifest["gpu_verification"]["probe_id"],
                      "worker_verification_id": manifest["worker_verification_id"]}
        return save_prediction(output, participant_id=entry["participant_id"],
                               recording_id=record_id,
                               epoch_index=np.arange(entry["n_epochs"]),
                               onset_seconds=np.arange(entry["n_epochs"]) * 30.0,
                               hard_label=hard, probabilities=probabilities,
                               model_id="native-attnsleep-cuda-v1",
                               protocol_hash=manifest["protocol_hash"],
                               registry_hash=manifest["registry_hash"], provenance=provenance)


def _snapshot_sources(root: Path, output: Path, hashes: dict[str, str]) -> None:
    for relative, expected in hashes.items():
        source = root / relative
        target = output / "source_snapshot" / relative
        if file_sha256(source) != expected:
            raise ValueError("AttnSleep CUDA frozen source changed")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if file_sha256(target) != expected:
            raise ValueError("AttnSleep CUDA source snapshot changed")
    atomic_json(output / "source_snapshot/manifest.json", {"files": hashes}, immutable=True)


def _prediction(path: Path, record: dict, manifest: dict, checkpoint_sha256: str) -> None:
    metadata, arrays = load_prediction(path)
    provenance = metadata["provenance"]
    n = record["n_epochs"]
    if (metadata["model_id"] != "native-attnsleep-cuda-v1" or
            metadata["protocol_hash"] != manifest["protocol_hash"] or
            metadata["registry_hash"] != manifest["registry_hash"] or
            arrays["recording_id"] != record["recording_id"] or
            arrays["participant_id"] != record["participant_id"] or
            arrays["participant_id"] not in manifest["validation_participants"] or
            arrays["participant_id"] in manifest["train_participants"] or
            not np.array_equal(arrays["epoch_index"], np.arange(n)) or
            not np.array_equal(arrays["onset_seconds"], np.arange(n) * 30.0) or
            not np.array_equal(arrays["hard_label"], np.argmax(arrays["probabilities"], axis=1)) or
            provenance.get("fit_manifest_id") != manifest["manifest_id"] or
            provenance.get("checkpoint_sha256") != checkpoint_sha256 or
            provenance.get("source_psg_sha256") != record["psg_sha256"] or
            provenance.get("training_participants") != manifest["train_participants"] or
            provenance.get("fold_id") != manifest["fold_id"] or
            provenance.get("seed") != manifest["seed"] or
            provenance.get("input_channel") != native.CHANNEL or
            provenance.get("input_unit") != "uV" or
            provenance.get("sample_rate_hz") != 100 or
            provenance.get("selection_rule") != "fixed_final_epoch_100" or
            provenance.get("gpu_probe_id") != manifest["gpu_verification"]["probe_id"] or
            provenance.get("worker_verification_id") != manifest["worker_verification_id"]):
        raise ValueError("AttnSleep CUDA OOF prediction differs from original timeline or fit")


def run_development(root: Path, cache: Path, labels: Path, seed: int) -> dict:
    root = root.resolve()
    with compute_lease(root, f"attnsleep-cuda-controller-seed-{seed}"):
        return _run_development_locked(root, cache.resolve(), labels.resolve(), seed)


def _run_development_locked(root: Path, cache: Path, labels: Path, seed: int) -> dict:
    if seed not in SEEDS:
        raise ValueError("Use a registered original development seed")
    _require_lease(root, f"attnsleep-cuda-controller-seed-{seed}")
    started = time.monotonic()
    gpu = cuda_verify.require_verification(root)
    worker_receipt = require_worker_verification(root)
    protocol, split, records = development_records(root)
    if [fold["fold_id"] for fold in split["folds"]] != list(range(5)):
        raise ValueError("AttnSleep CUDA requires all five original D folds")
    manifests = [_manifest(root, fold["fold_id"], cache, labels, seed)
                 for fold in split["folds"]]
    sources = {name: file_sha256(root / name) for name in SOURCE_FILES}
    sources.update({"vendor/attnsleep/" + name: digest
                    for name, digest in manifests[0]["source"]["files"].items()})
    config = {"experiment": "native_attnsleep_cuda_v1", "seed": seed,
              "protocol_hash": protocol["protocol_hash"], "split_id": split["split_id"],
              "epochs": 100, "selection": "fixed_final_epoch_100",
              "channel": native.CHANNEL, "fold_manifests": manifests,
              "generic_cuda_verification": gpu,
              "worker_specific_verification": worker_receipt,
              "sources": sources,
              "truth_manifest_sha256": file_sha256(root / "research/development-truth-v1.json")}
    run_id = f"attnsleep-cuda-s{seed}-{content_id(config)[-10:]}"
    output = root / "runs" / run_id
    native._writable_path(output)
    atomic_json(output / "config.json", config, immutable=True)
    _snapshot_sources(root, output, sources)
    predictions, truths, fits = [], [], []
    for fold, manifest in zip(split["folds"], manifests):
        fold_id = fold["fold_id"]
        fold_dir = output / f"fold-{fold_id}"
        if (any(file_sha256(root / name) != digest for name, digest in sources.items()) or
                cuda_verify.require_verification(root) != gpu or
                require_worker_verification(root) != worker_receipt or
                _manifest(root, fold_id, cache, labels, seed) != manifest):
            raise ValueError("AttnSleep CUDA frozen source, input or verification changed")
        trained = train_or_load(root, fold_id, cache, labels, fold_dir, seed, started)
        inspection = inspect(root, fold_id, cache, labels, fold_dir, seed, started)
        checkpoint_hash = file_sha256(fold_dir / "final.pt")
        if any(row.get("manifest_id") != manifest["manifest_id"] or
               row.get("completed_epochs") != 100 or row.get("checkpoint_sha256") != checkpoint_hash
               for row in (trained, inspection)):
            raise ValueError("AttnSleep CUDA final checkpoint differs from complete fold fit")
        fit = {"fold_id": fold_id, "seed": seed, "manifest_id": manifest["manifest_id"],
               "checkpoint_sha256": checkpoint_hash, "completed_epochs": 100,
               "training_participants": manifest["train_participants"],
               "validation_participants": manifest["validation_participants"],
               "selection_rule": "fixed_final_epoch_100", "native_inspection": inspection}
        atomic_json(fold_dir / "fit.json", fit, immutable=True)
        fits.append(fit)
        for record in records:
            if record["participant_id"] not in fold["validation"]:
                continue
            path = fold_dir / "predictions" / (record["recording_id"] + ".npz")
            if not path.exists():
                predict(root, fold_id, cache, labels, fold_dir, record["recording_id"],
                        path, seed, started)
            _prediction(path, record, manifest, checkpoint_hash)
            load_development_truth(record, split)
            predictions.append(path)
            truths.append(Path(record["truth_path"]))
        if file_sha256(fold_dir / "final.pt") != checkpoint_hash:
            raise ValueError("AttnSleep CUDA checkpoint changed during OOF inference")
        append_event(output / "events.jsonl", {"event": "fold_complete", "fold_id": fold_id,
                     "checkpoint_sha256": checkpoint_hash,
                     "elapsed_seconds_this_invocation": time.monotonic() - started})
    if (len(predictions) != len(records) or len(set(predictions)) != len(records) or
            any(file_sha256(root / name) != digest for name, digest in sources.items())):
        raise ValueError("AttnSleep CUDA OOF coverage or frozen source differs")
    _snapshot_sources(root, output, sources)
    metrics = evaluate_saved_records(truths, predictions, protocol_hash=protocol["protocol_hash"],
                                     registry_hash=protocol["registry_hash"],
                                     model_id="native-attnsleep-cuda-v1")
    result = {"status": "DEVELOPMENT_OOF_COMPLETE", "confirmatory": False,
              "run_id": run_id, "config_hash": content_id(config),
              "protocol_hash": protocol["protocol_hash"], "metrics": metrics,
              "folds": fits, "prediction_paths": [str(path.resolve()) for path in predictions],
              "truth_paths": [str(path.resolve()) for path in truths],
              "gate_A": "NOT_RUN", "gate_B": "NOT_RUN"}
    atomic_json(output / "result.json", result, immutable=True)
    append_event(root / "runs/experiments.jsonl", {"event": "development_run_complete",
                 "run_id": run_id, "report_sha256": file_sha256(output / "result.json"),
                 "macro_f1": metrics["macro_f1"], "confirmatory": False})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify-worker", "run-development"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/attnsleep/eeg-cache"))
    parser.add_argument("--label-dir", type=Path, default=Path("runs/attnsleep/labels"))
    parser.add_argument("--seed", type=int, choices=SEEDS, default=17)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "verify-worker":
        result = run_worker_preflight(root)
    else:
        cache = args.cache_dir if args.cache_dir.is_absolute() else root / args.cache_dir
        labels = args.label_dir if args.label_dir.is_absolute() else root / args.label_dir
        result = run_development(root, cache, labels, args.seed)
    print(json_text(result))


if __name__ == "__main__":
    main()
