"""Bounded, resumable D-only U-Time signal and label preparation.

The numerical producer is sleepedf.tf_native.prepare_development. This module
only binds its inputs, supervises one child, and verifies exported artifacts.
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
import subprocess
import time
import uuid

import numpy as np

from .contracts import content_id, json_text, read_json
from .protocol import development_records, load_development_truth
from .research import atomic_json, compute_lease, file_sha256
from .tf_native import _preprocess_identity, load_signal_cache
from .tf_preprocessing_verify import (_monitor, _parent_resource_bound,
                                      _worker_identity, require_verification as require_parity)
from .tf_verify import require_verification as require_native


BASE = "runs/utime/full-record-v2"
LEASE = "utime-full-record-v2-prepare"
WORKER_PREFIX = "UTIME_PREPARE_RESULT "
MAX_SECONDS = 1800
THREADS = 4
SOURCES = ("sleepedf/tf_prepare.py", "tools/tf_prepare_worker.py",
           "tools/verify_tf_preprocessing.py",
           "docs/NATIVE_TF_PREPARE.md", "sleepedf/tf_native.py",
           "sleepedf/tf_preprocessing_verify.py", "sleepedf/tf_verify.py",
           "sleepedf/protocol.py", "sleepedf/predictions.py", "sleepedf/splits.py",
           "sleepedf/dataset.py", "sleepedf/readers.py", "sleepedf/contracts.py",
           "sleepedf/research.py")


def _paths(root: Path, data_root: Path) -> tuple[Path, Path]:
    root, data_root = root.resolve(), data_root.resolve()
    if not data_root.is_dir() or data_root == root or root.is_relative_to(data_root):
        raise ValueError("Explicit original data root is missing or includes the project")
    base = root / BASE
    cache, labels = base / "eeg128-cache", base / "labels"
    write_targets = (root / "runs", root / "runs/compute.lock",
                     root / "runs/scheduler.jsonl", base, base / "attempts",
                     base / "receipts", base / "current.json", cache, labels)
    if any(path.resolve().is_relative_to(data_root) for path in write_targets):
        raise ValueError("Preparation output path enters original data")
    return cache, labels


def _records(root: Path) -> tuple[dict, dict, list[dict], list[dict]]:
    protocol, split, records = development_records(root)
    participants = split["participants"]["development"]
    ids = [r["recording_id"] for r in records]
    if (len(participants) != 60 or len(set(participants)) != 60 or
            len(records) != 119 or len(set(ids)) != 119 or
            {r["participant_id"] for r in records} != set(participants)):
        raise ValueError("U-Time preparation requires exactly 60 D participants and 119 recordings")
    rows = []
    for rec in records:
        frozen = rec.get("frozen_truth")
        duration = rec.get("duration_seconds")
        if (re.fullmatch(r"(?:SC|ST)[0-9]{4}", rec["recording_id"]) is None or
                rec.get("checks", {}).get("signals_verified") is not True or
                re.fullmatch(r"[0-9a-f]{64}", str(rec.get("psg_sha256"))) is None or
                type(rec.get("n_epochs")) is not int or rec["n_epochs"] < 1 or
                type(duration) not in (int, float) or not math.isfinite(duration) or
                int(duration // 30) != rec["n_epochs"] or
                type(frozen) is not dict or
                any(re.fullmatch(r"[0-9a-f]{64}", str(frozen.get(key))) is None
                    for key in ("payload_sha256", "sidecar_sha256"))):
            raise ValueError("D recording has unresolved grid, signal or frozen truth ancestry")
        rows.append({"recording_id": rec["recording_id"],
                     "participant_id": rec["participant_id"], "n_epochs": rec["n_epochs"],
                     "duration_seconds": duration, "psg_sha256": rec["psg_sha256"],
                     "truth_payload_sha256": frozen["payload_sha256"],
                     "truth_sidecar_sha256": frozen["sidecar_sha256"]})
    return protocol, split, records, sorted(rows, key=lambda row: row["recording_id"])


def _snapshot(root: Path, data_root: Path, *, started: float) -> dict:
    _parent_resource_bound(root, started)
    cache, labels = _paths(root, data_root)
    protocol, split, _records_in, rows = _records(root)
    native = require_native(root)
    parity = require_parity(root, data_root)
    parity_rows = [{**item, "receipt_sha256": file_sha256(Path(item["receipt_path"]))}
                   for item in parity["records"]]
    preflight_currents = {mode: file_sha256(root / "runs/utime-native-preflight" / mode / "current.json")
                          for mode in ("numerical", "profile")}
    # The preflight verifier checks exact installed packages and its source tree.
    result = {"schema_version": "1.0", "artifact_type": "utime_full_record_prepare_inputs",
              "scope": "development_60_participants_119_recordings_only",
              "data_root": str(data_root.resolve()), "cache_dir": str(cache.resolve()),
              "label_dir": str(labels.resolve()), "protocol_hash": protocol["protocol_hash"],
              "split_id": split["split_id"], "readiness_sha256": protocol["readiness_sha256"],
              "split_sha256": protocol["split_sha256"],
              "truth_manifest_sha256": file_sha256(root / "research/development-truth-v1.json"),
              "recordings": rows, "recordings_id": content_id(rows),
              "preprocess_identity": _preprocess_identity(),
              "native_preflight": native, "native_preflight_currents_sha256": preflight_currents,
              "two_record_parity_id": parity["identity"], "two_record_receipts": parity_rows,
              "sources_sha256": {name: file_sha256(root / name) for name in SOURCES},
              "runtime_lock_sha256": file_sha256(root / "requirements/utime.lock.txt"),
              "runtime_install_sha256": file_sha256(root / "research/runtimes/utime/install.json"),
              "native_executable_sha256": file_sha256(root / ".venvs/utime/Scripts/python.exe"),
              "max_seconds": MAX_SECONDS, "threads": THREADS,
              "max_combined_rss_bytes": 10 * 1024**3,
              "min_free_ram_bytes": 4 * 1024**3,
              "min_free_disk_bytes": 20 * 1024**3}
    _parent_resource_bound(root, started)
    return result


def _inventory(root: Path, records: list[dict], split: dict, cache: Path,
               labels: Path, *, started: float) -> dict:
    rows = []
    for rec in records:
        _parent_resource_bound(root, started)
        rid, n = rec["recording_id"], rec["n_epochs"]
        meta = load_signal_cache(rec, cache, metadata_only=True)
        signal = np.load(cache / (rid + ".npy"), mmap_mode="r", allow_pickle=False)
        if signal.shape != (n, 3840, 1) or signal.dtype != np.float32 or not np.isfinite(signal).all():
            raise ValueError("Prepared signal is not finite float32 on the original grid")
        truth = load_development_truth(rec, split)
        label_path = labels / (rid + ".npy")
        exported = np.load(label_path, allow_pickle=False)
        reference = np.asarray(truth["reference_label"])
        mask = np.asarray(truth["valid_mask"])
        if (exported.shape != (n,) or exported.dtype != np.int8 or
                reference.shape != (n,) or mask.shape != (n,) or
                not np.issubdtype(reference.dtype, np.integer) or mask.dtype != np.bool_ or
                not np.array_equal(truth["epoch_index"], np.arange(n)) or
                not np.array_equal(truth["onset_seconds"], np.arange(n) * 30.0) or
                not np.array_equal(exported, np.where(mask, reference, -1).astype(np.int8)) or
                np.any(mask & ((reference < 0) | (reference > 4)))):
            raise ValueError("Prepared D labels differ from frozen original-grid truth")
        rows.append({"recording_id": rid, "participant_id": rec["participant_id"],
                     "n_epochs": n, "valid_epochs": int(np.count_nonzero(mask)),
                     "invalid_epochs": int(n - np.count_nonzero(mask)),
                     "signal_meta_sha256": file_sha256(cache / (rid + ".json")),
                     "signal_payload_sha256": meta["payload_sha256"],
                     "label_sha256": file_sha256(label_path),
                     "truth_payload_sha256": rec["frozen_truth"]["payload_sha256"],
                     "truth_sidecar_sha256": rec["frozen_truth"]["sidecar_sha256"]})
    rows.sort(key=lambda row: row["recording_id"])
    ids = {row["recording_id"] for row in rows}
    expected_cache = {rid + suffix for rid in ids for suffix in (".npy", ".json")}
    expected_labels = {rid + ".npy" for rid in ids}
    if ({path.name for path in cache.iterdir()} != expected_cache or
            {path.name for path in labels.iterdir()} != expected_labels):
        raise ValueError("Prepared directories contain missing or extra recording artifacts")
    return {"recordings": rows, "recordings_id": content_id(rows),
            "recording_count": len(rows),
            "complete_epochs": sum(row["n_epochs"] for row in rows),
            "valid_epochs": sum(row["valid_epochs"] for row in rows)}


def _worker_result(log: Path) -> dict:
    lines = [line[len(WORKER_PREFIX):] for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
             if line.startswith(WORKER_PREFIX)]
    if len(lines) != 1:
        raise ValueError("Preparation worker emitted no unique result")
    def unique(pairs):
        item = {}
        for key, value in pairs:
            if key in item:
                raise ValueError("Duplicate preparation result metadata key")
            item[key] = value
        return item
    result = json.loads(lines[0], object_pairs_hook=unique,
                        parse_constant=lambda _: (_ for _ in ()).throw(
        ValueError("Non-finite preparation result")))
    if type(result) is not dict:
        raise ValueError("Preparation worker result is not an object")
    json_text(result)
    return result


def run(root: Path, data_root: Path) -> dict:
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    _paths(root, data_root)
    with compute_lease(root, LEASE):
        return _run_locked(root, data_root)


def _run_locked(root: Path, data_root: Path) -> dict:
    _paths(root, data_root)
    base = root / BASE
    attempt = base / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
                                   "-" + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    current = base / "current.json"
    atomic_json(current, {"status": "PENDING", "attempt_path": str((attempt / "attempt.json").resolve())})
    started = time.monotonic()
    before = after = result = monitor = inventory = invocation = None
    failure = None
    try:
        before = _snapshot(root, data_root, started=started)
        request = {"schema_version": "1.0", "artifact_type": "utime_full_record_prepare_request",
                   "root": str(root), "data_root": str(data_root),
                   "attempt_dir": str(attempt), "snapshot": before,
                   "request_id": content_id(before), "lease_run_id": LEASE}
        atomic_json(attempt / "request.json", request, immutable=True)
        log_path = attempt / "worker.log"
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   NUMEXPR_NUM_THREADS="4", TF_NUM_INTRAOP_THREADS="4",
                   TF_NUM_INTEROP_THREADS="4", CUDA_VISIBLE_DEVICES="-1")
        lease = read_json(root / "runs/compute.lock")
        env["UTIME_LEASE_PARENT_PID"] = str(lease["pid"])
        env["UTIME_LEASE_PARENT_BIRTH"] = str(lease["process_start"])
        command = [str(root / ".venvs/utime/Scripts/python.exe"), "-I",
                   str(root / "tools/tf_prepare_worker.py"), "--request", str(attempt / "request.json")]
        invocation = {"command": command, "request_sha256": file_sha256(attempt / "request.json"),
                      "thread_environment": {key: env[key] for key in (
                          "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                          "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
                          "TF_NUM_INTEROP_THREADS", "CUDA_VISIBLE_DEVICES")},
                      "exit_code": None}
        _parent_resource_bound(root, started)
        with log_path.open("xb") as output:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=output,
                                     stderr=subprocess.STDOUT)
            monitor = _monitor(child, root, started, log_path)
        invocation["exit_code"] = child.returncode
        after = _snapshot(root, data_root, started=started)
        if after != before or child.returncode != 0:
            raise ValueError("Preparation inputs changed or worker failed; inspect immutable log")
        result = _worker_result(log_path)
        if result != {"artifact_type": "utime_development_prepare", "split_id": before["split_id"],
                      "recordings": 119, "cache_dir": before["cache_dir"],
                      "label_dir": before["label_dir"]}:
            raise ValueError("Worker did not return exact D-only preparation result")
        _, split, records, _ = _records(root)
        inventory = _inventory(root, records, split, Path(before["cache_dir"]),
                               Path(before["label_dir"]), started=started)
        if inventory["recording_count"] != 119:
            raise ValueError("Preparation omitted development recordings")
        _parent_resource_bound(root, started)
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    elapsed = time.monotonic() - started
    if failure is None and elapsed > MAX_SECONDS:
        failure = {"type": "TimeoutError", "message": "Preparation exceeded the per-invocation deadline"}
    log = attempt / "worker.log"
    record = {"schema_version": "1.0", "artifact_type": "utime_full_record_prepare_attempt",
              "status": "SUCCESS" if failure is None else "FAILED", "before": before,
              "after": after, "result": result, "inventory": inventory,
              "monitor": monitor, "invocation": invocation, "failure": failure,
              "elapsed_seconds": elapsed,
              "log_sha256": file_sha256(log) if log.exists() else None}
    atomic_json(attempt / "attempt.json", record, immutable=True)
    if failure is not None:
        atomic_json(current, {"status": "FAILED", "attempt_path": str((attempt / "attempt.json").resolve())})
        raise RuntimeError("U-Time D preparation failed; immutable attempt: " + str(attempt))
    receipt = {"schema_version": "1.0", "artifact_type": "utime_full_record_prepare_receipt",
               "request_id": content_id(before), "attempt_path": str((attempt / "attempt.json").resolve()),
               "attempt_sha256": file_sha256(attempt / "attempt.json")}
    receipt_path = base / "receipts" / (attempt.name + ".json")
    atomic_json(receipt_path, receipt, immutable=True)
    atomic_json(current, {"status": "SUCCESS", "receipt_path": str(receipt_path.resolve()),
                          "receipt_sha256": file_sha256(receipt_path)})
    try:
        return verify_receipt(root, data_root, started=started)
    except BaseException:
        atomic_json(current, {"status": "FAILED", "attempt_path": str((attempt / "attempt.json").resolve())})
        raise


def _receipt_lineage(root: Path, data_root: Path, started: float) -> tuple[dict, Path, dict]:
    snapshot = _snapshot(root, data_root, started=started)
    base = root / BASE
    current = read_json(base / "current.json")
    if (type(current) is not dict or set(current) != {"status", "receipt_path", "receipt_sha256"} or
            current.get("status") != "SUCCESS"):
        raise ValueError("No current successful D preparation")
    receipt_path = Path(current["receipt_path"])
    if (not receipt_path.is_absolute() or receipt_path != receipt_path.resolve() or
            receipt_path.parent != (base / "receipts").resolve() or
            file_sha256(receipt_path) != current.get("receipt_sha256")):
        raise ValueError("Preparation receipt path or bytes changed")
    receipt = read_json(receipt_path)
    attempt_path = Path(receipt["attempt_path"])
    latest = max((p for p in (base / "attempts").iterdir() if p.is_dir()), default=None)
    if (not attempt_path.is_absolute() or attempt_path != attempt_path.resolve() or
            attempt_path.parent.parent != (base / "attempts").resolve() or
            attempt_path.parent != latest or attempt_path.name != "attempt.json" or
            receipt.get("schema_version") != "1.0" or
            receipt.get("artifact_type") != "utime_full_record_prepare_receipt" or
            file_sha256(attempt_path) != receipt.get("attempt_sha256") or
            receipt.get("request_id") != content_id(snapshot)):
        raise ValueError("Preparation attempt is stale or altered")
    attempt = read_json(attempt_path)
    request = read_json(attempt_path.parent / "request.json")
    command = [str(root / ".venvs/utime/Scripts/python.exe"), "-I",
               str(root / "tools/tf_prepare_worker.py"), "--request",
               str(attempt_path.parent / "request.json")]
    expected_result = {"artifact_type": "utime_development_prepare", "split_id": snapshot["split_id"],
                       "recordings": 119, "cache_dir": snapshot["cache_dir"],
                       "label_dir": snapshot["label_dir"]}
    invocation = attempt.get("invocation")
    if (request != {"schema_version": "1.0", "artifact_type": "utime_full_record_prepare_request",
                    "root": str(root), "data_root": str(data_root),
                    "attempt_dir": str(attempt_path.parent), "snapshot": snapshot,
                    "request_id": content_id(snapshot), "lease_run_id": LEASE} or
            type(invocation) is not dict or invocation.get("command") != command or
            invocation.get("request_sha256") != file_sha256(attempt_path.parent / "request.json") or
            invocation.get("exit_code") != 0 or
            invocation.get("thread_environment") != {
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                "NUMEXPR_NUM_THREADS": "4", "TF_NUM_INTRAOP_THREADS": "4",
                "TF_NUM_INTEROP_THREADS": "4", "CUDA_VISIBLE_DEVICES": "-1"} or
            attempt.get("result") != expected_result or
            attempt.get("status") != "SUCCESS" or attempt.get("before") != snapshot or
            attempt.get("after") != snapshot or attempt.get("failure") is not None or
            type(attempt.get("elapsed_seconds")) not in (int, float) or
            not math.isfinite(attempt["elapsed_seconds"]) or
            not 0 < attempt["elapsed_seconds"] <= MAX_SECONDS or
            type(attempt.get("monitor")) is not dict or
            attempt["monitor"].get("worker_identity") != _worker_identity(attempt_path.parent / "worker.log") or
            type(attempt["monitor"].get("peak_combined_rss_bytes")) is not int or
            not 0 < attempt["monitor"]["peak_combined_rss_bytes"] <= 10 * 1024**3 or
            type(attempt["monitor"].get("elapsed_seconds")) not in (int, float) or
            not math.isfinite(attempt["monitor"]["elapsed_seconds"]) or
            not 0 < attempt["monitor"]["elapsed_seconds"] <= attempt["elapsed_seconds"] + 0.1 or
            file_sha256(attempt_path.parent / "worker.log") != attempt.get("log_sha256") or
            _worker_result(attempt_path.parent / "worker.log") != attempt.get("result")):
        raise ValueError("Preparation attempt execution or lineage differs")
    return snapshot, receipt_path, attempt


def verify_fit_attestation(root: Path, data_root: Path) -> dict:
    """Hash-only D preparation gate; never decode outer-fold labels or signal arrays."""
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    started = time.monotonic()
    snapshot, receipt_path, attempt = _receipt_lineage(root, data_root, started)
    inventory = attempt.get("inventory")
    rows = inventory.get("recordings") if type(inventory) is dict else None
    expected = snapshot["recordings"]
    cache, labels = Path(snapshot["cache_dir"]), Path(snapshot["label_dir"])
    if (type(rows) is not list or len(rows) != 119 or
            inventory.get("recording_count") != 119 or
            inventory.get("recordings_id") != content_id(rows) or
            [row.get("recording_id") for row in rows] !=
            [row["recording_id"] for row in expected] or
            inventory.get("complete_epochs") != sum(row["n_epochs"] for row in expected) or
            type(inventory.get("valid_epochs")) is not int or
            not 0 <= inventory["valid_epochs"] <= inventory["complete_epochs"]):
        raise ValueError("Preparation inventory is incomplete or inconsistent")
    for row, source in zip(rows, expected):
        _parent_resource_bound(root, started)
        rid = source["recording_id"]
        if (type(row) is not dict or row.get("participant_id") != source["participant_id"] or
                row.get("n_epochs") != source["n_epochs"] or
                row.get("truth_payload_sha256") != source["truth_payload_sha256"] or
                row.get("truth_sidecar_sha256") != source["truth_sidecar_sha256"] or
                type(row.get("valid_epochs")) is not int or
                type(row.get("invalid_epochs")) is not int or
                row["valid_epochs"] < 0 or row["invalid_epochs"] < 0 or
                row["valid_epochs"] + row["invalid_epochs"] != source["n_epochs"] or
                file_sha256(cache / (rid + ".json")) != row.get("signal_meta_sha256") or
                file_sha256(cache / (rid + ".npy")) != row.get("signal_payload_sha256") or
                file_sha256(labels / (rid + ".npy")) != row.get("label_sha256")):
            raise ValueError("Prepared output hash or grid ancestry changed")
    names = {row["recording_id"] for row in expected}
    if ({path.name for path in cache.iterdir()} !=
            {rid + suffix for rid in names for suffix in (".json", ".npy")} or
            {path.name for path in labels.iterdir()} != {rid + ".npy" for rid in names} or
            sum(row["valid_epochs"] for row in rows) != inventory["valid_epochs"]):
        raise ValueError("Prepared output inventory changed")
    _parent_resource_bound(root, started)
    return {"artifact_type": "utime_fit_preparation_attestation",
            "receipt_path": str(receipt_path), "receipt_sha256": file_sha256(receipt_path),
            "request_id": content_id(snapshot), "data_root": snapshot["data_root"],
            "cache_dir": snapshot["cache_dir"], "label_dir": snapshot["label_dir"],
            "recordings_id": inventory["recordings_id"], "recording_count": 119,
            "complete_epochs": inventory["complete_epochs"],
            "valid_epochs": inventory["valid_epochs"]}


def verify_receipt(root: Path, data_root: Path, *, started: float | None = None) -> dict:
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    started = time.monotonic() if started is None else started
    snapshot, receipt_path, attempt = _receipt_lineage(root, data_root, started)
    _, split, records, _ = _records(root)
    inventory = _inventory(root, records, split, Path(snapshot["cache_dir"]),
                           Path(snapshot["label_dir"]), started=started)
    if inventory != attempt.get("inventory") or inventory["recording_count"] != 119:
        raise ValueError("Prepared D cache or labels changed after receipt")
    _parent_resource_bound(root, started)
    return {"artifact_type": "utime_full_record_prepare_verification",
            "scope": "119_development_recordings_only_no_fit_or_gate",
            "receipt_path": str(receipt_path), "receipt_sha256": file_sha256(receipt_path),
            "recordings_id": inventory["recordings_id"],
            "recording_count": 119, "complete_epochs": inventory["complete_epochs"],
            "valid_epochs": inventory["valid_epochs"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    action = verify_receipt if args.verify_only else run
    print(json_text(action(args.root, args.data_root)))


if __name__ == "__main__":
    main()
