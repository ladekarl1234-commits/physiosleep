"""Lease-bound synthetic native AttnSleep verification and fit precondition."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from . import attnsleep_baseline as native
from .contracts import content_id, json_text, read_json
from .research import atomic_json, compute_lease, file_sha256

PREFIX = "ATTN37_VERIFY_RESULT "
RUN_ID = "attn37-native-verify"
MAX_SECONDS = 120
MAX_BYTES = 10 * 1024**3
MIN_AVAILABLE = 4 * 1024**3


def _snapshot(root: Path) -> dict:
    source = native._source(root)
    runtime = native._runtime(root)
    return {"source": source, "runtime": runtime,
            "runtime_provenance_sha256": file_sha256(root / "research/runtimes/attn37/provenance.json"),
            "controller_sha256": file_sha256(Path(__file__)),
            "probe_sha256": file_sha256(root / "tools/verify_attn37_native.py"),
            "worker_sha256": file_sha256(root / "tools/attn37_worker.py"),
            "adapter_sha256": file_sha256(root / "sleepedf/attnsleep_baseline.py")}


def _parse_output(output: str) -> dict:
    lines = [line[len(PREFIX):] for line in output.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise ValueError("Native AttnSleep probe returned no unique machine result")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate native verification result key")
            result[key] = value
        return result
    result = json.loads(lines[0], object_pairs_hook=unique,
                        parse_constant=lambda _: (_ for _ in ()).throw(
                            ValueError("Non-finite native verification result")))
    json_text(result)
    if type(result) is not dict:
        raise ValueError("Native AttnSleep result must be an object")
    return result


def _check_result(result: dict, snapshot: dict) -> None:
    required = {"schema_version", "artifact_type", "synthetic_scope", "edf_or_truth_read",
                "native_model", "native_loss", "batch_size", "threads", "steps",
                "loss_abs_error", "invalid_gradient_max", "one_update_max_abs_error",
                "second_step_loss_abs_error", "checkpoint_in_memory_sha256", "resume_exact",
                "post_epoch_10_lr_restored", "peak_child_plus_parent_rss_bytes", "wall_seconds",
                "limits", "source_manifest_sha256", "runtime_provenance_sha256",
                "runtime_freeze_sha256", "python_executable_sha256", "worker_sha256",
                "verifier_sha256", "torch_version", "python_version", "pinned_files_verified"}
    if (set(result) != required or any(type(result[key]) is not bool or not result[key] for key in (
            "synthetic_scope", "resume_exact", "post_epoch_10_lr_restored")) or
            result["edf_or_truth_read"] is not False or
            result["schema_version"] != "1.0" or
            result["artifact_type"] != "attn37_native_synthetic_verification" or
            result["native_model"] != "pinned AttnSleep" or
            result["native_loss"] != "weighted_CrossEntropyLoss" or
            result["batch_size"] != 3 or result["threads"] != 4 or result["steps"] != 2 or
            result["limits"] != {"wall_seconds": MAX_SECONDS, "host_rss_bytes": MAX_BYTES,
                                 "host_available_floor_bytes": MIN_AVAILABLE}):
        raise ValueError("Native AttnSleep synthetic scope or probe contract differs")
    for key, maximum in (("loss_abs_error", 1e-7), ("invalid_gradient_max", 0),
                         ("one_update_max_abs_error", 1e-7), ("second_step_loss_abs_error", 0),
                         ("wall_seconds", MAX_SECONDS),
                         ("peak_child_plus_parent_rss_bytes", MAX_BYTES)):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError("Native AttnSleep numerical or resource check failed: " + key)
    if (result["peak_child_plus_parent_rss_bytes"] < 1 or result["wall_seconds"] <= 0 or
            type(result["pinned_files_verified"]) is not int or
            result["pinned_files_verified"] != len(snapshot["source"]["files"]) or
            not re.fullmatch(r"[0-9a-f]{64}", result["checkpoint_in_memory_sha256"])):
        raise ValueError("Native AttnSleep numerical evidence is incomplete")
    expected = {"source_manifest_sha256": snapshot["source"]["source_manifest_sha256"],
                "runtime_provenance_sha256": snapshot["runtime_provenance_sha256"],
                "runtime_freeze_sha256": snapshot["runtime"]["freeze_sha256"],
                "python_executable_sha256": snapshot["runtime"]["python_sha256"],
                "worker_sha256": snapshot["worker_sha256"],
                "verifier_sha256": snapshot["probe_sha256"],
                "torch_version": "1.4.0+cpu", "python_version": "3.7.9"}
    if any(result[key] != value for key, value in expected.items()):
        raise ValueError("Native AttnSleep probe result is not bound to current pinned inputs")


def _identity(snapshot: dict) -> str:
    return content_id({"schema_version": "1.0", "artifact_type": "attn37_native_verified_contract",
                       "inputs": snapshot, "synthetic_only": True,
                       "checks": ["native_weighted_ce", "invalid_gradient_zero",
                                  "one_adam_update", "dropout_bn_rng_optimizer_resume"],
                       "batch_size": 3, "threads": 4, "steps": 2})


def _record_attempt(path: Path, value: dict, output: str) -> None:
    path.mkdir(parents=True, exist_ok=False)
    log = path / "stdout.log"
    with log.open("xb") as stream:
        stream.write(output.encode("utf-8", errors="replace"))
        stream.flush()
        os.fsync(stream.fileno())
    value["stdout_sha256"] = file_sha256(log)
    value["stdout_path"] = str(log.resolve())
    atomic_json(path / "attempt.json", value, immutable=True)


def _recorded_path(value: object, parent: Path, filename: str | None = None) -> Path:
    """Accept only an already-canonical absolute path in the expected run tree."""
    if type(value) is not str:
        raise ValueError("Native verification path is not an absolute string")
    path = Path(value)
    if (not path.is_absolute() or path != path.resolve() or
            not path.is_relative_to(parent.resolve()) or
            (filename is not None and path.name != filename)):
        raise ValueError("Native verification path escapes its run directory")
    return path


def run(root: Path) -> dict:
    root = root.resolve()
    with compute_lease(root, RUN_ID):
        return _run_locked(root)


def _run_locked(root: Path) -> dict:
    attempts = root / "runs" / "attn37-native-verify" / "attempts"
    path = attempts / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
    current = root / "runs" / "attn37-native-verify" / "current.json"
    atomic_json(current, {"schema_version": "1.0", "artifact_type": "attn37_native_verification_pending",
                          "attempt_path": str((path / "attempt.json").resolve()), "status": "IN_PROGRESS"})
    started = time.monotonic()
    command = [str(root / ".venvs" / "attn37" / "python.exe"), "-I",
               str(root / "tools" / "verify_attn37_native.py")]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="")
    output = ""
    before = after = result = None
    exit_code = None
    status = "FAILED"
    error = None
    try:
        before = _snapshot(root)
        remaining = MAX_SECONDS - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Native AttnSleep preflight exceeded the 120-second deadline")
        completed = subprocess.run(command, cwd=root, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, timeout=remaining)
        output, exit_code = completed.stdout, completed.returncode
        after = _snapshot(root)
        if before != after:
            raise ValueError("Pinned native source or runtime changed during probe")
        if exit_code != 0:
            raise RuntimeError("Native AttnSleep probe exited unsuccessfully")
        result = _parse_output(output)
        _check_result(result, before)
        if time.monotonic() - started > MAX_SECONDS:
            raise TimeoutError("Native AttnSleep probe exceeded the 120-second deadline")
        status = "SUCCESS"
    except subprocess.TimeoutExpired as exc:
        output = exc.output.decode("utf-8", errors="replace") if isinstance(exc.output, bytes) else (exc.output or "")
        error = {"type": "TimeoutExpired", "message": str(exc)}
        status = "TIMEOUT"
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    attempt = {"schema_version": "1.0", "artifact_type": "attn37_native_probe_attempt",
               "status": status, "command": command, "exit_code": exit_code,
               "deadline_seconds": MAX_SECONDS, "elapsed_seconds": time.monotonic() - started,
               "before": before, "after": after, "result": result,
               "error": error}
    _record_attempt(path, attempt, output)
    if status != "SUCCESS":
        atomic_json(current, {"schema_version": "1.0", "artifact_type": "attn37_native_verification_failed",
                              "attempt_path": str((path / "attempt.json").resolve()), "status": status})
        raise RuntimeError("AttnSleep native verification failed; immutable attempt: " + str(path))
    attempt_path = path / "attempt.json"
    verification_id = _identity(before)
    receipt = {"schema_version": "1.0", "artifact_type": "attn37_native_verification_receipt",
               "verification_id": verification_id, "inputs": before,
               "result": result, "attempt_path": str(attempt_path.resolve()),
               "attempt_sha256": file_sha256(attempt_path)}
    receipts = root / "runs" / "attn37-native-verify" / "receipts"
    receipt_path = receipts / (path.name + ".json")
    atomic_json(receipt_path, receipt, immutable=True)
    pointer = {"schema_version": "1.0", "artifact_type": "attn37_native_current_verification",
               "verification_id": verification_id, "receipt_path": str(receipt_path.resolve()),
               "receipt_sha256": file_sha256(receipt_path)}
    atomic_json(current, pointer)
    return require_verification(root)


def require_verification(root: Path) -> dict:
    """Return stable fit identity only for a current, successful native probe."""
    root = root.resolve()
    pointer = read_json(root / "runs" / "attn37-native-verify" / "current.json")
    if (pointer.get("artifact_type") != "attn37_native_current_verification" or
            pointer.get("schema_version") != "1.0"):
        raise ValueError("AttnSleep native verification pointer is missing")
    receipt_path = _recorded_path(pointer["receipt_path"],
                                  root / "runs" / "attn37-native-verify" / "receipts")
    if (receipt_path.parent != (root / "runs" / "attn37-native-verify" / "receipts").resolve() or
            file_sha256(receipt_path) != pointer["receipt_sha256"]):
        raise ValueError("AttnSleep native verification receipt changed")
    receipt = read_json(receipt_path)
    snapshot = _snapshot(root)
    if (receipt.get("artifact_type") != "attn37_native_verification_receipt" or
            receipt.get("schema_version") != "1.0" or
            receipt.get("verification_id") != pointer["verification_id"] or
            receipt["verification_id"] != _identity(snapshot) or receipt["inputs"] != snapshot):
        raise ValueError("AttnSleep native verification is stale")
    attempt_path = _recorded_path(receipt["attempt_path"],
                                  root / "runs" / "attn37-native-verify" / "attempts",
                                  "attempt.json")
    attempts = (root / "runs" / "attn37-native-verify" / "attempts").resolve()
    latest = max((path for path in attempts.iterdir() if path.is_dir()), default=None)
    if (attempt_path.parent.parent != attempts or latest != attempt_path.parent or
            file_sha256(attempt_path) != receipt["attempt_sha256"]):
        raise ValueError("AttnSleep native verification attempt changed")
    attempt = read_json(attempt_path)
    log = _recorded_path(attempt["stdout_path"], attempt_path.parent, "stdout.log")
    expected_command = [str(root / ".venvs" / "attn37" / "python.exe"), "-I",
                        str(root / "tools" / "verify_attn37_native.py")]
    if (attempt.get("status") != "SUCCESS" or attempt.get("exit_code") != 0 or
            attempt.get("command") != expected_command or
            attempt.get("deadline_seconds") != MAX_SECONDS or
            type(attempt.get("elapsed_seconds")) not in (int, float) or
            not math.isfinite(attempt["elapsed_seconds"]) or
            not 0 <= attempt["elapsed_seconds"] <= MAX_SECONDS or
            attempt.get("error") is not None or
            attempt.get("before") != snapshot or attempt.get("after") != snapshot or
            attempt.get("result") != receipt["result"] or
            log != attempt_path.parent / "stdout.log" or
            file_sha256(log) != attempt["stdout_sha256"] or
            _parse_output(log.read_text(encoding="utf-8")) != receipt["result"]):
        raise ValueError("AttnSleep native attempt does not verify its receipt")
    _check_result(receipt["result"], snapshot)
    return {"artifact_type": "attn37_native_verified_contract",
            "verification_id": receipt["verification_id"],
            "source_manifest_sha256": snapshot["source"]["source_manifest_sha256"],
            "runtime_freeze_sha256": snapshot["runtime"]["freeze_sha256"],
            "controller_sha256": snapshot["controller_sha256"],
            "probe_sha256": snapshot["probe_sha256"],
            "worker_sha256": snapshot["worker_sha256"],
            "adapter_sha256": snapshot["adapter_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json_text(run(args.root)))


if __name__ == "__main__":
    main()
