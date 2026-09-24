"""Fail-closed, synthetic-only preflight for the pinned original U-Sleep slot.

TensorFlow is imported only by the isolated child after a live compute lease.
No receipt authorizes preprocessing, a fit, or an audit read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import time
import uuid
import zipfile

import numpy as np

from .contracts import content_id, json_text, read_json
from .research import atomic_json, compute_lease, file_sha256
from .tf_native import locked_packages
from .usleep_native import source_contract


BASE = "runs/usleep-native-preflight"
MODES = ("numerical", "batch64", "one_shot")
STAGES = {"numerical": ("loss", "full", "first", "resume", "inference"),
          "batch64": ("profile",), "one_shot": ("profile",)}
PREFIX = "USLEEP_NATIVE_PREFLIGHT_RESULT "
WORKER_PREFIX = "USLEEP_NATIVE_PREFLIGHT_WORKER "
MAX_SECONDS = 1800
MAX_BYTES = 10 * 1024**3
MIN_FREE_BYTES = 4 * 1024**3
MIN_DISK_BYTES = 20 * 1024**3
THREADS = 4
ATOL = 1e-7
RTOL = 1e-5
SEED = 17
READINESS_SHA256 = "09f2db2fc6fe8384d9bc4fdecea8a18bb6cde8e7f9ac56931747a45c551bda9c"
MAX_BLOCK_EPOCHS = 2880
SOURCE_FILES = ("sleepedf/usleep_verify.py", "tools/verify_usleep_native.py",
                "docs/NATIVE_USLEEP_PREFLIGHT.md", "sleepedf/usleep_native.py",
                "tools/tf_train_worker.py", "sleepedf/contracts.py",
                "sleepedf/research.py", "sleepedf/tf_native.py")


def _capacity(root: Path) -> int:
    path = root / "reports/data-readiness.json"
    if file_sha256(path) != READINESS_SHA256:
        raise ValueError("U-Sleep complete-dataset capacity metadata changed")
    report = read_json(path)
    records = report.get("records")
    if (report.get("complete_dataset_scope") is not True or
            type(records) is not list or len(records) != 197 or
            any(type(row) is not dict or type(row.get("n_epochs")) is not int or
                row["n_epochs"] < 1 or type(row.get("duration_seconds")) not in (int, float) or
                not math.isfinite(row["duration_seconds"]) or row["duration_seconds"] <= 0 or
                row["n_epochs"] != int(row["duration_seconds"] // 30)
                for row in records)):
        raise ValueError("U-Sleep required one-shot capacity is not verified metadata")
    maximum = max(row["n_epochs"] for row in records)
    if maximum != MAX_BLOCK_EPOCHS:
        raise ValueError("U-Sleep full-record one-shot capacity changed")
    return maximum


def _snapshot(root: Path) -> dict:
    root = Path(root).resolve()
    contract = source_contract(root)
    if contract["slot"] != "usleep_official" or contract["training_status"] != "NOT_READY_SOURCE_ONLY":
        raise ValueError("U-Sleep preflight source contract changed")
    capacity = _capacity(root)
    lock = root / "requirements/utime.lock.txt"
    packages = locked_packages(lock)
    observed = {}
    site = root / ".venvs/utime/Lib/site-packages"
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        if name in observed:
            raise ValueError("U-Sleep runtime has duplicate distributions")
        observed[name] = distribution.version
    if observed != packages:
        raise ValueError("U-Sleep installed package set differs from exact lock")
    install = root / "research/runtimes/utime/install.json"
    if read_json(install).get("freeze_sha256") != file_sha256(lock):
        raise ValueError("U-Sleep runtime install differs from exact lock")
    return {"schema_version": "1.0", "artifact_type": "usleep_synthetic_preflight_inputs",
            "synthetic_scope": "no_edf_truth_weights_preprocessing_or_fit",
            "source_contract_id": contract["contract_id"],
            "source_manifest_sha256": contract["source_manifest_sha256"],
            "readiness_metadata_sha256": READINESS_SHA256,
            "required_max_block_epochs": capacity,
            "runtime_lock_sha256": file_sha256(lock),
            "runtime_install_sha256": file_sha256(install),
            "native_executable_sha256": file_sha256(root / ".venvs/utime/Scripts/python.exe"),
            "packages": packages,
            "backend_environment": {"TF_ENABLE_ONEDNN_OPTS":
                                    os.environ.get("TF_ENABLE_ONEDNN_OPTS"),
                                    "TF_DETERMINISTIC_OPS": "1",
                                    "CUDA_VISIBLE_DEVICES": "-1",
                                    "CPU_THREADS": "4"},
            "source_files_sha256": {name: file_sha256(root / name) for name in SOURCE_FILES},
            "tensorflow_version": "2.13.1", "threads": THREADS,
            "training_execution_mode": "native_compiled_usleep_amsgrad_v1",
            "checkpoint_schema": "model_adam_m_v_vhat_generator_loss_mean_counter_rng_v1",
            "batch_size": 64, "sample_rate_hz": 128, "epoch_samples": 3840,
            "channels": 2, "classes": 5, "atol": ATOL, "rtol": RTOL,
            "seed": SEED, "max_seconds": MAX_SECONDS, "max_bytes": MAX_BYTES,
            "min_free_bytes": MIN_FREE_BYTES, "min_disk_bytes": MIN_DISK_BYTES}


def _lease_name(mode: str) -> str:
    if mode not in MODES:
        raise ValueError("Unknown U-Sleep preflight mode")
    return "usleep-native-verify-" + mode


def _identity(snapshot: dict, mode: str) -> str:
    return content_id({"schema_version": "1.0", "artifact_type": "usleep_preflight_contract",
                       "snapshot": snapshot, "mode": mode, "stages": list(STAGES[mode])})


def _strict_output(output: str) -> dict:
    lines = [line[len(PREFIX):] for line in output.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise ValueError("U-Sleep probe did not emit exactly one result")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate U-Sleep probe metadata key")
            result[key] = value
        return result
    value = json.loads(lines[0], object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite probe metadata")))
    json_text(value)
    if type(value) is not dict:
        raise ValueError("U-Sleep probe result must be an object")
    return value


def _worker_identity(output: str) -> dict | None:
    lines = [line[len(WORKER_PREFIX):] for line in output.splitlines()
             if line.startswith(WORKER_PREFIX)]
    if not lines:
        return None
    if len(lines) != 1:
        raise ValueError("U-Sleep worker emitted duplicate process identities")
    identity = json.loads(lines[0], parse_constant=lambda _: (_ for _ in ()).throw(
        ValueError("Non-finite U-Sleep worker identity")))
    if (type(identity) is not dict or set(identity) != {"pid", "process_start", "parent_pid"} or
            type(identity["pid"]) is not int or type(identity["parent_pid"]) is not int or
            type(identity["process_start"]) not in (int, float) or
            not math.isfinite(identity["process_start"])):
        raise ValueError("Invalid U-Sleep worker process identity")
    return identity


def _parent_resource_bound(root: Path, started: float) -> None:
    import psutil
    if (time.monotonic() - started > MAX_SECONDS or
            psutil.Process().memory_info().rss > MAX_BYTES or
            psutil.virtual_memory().available < MIN_FREE_BYTES or
            psutil.disk_usage(str(root)).free < MIN_DISK_BYTES):
        raise TimeoutError("U-Sleep parent exceeded time, memory or disk bound")


def _recorded_path(value: object, parent: Path, name: str) -> Path:
    if type(value) is not str:
        raise ValueError("U-Sleep probe path is not absolute")
    path = Path(value)
    if (not path.is_absolute() or path != path.resolve() or
            path.parent != parent.resolve() or path.name != name):
        raise ValueError("U-Sleep probe path escapes its immutable attempt")
    return path


def _metric(value: object) -> None:
    if (type(value) is not dict or set(value) != {"max_abs_error", "max_normalized_error"} or
            any(type(number) not in (int, float) or not math.isfinite(number) or number < 0
                for number in value.values()) or value["max_normalized_error"] > 1):
        raise ValueError("U-Sleep numerical tolerance or metric schema differs")


def _rng_state(value: object) -> None:
    if type(value) is not dict or set(value) != {"python", "numpy", "sampler_events", "channel_events"}:
        raise ValueError("U-Sleep shared native RNG and draw counters are incomplete")
    if (type(value["sampler_events"]) is not int or value["sampler_events"] < 1 or
            type(value["channel_events"]) is not int or
            value["channel_events"] < value["sampler_events"]):
        raise ValueError("U-Sleep sampler/channel RNG draw trace is incomplete")
    for name in ("numpy",):
        state = value[name]
        if (type(state) is not dict or
                set(state) != {"name", "keys", "position", "has_gauss", "cached_gauss"} or
                state["name"] != "MT19937" or type(state["keys"]) is not list or
                len(state["keys"]) != 624 or
                any(type(key) is not int or not 0 <= key < 2**32 for key in state["keys"]) or
                type(state["position"]) is not int or not 0 <= state["position"] <= 624 or
                type(state["has_gauss"]) is not int or state["has_gauss"] not in (0, 1) or
                type(state["cached_gauss"]) not in (int, float) or
                not math.isfinite(state["cached_gauss"])):
            raise ValueError("U-Sleep NumPy/sampler/channel RNG state is invalid")
        np.random.RandomState().set_state((state["name"],
                                         np.asarray(state["keys"], dtype=np.uint32),
                                         state["position"], state["has_gauss"],
                                         state["cached_gauss"]))
    def tuples(item):
        return tuple(tuples(child) for child in item) if isinstance(item, list) else item
    try:
        random.Random().setstate(tuples(value["python"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("U-Sleep Python RNG state is invalid") from exc


def _monitor(child: subprocess.Popen, started: float, root: Path, log_path: Path) -> dict:
    import psutil
    observed = {}
    peak = 0
    worker = None
    try:
        while True:
            try:
                parent = psutil.Process(child.pid)
                for process in [parent] + parent.children(recursive=True):
                    observed[process.pid] = process.create_time()
            except psutil.NoSuchProcess:
                pass
            if worker is None and log_path.exists():
                identity = _worker_identity(log_path.read_text(encoding="utf-8", errors="replace"))
                if identity is not None:
                    if identity["parent_pid"] not in (child.pid, os.getpid()):
                        raise ValueError("U-Sleep worker is not associated with launcher")
                    process = psutil.Process(identity["pid"])
                    if process.create_time() != identity["process_start"]:
                        raise ValueError("U-Sleep worker PID/birth handshake differs")
                    observed[process.pid] = process.create_time()
                    worker = identity
            if worker is not None:
                try:
                    process = psutil.Process(worker["pid"])
                    if process.create_time() == worker["process_start"]:
                        for descendant in process.children(recursive=True):
                            observed[descendant.pid] = descendant.create_time()
                except psutil.NoSuchProcess:
                    pass
            live = []
            for pid, birth in observed.items():
                try:
                    process = psutil.Process(pid)
                    if (process.create_time() == birth and process.is_running() and
                            process.status() != psutil.STATUS_ZOMBIE):
                        live.append(process)
                except psutil.NoSuchProcess:
                    pass
            rss = psutil.Process().memory_info().rss + sum(p.memory_info().rss for p in live)
            peak = max(peak, rss)
            if (rss > MAX_BYTES or psutil.virtual_memory().available < MIN_FREE_BYTES or
                    psutil.disk_usage(str(root)).free < MIN_DISK_BYTES or
                    time.monotonic() - started > MAX_SECONDS):
                raise TimeoutError("U-Sleep probe exceeded memory, disk or time bound")
            if child.poll() is not None and worker is None and time.monotonic() - started > 30:
                raise RuntimeError("U-Sleep launcher exited without worker PID/birth handshake")
            if child.poll() is not None and worker is not None and not live:
                return {"peak_combined_rss_bytes": peak,
                        "elapsed_seconds": time.monotonic() - started,
                        "worker_identity": worker}
            time.sleep(0.25)
    except BaseException:
        survivors = []
        for pid, birth in observed.items():
            try:
                process = psutil.Process(pid)
                if process.create_time() == birth and process.is_running():
                    survivors.append(process)
            except psutil.NoSuchProcess:
                pass
        for process in reversed(survivors):
            try:
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
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=3)
        raise


def _run_stage(root: Path, mode: str, stage: str, snapshot: dict,
               attempt_dir: Path, started: float) -> tuple[dict, dict]:
    request_path = attempt_dir / (stage + ".request.json")
    log_path = attempt_dir / (stage + ".log")
    request = {"schema_version": "1.0", "artifact_type": "usleep_preflight_request",
               "mode": mode, "stage": stage, "root": str(root),
               "attempt_dir": str(attempt_dir), "snapshot": snapshot,
               "verification_id": _identity(snapshot, mode),
               "lease_run_id": _lease_name(mode)}
    atomic_json(request_path, request, immutable=True)
    command = [str(root / ".venvs/utime/Scripts/python.exe"), "-I",
               str(root / "tools/verify_usleep_native.py"), "--request", str(request_path)]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       NUMEXPR_NUM_THREADS="4", NUMBA_NUM_THREADS="4",
                       TF_NUM_INTRAOP_THREADS="4", TF_NUM_INTEROP_THREADS="4",
                       TF_DETERMINISTIC_OPS="1", CUDA_VISIBLE_DEVICES="-1")
    lease_owner = read_json(root / "runs/compute.lock")
    environment["USLEEP_LEASE_PARENT_PID"] = str(lease_owner["pid"])
    environment["USLEEP_LEASE_PARENT_BIRTH"] = str(lease_owner["process_start"])
    with log_path.open("xb") as log:
        child = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT)
        monitor = _monitor(child, started, root, log_path)
    if child.returncode != 0:
        raise RuntimeError("U-Sleep synthetic stage failed: " + str(log_path))
    result = _strict_output(log_path.read_text(encoding="utf-8", errors="replace"))
    _stage_schema(result, mode, stage, request["verification_id"])
    return result, {"path": str(log_path.resolve()), "sha256": file_sha256(log_path), **monitor}


def _stage_schema(item: object, mode: str, stage: str, identity: str) -> None:
    fields = {
        "loss": {"checks", "errors"},
        "full": {"checks", "errors", "inventory", "first_batch", "second_batch",
                 "first_rng_sha256", "final_rng_sha256", "final_state_path",
                 "final_state_sha256", "final_iterations", "final_train_counter"},
        "first": {"first_batch", "first_iterations", "first_train_counter",
                  "rng_state_sha256", "checkpoint_files", "checkpoint_manifest_sha256"},
        "resume": {"second_batch", "restored_iterations", "restored_train_counter",
                   "restored_run_identity", "checkpoint_consumed", "final_rng_sha256",
                   "final_state_path", "final_state_sha256", "final_iterations",
                   "final_train_counter"},
        "inference": {"checks", "blocks"},
        "profile": ({"batch_size", "warmup_updates", "measured_updates", "iterations",
                     "train_counter", "batch_traces", "times_seconds", "checkpoint_files",
                     "checks"} if mode == "batch64" else
                    {"lengths", "blocks", "times_seconds", "checks"}),
    }
    common = {"schema_version", "artifact_type", "mode", "stage", "verification_id",
              "synthetic_scope"}
    if (mode not in MODES or stage not in STAGES[mode] or type(item) is not dict or
            set(item) != common | fields[stage] or item["schema_version"] != "1.0" or
            item["artifact_type"] != "usleep_native_preflight_stage" or
            item["mode"] != mode or item["stage"] != stage or
            item["verification_id"] != identity or
            item["synthetic_scope"] != "no_edf_truth_weights_preprocessing_or_fit"):
        raise ValueError("U-Sleep worker stage schema or synthetic scope differs")


def _hex(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _exact_int(value: object, expected: int) -> bool:
    return type(value) is int and value == expected


def _batch_trace(value: object, maximum: int) -> None:
    if (type(value) is not dict or
            set(value) != {"raw_sha256", "augmented_sha256", "valid_targets",
                           "reported_mean_loss", "selection_sha256", "selection",
                           "augmenter_effects"} or
            any(not _hex(value[key]) for key in
                ("raw_sha256", "augmented_sha256", "selection_sha256")) or
            type(value["valid_targets"]) is not int or
            not 1 <= value["valid_targets"] <= maximum or
            type(value["reported_mean_loss"]) not in (int, float) or
            not math.isfinite(value["reported_mean_loss"]) or
            value["reported_mean_loss"] < 0 or
            type(value["selection"]) is not list or len(value["selection"]) != 64 or
            hashlib.sha256(json_text(value["selection"]).encode()).hexdigest() !=
            value["selection_sha256"]):
        raise ValueError("U-Sleep batch/sampler trace is incomplete")
    effects = value["augmenter_effects"]
    if (type(effects) is not list or len(effects) != 2 or
            [item.get("name") if type(item) is dict else None for item in effects] !=
            ["RegionalErase", "ChannelDropout"]):
        raise ValueError("U-Sleep native augmenter trace is incomplete")
    for effect in effects:
        if (set(effect) != {"name", "before_sha256", "after_sha256", "changed_contexts"} or
                not _hex(effect["before_sha256"]) or not _hex(effect["after_sha256"]) or
                type(effect["changed_contexts"]) is not int or
                not 0 <= effect["changed_contexts"] <= 64 or
                (effect["changed_contexts"] == 0) !=
                (effect["before_sha256"] == effect["after_sha256"])):
            raise ValueError("U-Sleep native augmenter effect evidence differs")
    required = {"cohort", "record_id", "stage", "center", "offset", "view", "retries"}
    for row in value["selection"]:
        if (type(row) is not dict or set(row) != required or
                row["cohort"] not in ("SC", "ST") or
                type(row["record_id"]) is not str or
                not row["record_id"].startswith(row["cohort"]) or
                type(row["stage"]) is not int or not 0 <= row["stage"] <= 4 or
                type(row["center"]) is not int or row["center"] < 0 or
                type(row["offset"]) is not int or not -17 <= row["offset"] <= 17 or
                row["view"] not in ("fpz_eog", "pz_eog") or
                type(row["retries"]) is not int or not 0 <= row["retries"] <= 1000):
            raise ValueError("U-Sleep native sampler/channel event trace differs")


def _files(value: object, parent: Path, prefix: str) -> None:
    if type(value) is not dict or len(value) < 2:
        raise ValueError("U-Sleep checkpoint lacks data shards")
    names = []
    for raw, sha in value.items():
        path = _recorded_path(raw, parent, Path(raw).name)
        if not _hex(sha) or not path.is_file() or path.stat().st_size == 0 or file_sha256(path) != sha:
            raise ValueError("U-Sleep checkpoint bytes changed")
        names.append(path.name)
    if names.count(prefix + ".index") != 1:
        raise ValueError("U-Sleep checkpoint index is missing")
    shards = [re.fullmatch(re.escape(prefix) + r"\.data-(\d{5})-of-(\d{5})", name)
              for name in names if name != prefix + ".index"]
    if (not shards or any(match is None for match in shards) or
            len({int(match.group(2)) for match in shards}) != 1 or
            int(shards[0].group(2)) != len(shards) or
            {int(match.group(1)) for match in shards} != set(range(len(shards)))):
        raise ValueError("U-Sleep checkpoint shards are incomplete")
    actual = {str(path.resolve()) for path in parent.glob(prefix + ".*")
              if path.name != prefix + ".json"}
    if actual != set(value):
        raise ValueError("U-Sleep checkpoint has extra or missing files")


def _state(path_value: object, sha: object, parent: Path) -> dict[str, np.ndarray]:
    path = _recorded_path(path_value, parent, Path(str(path_value)).name)
    if path.suffix != ".npz" or not _hex(sha) or file_sha256(path) != sha:
        raise ValueError("U-Sleep synthetic state archive changed")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if (not members or len(members) > 1000 or
                sum(item.file_size for item in members) > 2 * 1024**3 or
                any(item.file_size > 512 * 1024**2 or "/" in item.filename or
                    "\\" in item.filename or not item.filename.endswith(".npy")
                    for item in members)):
            raise ValueError("U-Sleep synthetic state archive is not bounded")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("U-Sleep synthetic state contains non-finite values")
    return arrays


def _same_array_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    return (left.shape == right.shape and left.dtype == right.dtype and
            left.tobytes(order="C") == right.tobytes(order="C"))


def _expected_model() -> dict[str, tuple[tuple[int, ...], str]]:
    result = {}
    def conv(name: str, kernel: int, inputs: int, outputs: int) -> None:
        result[name + "/kernel"] = ((kernel, 1, inputs, outputs), "trainable")
        result[name + "/bias"] = ((outputs,), "trainable")
    def bn(name: str, width: int) -> None:
        for name_part in ("gamma", "beta"):
            result[name + "/" + name_part] = ((width,), "trainable")
        for name_part in ("moving_mean", "moving_variance"):
            result[name + "/" + name_part] = ((width,), "batchnorm_moving")
    cf = math.sqrt(1.67)
    filt = 5
    previous = 2
    encoder = []
    for level in range(12):
        width = int(filt * cf)
        conv(f"encoder_L{level}_conv1", 9, previous, width)
        bn(f"encoder_L{level}_BN1", width)
        encoder.append(width)
        previous = width
        filt = int(filt * math.sqrt(2))
    bottom = int(filt * cf)
    conv("bottom_conv1", 9, previous, bottom)
    bn("bottom_BN1", bottom)
    previous = bottom
    for level, residual in enumerate(reversed(encoder)):
        filt = math.ceil(filt / math.sqrt(2))
        width = int(filt * cf)
        if width != residual:
            raise ValueError("U-Sleep source-derived decoder width differs")
        stem = f"upsample_L{level}"
        conv(stem + "_conv1", 2, previous, width)
        bn(stem + "_BN1", width)
        conv(stem + "_conv2", 9, 2 * width, width)
        bn(stem + "_BN2", width)
        previous = width
    dense = int(5 * cf)
    conv("dense_classifier_out", 1, previous, dense)
    conv("sequence_conv_out_1", 1, dense, 5)
    conv("sequence_conv_out_2", 1, 5, 5)
    if len(result) != 228:
        raise ValueError("U-Sleep source-derived model variable count differs")
    return result


def _inventory(value: object, arrays: dict[str, np.ndarray] | None = None) -> None:
    expected = _expected_model()
    if (type(value) is not dict or set(value) != {"model", "adam", "tf_generator", "compiled"} or
            type(value["model"]) is not list or len(value["model"]) != 228 or
            type(value["adam"]) is not list or len(value["adam"]) != 463):
        raise ValueError("U-Sleep architecture/AMSGrad inventory is incomplete")
    seen = set()
    trainable = []
    for index, item in enumerate(value["model"]):
        if type(item) is not dict or set(item) != {"name", "shape", "dtype", "role"}:
            raise ValueError("U-Sleep model variable metadata is incomplete")
        name = item["name"]
        key = name.removesuffix(":0") if type(name) is str else ""
        if (name != key + ":0" or key not in expected or key in seen or
                item["shape"] != list(expected[key][0]) or
                item["role"] != expected[key][1] or item["dtype"] != "float32"):
            raise ValueError("U-Sleep model variable differs from source architecture")
        seen.add(key)
        if item["role"] == "trainable":
            trainable.append(item)
        if arrays is not None and (arrays[f"model_{index:05d}"].shape != tuple(item["shape"]) or
                                   arrays[f"model_{index:05d}"].dtype != np.float32):
            raise ValueError("U-Sleep model archive array differs")
    if seen != set(expected) or len(trainable) != 154:
        raise ValueError("U-Sleep trainable/BN state differs from source")
    for index, item in enumerate(value["adam"]):
        if (type(item) is not dict or
                set(item) != {"name", "shape", "dtype", "role", "for_variable"} or
                type(item["name"]) is not str):
            raise ValueError("U-Sleep AMSGrad slot metadata is incomplete")
        if index == 0:
            if (item["shape"] != [] or item["dtype"] != "int64" or
                    item["role"] != "iteration" or item["for_variable"] is not None or
                    not item["name"].endswith("iter:0")):
                raise ValueError("U-Sleep Adam iteration state differs")
        else:
            role = ("m", "v", "vhat")[(index - 1) // 154]
            variable = trainable[(index - 1) % 154]
            if (item["role"] != role or item["for_variable"] != variable["name"] or
                    item["shape"] != variable["shape"] or item["dtype"] != "float32" or
                    not item["name"].endswith(variable["name"].removesuffix(":0") +
                                                   "/" + role + ":0")):
                raise ValueError("U-Sleep AMSGrad m/v/vhat slot identity differs")
        if arrays is not None and (arrays[f"adam_{index:05d}"].shape != tuple(item["shape"]) or
                                   arrays[f"adam_{index:05d}"].dtype.name != item["dtype"]):
            raise ValueError("U-Sleep Adam archive array differs")
    if value["tf_generator"] != {"shape": [3], "dtype": "int64"}:
        raise ValueError("U-Sleep TensorFlow generator state differs")
    if value["compiled"] != {"loss_metric_total": {"shape": [], "dtype": "float32"},
                             "loss_metric_count": {"shape": [], "dtype": "float32"},
                             "train_counter": {"shape": [], "dtype": "int64"}}:
        raise ValueError("U-Sleep compiled metric/counter state differs")
    if arrays is not None:
        required = ({f"model_{index:05d}" for index in range(228)} |
                    {f"adam_{index:05d}" for index in range(463)} |
                    {"tf_generator", "loss_metric_total", "loss_metric_count", "train_counter"})
        if (set(arrays) != required or arrays["tf_generator"].shape != (3,) or
                arrays["tf_generator"].dtype != np.int64 or
                any(arrays[name].shape != () or arrays[name].dtype != dtype for name, dtype in
                    (("loss_metric_total", np.float32), ("loss_metric_count", np.float32),
                     ("train_counter", np.int64))) or
                arrays["loss_metric_total"].item() < 0 or
                arrays["loss_metric_count"].item() <= 0):
            raise ValueError("U-Sleep full synthetic state archive is incomplete")


def _check_block(item: object, parent: Path, expected_length: int) -> None:
    from .usleep_native import aggregate_channel_views
    if (type(item) is not dict or
            set(item) != {"length_epochs", "archive_path", "archive_sha256",
                           "state_before_sha256", "state_after_sha256"} or
            not _exact_int(item["length_epochs"], expected_length) or
            not _hex(item["state_before_sha256"]) or
            item["state_before_sha256"] != item["state_after_sha256"]):
        raise ValueError("U-Sleep one-shot block metadata or inference state differs")
    arrays = _state(item["archive_path"], item["archive_sha256"], parent)
    if (set(arrays) != {"fpz_eog", "pz_eog", "epoch_index", "prediction"} or
            arrays["fpz_eog"].shape != (expected_length, 5) or
            arrays["pz_eog"].shape != (expected_length, 5) or
            arrays["fpz_eog"].dtype != np.float32 or
            arrays["pz_eog"].dtype != np.float32 or
            arrays["epoch_index"].dtype.kind not in "iu" or
            arrays["prediction"].shape != (expected_length,) or
            arrays["prediction"].dtype.kind not in "iu"):
        raise ValueError("U-Sleep one-shot output has wrong full-grid shape or dtype")
    scores = {name: arrays[name] for name in ("fpz_eog", "pz_eog")}
    indices = {name: arrays["epoch_index"] for name in scores}
    expected = aggregate_channel_views(scores, indices)
    if not np.array_equal(arrays["prediction"], expected):
        raise ValueError("U-Sleep two-view score sum or original-grid inference differs")


def _check_numerical(results: dict, parent: Path) -> dict:
    if type(results) is not dict or set(results) != set(STAGES["numerical"]):
        raise ValueError("U-Sleep numerical stage coverage is incomplete")
    loss, full, first, resume, inference = (results[stage] for stage in STAGES["numerical"])
    loss_checks = {"native_none_masked_sum_gradient", "invalid_gradient_zero",
                   "extreme_wrong_class_finite", "all_invalid_rejected",
                   "legacy_amsgrad_two_steps_vhat_exercised"}
    full_checks = {"independent_native_vs_production_compiled_update",
                   "all_model_bn_m_v_vhat_metric_counter_compared",
                   "nonzero_learning_rate_normalized_update", "no_regularization",
                   "full_35_epoch_grid"}
    if (type(loss["checks"]) is not dict or set(loss["checks"]) != loss_checks or
            any(value is not True for value in loss["checks"].values()) or
            type(full["checks"]) is not dict or set(full["checks"]) != full_checks or
            any(value is not True for value in full["checks"].values())):
        raise ValueError("U-Sleep actual numerical check inventory is incomplete")
    loss_errors = {name + suffix for name in ("all_valid", "mixed", "extreme")
                   for suffix in ("_loss", "_gradient")}
    loss_errors |= {f"amsgrad_{step}_{role}" for step in (1, 2)
                    for role in ("parameter", "parameter_delta_over_lr", "iteration", "m", "v", "vhat")}
    if type(loss["errors"]) is not dict or set(loss["errors"]) != loss_errors:
        raise ValueError("U-Sleep native loss/AMSGrad metric coverage differs")
    for value in loss["errors"].values():
        _metric(value)
    _inventory(full["inventory"])
    expected_full = ({f"model_{i:05d}" for i in range(228)} |
                     {f"adam_{i:05d}" for i in range(463)} |
                     {"loss_metric_total", "loss_metric_count", "train_counter",
                      "reported_mean_loss"} |
                     {f"trainable_delta_{i:05d}" for i, item in
                      enumerate(full["inventory"]["model"]) if item["role"] == "trainable"} |
                     {f"trainable_delta_over_lr_{i:05d}" for i, item in
                      enumerate(full["inventory"]["model"]) if item["role"] == "trainable"} |
                     {f"batchnorm_delta_{i:05d}" for i, item in
                      enumerate(full["inventory"]["model"]) if item["role"] == "batchnorm_moving"})
    if type(full["errors"]) is not dict or set(full["errors"]) != expected_full:
        raise ValueError("U-Sleep full model error inventory is incomplete")
    for value in full["errors"].values():
        _metric(value)
    for item in (full["first_batch"], full["second_batch"], first["first_batch"],
                 resume["second_batch"]):
        _batch_trace(item, 35)
    pair = (full["first_batch"], full["second_batch"])
    if (set(row["view"] for trace in pair for row in trace["selection"]) !=
            {"fpz_eog", "pz_eog"} or
            not any(row["retries"] > 0 for trace in pair for row in trace["selection"]) or
            any(sum(trace["augmenter_effects"][index]["changed_contexts"]
                    for trace in pair) <= 0 for index in (0, 1))):
        raise ValueError("U-Sleep synthetic sampling/augmentation coverage is incomplete")
    if (full["first_batch"] != first["first_batch"] or
            full["second_batch"] != resume["second_batch"] or
            not _exact_int(first["first_iterations"], 1) or
            not _exact_int(first["first_train_counter"], 1) or
            not _exact_int(resume["restored_iterations"], 1) or
            not _exact_int(resume["restored_train_counter"], 1) or
            not _exact_int(resume["final_iterations"], 2) or
            not _exact_int(full["final_iterations"], 2) or
            not _exact_int(resume["final_train_counter"], 2) or
            not _exact_int(full["final_train_counter"], 2) or
            resume["checkpoint_consumed"] is not True or
            resume["restored_run_identity"] is not True or
            not _hex(full["first_rng_sha256"]) or
            full["first_rng_sha256"] != first["rng_state_sha256"] or
            not _hex(full["final_rng_sha256"]) or
            full["final_rng_sha256"] != resume["final_rng_sha256"]):
        raise ValueError("U-Sleep fresh-process continuation differs")
    checkpoint_path = parent / "synthetic-checkpoint.json"
    if file_sha256(checkpoint_path) != first["checkpoint_manifest_sha256"]:
        raise ValueError("U-Sleep checkpoint manifest changed")
    checkpoint = read_json(checkpoint_path)
    if (set(checkpoint) != {"schema_version", "artifact_type", "verification_id",
                           "files", "rng", "rng_sha256"} or
            checkpoint["schema_version"] != "1.0" or
            checkpoint["artifact_type"] != "usleep_synthetic_checkpoint" or
            checkpoint["verification_id"] != first["verification_id"] or
            checkpoint["files"] != first["checkpoint_files"] or
            not _hex(checkpoint["rng_sha256"]) or
            checkpoint["rng_sha256"] != first["rng_state_sha256"] or
            hashlib.sha256(json_text(checkpoint["rng"]).encode()).hexdigest() !=
            checkpoint["rng_sha256"]):
        raise ValueError("U-Sleep checkpoint RNG ancestry differs")
    _rng_state(checkpoint["rng"])
    _files(first["checkpoint_files"], parent, "synthetic-checkpoint")
    uninterrupted = _state(full["final_state_path"], full["final_state_sha256"], parent)
    restored = _state(resume["final_state_path"], resume["final_state_sha256"], parent)
    _inventory(full["inventory"], uninterrupted)
    _inventory(full["inventory"], restored)
    if any(not _same_array_bytes(uninterrupted[key], restored[key]) for key in uninterrupted):
        raise ValueError("U-Sleep full model/BN/AMSGrad/vhat exact resume differs")
    if (int(uninterrupted["adam_00000"]) != 2 or
            int(uninterrupted["train_counter"]) != 2 or
            float(uninterrupted["loss_metric_count"]) !=
            full["second_batch"]["valid_targets"] or
            float(np.float32(uninterrupted["loss_metric_total"] /
                             uninterrupted["loss_metric_count"])) !=
            float(np.float32(full["second_batch"]["reported_mean_loss"]))):
        raise ValueError("U-Sleep archived optimizer/compiled metric state differs")
    if (type(inference["checks"]) is not dict or
            inference["checks"] != {"full_grid_no_tiling": True,
                                    "both_views_signal_only": True,
                                    "no_inference_state_mutation": True} or
            type(inference["blocks"]) is not list or len(inference["blocks"]) != 4):
        raise ValueError("U-Sleep short/full-block inference coverage differs")
    for item, length in zip(inference["blocks"], (1, 34, 35, 36)):
        _check_block(item, parent, length)
    return {"state_arrays": 695, "compared_update_metrics": len(full["errors"]),
            "checkpoint_exact_resume": True, "inference_lengths": [1, 34, 35, 36]}


def _check_profile(results: dict, parent: Path, mode: str, elapsed: float) -> dict:
    if type(results) is not dict or set(results) != {"profile"}:
        raise ValueError("U-Sleep profile stage is incomplete")
    item = results["profile"]
    timing = item["times_seconds"]
    if (type(timing) is not dict or
            set(timing) != {"initialization", "warmup", "update1", "update2",
                            "inference", "checkpoint"} or
            any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in timing.values()) or
            sum(timing.values()) > elapsed + 0.1):
        raise ValueError("U-Sleep profile timing exceeds monitored elapsed time")
    if mode == "batch64":
        if (type(item["checks"]) is not dict or
                set(item["checks"]) != {"full_native_batch64",
                                        "finite_model_optimizer_metric",
                                        "checkpoint_complete"} or
                any(value is not True for value in item["checks"].values()) or
                not _exact_int(item["batch_size"], 64) or
                not _exact_int(item["warmup_updates"], 1) or
                not _exact_int(item["measured_updates"], 2) or
                not _exact_int(item["iterations"], 3) or
                not _exact_int(item["train_counter"], 3) or
                any(timing[key] <= 0 for key in timing) or
                type(item["batch_traces"]) is not list or
                len(item["batch_traces"]) != 3):
            raise ValueError("U-Sleep native batch64 resource profile differs")
        for trace in item["batch_traces"]:
            _batch_trace(trace, 64 * 35)
        _files(item["checkpoint_files"], parent, "profile-checkpoint")
        return {"batch_size": 64, "warmup_updates": 1, "measured_updates": 2}
    if (type(item["checks"]) is not dict or
            set(item["checks"]) != {"full_block_no_tiling",
                                    "finite_two_view_probabilities",
                                    "no_inference_state_mutation"} or
            any(value is not True for value in item["checks"].values()) or
            timing["inference"] <= 0 or
            item["lengths"] != [1, 34, 35, 36, MAX_BLOCK_EPOCHS] or
            type(item["blocks"]) is not list or len(item["blocks"]) != 5):
        raise ValueError("U-Sleep full-record one-shot profile differs")
    for block, length in zip(item["blocks"], item["lengths"]):
        _check_block(block, parent, length)
    return {"max_block_epochs": MAX_BLOCK_EPOCHS,
            "input_shape": [1, MAX_BLOCK_EPOCHS, 3840, 2]}


def run(root: Path, mode: str) -> dict:
    root = Path(root).resolve()
    _lease_name(mode)
    with compute_lease(root, _lease_name(mode)):
        return _run_locked(root, mode)


def _run_locked(root: Path, mode: str) -> dict:
    base = root / BASE / mode
    attempt_dir = base / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
                                       "-" + uuid.uuid4().hex[:8])
    attempt_dir.mkdir(parents=True, exist_ok=False)
    current = base / "current.json"
    atomic_json(current, {"schema_version": "1.0", "artifact_type": "usleep_preflight_pending",
                          "attempt_path": str((attempt_dir / "attempt.json").resolve())})
    started = time.monotonic()
    before = after = results = summary = None
    monitors = {}
    failure = None
    status = "FAILED"
    try:
        before = _snapshot(root)
        results = {}
        for stage in STAGES[mode]:
            result, monitor = _run_stage(root, mode, stage, before, attempt_dir, started)
            results[stage] = result
            monitors[stage] = monitor
        summary = (_check_numerical(results, attempt_dir) if mode == "numerical" else
                   _check_profile(results, attempt_dir, mode,
                                  monitors["profile"]["elapsed_seconds"]))
        after = _snapshot(root)
        _parent_resource_bound(root, started)
        if before != after:
            raise ValueError("U-Sleep source/runtime changed or mode deadline elapsed")
        status = "SUCCESS"
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    logs = {stage: {"path": str(path.resolve()), "sha256": file_sha256(path),
                    **{key: monitors.get(stage, {}).get(key) for key in
                       ("peak_combined_rss_bytes", "elapsed_seconds", "worker_identity")}}
            for stage in STAGES[mode] if (path := attempt_dir / (stage + ".log")).exists()}
    if status == "SUCCESS":
        summary["peak_combined_rss_bytes"] = max(
            logs[stage]["peak_combined_rss_bytes"] for stage in STAGES[mode])
    attempt = {"schema_version": "1.0", "artifact_type": "usleep_native_preflight_attempt",
               "mode": mode, "status": status, "before": before, "after": after,
               "results": results, "summary": summary, "logs": logs,
               "elapsed_seconds": time.monotonic() - started,
               "deadline_seconds": MAX_SECONDS, "failure": failure}
    atomic_json(attempt_dir / "attempt.json", attempt, immutable=True)
    if status != "SUCCESS":
        atomic_json(current, {"schema_version": "1.0", "artifact_type": "usleep_preflight_failed",
                              "attempt_path": str((attempt_dir / "attempt.json").resolve())})
        raise RuntimeError("U-Sleep synthetic preflight failed; attempt: " + str(attempt_dir))
    receipt = {"schema_version": "1.0", "artifact_type": "usleep_native_preflight_receipt",
               "mode": mode, "verification_id": _identity(before, mode),
               "snapshot": before, "attempt_path": str((attempt_dir / "attempt.json").resolve()),
               "attempt_sha256": file_sha256(attempt_dir / "attempt.json")}
    receipt_path = base / "receipts" / (attempt_dir.name + ".json")
    atomic_json(receipt_path, receipt, immutable=True)
    atomic_json(current, {"schema_version": "1.0", "artifact_type": "usleep_preflight_current",
                          "receipt_path": str(receipt_path.resolve()),
                          "receipt_sha256": file_sha256(receipt_path)})
    try:
        return _verify_mode(root, mode, started=started)
    except BaseException:
        atomic_json(current, {"schema_version": "1.0", "artifact_type": "usleep_preflight_failed",
                              "attempt_path": str((attempt_dir / "attempt.json").resolve())})
        raise


def _verify_mode(root: Path, mode: str, *, started: float | None = None) -> dict:
    _lease_name(mode)
    started = time.monotonic() if started is None else started
    base = root / BASE / mode
    current = read_json(base / "current.json")
    if (type(current) is not dict or set(current) !=
            {"schema_version", "artifact_type", "receipt_path", "receipt_sha256"} or
            current["schema_version"] != "1.0" or
            current["artifact_type"] != "usleep_preflight_current"):
        raise ValueError("U-Sleep has no current successful " + mode + " preflight")
    receipt_path = _recorded_path(current["receipt_path"], (base / "receipts").resolve(),
                                  Path(current["receipt_path"]).name)
    if not _hex(current["receipt_sha256"]) or file_sha256(receipt_path) != current["receipt_sha256"]:
        raise ValueError("U-Sleep current receipt bytes changed")
    receipt = read_json(receipt_path)
    snapshot = _snapshot(root)
    if (type(receipt) is not dict or set(receipt) !=
            {"schema_version", "artifact_type", "mode", "verification_id", "snapshot",
             "attempt_path", "attempt_sha256"} or
            receipt["schema_version"] != "1.0" or
            receipt["artifact_type"] != "usleep_native_preflight_receipt" or
            receipt["mode"] != mode or receipt["snapshot"] != snapshot or
            receipt["verification_id"] != _identity(snapshot, mode)):
        raise ValueError("U-Sleep current receipt is stale or wrong scope")
    candidate = Path(receipt["attempt_path"])
    attempts = (base / "attempts").resolve()
    if (not candidate.is_absolute() or candidate != candidate.resolve() or
            candidate.parent.parent != attempts or
            re.fullmatch(r"[0-9]{8}T[0-9]{12}Z-[0-9a-f]{8}", candidate.parent.name) is None):
        raise ValueError("U-Sleep attempt path escapes its run tree")
    attempt_path = _recorded_path(receipt["attempt_path"], candidate.parent, "attempt.json")
    latest = max((path for path in attempts.iterdir() if path.is_dir()), default=None)
    if (candidate.parent != latest or not _hex(receipt["attempt_sha256"]) or
            file_sha256(attempt_path) != receipt["attempt_sha256"]):
        raise ValueError("U-Sleep attempt bytes changed or newer attempt superseded it")
    attempt = read_json(attempt_path)
    if (type(attempt) is not dict or set(attempt) !=
            {"schema_version", "artifact_type", "mode", "status", "before", "after",
             "results", "summary", "logs", "elapsed_seconds", "deadline_seconds", "failure"} or
            attempt["schema_version"] != "1.0" or
            attempt["artifact_type"] != "usleep_native_preflight_attempt" or
            attempt["mode"] != mode or attempt["status"] != "SUCCESS" or
            attempt["before"] != snapshot or attempt["after"] != snapshot or
            attempt["failure"] is not None or attempt["deadline_seconds"] != MAX_SECONDS or
            type(attempt["elapsed_seconds"]) not in (int, float) or
            not math.isfinite(attempt["elapsed_seconds"]) or
            not 0 <= attempt["elapsed_seconds"] <= MAX_SECONDS):
        raise ValueError("U-Sleep successful attempt evidence is incomplete")
    results, logs = attempt["results"], attempt["logs"]
    if type(results) is not dict or type(logs) is not dict or set(results) != set(STAGES[mode]) or set(logs) != set(STAGES[mode]):
        raise ValueError("U-Sleep attempt stage coverage differs")
    for stage in STAGES[mode]:
        item = results[stage]
        log = _recorded_path(logs[stage]["path"], candidate.parent, stage + ".log")
        if (file_sha256(log) != logs[stage]["sha256"] or
                _strict_output(log.read_text(encoding="utf-8", errors="replace")) != item or
                type(logs[stage].get("worker_identity")) is not dict or
                logs[stage].get("worker_identity") != _worker_identity(
                    log.read_text(encoding="utf-8", errors="replace")) or
                type(logs[stage]["peak_combined_rss_bytes"]) is not int or
                not 0 < logs[stage]["peak_combined_rss_bytes"] <= MAX_BYTES or
                type(logs[stage]["elapsed_seconds"]) not in (int, float) or
                not math.isfinite(logs[stage]["elapsed_seconds"]) or
                not 0 <= logs[stage]["elapsed_seconds"] <= attempt["elapsed_seconds"] + 0.1):
            raise ValueError("U-Sleep attempt log/resource evidence differs")
        _stage_schema(item, mode, stage, receipt["verification_id"])
    summary = (_check_numerical(results, candidate.parent) if mode == "numerical" else
               _check_profile(results, candidate.parent, mode,
                              logs["profile"]["elapsed_seconds"]))
    summary["peak_combined_rss_bytes"] = max(logs[stage]["peak_combined_rss_bytes"] for stage in STAGES[mode])
    if summary != attempt["summary"]:
        raise ValueError("U-Sleep receipt summary was not recomputed from evidence")
    _parent_resource_bound(root, started)
    return {"mode": mode, "verification_id": receipt["verification_id"],
            "summary": summary, "receipt_path": str(receipt_path)}


def require_verification(root: Path, required_max_block_length: int) -> dict:
    """Require all synthetic receipts; caller must bind length to frozen records."""
    if (type(required_max_block_length) is not int or
            not 1 <= required_max_block_length <= MAX_BLOCK_EPOCHS):
        raise ValueError("U-Sleep required one-shot block length is not frozen/capacity-bound")
    root = Path(root).resolve()
    verified = {mode: _verify_mode(root, mode) for mode in MODES}
    if verified["one_shot"]["summary"]["max_block_epochs"] < required_max_block_length:
        raise ValueError("U-Sleep required one-shot length exceeds qualified profile")
    contract = {"artifact_type": "usleep_native_preflight_contract",
                "engine": "usleep_official", "synthetic_scope":
                "no_edf_truth_weights_preprocessing_or_fit",
                "numerical_id": verified["numerical"]["verification_id"],
                "batch64_id": verified["batch64"]["verification_id"],
                "one_shot_id": verified["one_shot"]["verification_id"],
                "required_max_block_length": required_max_block_length,
                "profiled_max_block_length": MAX_BLOCK_EPOCHS,
                "threads": THREADS, "max_bytes": MAX_BYTES,
                "atol": ATOL, "rtol": RTOL}
    contract["verification_id"] = content_id(contract)
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args()
    print(json_text(run(args.root, args.mode)))


if __name__ == "__main__":
    main()
