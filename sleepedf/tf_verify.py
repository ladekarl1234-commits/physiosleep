"""Bounded synthetic numerical preflight for the pinned native U-Time runtime.

Both numerical and batch-12 profile receipts are required before an original fit.
This module imports no TensorFlow and never reads EDF, annotations, or model weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time
import uuid
import zipfile

import numpy as np

from .contracts import content_id, json_text, read_json
from .research import atomic_json, compute_lease, file_sha256
from .tf_native import locked_packages

BASE = "runs/utime-native-preflight"
MODES = ("numerical", "profile")
STAGES = {"numerical": ("loss", "full", "first", "resume"), "profile": ("profile",)}
PREFIX = "UTIME_NATIVE_PREFLIGHT_RESULT "
MAX_SECONDS = 1800
MAX_BYTES = 10 * 1024**3
MIN_FREE_BYTES = 4 * 1024**3
THREADS = 4
ATOL = 1e-7
RTOL = 1e-5
SEED = 17
SOURCE_FILES = ("sleepedf/tf_verify.py", "tools/verify_tf_native.py",
                "docs/NATIVE_TF_PREFLIGHT.md",
                "sleepedf/tf_native.py", "sleepedf/tf_experiment.py",
                "tools/tf_train_worker.py", "sleepedf/contracts.py",
                "sleepedf/research.py", "vendor/utime/utime/bin/defaults/utime/hparams.yaml")


def _snapshot(root: Path) -> dict:
    root = root.resolve()
    lock = root / "requirements/utime.lock.txt"
    packages = locked_packages(lock)
    site = root / ".venvs/utime/Lib/site-packages"
    observed = {}
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        if name in observed:
            raise ValueError("U-Time native runtime has duplicate installed packages")
        observed[name] = distribution.version
    if observed != packages:
        raise ValueError("U-Time native installed packages differ from exact lock")
    installed = read_json(root / "research/runtimes/utime/install.json")
    if installed.get("freeze_sha256") != file_sha256(lock):
        raise ValueError("U-Time native runtime differs from its installed lock")
    source_path = root / "research/sources/utime.json"
    source = read_json(source_path)
    if source.get("weights_downloaded") is not False:
        raise ValueError("U-Time preflight cannot use external weights")
    for relative, item in source["files"].items():
        if file_sha256(root / "vendor/utime" / relative) != item["sha256"]:
            raise ValueError("Pinned U-Time source changed")
    files = {name: file_sha256(root / name) for name in SOURCE_FILES}
    return {"schema_version": "1.0", "artifact_type": "utime_synthetic_preflight_inputs",
            "scope": "synthetic_only_no_edf_truth_or_weights",
            "source_manifest_sha256": file_sha256(source_path),
            "runtime_lock_sha256": file_sha256(lock),
            "runtime_install_sha256": file_sha256(root / "research/runtimes/utime/install.json"),
            "native_executable_sha256": file_sha256(root / ".venvs/utime/Scripts/python.exe"),
            "packages": packages, "source_files_sha256": files,
            "tensorflow_version": "2.13.1", "threads": THREADS,
            "training_execution_mode": "native_compiled_train_on_batch_v1",
            "training_checkpoint_schema": "model_adam_generator_loss_mean_train_counter_v1",
            "max_seconds": MAX_SECONDS, "max_bytes": MAX_BYTES,
            "min_free_bytes": MIN_FREE_BYTES, "atol": ATOL, "rtol": RTOL,
            "seed": SEED}


def _lease_name(mode: str) -> str:
    if mode not in MODES:
        raise ValueError("Unknown U-Time preflight mode")
    return "utime-native-verify-" + mode


def _identity(snapshot: dict, mode: str) -> str:
    return content_id({"schema_version": "1.0", "artifact_type": "utime_native_preflight_contract",
                       "snapshot": snapshot, "mode": mode, "stages": list(STAGES[mode])})


def _parse_output(output: str) -> dict:
    lines = [line[len(PREFIX):] for line in output.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise ValueError("U-Time native preflight worker did not produce one result")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate U-Time preflight metadata key")
            result[key] = value
        return result
    result = json.loads(lines[0], object_pairs_hook=unique,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite preflight metadata")))
    json_text(result)
    if type(result) is not dict:
        raise ValueError("U-Time preflight result must be an object")
    return result


def _check_stage_metadata(item: object, mode: str, stage: str, identity: str) -> None:
    common = {"schema_version", "artifact_type", "mode", "stage", "verification_id",
              "synthetic_scope"}
    fields = {
        "loss": {"checks", "errors"},
        "full": {"checks", "errors", "state_inventory", "first_batch", "second_batch",
                 "first_rng_sha256", "final_rng_sha256", "final_iterations",
                 "final_train_counter",
                 "final_state_path", "final_state_sha256"},
        "first": {"first_batch", "checkpoint_files", "checkpoint_manifest_sha256",
                  "first_iterations", "first_train_counter", "rng_state_sha256"},
        "resume": {"second_batch", "restored_iterations", "restored_run_identity",
                   "restored_train_counter", "checkpoint_consumed", "final_rng_sha256", "final_iterations",
                   "final_train_counter",
                   "final_state_path", "final_state_sha256"},
        "profile": {"batch_size", "warmup_updates", "measured_updates", "inference_batches",
                    "optimizer_iterations", "train_counter", "initialization_seconds", "warmup_seconds",
                    "update1_seconds", "update2_seconds", "inference_seconds",
                    "checkpoint_seconds", "profile_batch_hashes", "checkpoint_files"}}
    if (type(item) is not dict or set(item) != common | fields[stage] or
            item["schema_version"] != "1.0" or
            item["artifact_type"] != "utime_native_preflight_stage" or
            item["mode"] != mode or item["stage"] != stage or
            item["verification_id"] != identity or
            item["synthetic_scope"] != "no_edf_truth_weights_or_fit"):
        raise ValueError("U-Time native preflight stage schema or scope differs")


def _supervise(child: subprocess.Popen, started: float) -> dict:
    """Bound and reap the worker launcher and all observed descendants."""
    import psutil
    observed = {}
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
            rss = psutil.Process().memory_info().rss + sum(process.memory_info().rss for process in live)
            peak = max(peak, rss)
            if (rss > MAX_BYTES or psutil.virtual_memory().available < MIN_FREE_BYTES or
                    time.monotonic() - started > MAX_SECONDS):
                raise TimeoutError("U-Time native preflight exceeded time or host memory bound")
            if child.poll() is not None and not live:
                break
            time.sleep(0.25)
        child.wait()
        return {"peak_combined_rss_bytes": peak,
                "elapsed_seconds": time.monotonic() - started}
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
        child.wait()
        raise


def _run_stage(root: Path, mode: str, stage: str, snapshot: dict,
               attempt_dir: Path, started: float) -> tuple[dict, dict]:
    request_path = attempt_dir / (stage + ".request.json")
    log_path = attempt_dir / (stage + ".log")
    request = {"schema_version": "1.0", "artifact_type": "utime_native_preflight_request",
               "mode": mode, "stage": stage, "root": str(root), "attempt_dir": str(attempt_dir),
               "snapshot": snapshot, "verification_id": _identity(snapshot, mode),
               "lease_run_id": _lease_name(mode)}
    atomic_json(request_path, request, immutable=True)
    command = [str(root / ".venvs/utime/Scripts/python.exe"), "-I",
               str(root / "tools/verify_tf_native.py"), "--request", str(request_path.resolve())]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       NUMEXPR_NUM_THREADS="4", NUMBA_NUM_THREADS="4",
                       TF_NUM_INTRAOP_THREADS="4", TF_NUM_INTEROP_THREADS="4",
                       TF_DETERMINISTIC_OPS="1",
                       CUDA_VISIBLE_DEVICES="-1")
    with log_path.open("xb") as log:
        child = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT)
        monitor = _supervise(child, started)
    if child.returncode != 0:
        raise RuntimeError("U-Time preflight stage failed: " + str(log_path))
    result = _parse_output(log_path.read_text(encoding="utf-8", errors="replace"))
    _check_stage_metadata(result, mode, stage, request["verification_id"])
    return result, {"path": str(log_path.resolve()), "sha256": file_sha256(log_path),
                    **monitor}


def _array_state(path: Path, expected_sha: str, parent: Path) -> dict[str, np.ndarray]:
    path = _recorded_path(str(path), parent, path.name)
    if path.suffix != ".npz" or file_sha256(path) != expected_sha:
        raise ValueError("U-Time synthetic state artifact changed")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if (not members or len(members) > 5000 or
                sum(member.file_size for member in members) > 1024**3 or
                any(member.file_size > 512 * 1024**2 for member in members)):
            raise ValueError("U-Time synthetic state exceeds bounded archive size")
    with np.load(path, allow_pickle=False) as archive:
        if not archive.files or any(not np.isfinite(archive[name]).all() for name in archive.files):
            raise ValueError("U-Time synthetic state contains non-finite arrays")
        return {name: archive[name] for name in archive.files}


def _metric(value: object) -> None:
    if (type(value) is not dict or set(value) != {"max_abs_error", "max_normalized_error"} or
            any(type(number) not in (int, float) or not math.isfinite(number) or number < 0
                for number in value.values()) or value["max_normalized_error"] > 1):
        raise ValueError("U-Time parity error exceeds frozen tolerance or has invalid schema")


def _batch_trace(value: object, maximum: int) -> None:
    if (type(value) is not dict or
            set(value) != {"raw_sha256", "augmented_sha256", "valid_targets", "reported_mean_loss"} or
            any(type(value[key]) is not str or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None
                for key in ("raw_sha256", "augmented_sha256")) or
            type(value["valid_targets"]) is not int or not 1 <= value["valid_targets"] <= maximum or
            type(value["reported_mean_loss"]) not in (int, float) or
            not math.isfinite(value["reported_mean_loss"]) or value["reported_mean_loss"] < 0):
        raise ValueError("U-Time synthetic batch trace is incomplete")


def _checkpoint_files(value: object, parent: Path, prefix: str) -> None:
    if type(value) is not dict or len(value) < 2:
        raise ValueError("U-Time synthetic checkpoint lacks data shards")
    names = []
    for raw, sha in value.items():
        if type(raw) is not str or type(sha) is not str or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
            raise ValueError("U-Time synthetic checkpoint file evidence is invalid")
        path = _recorded_path(raw, parent, Path(raw).name)
        if not path.is_file() or file_sha256(path) != sha or path.stat().st_size == 0:
            raise ValueError("U-Time synthetic checkpoint bytes changed")
        names.append(path.name)
    if names.count(prefix + ".index") != 1:
        raise ValueError("U-Time synthetic checkpoint index is incomplete")
    shards = [re.fullmatch(re.escape(prefix) + r"\.data-(\d{5})-of-(\d{5})", name)
              for name in names if name != prefix + ".index"]
    if (not shards or any(match is None for match in shards) or
            len({int(match.group(2)) for match in shards}) != 1 or
            int(shards[0].group(2)) != len(shards) or
            {int(match.group(1)) for match in shards} != set(range(len(shards)))):
        raise ValueError("U-Time synthetic checkpoint shards are incomplete")
    actual = {str(path.resolve()) for path in parent.glob(prefix + ".*")
              if path.name != prefix + ".json"}
    if actual != set(value):
        raise ValueError("U-Time synthetic checkpoint has extra or missing files")


def _rng_state(value: object) -> None:
    if type(value) is not dict or set(value) != {"python", "numpy"}:
        raise ValueError("U-Time synthetic Python/NumPy RNG state is incomplete")
    numpy = value["numpy"]
    if (type(numpy) is not dict or
            set(numpy) != {"name", "keys", "position", "has_gauss", "cached_gauss"} or
            numpy["name"] != "MT19937" or type(numpy["keys"]) is not list or
            len(numpy["keys"]) != 624 or
            any(type(key) is not int or not 0 <= key < 2**32 for key in numpy["keys"]) or
            type(numpy["position"]) is not int or not 0 <= numpy["position"] <= 624 or
            type(numpy["has_gauss"]) is not int or numpy["has_gauss"] not in (0, 1) or
            type(numpy["cached_gauss"]) not in (int, float) or
            not math.isfinite(numpy["cached_gauss"])):
        raise ValueError("U-Time synthetic NumPy RNG state is invalid")
    np.random.RandomState().set_state((numpy["name"], np.asarray(numpy["keys"], dtype=np.uint32),
                                      numpy["position"], numpy["has_gauss"],
                                      numpy["cached_gauss"]))
    def tuples(item):
        return tuple(tuples(child) for child in item) if isinstance(item, list) else item
    try:
        random.Random().setstate(tuples(value["python"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("U-Time synthetic Python RNG state is invalid") from exc


def _expected_model_shapes() -> dict[str, tuple[tuple[int, ...], str]]:
    result = {}
    def conv(name: str, kernel: int, inputs: int, outputs: int) -> None:
        result[name + "/kernel"] = ((kernel, 1, inputs, outputs), "trainable")
        result[name + "/bias"] = ((outputs,), "trainable")
    def bn(name: str, width: int) -> None:
        for field in ("gamma", "beta"):
            result[name + "/" + field] = ((width,), "trainable")
        for field in ("moving_mean", "moving_variance"):
            result[name + "/" + field] = ((width,), "batchnorm_moving")
    widths = (22, 45, 90, 181, 362)
    previous = 1
    for level, width in enumerate(widths[:4]):
        stem = f"encoder_L{level}"
        conv(stem + "_conv1", 5, previous, width)
        bn(stem + "_BN1", width)
        conv(stem + "_conv2", 5, width, width)
        bn(stem + "_BN2", width)
        previous = width
    conv("bottom_conv1", 5, 181, 362)
    bn("bottom_BN1", 362)
    conv("bottom_conv2", 5, 362, 362)
    bn("bottom_BN2", 362)
    previous = 362
    for level, (width, kernel) in enumerate(zip((181, 90, 45, 22), (4, 6, 8, 10))):
        stem = f"upsample_L{level}"
        conv(stem + "_conv1", kernel, previous, width)
        bn(stem + "_BN1", width)
        conv(stem + "_conv2", 5, 2 * width, width)
        bn(stem + "_BN2", width)
        conv(stem + "_conv3", 5, width, width)
        bn(stem + "_BN3", width)
        previous = width
    conv("dense_classifier_out", 1, 22, 7)
    conv("sequence_conv_out_1", 1, 7, 5)
    conv("sequence_conv_out_2", 1, 5, 5)
    return result


def _state_inventory(value: object, arrays: dict[str, np.ndarray] | None = None) -> None:
    expected = _expected_model_shapes()
    if (type(value) is not dict or set(value) != {"model", "adam", "tf_generator", "compiled"} or
            type(value["model"]) is not list or len(value["model"]) != 138 or
            type(value["adam"]) is not list or len(value["adam"]) != 189):
        raise ValueError("U-Time full-model state inventory is incomplete")
    seen = set()
    trainable_shapes = []
    trainable_names = []
    for index, item in enumerate(value["model"]):
        if type(item) is not dict or set(item) != {"name", "shape", "dtype", "role"}:
            raise ValueError("U-Time model variable metadata is incomplete")
        name = item["name"]
        key = name.removesuffix(":0") if type(name) is str else ""
        if (name != key + ":0" or key not in expected or key in seen or
                item["shape"] != list(expected[key][0]) or item["role"] != expected[key][1] or
                item["dtype"] != "float32"):
            raise ValueError("U-Time model variable differs from pinned architecture")
        seen.add(key)
        if item["role"] == "trainable":
            trainable_shapes.append(tuple(item["shape"]))
            trainable_names.append(name)
        if arrays is not None and (arrays[f"model_{index:05d}"].shape != tuple(item["shape"]) or
                                   arrays[f"model_{index:05d}"].dtype != np.float32):
            raise ValueError("U-Time model state array shape or dtype differs")
    if seen != set(expected) or len(trainable_shapes) != 94:
        raise ValueError("U-Time model variable inventory differs from source")
    slots = []
    names = set()
    for index, item in enumerate(value["adam"]):
        if (type(item) is not dict or
                set(item) != {"name", "shape", "dtype", "role", "for_variable"} or
                type(item["name"]) is not str or not item["name"] or item["name"] in names or
                type(item["shape"]) is not list):
            raise ValueError("U-Time Adam variable metadata is incomplete")
        names.add(item["name"])
        shape = tuple(item["shape"])
        if index == 0:
            if (shape != () or item["dtype"] != "int64" or item["role"] != "iteration" or
                    item["for_variable"] is not None or not item["name"].endswith("iter:0")):
                raise ValueError("U-Time Adam iterations state differs")
        else:
            slot_role = "m" if index <= 94 else "v"
            variable_name = trainable_names[(index - 1) % 94]
            if (item["dtype"] != "float32" or item["role"] != slot_role or
                    item["for_variable"] != variable_name or
                    not item["name"].endswith(variable_name.removesuffix(":0") +
                                                    "/" + slot_role + ":0") or
                    shape != trainable_shapes[(index - 1) % 94]):
                raise ValueError("U-Time Adam slot dtype differs")
            slots.append(shape)
        if arrays is not None and (arrays[f"adam_{index:05d}"].shape != shape or
                                   arrays[f"adam_{index:05d}"].dtype.name != item["dtype"]):
            raise ValueError("U-Time Adam state array shape or dtype differs")
    if Counter(slots) != Counter({shape: 2 * count for shape, count in
                                  Counter(trainable_shapes).items()}):
        raise ValueError("U-Time Adam slot inventory differs from trainables")
    generator = value["tf_generator"]
    if (type(generator) is not dict or set(generator) != {"shape", "dtype"} or
            generator["dtype"] != "int64" or generator["shape"] != [3]):
        raise ValueError("U-Time TensorFlow generator inventory differs")
    compiled = value["compiled"]
    if (type(compiled) is not dict or
            set(compiled) != {"loss_metric_total", "loss_metric_count", "train_counter"} or
            any(compiled[key] != {"shape": [], "dtype": dtype} for key, dtype in
                (("loss_metric_total", "float32"), ("loss_metric_count", "float32"),
                 ("train_counter", "int64")))):
        raise ValueError("U-Time compiled metric or train counter inventory differs")
    if arrays is not None and (set(arrays) != {f"model_{i:05d}" for i in range(138)} |
                                  {f"adam_{i:05d}" for i in range(189)} |
                                  {"tf_generator", "loss_metric_total", "loss_metric_count",
                                   "train_counter"} or
                               arrays["tf_generator"].shape != (3,) or
                               arrays["tf_generator"].dtype != np.int64 or
                               any(arrays[key].shape != () or arrays[key].dtype != dtype for key, dtype in
                                   (("loss_metric_total", np.float32),
                                    ("loss_metric_count", np.float32),
                                    ("train_counter", np.int64))) or
                               any(not np.isfinite(arrays[key]).all() for key in
                                   ("loss_metric_total", "loss_metric_count")) or
                               arrays["loss_metric_total"].item() < 0 or
                               arrays["loss_metric_count"].item() <= 0):
        raise ValueError("U-Time synthetic state lacks full model, Adam or RNG")


def _check_numerical(results: dict, attempt_dir: Path) -> dict:
    if set(results) != set(STAGES["numerical"]):
        raise ValueError("U-Time native numerical stages are incomplete")
    loss = results["loss"]
    full = results["full"]
    expected_checks = {
        "loss": {"native_loss_and_gradients", "invalid_logit_gradients_zero",
                 "all_invalid_rejected", "native_modern_adam_decay_rejected",
                 "legacy_adam_two_updates"},
        "full": {"full_native_graph_vs_production_compiled_update",
                 "all_trainable_batchnorm_adam_metric_counter_compared",
                 "no_regularization_loss", "full_original_grid_shape"}}
    for stage in ("loss", "full"):
        checks = results[stage].get("checks")
        if type(checks) is not dict or set(checks) != expected_checks[stage] or any(
                value is not True for value in checks.values()):
            raise ValueError("U-Time native numerical parity check inventory differs")
    loss_keys = {name + suffix for name in ("all_valid", "mixed", "extreme")
                 for suffix in ("_loss", "_gradient")}
    loss_keys |= {f"adam_parameter_step_{step}" for step in (1, 2)}
    loss_keys |= {f"adam_parameter_delta_step_{step}" for step in (1, 2)}
    loss_keys |= {f"adam_slot_{step}_{index}" for step in (1, 2) for index in range(3)}
    if type(loss.get("errors")) is not dict or set(loss["errors"]) != loss_keys:
        raise ValueError("U-Time native loss and Adam error inventory differs")
    for metric in loss["errors"].values():
        _metric(metric)
    _state_inventory(full.get("state_inventory"))
    expected_full_keys = ({f"model_{index:05d}" for index in range(138)} |
                          {f"adam_{index:05d}" for index in range(189)} |
                          {"loss_metric_total", "loss_metric_count", "train_counter", "reported_mean_loss"} |
                          {f"trainable_delta_{index:05d}" for index, item in
                           enumerate(full["state_inventory"]["model"]) if item["role"] == "trainable"} |
                          {f"batchnorm_delta_{index:05d}" for index, item in
                           enumerate(full["state_inventory"]["model"]) if item["role"] == "batchnorm_moving"})
    if type(full.get("errors")) is not dict or set(full["errors"]) != expected_full_keys:
        raise ValueError("U-Time full-model numerical error inventory differs")
    for metric in full["errors"].values():
        _metric(metric)
    first = results["first"]
    resumed = results["resume"]
    checkpoint_manifest = attempt_dir / "synthetic-checkpoint.json"
    if (file_sha256(checkpoint_manifest) != first.get("checkpoint_manifest_sha256") or
            type(first.get("first_iterations")) is not int or first["first_iterations"] != 1 or
            type(first.get("first_train_counter")) is not int or first["first_train_counter"] != 1):
        raise ValueError("U-Time synthetic first-step checkpoint manifest changed")
    checkpoint = read_json(checkpoint_manifest)
    _rng_state(checkpoint.get("rng"))
    if (set(checkpoint) != {"schema_version", "artifact_type", "verification_id",
                            "stage", "files", "rng"} or
            checkpoint.get("schema_version") != "1.0" or
            checkpoint.get("artifact_type") != "utime_synthetic_checkpoint" or
            checkpoint.get("files") != first.get("checkpoint_files") or
            checkpoint.get("stage") != "first" or
            checkpoint.get("verification_id") != first.get("verification_id") or
            hashlib.sha256(json_text(checkpoint.get("rng")).encode()).hexdigest() !=
            first.get("rng_state_sha256")):
        raise ValueError("U-Time synthetic checkpoint ancestry differs")
    _checkpoint_files(first.get("checkpoint_files"), attempt_dir, "synthetic-checkpoint")
    actual_files = {str(path.resolve()): file_sha256(path)
                    for path in attempt_dir.glob("synthetic-checkpoint.*")
                    if path.name != "synthetic-checkpoint.json"}
    if actual_files != first["checkpoint_files"]:
        raise ValueError("U-Time synthetic checkpoint bytes changed")
    for item in (first.get("first_batch"), full.get("first_batch"),
                 full.get("second_batch"), resumed.get("second_batch")):
        _batch_trace(item, 35)
    if (first.get("first_batch") != full.get("first_batch") or
            resumed.get("second_batch") != full.get("second_batch") or
            first.get("rng_state_sha256") != full.get("first_rng_sha256") or
            resumed.get("final_rng_sha256") != full.get("final_rng_sha256") or
            not all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in
                    (first.get("rng_state_sha256"), resumed.get("final_rng_sha256"))) or
            type(first.get("first_iterations")) is not int or
            type(resumed.get("restored_iterations")) is not int or
            resumed.get("restored_iterations") != 1 or
            type(resumed.get("restored_train_counter")) is not int or
            resumed.get("restored_train_counter") != 1 or
            resumed.get("restored_run_identity") is not True or
            resumed.get("checkpoint_consumed") is not True or
            type(full.get("final_iterations")) is not int or
            type(resumed.get("final_iterations")) is not int or
            type(full.get("final_train_counter")) is not int or
            type(resumed.get("final_train_counter")) is not int or
            full.get("final_iterations") != 2 or resumed.get("final_iterations") != 2 or
            full.get("final_train_counter") != 2 or resumed.get("final_train_counter") != 2):
        raise ValueError("U-Time fresh-process sampling or checkpoint resume differs")
    if (Path(full["final_state_path"]).name != "full-final.npz" or
            Path(resumed["final_state_path"]).name != "resume-final.npz"):
        raise ValueError("U-Time synthetic final state names differ from declared stages")
    left = _array_state(Path(full["final_state_path"]), full["final_state_sha256"], attempt_dir)
    right = _array_state(Path(resumed["final_state_path"]), resumed["final_state_sha256"], attempt_dir)
    _state_inventory(full["state_inventory"], left)
    _state_inventory(full["state_inventory"], right)
    if any(type(arrays[key].item()) is not int or arrays[key].item() != 2
           for arrays in (left, right) for key in ("adam_00000", "train_counter")):
        raise ValueError("U-Time archived Adam or train counter differs from completed steps")
    for arrays in (left, right):
        if (arrays["loss_metric_count"].item() != resumed["second_batch"]["valid_targets"] or
                np.float32(arrays["loss_metric_total"].item() /
                           arrays["loss_metric_count"].item()) !=
                np.float32(resumed["second_batch"]["reported_mean_loss"])):
            raise ValueError("U-Time archived compiled metric differs from final batch")
    if set(left) != set(right) or any(left[name].dtype != right[name].dtype or
                                     not np.array_equal(left[name], right[name]) for name in left):
        raise ValueError("U-Time uninterrupted and fresh-process restored state differ")
    return {"state_arrays_compared": len(left), "exact_state_equality": True,
            "first_batch": first["first_batch"], "second_batch": resumed["second_batch"]}


def _check_profile(results: dict, attempt_dir: Path) -> dict:
    item = results.get("profile")
    if (set(results) != {"profile"} or type(item) is not dict or
            type(item.get("batch_size")) is not int or item.get("batch_size") != 12 or
            type(item.get("warmup_updates")) is not int or
            item.get("warmup_updates") != 1 or item.get("measured_updates") != 2 or
            item.get("inference_batches") != 1):
        raise ValueError("U-Time native production-batch profile is incomplete")
    if (type(item.get("measured_updates")) is not int or
            type(item.get("inference_batches")) is not int or
            type(item.get("optimizer_iterations")) is not int or
            type(item.get("profile_batch_hashes")) is not list or
            len(item["profile_batch_hashes"]) != 3):
        raise ValueError("U-Time native production-batch profile trace is incomplete")
    for trace in item["profile_batch_hashes"]:
        _batch_trace(trace, 420)
    for key in ("initialization_seconds", "warmup_seconds", "update1_seconds",
                "update2_seconds", "inference_seconds", "checkpoint_seconds"):
        value = item.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= MAX_SECONDS:
            raise ValueError("U-Time native production-batch profile timing is invalid")
    expected_files = item.get("checkpoint_files")
    files = {str(path.resolve()): file_sha256(path)
             for path in attempt_dir.glob("profile-synthetic-checkpoint.*")}
    _checkpoint_files(expected_files, attempt_dir, "profile-synthetic-checkpoint")
    if files != expected_files or item.get("optimizer_iterations") != 3 or item.get("train_counter") != 3:
        raise ValueError("U-Time native production-batch checkpoint changed")
    return {key: item[key] for key in ("batch_size", "warmup_updates", "measured_updates",
                                      "inference_batches", "initialization_seconds",
                                      "warmup_seconds", "update1_seconds", "update2_seconds",
                                      "inference_seconds", "checkpoint_seconds")}


def _profile_timing(item: dict, elapsed: float) -> None:
    keys = ("initialization_seconds", "warmup_seconds", "update1_seconds",
            "update2_seconds", "inference_seconds", "checkpoint_seconds")
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or
            not 0 <= elapsed <= MAX_SECONDS or sum(item[key] for key in keys) > elapsed + 0.001):
        raise ValueError("U-Time production-batch timing exceeds monitored wall time")


def _monitor_coverage(monitors: dict, mode: str, elapsed: float) -> None:
    if set(monitors) != set(STAGES[mode]):
        raise ValueError("U-Time preflight monitored stage coverage differs")
    previous = 0.0
    for stage in STAGES[mode]:
        monitor = monitors[stage]
        peak = monitor.get("peak_combined_rss_bytes")
        seconds = monitor.get("elapsed_seconds")
        if (type(peak) is not int or not 0 < peak <= MAX_BYTES or
                type(seconds) not in (int, float) or not math.isfinite(seconds) or
                not 0 <= seconds <= MAX_SECONDS or seconds + 0.02 < previous or
                seconds > elapsed + 0.02):
            raise ValueError("U-Time native preflight monitor timing or resource bound differs")
        previous = seconds


def _recorded_path(value: str, parent: Path, name: str) -> Path:
    if type(value) is not str:
        raise ValueError("U-Time preflight artifact path is not a string")
    path = Path(value)
    if (not path.is_absolute() or path != path.resolve() or
            path.parent != parent.resolve() or path.name != name):
        raise ValueError("U-Time preflight artifact path escapes its attempt")
    return path


def run(root: Path, mode: str) -> dict:
    root = root.resolve()
    _lease_name(mode)
    with compute_lease(root, _lease_name(mode)):
        return _run_locked(root, mode)


def _run_locked(root: Path, mode: str) -> dict:
    base = root / BASE / mode
    attempt_dir = base / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
                                       "-" + uuid.uuid4().hex[:8])
    attempt_dir.mkdir(parents=True, exist_ok=False)
    current = base / "current.json"
    atomic_json(current, {"schema_version": "1.0", "artifact_type": "utime_preflight_pending",
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
            _check_stage_metadata(result, mode, stage, _identity(before, mode))
            results[stage] = result
            monitors[stage] = {key: monitor[key] for key in
                               ("peak_combined_rss_bytes", "elapsed_seconds")}
        summary = (_check_numerical(results, attempt_dir) if mode == "numerical" else
                   _check_profile(results, attempt_dir))
        _monitor_coverage(monitors, mode, time.monotonic() - started)
        if mode == "profile":
            _profile_timing(results["profile"], monitors["profile"]["elapsed_seconds"])
        after = _snapshot(root)
        if before != after or time.monotonic() - started > MAX_SECONDS:
            raise ValueError("U-Time preflight inputs changed or total deadline expired")
        status = "SUCCESS"
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    logs = {stage: {"path": str(path.resolve()), "sha256": file_sha256(path),
                    **monitors.get(stage, {})}
            for stage in STAGES[mode] if (path := attempt_dir / (stage + ".log")).exists()}
    if status == "SUCCESS":
        summary["peak_combined_rss_bytes"] = max(
            logs[stage]["peak_combined_rss_bytes"] for stage in STAGES[mode])
    attempt = {"schema_version": "1.0", "artifact_type": "utime_native_preflight_attempt",
               "mode": mode, "status": status, "before": before, "after": after,
               "results": results, "summary": summary, "logs": logs,
               "elapsed_seconds": time.monotonic() - started,
               "deadline_seconds": MAX_SECONDS, "failure": failure}
    atomic_json(attempt_dir / "attempt.json", attempt, immutable=True)
    if status != "SUCCESS":
        atomic_json(current, {"schema_version": "1.0", "artifact_type": "utime_preflight_failed",
                              "attempt_path": str((attempt_dir / "attempt.json").resolve())})
        raise RuntimeError("U-Time native preflight failed; attempt: " + str(attempt_dir))
    receipt = {"schema_version": "1.0", "artifact_type": "utime_native_preflight_receipt",
               "mode": mode, "verification_id": _identity(before, mode), "snapshot": before,
               "attempt_path": str((attempt_dir / "attempt.json").resolve()),
               "attempt_sha256": file_sha256(attempt_dir / "attempt.json")}
    receipt_path = base / "receipts" / (attempt_dir.name + ".json")
    atomic_json(receipt_path, receipt, immutable=True)
    atomic_json(current, {"schema_version": "1.0", "artifact_type": "utime_preflight_current",
                          "receipt_path": str(receipt_path.resolve()),
                          "receipt_sha256": file_sha256(receipt_path)})
    return _verify_mode(root, mode)


def _verify_mode(root: Path, mode: str) -> dict:
    base = root / BASE / mode
    current = read_json(base / "current.json")
    if (current.get("schema_version") != "1.0" or
            current.get("artifact_type") != "utime_preflight_current"):
        raise ValueError("U-Time native preflight has no current successful " + mode + " receipt")
    receipt_path = _recorded_path(current["receipt_path"], (base / "receipts").resolve(),
                                  Path(current["receipt_path"]).name)
    if file_sha256(receipt_path) != current["receipt_sha256"]:
        raise ValueError("U-Time native preflight receipt bytes changed")
    receipt = read_json(receipt_path)
    snapshot = _snapshot(root)
    if (receipt.get("schema_version") != "1.0" or
            receipt.get("artifact_type") != "utime_native_preflight_receipt" or
            receipt.get("mode") != mode or receipt.get("snapshot") != snapshot or
            receipt.get("verification_id") != _identity(snapshot, mode)):
        raise ValueError("U-Time native preflight source or runtime is stale")
    attempts = (base / "attempts").resolve()
    candidate = Path(receipt["attempt_path"])
    if (not candidate.is_absolute() or candidate != candidate.resolve() or
            candidate.parent.parent != attempts or
            re.fullmatch(r"[0-9]{8}T[0-9]{12}Z-[0-9a-f]{8}", candidate.parent.name) is None):
        raise ValueError("U-Time native preflight attempt path escapes run tree")
    attempt_path = _recorded_path(receipt["attempt_path"], candidate.parent,
                                  "attempt.json")
    newest = max((path for path in attempts.iterdir() if path.is_dir()), default=None)
    if (attempt_path.parent != newest or
            file_sha256(attempt_path) != receipt["attempt_sha256"]):
        raise ValueError("U-Time native preflight attempt was altered or superseded")
    attempt = read_json(attempt_path)
    if (attempt.get("schema_version") != "1.0" or
            attempt.get("artifact_type") != "utime_native_preflight_attempt" or
            attempt.get("mode") != mode or attempt.get("status") != "SUCCESS" or
            attempt.get("before") != snapshot or attempt.get("after") != snapshot or
            attempt.get("failure") is not None or attempt.get("deadline_seconds") != MAX_SECONDS or
            type(attempt.get("elapsed_seconds")) not in (int, float) or
            not math.isfinite(attempt["elapsed_seconds"]) or
            not 0 <= attempt["elapsed_seconds"] <= MAX_SECONDS):
        raise ValueError("U-Time native preflight attempt did not complete")
    results, logs = attempt["results"], attempt["logs"]
    if set(results) != set(STAGES[mode]) or set(logs) != set(STAGES[mode]):
        raise ValueError("U-Time native preflight stage coverage differs")
    for stage in STAGES[mode]:
        result = results[stage]
        log = _recorded_path(logs[stage]["path"], attempt_path.parent, stage + ".log")
        if (file_sha256(log) != logs[stage]["sha256"] or
                _parse_output(log.read_text(encoding="utf-8", errors="replace")) != result or
                result.get("verification_id") != _identity(snapshot, mode) or
                result.get("synthetic_scope") != "no_edf_truth_weights_or_fit" or
                type(logs[stage].get("peak_combined_rss_bytes")) is not int or
                not 0 < logs[stage]["peak_combined_rss_bytes"] <= MAX_BYTES or
                type(logs[stage].get("elapsed_seconds")) not in (int, float) or
                not math.isfinite(logs[stage]["elapsed_seconds"]) or
                not 0 <= logs[stage]["elapsed_seconds"] <= MAX_SECONDS):
            raise ValueError("U-Time native preflight worker log or scope changed")
        _check_stage_metadata(result, mode, stage, _identity(snapshot, mode))
    _monitor_coverage(logs, mode, attempt["elapsed_seconds"])
    summary = (_check_numerical(results, attempt_path.parent) if mode == "numerical" else
               _check_profile(results, attempt_path.parent))
    if mode == "profile":
        _profile_timing(results["profile"], logs["profile"]["elapsed_seconds"])
    summary["peak_combined_rss_bytes"] = max(
        logs[stage]["peak_combined_rss_bytes"] for stage in STAGES[mode])
    if summary != attempt["summary"]:
        raise ValueError("U-Time native preflight numerical summary changed")
    return {"mode": mode, "verification_id": receipt["verification_id"],
            "summary": summary, "receipt_path": str(receipt_path)}


def require_verification(root: Path, engine: str = "utime") -> dict:
    """Require current source-bound numerical and production batch profile receipts."""
    if engine != "utime":
        raise ValueError("Only pinned original U-Time has this native preflight")
    root = root.resolve()
    numerical = _verify_mode(root, "numerical")
    profile = _verify_mode(root, "profile")
    contract = {"artifact_type": "utime_native_preflight_contract", "engine": engine,
                "numerical_verification_id": numerical["verification_id"],
                "profile_verification_id": profile["verification_id"],
                "batch_size": 12, "threads": THREADS, "memory_limit_bytes": MAX_BYTES,
                "synthetic_scope": "no_edf_truth_weights_or_fit"}
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
