"""Isolated, synthetic CUDA 2.10 native U-Time batch-12 capacity check.

This is a capacity observation, not CPU/GPU numerical parity or fit authority.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid

from .contracts import content_id, json_text, read_json
from .research import atomic_json, compute_lease, file_sha256

BASE = Path("runs/utime-cuda210-capacity")
LEASE = "utime-cuda210-capacity"
PREFIX = "UTIME_CUDA210_CAPACITY_RESULT "
FAIL_PREFIX = "UTIME_CUDA210_CAPACITY_FAILURE "
IDENTITY_PREFIX = "UTIME_CUDA210_WORKER_IDENTITY "
PROFILE_BEGIN = "UTIME_CUDA210_NATIVE_PROFILE_BEGIN"
PROFILE_END = "UTIME_CUDA210_NATIVE_PROFILE_END"
SECONDS = 1800
HOST_BYTES = 10 * 1024**3
FREE_RAM = 4 * 1024**3
FREE_DISK = 20 * 1024**3
GPU_MIB = 3072
GPU_HEADROOM_MIB = GPU_MIB + 256
SOURCE_FILES = (
    "sleepedf/tf_cuda210_verify.py", "tools/verify_tf_cuda210.py",
    "tests/test_tf_cuda210_verify.py", "docs/NATIVE_TF_CUDA210_QUALIFICATION.md",
    "tools/verify_tf_native.py", "tools/tf_train_worker.py",
    "sleepedf/tf_native.py", "sleepedf/research.py", "sleepedf/contracts.py",
    "vendor/utime/utime/bin/defaults/utime/hparams.yaml",
)
RUNTIME = Path("research/runtimes/utime-cuda210")
NATIVE_RESULT = RUNTIME / "native-install-attempts/20260924T073456394927Z/result.json"
ENVIRONMENT = {name: "4" for name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")}
ENVIRONMENT.update(CUDA_VISIBLE_DEVICES="0", TF_DETERMINISTIC_OPS="1",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", TF_ENABLE_ONEDNN_OPTS="0",
                   TF_CPP_MIN_LOG_LEVEL="0")


def _output_base(root: Path) -> Path:
    root = root.resolve()
    runs = root / "runs"
    base = root / BASE
    targets = (runs, runs / "compute.lock", base, base / "attempts", base / "current.json")
    if any(path.resolve() != path for path in targets):
        raise ValueError("CUDA capacity output redirects outside canonical project runs")
    return base


def _limits(root: Path, deadline: float, observed_rss: int = 0) -> None:
    import psutil
    if (time.monotonic() >= deadline or
            psutil.Process().memory_info().rss + observed_rss > HOST_BYTES or
            psutil.virtual_memory().available < FREE_RAM or
            shutil.disk_usage(root).free < FREE_DISK):
        raise RuntimeError("CUDA capacity exceeded time, host memory or disk bound")


def _sha(path: Path, root: Path, deadline: float) -> str:
    import hashlib
    digest = hashlib.sha256()
    since_check = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            since_check += len(chunk)
            if since_check >= 16 * 1024**2:
                _limits(root, deadline)
                since_check = 0
    if time.monotonic() >= deadline:
        raise RuntimeError("CUDA capacity source hash exceeded deadline")
    return digest.hexdigest()


def _locked_packages(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, version = line.split("==")
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key in result or not version:
            raise ValueError("CUDA runtime lock has duplicate or invalid package")
        result[key] = version
    return result


def _package_versions(site: Path) -> dict[str, str]:
    result = {}
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        key = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        if key in result:
            raise ValueError("CUDA runtime has duplicate installed package")
        result[key] = distribution.version
    return result


def _installed_path(site: Path, venv: Path, relative: str) -> Path:
    """Resolve wheel RECORD paths, including legitimate ../../Scripts entries."""
    if type(relative) is not str:
        raise ValueError("CUDA installed package RECORD path is not a string")
    item = Path(relative)
    if item.is_absolute() or item.drive:
        raise ValueError("CUDA installed package RECORD path is not relative")
    path = (site / item).resolve()
    if not path.is_relative_to(venv.resolve()):
        raise ValueError("CUDA installed package RECORD path escapes GPU venv")
    return path


def _require_install_evidence(root: Path, installed: dict, path: Path,
                              deadline: float) -> str:
    expected = installed.get("evidence", {}).get(path.relative_to(root).as_posix())
    observed = _sha(path, root, deadline)
    if type(expected) is not str or expected != observed:
        raise ValueError("CUDA installation evidence hash differs: " + path.name)
    return observed


def _snapshot(root: Path, deadline: float) -> dict:
    """Verify installed package and private CUDA bytes against archived evidence."""
    root = root.resolve()
    _limits(root, deadline)
    lock = root / "requirements/utime-cuda210.lock.txt"
    installation = root / RUNTIME / "install.json"
    installed_files = root / RUNTIME / "installed-files.json"
    native_path = root / NATIVE_RESULT
    authorization = root / RUNTIME / "authorization.json"
    source_manifest = root / "research/sources/utime.json"
    installed = read_json(installation)
    if (installed.get("status") !=
            "INSTALLED_SYNTHETIC_DEPENDENCY_CHECK_PASS_NATIVE_QUALIFICATION_NOT_RUN" or
            installed.get("freeze_sha256") != _sha(lock, root, deadline)):
        raise ValueError("CUDA installation evidence differs from exact lock")
    installed_files_sha = _require_install_evidence(root, installed, installed_files, deadline)
    native_sha = _require_install_evidence(root, installed, native_path, deadline)
    authorization_sha = _require_install_evidence(root, installed, authorization, deadline)
    expected = _locked_packages(lock)
    venv = root / ".venvs/utime-cuda210"
    site = venv / "Lib/site-packages"
    if venv.resolve() != venv:
        raise ValueError("CUDA runtime venv redirects outside canonical location")
    if (len(expected) != 92 or _package_versions(site) != expected or
            installed.get("package_count") != 92 or installed.get("python") != "3.9.25"):
        raise ValueError("CUDA installed package versions differ from exact lock")
    inventory = read_json(installed_files)
    if len(inventory.get("packages", [])) != 92:
        raise ValueError("CUDA installed-file inventory is incomplete")
    checked = 0
    for package in inventory["packages"]:
        for relative, expected_sha in package["verified_files"]:
            path = _installed_path(site, venv, relative)
            if _sha(path, root, deadline) != expected_sha:
                raise ValueError("CUDA installed package file changed")
            checked += 1
            if checked % 64 == 0:
                _limits(root, deadline)
    if checked != sum(item["verified_file_count"] for item in inventory["packages"]):
        raise ValueError("CUDA installed package file count changed")
    native = read_json(native_path)
    destination = root / native["destination"]
    if (native.get("status") != "EXTRACTED_HASHED_NOT_GPU_QUALIFIED" or
            len(native.get("files", [])) != 97 or destination.resolve() != destination or
            installed.get("native_hashed_file_count") != 97):
        raise ValueError("Private CUDA library installation differs")
    for item in native["files"]:
        path = root / item["path"]
        if (path.resolve() != path or not path.is_relative_to(destination) or
                path.stat().st_size != item["size"] or
                _sha(path, root, deadline) != item["sha256"]):
            raise ValueError("Private CUDA library hash differs")
        _limits(root, deadline)
    source = read_json(source_manifest)
    if source.get("weights_downloaded") is not False:
        raise ValueError("Pinned native U-Time source enables external weights")
    for relative, item in source["files"].items():
        if _sha(root / "vendor/utime" / relative, root, deadline) != item["sha256"]:
            raise ValueError("Pinned native U-Time source changed")
    files = {name: _sha(root / name, root, deadline) for name in SOURCE_FILES}
    _limits(root, deadline)
    return {"schema_version": "1.0", "artifact_type": "utime_cuda210_capacity_inputs",
            "scope": "synthetic_batch12_capacity_only",
            "source_sha256": files, "source_manifest_sha256": file_sha256(source_manifest),
            "runtime_lock_sha256": file_sha256(lock),
            "runtime_install_sha256": file_sha256(installation),
            "installed_files_sha256": installed_files_sha,
            "installed_file_count": checked, "package_versions": expected,
            "native_install_sha256": native_sha,
            "authorization_sha256": authorization_sha,
            "native_file_count": len(native["files"]),
            "gpu_executable_sha256": file_sha256(root / ".venvs/utime-cuda210/Scripts/python.exe"),
            "tensorflow": "2.10.1", "batch_size": 12, "logical_gpu_limit_mib": GPU_MIB,
            "threads": 4, "max_seconds": SECONDS, "max_host_bytes": HOST_BYTES,
            "min_free_ram_bytes": FREE_RAM, "min_free_disk_bytes": FREE_DISK}


def _gpu_observation() -> dict:
    output = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            check=True, timeout=30).stdout.strip().splitlines()
    if len(output) != 1:
        raise RuntimeError("CUDA capacity needs exactly one visible physical GPU")
    columns = [item.strip() for item in output[0].split(",")]
    if len(columns) != 4:
        raise ValueError("CUDA capacity GPU inventory has four expected columns")
    total, free = int(columns[2]), int(columns[3])
    if not columns[0] or not columns[1] or total <= 0 or not 0 <= free <= total:
        raise ValueError("CUDA capacity GPU inventory is invalid")
    return {"name": columns[0], "driver": columns[1],
            "total_mib": total, "free_mib_before": free}


class GPUHeadroomUnavailable(RuntimeError):
    pass


def _require_gpu_headroom(observation: dict) -> None:
    if observation["name"] != "NVIDIA T500":
        raise ValueError("CUDA capacity requires the recorded NVIDIA T500")
    if observation["free_mib_before"] < GPU_HEADROOM_MIB:
        raise GPUHeadroomUnavailable(
            "CUDA capacity prelaunch GPU memory is below " + str(GPU_HEADROOM_MIB) +
            " MiB: observed " + str(observation["free_mib_before"]) + " MiB")


def _expose_keras210_jit_setting(model_type, tensorflow_version: str) -> None:
    """Expose Keras 2.10's actual compile flag for the frozen CPU assertion."""
    if tensorflow_version != "2.10.1" or hasattr(model_type, "jit_compile"):
        raise ValueError("Keras compile-setting compatibility scope changed")

    def setting(model):
        value = model._jit_compile
        if type(value) is not bool:
            raise ValueError("Keras compile flag is not an explicit boolean")
        return value

    model_type.jit_compile = property(setting)


def _request_path(root: Path, request_path: Path) -> Path:
    base = _output_base(root) / "attempts"
    if (not request_path.is_absolute() or request_path != request_path.resolve() or
            request_path.name != "request.json" or request_path.parent.parent != base):
        raise ValueError("CUDA capacity request escapes its immutable attempt")
    return request_path.parent


def _worker_identity(root: Path, request_path: Path, request: dict) -> dict:
    import psutil
    attempt = _request_path(root, request_path)
    lease = read_json(root / "runs/compute.lock")
    owner_pid, owner_birth = request.get("owner_pid"), request.get("owner_birth")
    command = [str(root / ".venvs/utime-cuda210/Scripts/python.exe"), "-I", "-B",
               str(root / "tools/verify_tf_cuda210.py"), "--child", str(request_path)]
    if (request.get("schema_version") != "1.0" or
            request.get("artifact_type") != "utime_cuda210_capacity_request" or
            request.get("root") != str(root) or
            request.get("attempt_dir") != str(attempt) or request.get("command") != command or
            request.get("lease_run_id") != LEASE or request.get("scope") !=
            "synthetic_batch12_capacity_only" or type(owner_pid) is not int or
            type(owner_birth) not in (int, float) or not math.isfinite(owner_birth) or
            lease.get("pid") != owner_pid or lease.get("process_start") != owner_birth or
            lease.get("host") != socket.gethostname() or lease.get("run_id") != LEASE):
        raise RuntimeError("CUDA capacity request has no exact compute lease")
    owner = psutil.Process(owner_pid)
    current = psutil.Process()
    parent = current.parent()
    if owner.create_time() != owner_birth or parent is None:
        raise RuntimeError("CUDA capacity owner PID/birth is no longer live")
    if parent.pid != owner_pid:
        if (Path(parent.exe()).resolve() != Path(command[0]).resolve() or
                parent.cmdline() != command or parent.parent() is None or
                parent.parent().pid != owner_pid):
            raise RuntimeError("CUDA capacity worker lacks exact venv launcher ancestry")
    if request.get("request_sha256") is not None:
        raise ValueError("CUDA capacity request cannot self-declare its digest")
    return {"pid": current.pid, "birth": current.create_time(), "ppid": parent.pid,
            "owner_pid": owner_pid, "owner_birth": owner_birth,
            "request_sha256": file_sha256(request_path)}


def _classify(exc: BaseException, phase: str) -> str:
    if phase == "authorization":
        return "PROVENANCE"
    if type(exc).__name__ in ("ResourceExhaustedError", "OutOfMemoryError", "MemoryError"):
        return "OOM"
    if isinstance(exc, ImportError) or (isinstance(exc, OSError) and phase == "import"):
        return "IMPORT"
    if isinstance(exc, (AttributeError, TypeError)) or type(exc).__name__ in (
            "InvalidArgumentError", "UnimplementedError", "NotFoundError"):
        return "API"
    return "WORKER"


def _strict_line(text: str) -> dict:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate CUDA capacity metadata key")
            value[key] = item
        return value
    result = json.loads(text, object_pairs_hook=unique,
                        parse_constant=lambda _: (_ for _ in ()).throw(
                            ValueError("Non-finite CUDA capacity metadata")))
    if type(result) is not dict:
        raise ValueError("CUDA capacity marker is not an object")
    json_text(result)
    return result


def _placement_from_log(path: Path) -> dict[str, int]:
    """Require scoped forward/backward device lines from the actual profile."""
    started = ended = False
    counts = {"forward_gpu_lines": 0, "backward_gpu_lines": 0,
              "other_device_lines": 0, "unplaced_lines": 0,
              "scoped_forward_gpu_lines": 0, "scoped_backward_gpu_lines": 0}
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.strip() == PROFILE_BEGIN:
                if started or ended:
                    raise ValueError("CUDA native profile begin marker is duplicated")
                started = True
                continue
            if line.strip() == PROFILE_END:
                if not started or ended:
                    raise ValueError("CUDA native profile end marker is misplaced")
                ended = True
                continue
            if not started or ended or re.search(
                    r"\b(?:_FusedConv2D|Conv2D(?:Backprop(?:Input|Filter)(?:V2)?)?)\b",
                    line) is None:
                continue
            scoped = any(name in line for name in ("encoder_", "decoder_", "sequence_conv_"))
            if "/device:GPU:0" in line:
                name = "backward_gpu_lines" if "Conv2DBackprop" in line else "forward_gpu_lines"
                counts[name] += 1
                if scoped:
                    counts["scoped_" + name] += 1
            elif "/device:" in line:
                counts["other_device_lines"] += 1
            else:
                counts["unplaced_lines"] += 1
    if (not started or not ended or counts["forward_gpu_lines"] < 1 or
            counts["backward_gpu_lines"] < 1 or
            counts["scoped_forward_gpu_lines"] < 1 or
            counts["scoped_backward_gpu_lines"] < 1 or
            counts["other_device_lines"] or counts["unplaced_lines"]):
        raise ValueError("Actual native profile Conv2D/backprop GPU placement is unknown or mixed")
    return counts


def worker_main(request_path: Path) -> None:
    """Authenticate first, then import TF and invoke the unchanged native profile."""
    root = Path(__file__).resolve().parents[1]
    phase = "authorization"
    try:
        request_path = request_path.resolve()
        request = read_json(request_path)
        identity = _worker_identity(root, request_path, request)
        if (Path(sys.prefix).resolve() != (root / ".venvs/utime-cuda210").resolve() or
                sys.version_info[:2] != (3, 9) or
                any(os.environ.get(name) != value for name, value in ENVIRONMENT.items())):
            raise RuntimeError("CUDA capacity worker runtime or deterministic environment differs")
        remaining = request.get("deadline_remaining_seconds")
        if (type(remaining) not in (int, float) or not math.isfinite(remaining) or
                not 0 < remaining <= SECONDS):
            raise ValueError("CUDA capacity request has invalid remaining deadline")
        deadline = time.monotonic() + remaining
        if request.get("snapshot") != _snapshot(root, deadline):
            raise ValueError("CUDA capacity source or installed runtime changed")
        print(IDENTITY_PREFIX + json.dumps(identity, sort_keys=True, allow_nan=False), flush=True)
        phase = "import"
        native = read_json(root / NATIVE_RESULT)
        base = root / native["destination"]
        folders = [base / "cudatoolkit/Library/bin", base / "cudnn/Library/bin"]
        handles = [os.add_dll_directory(str(path)) for path in folders]
        os.environ["PATH"] = os.pathsep.join(map(str, folders)) + os.pathsep + os.environ.get("PATH", "")
        import tensorflow as tf
        if tf.__version__ != "2.10.1":
            raise ImportError("CUDA capacity requires TensorFlow 2.10.1")
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(4)
        tf.config.experimental.enable_op_determinism()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.config.set_soft_device_placement(False)
        devices = tf.config.list_physical_devices("GPU")
        if len(devices) != 1:
            raise RuntimeError("CUDA capacity has no single GPU device")
        tf.config.set_logical_device_configuration(devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MIB)])
        phase = "profile"
        sys.path.insert(0, str(root / "vendor/utime"))
        sys.path.insert(0, str(root / "tools"))
        from verify_tf_native import _profile_stage
        from utime.models import UTime
        _expose_keras210_jit_setting(UTime, tf.__version__)
        print(PROFILE_BEGIN, flush=True)
        tf.debugging.set_log_device_placement(True)
        try:
            profile = _profile_stage(tf, request)
        finally:
            tf.debugging.set_log_device_placement(False)
        print(PROFILE_END, flush=True)
        memory = tf.config.experimental.get_memory_info("GPU:0")
        if (memory["peak"] > GPU_MIB * 1024**2 or not handles or
                tf.config.experimental.tensor_float_32_execution_enabled() or
                tf.keras.mixed_precision.global_policy().name != "float32"):
            raise RuntimeError("CUDA capacity logical memory or float32 policy differs")
        maps = [str(item.path) for item in __import__("psutil").Process().memory_maps()]
        loaded = [path for path in maps if Path(path).name.lower().startswith(
                  ("cudnn", "cublas", "cudart"))]
        if (not any(Path(path).name.lower().startswith("cudnn") for path in loaded) or
                not any(Path(path).name.lower().startswith("cublas") for path in loaded) or
                any(not Path(path).resolve().is_relative_to(base) for path in loaded)):
            raise RuntimeError("CUDA capacity did not load only private cuDNN/cuBLAS libraries")
        result = {"schema_version": "1.0", "artifact_type": "utime_cuda210_capacity_result",
                  "scope": "synthetic_batch12_capacity_only", "status": "PASS_CAPACITY_ONLY",
                  "snapshot_id": content_id(request["snapshot"]),
                  "batch_size": 12, "logical_gpu_limit_mib": GPU_MIB,
                  "tensorflow": tf.__version__,
                  "keras210_jit_compile_readonly_alias": True,
                  "placement_policy": "native_placer_host_ops_gpu_convolutions_verified_v1",
                  "profile": profile, "gpu_memory": memory,
                  "loaded_cuda_libraries": loaded,
                  "tf32": False, "float32_policy": "float32", "fit_authorized": False}
        print(PREFIX + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        print(FAIL_PREFIX + json.dumps({"class": _classify(exc, phase),
              "phase": phase, "type": type(exc).__name__, "message": str(exc)},
              sort_keys=True, allow_nan=False), flush=True)
        raise SystemExit(1)


def _validate_result(result: dict, snapshot: dict, attempt: Path) -> None:
    if (result.get("artifact_type") != "utime_cuda210_capacity_result" or
            result.get("status") != "PASS_CAPACITY_ONLY" or
            result.get("scope") != "synthetic_batch12_capacity_only" or
            result.get("snapshot_id") != content_id(snapshot) or
            result.get("batch_size") != 12 or result.get("logical_gpu_limit_mib") != GPU_MIB or
            result.get("tensorflow") != "2.10.1" or result.get("fit_authorized") is not False or
            result.get("keras210_jit_compile_readonly_alias") is not True or
            result.get("placement_policy") != "native_placer_host_ops_gpu_convolutions_verified_v1" or
            result.get("tf32") is not False or result.get("float32_policy") != "float32"):
        raise ValueError("CUDA capacity worker result identity differs")
    placement = result["native_profile_placement"]
    if (placement.get("forward_gpu_lines", 0) < 1 or
            placement.get("backward_gpu_lines", 0) < 1 or
            placement.get("other_device_lines", 0) != 0 or
            placement.get("unplaced_lines", 0) != 0 or
            placement.get("scoped_forward_gpu_lines", 0) < 1 or
            placement.get("scoped_backward_gpu_lines", 0) < 1):
        raise ValueError("Actual native Conv2D/backprop lacked unambiguous GPU placement")
    profile = result["profile"]
    if (profile.get("batch_size") != 12 or profile.get("warmup_updates") != 1 or
            profile.get("measured_updates") != 2 or profile.get("inference_batches") != 1 or
            profile.get("optimizer_iterations") != 3 or profile.get("train_counter") != 3 or
            len(profile.get("profile_batch_hashes", [])) != 3):
        raise ValueError("Native batch-12 profile coverage differs")
    for name in ("initialization_seconds", "warmup_seconds", "update1_seconds",
                 "update2_seconds", "inference_seconds", "checkpoint_seconds"):
        value = profile.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("CUDA capacity timing is non-finite or negative")
    for item in profile["profile_batch_hashes"]:
        if (type(item) is not dict or type(item.get("valid_targets")) is not int or
                not 0 < item["valid_targets"] <= 12 * 35 or
                type(item.get("reported_mean_loss")) not in (int, float) or
                not math.isfinite(item["reported_mean_loss"]) or
                any(type(item.get(name)) is not str or
                    re.fullmatch(r"[0-9a-f]{64}", item[name]) is None for name in
                    ("raw_sha256", "augmented_sha256"))):
            raise ValueError("CUDA capacity synthetic batch evidence differs")
    files = profile.get("checkpoint_files")
    if (type(files) is not dict or len(files) < 2 or
            not any(name.endswith(".index") for name in files) or
            not any(".data-" in name for name in files)):
        raise ValueError("CUDA capacity checkpoint coverage differs")
    for name, expected in files.items():
        path = Path(name)
        if (path.parent != attempt or path != path.resolve() or
                not path.name.startswith("profile-synthetic-checkpoint.") or
                file_sha256(path) != expected):
            raise ValueError("CUDA capacity checkpoint hash differs")
    memory = result["gpu_memory"]
    native = read_json(attempt.parents[3] / NATIVE_RESULT)
    destination = (attempt.parents[3] / native["destination"]).resolve()
    loaded = result.get("loaded_cuda_libraries")
    if (type(memory.get("peak")) is not int or
            not 0 < memory["peak"] <= GPU_MIB * 1024**2 or
            type(memory.get("current")) is not int or memory["current"] < 0 or
            type(loaded) is not list or not loaded or
            not any(Path(path).name.lower().startswith("cudnn") for path in loaded) or
            not any(Path(path).name.lower().startswith("cublas") for path in loaded) or
            any(type(path) is not str or not Path(path).resolve().is_relative_to(destination)
                for path in loaded)):
        raise ValueError("CUDA capacity GPU memory or library evidence differs")


def _monitor(child: subprocess.Popen, log_path: Path, root: Path, request: dict,
             deadline: float) -> tuple[int, dict | None, dict | None]:
    import psutil
    owner = psutil.Process()
    observed: dict[int, float] = {}
    identity = result = failure = None
    peak = 0
    error = None
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        try:
            while True:
                for line in stream.readlines():
                    if line.startswith(IDENTITY_PREFIX):
                        if identity is not None:
                            raise RuntimeError("Duplicate CUDA capacity worker identity")
                        identity = _strict_line(line[len(IDENTITY_PREFIX):])
                        if (type(identity.get("pid")) is not int or identity["pid"] < 1 or
                                type(identity.get("birth")) not in (int, float) or
                                not math.isfinite(identity["birth"]) or
                                type(identity.get("ppid")) is not int):
                            raise RuntimeError("CUDA capacity worker PID/birth marker is invalid")
                        worker = psutil.Process(identity["pid"])
                        if (worker.create_time() != identity.get("birth") or
                                worker.ppid() != identity.get("ppid") or
                                identity.get("owner_pid") != owner.pid or
                                identity.get("owner_birth") != owner.create_time() or
                                identity.get("request_sha256") != request["request_sha256"] or
                                worker.ppid() not in (owner.pid, child.pid)):
                            raise RuntimeError("CUDA capacity worker PID/birth/lease handshake differs")
                        if worker.ppid() == child.pid and child.pid != owner.pid:
                            launcher = psutil.Process(child.pid)
                            launcher_parent = launcher.parent()
                            if (Path(launcher.exe()).resolve() != Path(request["command"][0]).resolve() or
                                    launcher.cmdline() != request["command"] or
                                    launcher_parent is None or launcher_parent.pid != owner.pid or
                                    launcher_parent.create_time() != owner.create_time()):
                                raise RuntimeError("CUDA capacity launcher ancestry differs")
                            observed[launcher.pid] = launcher.create_time()
                        observed[worker.pid] = worker.create_time()
                    elif line.startswith(PREFIX):
                        if result is not None or identity is None:
                            raise RuntimeError("CUDA capacity result has no unique live worker identity")
                        result = _strict_line(line[len(PREFIX):])
                    elif line.startswith(FAIL_PREFIX):
                        if failure is not None:
                            raise RuntimeError("Duplicate CUDA capacity failure marker")
                        failure = _strict_line(line[len(FAIL_PREFIX):])
                if log_path.stat().st_size > 256 * 1024**2:
                    raise RuntimeError("CUDA capacity worker log exceeded 256 MiB resource bound")
                try:
                    launcher = psutil.Process(child.pid)
                    if launcher.is_running():
                        observed[launcher.pid] = launcher.create_time()
                        for item in launcher.children(recursive=True):
                            observed[item.pid] = item.create_time()
                except psutil.NoSuchProcess:
                    pass
                except psutil.AccessDenied as exc:
                    raise RuntimeError("CUDA capacity launcher tree became inaccessible") from exc
                if identity is not None:
                    try:
                        worker = psutil.Process(identity["pid"])
                        if worker.create_time() == identity["birth"]:
                            for item in worker.children(recursive=True):
                                observed[item.pid] = item.create_time()
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.AccessDenied as exc:
                        raise RuntimeError("CUDA capacity worker tree became inaccessible") from exc
                live = []
                for pid, birth in observed.items():
                    try:
                        item = psutil.Process(pid)
                        if item.create_time() == birth and item.is_running():
                            live.append(item)
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.AccessDenied as exc:
                        raise RuntimeError("CUDA capacity observed process inaccessible: " +
                                           repr((pid, birth))) from exc
                rss = 0
                for item in live:
                    try:
                        rss += item.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.AccessDenied as exc:
                        raise RuntimeError("CUDA capacity observed RSS inaccessible: " +
                                           repr((item.pid, observed[item.pid]))) from exc
                peak = max(peak, owner.memory_info().rss + rss)
                _limits(root, deadline, rss)
                if child.poll() is not None and not live:
                    break
                time.sleep(.25)
        except BaseException as exc:
            error = exc
    unresolved = []
    survivors = []
    for pid, birth in observed.items():
        try:
            item = psutil.Process(pid)
            if item.create_time() == birth and item.is_running():
                item.terminate()
                survivors.append(item)
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            unresolved.append((pid, birth))
    try:
        _, remaining = psutil.wait_procs(survivors, timeout=3)
    except psutil.AccessDenied:
        unresolved.extend((item.pid, observed.get(item.pid)) for item in survivors)
        remaining = survivors
    for item in remaining:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            unresolved.append((item.pid, observed.get(item.pid)))
    try:
        _, remaining = psutil.wait_procs(remaining, timeout=3)
    except psutil.AccessDenied:
        unresolved.extend((item.pid, observed.get(item.pid)) for item in remaining)
    if child.poll() is None:
        try:
            child.kill()
        except PermissionError:
            unresolved.append((child.pid, observed.get(child.pid)))
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        unresolved.append((child.pid, observed.get(child.pid)))
    if remaining or unresolved:
        raise RuntimeError("CUDA capacity cleanup left live or inaccessible identities: " +
                           repr([(item.pid, observed.get(item.pid)) for item in remaining] +
                                unresolved)) from error
    if error is not None:
        raise error
    return peak, result, failure


def run(root: Path) -> dict:
    root = root.resolve()
    base = _output_base(root)
    started = time.monotonic()
    deadline = started + SECONDS
    with compute_lease(root, LEASE) as lease:
        attempt = base / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
                                       "-" + uuid.uuid4().hex[:8])
        attempt.mkdir(parents=True, exist_ok=False)
        current = base / "current.json"
        atomic_json(current, {"status": "STARTED", "attempt_path": str(attempt.resolve())})
        before = after = result = gpu = None
        peak = 0
        failure = None
        status = "FAILED"
        log = attempt / "worker.log"
        try:
            before = _snapshot(root, deadline)
            gpu = _gpu_observation()
            _require_gpu_headroom(gpu)
            request_path = attempt / "request.json"
            command = [str(root / ".venvs/utime-cuda210/Scripts/python.exe"), "-I", "-B",
                       str(root / "tools/verify_tf_cuda210.py"), "--child", str(request_path.resolve())]
            request = {"schema_version": "1.0", "artifact_type": "utime_cuda210_capacity_request",
                       "scope": "synthetic_batch12_capacity_only", "attempt_dir": str(attempt.resolve()),
                       "root": str(root), "snapshot": before, "gpu_observation": gpu,
                       "lease_run_id": LEASE, "owner_pid": lease["pid"],
                       "owner_birth": lease["process_start"], "command": command,
                       "deadline_remaining_seconds": deadline - time.monotonic()}
            atomic_json(request_path, request, immutable=True)
            request["request_sha256"] = file_sha256(request_path)
            env = os.environ.copy()
            env.update(ENVIRONMENT)
            _limits(root, deadline)
            with log.open("xb") as handle:
                child = subprocess.Popen(command, cwd=root, env=env, stdout=handle,
                                         stderr=subprocess.STDOUT)
                peak, result, worker_failure = _monitor(child, log, root, request, deadline)
            if child.returncode != 0:
                if worker_failure is None:
                    raise RuntimeError("CUDA capacity worker failed without classified marker")
                raise RuntimeError("CUDA capacity worker " + worker_failure["class"] + ": " +
                                   worker_failure["type"] + ": " + worker_failure["message"])
            if worker_failure is not None or result is None:
                raise ValueError("CUDA capacity worker returned ambiguous result")
            result["native_profile_placement"] = _placement_from_log(log)
            _validate_result(result, before, attempt)
            after = _snapshot(root, deadline)
            _limits(root, deadline)
            if before != after:
                raise ValueError("CUDA capacity source or runtime changed during attempt")
            status = "PASS_CAPACITY_ONLY"
        except BaseException as exc:
            message = str(exc)
            kind = "RESOURCE" if isinstance(exc, GPUHeadroomUnavailable) or "bound" in message or "deadline" in message else (
                "OOM" if " OOM:" in message else "API" if " API:" in message else
                "IMPORT" if " IMPORT:" in message else "PROVENANCE" if
                " PROVENANCE:" in message else "WORKER")
            failure = {"class": kind, "type": type(exc).__name__, "message": message}
            if isinstance(exc, GPUHeadroomUnavailable):
                failure.update(stage="prelaunch_gpu", resource="free_gpu_memory_mib",
                               required_mib=GPU_HEADROOM_MIB,
                               observed_mib=gpu["free_mib_before"])
        elapsed = time.monotonic() - started
        if status == "PASS_CAPACITY_ONLY" and elapsed > SECONDS:
            status = "FAILED"
            failure = {"class": "RESOURCE", "type": "TimeoutError",
                       "message": "CUDA capacity total attempt exceeded 1800 seconds"}
        record = {"schema_version": "1.0", "artifact_type": "utime_cuda210_capacity_attempt",
                  "status": status, "scope": "synthetic_batch12_capacity_only",
                  "before": before, "after": after, "gpu_observation": gpu,
                  "result": result, "failure": failure, "peak_combined_rss_bytes": peak,
                  "elapsed_seconds": elapsed, "deadline_seconds": SECONDS,
                  "log_sha256": file_sha256(log) if log.exists() else None,
                  "request_sha256": file_sha256(attempt / "request.json") if
                  (attempt / "request.json").exists() else None,
                  "fit_authorized": False}
        atomic_json(attempt / "attempt.json", record, immutable=True)
        atomic_json(current, {"status": status,
                              "attempt_path": str((attempt / "attempt.json").resolve()),
                              "attempt_sha256": file_sha256(attempt / "attempt.json")})
        if status != "PASS_CAPACITY_ONLY":
            raise RuntimeError("CUDA capacity failed; immutable attempt: " + str(attempt))
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = run(args.root)
    print(json.dumps({"status": report["status"], "fit_authorized": False},
                     sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
