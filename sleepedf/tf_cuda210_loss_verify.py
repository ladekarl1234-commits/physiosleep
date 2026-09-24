"""Bounded native-loss placement experiment; synthetic tensors, no U-Time model."""

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
import sys
import time
import uuid

import numpy as np

from . import tf_cuda210_verify as capacity
from .contracts import content_id, read_json
from .research import atomic_json, compute_lease, file_sha256


BASE = Path("runs/utime-cuda210-loss-probe")
LEASE = "utime-cuda210-native-loss-probe"
IDENTITY = "UTIME_CUDA210_LOSS_IDENTITY "
READY = "UTIME_CUDA210_LOSS_READY "
RESULT = "UTIME_CUDA210_LOSS_RESULT "
FAILURE = "UTIME_CUDA210_LOSS_FAILURE "
PLACEMENT_BEGIN = "UTIME_CUDA210_LOSS_PLACEMENT_BEGIN"
PLACEMENT_END = "UTIME_CUDA210_LOSS_PLACEMENT_END"
CASE_BEGIN = "UTIME_CUDA210_LOSS_CASE_BEGIN "
CASE_END = "UTIME_CUDA210_LOSS_CASE_END "
TOTAL_SECONDS, STEP_SECONDS = 900, 120
SOURCE_FILES = ("sleepedf/tf_cuda210_loss_verify.py", "tools/verify_tf_cuda210_loss.py",
                "tests/test_tf_cuda210_loss_verify.py", "docs/NATIVE_TF_CUDA210_LOSS_PROBE.md")
STAGES = ("cpu", "gpu1", "gpu2")
ATOL, RTOL = 1e-7, 1e-5


def _base(root: Path) -> Path:
    root = root.resolve()
    base = root / BASE
    if any(path.resolve() != path for path in
           (root / "runs", root / "runs/compute.lock", base, base / "attempts")):
        raise ValueError("Loss probe output redirects outside canonical runs")
    return base


def _snapshot(root: Path, deadline: float) -> dict:
    original = capacity._snapshot(root, deadline)
    additional = {name: capacity._sha(root / name, root, deadline) for name in SOURCE_FILES}
    return {"capacity_runtime": original, "loss_probe_sources": additional,
            "scope": "native_loss_cpu_scope_synthetic_only", "total_seconds": TOTAL_SECONDS,
            "step_seconds": STEP_SECONDS, "fit_authorized": False}


def _fixture(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name not in ("all_valid", "mixed", "extreme"):
        raise ValueError("Unknown fixed native loss fixture")
    logits = np.random.default_rng(731 + ("all_valid", "mixed", "extreme").index(name)).normal(
        size=(12, 35, 5)).astype(np.float32)
    labels = (np.arange(12 * 35) % 5).astype(np.int32).reshape(12, 35, 1)
    if name == "mixed":
        labels.reshape(-1)[[0, 2, 31, 211]] = -1
        labels.reshape(-1)[[1, 3, 173, 419]] = 5
    if name == "extreme":
        logits[0, 0] = [100., -100., -90., -80., -70.]
        labels[0, 0, 0] = 1
    mask = (labels[..., 0] >= 0) & (labels[..., 0] < 5)
    return logits, labels, mask


def _require_valid(mask: np.ndarray) -> int:
    count = int(np.count_nonzero(mask))
    if mask.shape != (12, 35) or count == 0:
        raise ValueError("All-invalid native loss batch is rejected before update")
    return count


def _array_check(arrays: dict[str, np.ndarray]) -> None:
    expected = set()
    for name in ("all_valid", "mixed", "extreme"):
        expected.update({name + suffix for suffix in ("_logits", "_labels", "_mask", "_loss",
                                                         "_objective", "_gradient")})
    if set(arrays) != expected:
        raise ValueError("Native loss retained array keys differ")
    for name in ("all_valid", "mixed", "extreme"):
        logits, labels, mask = _fixture(name)
        stored = (arrays[name + "_logits"], arrays[name + "_labels"], arrays[name + "_mask"])
        if any(left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes()
               for left, right in zip(stored, (logits, labels, mask))):
            raise ValueError("Native loss fixture bytes, dtype or mask changed")
        valid = _require_valid(mask)
        losses, objective, gradient = (arrays[name + suffix] for suffix in
                                        ("_loss", "_objective", "_gradient"))
        if (losses.shape != (valid,) or objective.shape != () or gradient.shape != logits.shape or
            any(item.dtype != np.float32 or not np.isfinite(item).all()
                for item in (losses, objective, gradient)) or
            not np.any(gradient[mask] != 0) or
            np.any(gradient[~mask] != 0)):
            raise ValueError("Native loss/gradient shape, finite value or invalid-mask contract differs")


def _compare(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray], *, exact: bool) -> None:
    _array_check(candidate)
    _array_check(reference)
    for key in reference:
        left, right = candidate[key], reference[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("Native loss comparison changed array shape or dtype: " + key)
        if exact or not key.endswith(("_loss", "_objective", "_gradient")):
            if left.tobytes() != right.tobytes():
                raise ValueError("Fresh native loss repeats differ in bytes: " + key)
        elif not np.all(np.abs(left.astype(np.float64) - right.astype(np.float64)) <=
                        ATOL + RTOL * np.abs(right.astype(np.float64))):
            raise ValueError("Native loss CPU-scope numerical tolerance exceeded: " + key)


def _graph_inventory(concrete) -> dict:
    graph = concrete.graph.as_graph_def()
    def item(node):
        refs = []
        colocation = []
        for key, value in node.attr.items():
            if value.WhichOneof("value") == "func":
                refs.append(value.func.name)
            elif value.WhichOneof("value") == "list":
                refs.extend(function.name for function in value.list.func)
                if key == "_class":
                    colocation.extend(value.decode("utf-8", errors="replace") for value in value.list.s)
        return {"name": node.name, "op": node.op, "device": node.device,
                "function_refs": sorted(set(refs)), "colocation": colocation}
    top = [item(node) for node in graph.node]
    functions = {function.signature.name: [item(node) for node in function.node_def]
                 for function in graph.library.function}
    reachable = set()
    queue = [ref for node in top for ref in [node["op"], *node["function_refs"]]
             if ref in functions]
    while queue:
        name = queue.pop()
        if name in reachable:
            continue
        reachable.add(name)
        queue.extend(ref for node in functions[name] for ref in [node["op"], *node["function_refs"]]
                     if ref in functions and ref not in reachable)
    return {"top": top, "functions": functions, "reachable_functions": sorted(reachable)}


def _placement(log: Path, graph: dict, *, gpu: bool) -> dict:
    functions = graph.get("functions", {})
    reachable = graph.get("reachable_functions", [])
    if type(functions) is not dict or type(reachable) is not list or not reachable:
        raise ValueError("Native loss graph has no reachable nested functions")
    nodes = graph.get("top", []) + [node for name in reachable for node in functions[name]]
    if (not any(item.get("op") == "UnsortedSegmentSum" for item in nodes) or
        not any(item.get("op") == "GatherV2" for item in nodes) or
        not any(item.get("op") == "LogSoftmax" for item in nodes) or
        not any(item.get("op") == "Softmax" for item in nodes)):
        raise ValueError("Nested native loss forward/backward graph inventory is incomplete")
    required = {"mask_gather": set(), "loss_gather": set(), "log_softmax": set(),
                "backward_reduction": set(), "upstream_softmax": set(), "upstream_gradient": set()}
    started = ended = False
    cases = []
    current_case = None
    pattern = re.compile(r"(?P<name>[^\s:]+): \((?P<op>[A-Za-z0-9_]+)\):")
    device_pattern = re.compile(r"/device:(CPU|GPU):(\d+)")
    with log.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.strip() == PLACEMENT_BEGIN:
                if started or ended:
                    raise ValueError("Loss placement begin marker repeated")
                started = True
                continue
            if line.strip() == PLACEMENT_END:
                if not started or ended or current_case is not None:
                    raise ValueError("Loss placement end marker misplaced")
                ended = True
                continue
            if line.startswith(CASE_BEGIN):
                if not started or ended or current_case is not None:
                    raise ValueError("Loss fixture begin marker misplaced")
                current_case = line[len(CASE_BEGIN):].strip()
                if current_case not in ("all_valid", "mixed", "extreme") or current_case in cases:
                    raise ValueError("Loss fixture name is duplicated or unknown")
                continue
            if line.startswith(CASE_END):
                if current_case != line[len(CASE_END):].strip():
                    raise ValueError("Loss fixture end marker differs")
                cases.append(current_case)
                current_case = None
                continue
            if not started or ended:
                continue
            match = pattern.search(line)
            if match is None:
                continue
            name, op = match["name"], match["op"]
            native_scope = "sparse_categorical_crossentropy" in name.lower()
            upstream_scope = "probe_upstream_softmax" in name.lower()
            matches = [node for node in nodes if node.get("op") == op and
                       (name == node.get("name") or name.endswith("/" + node.get("name", "")))]
            role = None
            if native_scope and op == "GatherV2" and "boolean_mask" in name:
                role = "mask_gather"
            elif native_scope and op == "GatherV2" and "SparseSoftmaxCrossEntropyWithLogits" in name:
                role = "loss_gather"
            elif native_scope and op == "LogSoftmax" and "SparseSoftmaxCrossEntropyWithLogits" in name:
                role = "log_softmax"
            elif op == "UnsortedSegmentSum" and "gradients" in name and matches:
                role = "backward_reduction"
            elif upstream_scope and op == "Softmax":
                role = "upstream_softmax"
            elif (upstream_scope and "grad" in name.lower() and
                  op in ("Mul", "Sub", "Sum")):
                role = "upstream_gradient"
            if role is not None:
                device_match = device_pattern.search(line)
                if device_match is None:
                    raise ValueError("Native loss runtime placement is unknown for " + name)
                device = device_match[1] + ":" + device_match[2]
                if not matches:
                    raise ValueError("Loss runtime placement has no reachable graph node: " + name)
                expected = "GPU:0" if gpu and role.startswith("upstream_") else "CPU:0"
                if any(node.get("device") and not node["device"].endswith("/device:" + expected)
                       for node in matches):
                    raise ValueError("Loss graph requests a contradictory device for " + name)
                if device != expected:
                    raise ValueError("Native loss " + role + " placed on " + device + " instead of " + expected)
                required[role].add(name)
    if (not started or not ended or cases != ["all_valid", "mixed", "extreme"] or
        len(required["mask_gather"]) < 2 or
        any(not names for role, names in required.items() if role != "mask_gather")):
        raise ValueError("Native loss placement missing or unknown: " + repr(required))
    native_segments = [node for node in nodes if node.get("op") == "UnsortedSegmentSum" and
                       "gradients" in node.get("name", "")]
    if (not native_segments or
        not any("GatherV2_grad/UnsortedSegmentSum" in name for name in required["backward_reduction"]) or
        any(not any(name.endswith("/" + node["name"]) or name == node["name"]
                                  for name in required["backward_reduction"])
                                  for node in native_segments)):
        raise ValueError("Reachable native backward segment reduction lacks CPU runtime proof")
    return {role: len(names) for role, names in required.items()}


def _read_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    _array_check(arrays)
    return arrays


def _strict_marker(line: str) -> dict:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate native-loss marker key: " + key)
            value[key] = item
        return value
    value = json.loads(line, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite marker")))
    if type(value) is not dict:
        raise ValueError("Loss marker is not an object")
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Non-finite native-loss marker")
        if isinstance(item, dict):
            for nested in item.values():
                finite(nested)
        elif isinstance(item, list):
            for nested in item:
                finite(nested)
    finite(value)
    return value


def _request_dir(root: Path, request_path: Path) -> Path:
    if (not request_path.is_absolute() or request_path != request_path.resolve() or
        request_path.name != "request.json" or
        request_path.parent.parent.parent != _base(root) / "attempts" or
        request_path.parent.name not in STAGES):
        raise ValueError("Loss worker request escapes its attempt stage")
    return request_path.parent


def _worker_identity(root: Path, request_path: Path, request: dict) -> dict:
    import psutil
    stage_dir = _request_dir(root, request_path)
    command = [str(root / ".venvs/utime-cuda210/Scripts/python.exe"), "-I", "-B",
               str(root / "tools/verify_tf_cuda210_loss.py"), "--child", str(request_path)]
    lease = read_json(root / "runs/compute.lock")
    owner_pid, owner_birth = request.get("owner_pid"), request.get("owner_birth")
    if (request.get("schema_version") != "1.0" or
        request.get("artifact_type") != "utime_cuda210_loss_request" or
        request.get("scope") != "synthetic_native_loss_cpu_scope_only" or
        request.get("stage") != stage_dir.name or request.get("root") != str(root) or
        request.get("command") != command or request.get("lease_run_id") != LEASE or
        type(owner_pid) is not int or type(owner_birth) not in (int, float) or
        not math.isfinite(owner_birth) or lease.get("pid") != owner_pid or
        lease.get("process_start") != owner_birth or lease.get("host") != socket.gethostname() or
        lease.get("run_id") != LEASE):
        raise RuntimeError("Loss worker request has no exact parent compute lease")
    owner = psutil.Process(owner_pid)
    current = psutil.Process()
    immediate = current.parent()
    if owner.create_time() != owner_birth or immediate is None:
        raise RuntimeError("Loss worker parent lease owner is not live")
    if immediate.pid != owner_pid:
        grandparent = immediate.parent()
        if (Path(immediate.exe()).resolve() != Path(command[0]).resolve() or
            immediate.cmdline() != command or grandparent is None or
            grandparent.pid != owner_pid or grandparent.create_time() != owner_birth):
            raise RuntimeError("Loss worker lacks exact venv-launcher ancestry")
    return {"pid": current.pid, "birth": current.create_time(), "ppid": immediate.pid,
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


def worker_main(request_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    phase = "authorization"
    try:
        request_path = request_path.resolve()
        request = read_json(request_path)
        identity = _worker_identity(root, request_path, request)
        stage = request["stage"]
        environment = dict(capacity.ENVIRONMENT, CUDA_VISIBLE_DEVICES="-1" if stage == "cpu" else "0")
        if (Path(sys.prefix).resolve() != (root / ".venvs/utime-cuda210").resolve() or
            sys.version_info[:2] != (3, 9) or
            any(os.environ.get(name) != value for name, value in environment.items())):
            raise RuntimeError("Loss worker runtime or deterministic environment differs")
        remaining = request.get("deadline_remaining_seconds")
        if type(remaining) not in (int, float) or not math.isfinite(remaining) or not 0 < remaining <= TOTAL_SECONDS:
            raise ValueError("Loss worker has invalid total deadline")
        deadline = time.monotonic() + remaining
        if request.get("snapshot") != _snapshot(root, deadline):
            raise ValueError("Loss worker source or installed CUDA runtime changed")
        print(IDENTITY + json.dumps(identity, sort_keys=True, allow_nan=False), flush=True)
        phase = "import"
        if stage != "cpu":
            native = read_json(root / capacity.NATIVE_RESULT)
            folders = [root / native["destination"] / "cudatoolkit/Library/bin",
                       root / native["destination"] / "cudnn/Library/bin"]
            handles = [os.add_dll_directory(str(path)) for path in folders]
            os.environ["PATH"] = os.pathsep.join(map(str, folders)) + os.pathsep + os.environ.get("PATH", "")
        else:
            handles = []
        import tensorflow as tf
        if tf.__version__ != "2.10.1":
            raise ImportError("Loss worker requires TensorFlow 2.10.1")
        # Device-placement logging must precede logical-device enumeration, which initializes TF.
        tf.debugging.set_log_device_placement(True)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(4)
        tf.config.experimental.enable_op_determinism()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.config.set_soft_device_placement(False)
        physical = tf.config.list_physical_devices("GPU")
        if stage == "cpu":
            tf.config.set_visible_devices([], "GPU")
            if tf.config.list_logical_devices("GPU"):
                raise RuntimeError("CPU reference has a visible GPU")
        else:
            if len(physical) != 1 or not handles:
                raise RuntimeError("Loss candidate has no single private-DLL GPU")
            tf.config.set_logical_device_configuration(physical[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=capacity.GPU_MIB)])
        sys.path.insert(0, str(root / "vendor/utime"))
        sys.path.insert(0, str(root / "tools"))
        from verify_tf_native import _native_loss
        native_loss = _native_loss(tf)
        original_call = native_loss.call
        original_config = native_loss.get_config()
        if stage != "cpu":
            def cpu_call(y_true, y_pred):
                with tf.device("/CPU:0"):
                    return original_call(y_true, y_pred)
            native_loss.call = cpu_call
        if (native_loss.get_config() != original_config or
            native_loss.reduction != tf.keras.losses.Reduction.NONE):
            raise ValueError("Native loss class, configuration or reduction changed")
        device = "/CPU:0" if stage == "cpu" else "/GPU:0"

        @tf.function
        def step(logits, labels):
            with tf.GradientTape() as tape:
                tape.watch(logits)
                with tf.device(device):
                    with tf.name_scope("probe_upstream_softmax"):
                        probabilities = tf.nn.softmax(logits)
                losses = native_loss(labels, probabilities)
                objective = tf.reduce_sum(losses)
            gradient = tape.gradient(objective, logits)
            return losses, objective, gradient

        print(READY + json.dumps({"stage": stage, "identity": identity}, sort_keys=True), flush=True)
        phase = "probe"
        print(PLACEMENT_BEGIN, flush=True)
        try:
            concrete = step.get_concrete_function(tf.TensorSpec([12, 35, 5], tf.float32),
                                                   tf.TensorSpec([12, 35, 1], tf.int32))
            graph = _graph_inventory(concrete)
            stage_dir = request_path.parent
            graph_path = stage_dir / "graph.json"
            atomic_json(graph_path, graph, immutable=True)
            arrays = {}
            for name in ("all_valid", "mixed", "extreme"):
                logits, labels, mask = _fixture(name)
                _require_valid(mask)
                print(CASE_BEGIN + name, flush=True)
                losses, objective, gradient = step(tf.constant(logits), tf.constant(labels))
                print(CASE_END + name, flush=True)
                arrays.update({name + "_logits": logits, name + "_labels": labels,
                               name + "_mask": mask, name + "_loss": losses.numpy(),
                               name + "_objective": objective.numpy(),
                               name + "_gradient": gradient.numpy()})
            _array_check(arrays)
            try:
                _require_valid(np.zeros((12, 35), dtype=bool))
            except ValueError:
                all_invalid_rejected = True
            else:
                raise ValueError("All-invalid native loss fixture was accepted")
        finally:
            tf.debugging.set_log_device_placement(False)
        print(PLACEMENT_END, flush=True)
        temporary = stage_dir / "arrays.partial.npz"
        np.savez_compressed(temporary, **arrays)
        arrays_path = stage_dir / "arrays.npz"
        os.replace(temporary, arrays_path)
        if time.monotonic() >= deadline or tf.config.experimental.tensor_float_32_execution_enabled():
            raise RuntimeError("Loss worker deadline or TF32 policy changed")
        result = {"schema_version": "1.0", "artifact_type": "utime_cuda210_loss_stage_result",
                  "stage": stage, "scope": "synthetic_native_loss_cpu_scope_only",
                  "snapshot_id": content_id(request["snapshot"]),
                  "arrays_sha256": file_sha256(arrays_path), "graph_sha256": file_sha256(graph_path),
                  "all_invalid_rejected": all_invalid_rejected,
                  "native_loss_class": type(native_loss).__qualname__,
                  "native_loss_config": original_config, "gpu_cap_mib": capacity.GPU_MIB if stage != "cpu" else None,
                  "float32": tf.keras.mixed_precision.global_policy().name == "float32",
                  "tf32": tf.config.experimental.tensor_float_32_execution_enabled(),
                  "fit_authorized": False}
        print(RESULT + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    except BaseException as exc:
        failure = {"class": _classify(exc, phase), "phase": phase,
                   "type": type(exc).__name__, "message": str(exc)}
        print(FAILURE + json.dumps(failure, sort_keys=True, allow_nan=False), flush=True)
        raise


def _monitor(child: subprocess.Popen, log_path: Path, root: Path, request: dict,
             deadline: float, evidence: dict) -> tuple[int, dict | None, dict | None, dict | None]:
    import psutil
    owner = psutil.Process()
    observed = {}
    identity = result = failure = None
    ready_at = None
    peak = 0
    error = None
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        try:
            while True:
                for line in stream.readlines():
                    if line.startswith(IDENTITY):
                        if identity is not None:
                            raise RuntimeError("Duplicate native-loss worker identity")
                        identity = _strict_marker(line[len(IDENTITY):])
                        worker = psutil.Process(identity["pid"])
                        if (type(identity.get("pid")) is not int or
                            type(identity.get("birth")) not in (int, float) or
                            not math.isfinite(identity["birth"]) or
                            worker.create_time() != identity["birth"] or
                            worker.ppid() != identity.get("ppid") or
                            identity.get("owner_pid") != owner.pid or
                            identity.get("owner_birth") != owner.create_time() or
                            identity.get("request_sha256") != request["request_sha256"] or
                            worker.ppid() not in (owner.pid, child.pid)):
                            raise RuntimeError("Native-loss worker PID/birth/lease identity differs")
                        if worker.ppid() == child.pid:
                            launcher = psutil.Process(child.pid)
                            parent = launcher.parent()
                            if (Path(launcher.exe()).resolve() != Path(request["command"][0]).resolve() or
                                launcher.cmdline() != request["command"] or parent is None or
                                parent.pid != owner.pid or parent.create_time() != owner.create_time()):
                                raise RuntimeError("Native-loss venv launcher ancestry differs")
                        observed[worker.pid] = worker.create_time()
                    elif line.startswith(READY):
                        ready = _strict_marker(line[len(READY):])
                        if identity is None or ready_at is not None or ready != {
                                "stage": request["stage"], "identity": identity}:
                            raise RuntimeError("Native-loss authenticated setup marker differs")
                        ready_at = time.monotonic()
                    elif line.startswith(RESULT):
                        if result is not None or ready_at is None:
                            raise RuntimeError("Native-loss result lacks unique ready worker")
                        result = _strict_marker(line[len(RESULT):])
                    elif line.startswith(FAILURE):
                        if failure is not None:
                            raise RuntimeError("Duplicate native-loss failure marker")
                        failure = _strict_marker(line[len(FAILURE):])
                if log_path.stat().st_size > 256 * 1024**2:
                    raise RuntimeError("Native-loss placement log exceeded 256 MiB")
                try:
                    launcher = psutil.Process(child.pid)
                    observed[launcher.pid] = launcher.create_time()
                    for item in launcher.children(recursive=True):
                        observed[item.pid] = item.create_time()
                except psutil.NoSuchProcess:
                    pass
                if identity is not None:
                    try:
                        worker = psutil.Process(identity["pid"])
                        if worker.create_time() == identity["birth"]:
                            for item in worker.children(recursive=True):
                                observed[item.pid] = item.create_time()
                    except psutil.NoSuchProcess:
                        pass
                live = []
                for pid, birth in observed.items():
                    try:
                        item = psutil.Process(pid)
                        if item.create_time() == birth and item.is_running() and item.status() != psutil.STATUS_ZOMBIE:
                            live.append(item)
                    except psutil.NoSuchProcess:
                        pass
                try:
                    rss = sum(item.memory_info().rss for item in live)
                except psutil.NoSuchProcess:
                    continue
                peak = max(peak, owner.memory_info().rss + rss)
                evidence.update(worker_identity=identity, peak_combined_rss_bytes=peak,
                                observed_pid_birth=sorted(observed.items()),
                                result=result, failure=failure)
                capacity._limits(root, deadline, rss)
                if ready_at is not None and time.monotonic() - ready_at > STEP_SECONDS:
                    raise TimeoutError("Native-loss authenticated probe exceeded 120 seconds")
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
            if item.create_time() == birth and item.is_running() and item.status() != psutil.STATUS_ZOMBIE:
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
            if item.create_time() == observed[item.pid]:
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
    evidence.update(worker_identity=identity, peak_combined_rss_bytes=peak,
                    observed_pid_birth=sorted(observed.items()), result=result,
                    failure=failure, cleanup_survivors=sorted(
                        (item.pid, observed.get(item.pid)) for item in remaining) + unresolved)
    if remaining or unresolved:
        raise RuntimeError("Native-loss cleanup left live or inaccessible PID/birth: " +
                           repr([(item.pid, observed.get(item.pid)) for item in remaining] + unresolved)) from error
    if error is not None:
        raise error
    if identity is None or ready_at is None:
        raise RuntimeError("Native-loss stage never authenticated and became ready")
    return peak, result, failure, identity


def _stage_result(stage: str, folder: Path, result: dict, snapshot: dict) -> tuple[dict, dict[str, np.ndarray]]:
    if (type(result) is not dict or result.get("schema_version") != "1.0" or
        result.get("artifact_type") != "utime_cuda210_loss_stage_result" or
        result.get("stage") != stage or result.get("scope") != "synthetic_native_loss_cpu_scope_only" or
        result.get("snapshot_id") != content_id(snapshot) or
        result.get("all_invalid_rejected") is not True or result.get("float32") is not True or
        result.get("tf32") is not False or result.get("fit_authorized") is not False or
        result.get("gpu_cap_mib") != (None if stage == "cpu" else capacity.GPU_MIB) or
        result.get("arrays_sha256") != file_sha256(folder / "arrays.npz") or
        result.get("graph_sha256") != file_sha256(folder / "graph.json")):
        raise ValueError("Native-loss stage result or array/graph hashes differ")
    graph = read_json(folder / "graph.json")
    placement = _placement(folder / "worker.log", graph, gpu=stage != "cpu")
    return placement, _read_arrays(folder / "arrays.npz")


def run(root: Path) -> dict:
    root = root.resolve()
    base = _base(root)
    started = time.monotonic()
    deadline = started + TOTAL_SECONDS
    with compute_lease(root, LEASE) as lease:
        attempt = base / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
                                       "-" + uuid.uuid4().hex[:8])
        attempt.mkdir(parents=True, exist_ok=False)
        before = after = None
        reports = []
        failure = None
        status = "FAILED"
        try:
            before = _snapshot(root, deadline)
            reference = first = None
            for stage in STAGES:
                capacity._limits(root, deadline)
                report = {"stage": stage, "gpu_observation": None,
                          "request_sha256": None, "log_sha256": None,
                          "worker_identity": None, "peak_combined_rss_bytes": None,
                          "observed_pid_birth": [], "cleanup_survivors": []}
                reports.append(report)
                gpu_observation = None
                if stage != "cpu":
                    gpu_observation = capacity._gpu_observation()
                    report["gpu_observation"] = gpu_observation
                    capacity._require_gpu_headroom(gpu_observation)
                folder = attempt / stage
                folder.mkdir(exist_ok=False)
                request_path = (folder / "request.json").resolve()
                command = [str(root / ".venvs/utime-cuda210/Scripts/python.exe"), "-I", "-B",
                           str(root / "tools/verify_tf_cuda210_loss.py"), "--child", str(request_path)]
                request = {"schema_version": "1.0", "artifact_type": "utime_cuda210_loss_request",
                           "scope": "synthetic_native_loss_cpu_scope_only", "root": str(root),
                           "stage": stage, "snapshot": before, "lease_run_id": LEASE,
                           "owner_pid": lease["pid"], "owner_birth": lease["process_start"],
                           "command": command, "gpu_observation": gpu_observation,
                           "deadline_remaining_seconds": deadline - time.monotonic()}
                atomic_json(request_path, request, immutable=True)
                request["request_sha256"] = file_sha256(request_path)
                report["request_sha256"] = request["request_sha256"]
                env = os.environ.copy()
                env.update(capacity.ENVIRONMENT)
                env["CUDA_VISIBLE_DEVICES"] = "-1" if stage == "cpu" else "0"
                log_path = folder / "worker.log"
                with log_path.open("xb") as log:
                    child = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                             stderr=subprocess.STDOUT)
                    try:
                        peak, result, worker_failure, identity = _monitor(
                            child, log_path, root, request, deadline, report)
                    finally:
                        if log_path.stat().st_size <= 256 * 1024**2:
                            report["log_sha256"] = file_sha256(log_path)
                if child.returncode != 0 or worker_failure is not None:
                    kind = worker_failure.get("class", "WORKER") if worker_failure else "WORKER"
                    raise RuntimeError("Native-loss " + stage + " " + kind + " worker failed")
                placement, arrays = _stage_result(stage, folder, result, before)
                report["placement"] = placement
                if stage == "cpu":
                    reference = arrays
                else:
                    _compare(arrays, reference, exact=False)
                    if first is None:
                        first = arrays
                    else:
                        _compare(arrays, first, exact=True)
            after = _snapshot(root, deadline)
            capacity._limits(root, deadline)
            if before != after:
                raise ValueError("Native-loss source or installed runtime changed")
            status = "PASS_LOSS_PLACEMENT_ONLY"
        except BaseException as exc:
            message = str(exc)
            kind = "RESOURCE" if isinstance(exc, capacity.GPUHeadroomUnavailable) or (
                isinstance(exc, TimeoutError) or "bound" in message or "deadline" in message) else (
                "OOM" if " OOM " in message else "API" if " API " in message else
                "IMPORT" if " IMPORT " in message else "PROVENANCE" if " PROVENANCE " in message else
                "WORKER")
            failure = {"class": kind, "type": type(exc).__name__, "message": message}
        elapsed = time.monotonic() - started
        if status == "PASS_LOSS_PLACEMENT_ONLY" and elapsed > TOTAL_SECONDS:
            status = "FAILED"
            failure = {"class": "RESOURCE", "type": "TimeoutError",
                       "message": "Native-loss total invocation exceeded 900 seconds"}
        record = {"schema_version": "1.0", "artifact_type": "utime_cuda210_loss_attempt",
                  "scope": "synthetic_native_loss_cpu_scope_only", "status": status,
                  "before": before, "after": after, "stages": reports, "failure": failure,
                  "elapsed_seconds": elapsed, "fit_authorized": False}
        atomic_json(attempt / "attempt.json", record, immutable=True)
        if status != "PASS_LOSS_PLACEMENT_ONLY":
            raise RuntimeError("Native-loss probe failed; retained attempt: " + str(attempt))
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps({"status": result["status"], "fit_authorized": False}, sort_keys=True), flush=True)
