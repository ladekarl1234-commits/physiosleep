"""Lease-bound real D-source parity for the packaged Sleepyland/YASA bridge.

Only SC4362 and ST7011 are eligible. This does not load a classifier, train,
score labels, or use Audit A/B. A successful receipt is required before fitting.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import types
import uuid

import numpy as np

from . import sleepyland_yasa_bridge as bridge
from .contracts import content_id, json_text, read_json
from .research import atomic_json, compute_lease, file_sha256

RECORDINGS = ("SC4362", "ST7011")
RUN_ID = bridge.LEASE_NAME
PREFIX = "SLEEPYLAND_NATIVE_READER_RESULT "
FEATURE_PREFIX = "SLEEPYLAND_YASA_FEATURE_RESULT "
FEATURE_BOOTSTRAP = ("import sys; sys.path.insert(0, sys.argv[1]); "
                     "from sleepedf.sleepyland_yasa_verify import _feature_worker_main; "
                     "_feature_worker_main(sys.argv[2])")
MAX_SECONDS = 1800
MAX_RECORD_SECONDS = 900
MAX_BYTES = 10 * 1024**3
FEATURE_ATOL = 1e-6
FEATURE_RTOL = 1e-6
BASE = "runs/sleepyland-yasa-native-verify"


def _snapshot(root: Path, data_root: Path) -> dict:
    root, data_root = root.resolve(), data_root.resolve()
    protocol, split, records = bridge._development(root)
    chosen = {record["recording_id"]: record for record in records
              if record["recording_id"] in RECORDINGS}
    if set(chosen) != set(RECORDINGS):
        raise ValueError("Both forced parity nights must be original development recordings")
    config = bridge.feature_config(root)
    sources = {}
    for recording_id in RECORDINGS:
        record = chosen[recording_id]
        psg = bridge._safe_psg(data_root, record)
        sources[recording_id] = dict(bridge._checked_record(record),
                                     psg_path=str(psg),
                                     channel_names=list(bridge.CHANNELS))
    return {"schema_version": "1.0", "artifact_type": "sleepyland_yasa_parity_inputs",
            "scope": "two_original_development_recordings_no_truth",
            "data_root": str(data_root), "protocol_hash": protocol["protocol_hash"],
            "split_id": split["split_id"], "feature_config": config,
            "records": sources,
            "verifier_sha256": file_sha256(Path(__file__)),
            "probe_sha256": file_sha256(root / "tools/verify_sleepyland_yasa_reader.py"),
            "full_native_helper_sha256": file_sha256(
                root / "vendor/usleepyland/utime/bin/predict_one.py")}


def _identity(snapshot: dict) -> str:
    return content_id({"schema_version": "1.0",
                       "artifact_type": "sleepyland_yasa_native_parity_contract",
                       "inputs": snapshot, "recordings": list(RECORDINGS),
                       "checks": ["ast_vs_full_native_128hz_volts_exact",
                                  "both_group_yasa_100hz_features_and_nan_masks"],
                       "feature_atol": FEATURE_ATOL, "feature_rtol": FEATURE_RTOL,
                       "max_seconds": MAX_SECONDS, "max_bytes": MAX_BYTES})


def _parse_worker(output: str, prefix: str = PREFIX) -> dict:
    lines = [line[len(prefix):] for line in output.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError("Native Sleepyland reader returned no unique parity result")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate native parity key")
            value[key] = item
        return value
    item = json.loads(lines[0], object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite native parity result")))
    json_text(item)
    if type(item) is not dict:
        raise ValueError("Native parity result must be an object")
    return item


def _feature_result_line(value: dict) -> str:
    return FEATURE_PREFIX + json.dumps(value, sort_keys=True, allow_nan=False)


def _supervise(child: subprocess.Popen, started: float, begin: float) -> None:
    """Bound the lease owner's launcher and all observed descendants."""
    import psutil
    observed = {}
    try:
        while True:
            try:
                launcher = psutil.Process(child.pid)
                for process in [launcher] + launcher.children(recursive=True):
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
            if (rss > MAX_BYTES or psutil.virtual_memory().available < 4 * 1024**3 or
                    time.monotonic() - started > MAX_SECONDS or
                    time.monotonic() - begin > MAX_RECORD_SECONDS):
                raise RuntimeError("Sleepyland/YASA parity exceeded 10 GiB or remaining time bound")
            if child.poll() is not None and not live:
                break
            time.sleep(0.25)
        child.wait()
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


def _feature_worker_main(request_name: str) -> None:
    """Run only as a direct child of the current verifier lease owner."""
    import psutil
    from threadpoolctl import threadpool_limits
    request_path = Path(request_name)
    request = read_json(request_path)
    root = Path(request["root"])
    attempts = (root / BASE / "attempts").resolve()
    if (not root.is_absolute() or root != root.resolve() or
            not request_path.is_absolute() or request_path != request_path.resolve() or
            not request_path.is_relative_to(attempts) or
            request.get("schema_version") != "1.0" or
            request.get("artifact_type") != "sleepyland_yasa_feature_parity_request" or
            request.get("lease_run_id") != RUN_ID or
            request.get("recording_id") not in RECORDINGS or
            request.get("group_channel_types") !=
            {key: ["EEG", "EOG"] for key in bridge.GROUPS}):
        raise ValueError("Feature parity request scope differs")
    lease = read_json(root / "runs/compute.lock")
    immediate = psutil.Process(os.getppid())
    owner = immediate
    if immediate.pid != lease.get("pid"):
        expected = [str(root / ".venvs/research/Scripts/python.exe"), "-I", "-c",
                    FEATURE_BOOTSTRAP, str(root), str(request_path)]
        if (Path(immediate.exe()).resolve() != Path(expected[0]).resolve() or
                immediate.cmdline() != expected):
            raise RuntimeError("Feature parity launcher is not the exact parent command")
        owner = immediate.parent()
    if (owner is None or owner.pid != lease.get("pid") or
            owner.create_time() != lease.get("process_start") or
            lease.get("host") != socket.gethostname() or lease.get("run_id") != RUN_ID):
        raise RuntimeError("Feature parity has no live matching parent compute lease")
    bridge._verify_research_runtime(root)
    if bridge.feature_config(root) != request["feature_config"]:
        raise ValueError("Feature parity runtime, source or recipe changed")
    input_path = _recorded_path(request["input_path"], request_path.parent,
                                request["recording_id"] + ".native-signal.npz")
    if file_sha256(input_path) != request["input_sha256"]:
        raise ValueError("Feature parity native signal bytes changed")
    n_epochs = request["n_epochs"]
    if type(n_epochs) is not int or n_epochs <= 0:
        raise ValueError("Feature parity original grid length is invalid")
    with np.load(input_path, allow_pickle=False) as archive:
        if set(archive.files) != {"ast_signal", "native_signal", "epoch_index", "channel_names"}:
            raise ValueError("Feature parity signal transfer fields differ")
        ast, native = archive["ast_signal"], archive["native_signal"]
        if (ast.shape != (n_epochs * 30 * 128, 3) or native.shape != ast.shape or
                ast.dtype != np.float32 or native.dtype != np.float32 or
                not np.isfinite(ast).all() or not np.isfinite(native).all() or
                not np.array_equal(ast, native) or
                not np.array_equal(archive["epoch_index"], np.arange(n_epochs, dtype=np.int32)) or
                archive["channel_names"].tolist() != list(bridge.CHANNELS)):
            raise ValueError("Feature parity signal grid, units or exact native agreement differs")
    groups, group_evidence = {}, {}
    with threadpool_limits(limits=4):
        bridge._limit_cpu_threads()
        for group in bridge.GROUPS:
            ast_object, ast_frame = bridge._native_group(root, ast, group, n_epochs)
            native_object, native_frame = _native_yasa_branch(
                root, native, group, request["group_channel_types"][group])
            groups[group] = _compare_group(native_object, native_frame, ast_object, ast_frame)
            values = np.ascontiguousarray(ast_frame.to_numpy(dtype=np.float64))
            group_evidence[group] = {
                "columns": list(ast_frame.columns),
                "column_dtypes": [str(dtype) for dtype in ast_frame.dtypes],
                "native_nan_values": int(np.isnan(values).sum()),
                "values_sha256": hashlib.sha256(values.tobytes()).hexdigest()}
    result = {"schema_version": "1.0", "artifact_type": "sleepyland_yasa_feature_parity",
              "recording_id": request["recording_id"], "n_epochs": n_epochs,
              "input_sha256": request["input_sha256"], "config_id": request["feature_config"]["config_id"],
              "native_branch": "pinned_run_pred_on_until_classifier_boundary",
              "groups": groups,
              "bridge_evidence": {"recording_id": request["recording_id"],
                                  "participant_id": request["participant_id"],
                                  "source_psg_sha256": request["source_psg_sha256"],
                                  "n_epochs": n_epochs, "config_id": request["feature_config"]["config_id"],
                                  "signal_transfer_sha256": request["input_sha256"],
                                  "unit": "V", "sample_rate_hz": 128,
                                  "feature_rate_hz": 100, "groups": group_evidence}}
    print(_feature_result_line(result), flush=True)


def _compare_group(reference_native, reference_frame,
                   bridge_native, bridge_frame) -> dict:
    if (reference_native.sf != 100 or bridge_native.sf != 100 or
            reference_native.ch_names != bridge_native.ch_names or
            reference_native.ch_types != bridge_native.ch_types or
            reference_native.data.shape != bridge_native.data.shape or
            not np.array_equal(reference_native.data, bridge_native.data, equal_nan=True)):
        raise ValueError("Native YASA 100-Hz microvolt signal differs between routes")
    if (list(reference_frame.columns) != list(bridge_frame.columns) or
            [str(dtype) for dtype in reference_frame.dtypes] !=
            [str(dtype) for dtype in bridge_frame.dtypes] or
            reference_frame.shape != bridge_frame.shape):
        raise ValueError("Native YASA feature names, dtypes or original epoch coverage differ")
    left = reference_frame.to_numpy(dtype=np.float64)
    right = bridge_frame.to_numpy(dtype=np.float64)
    if (np.isinf(left).any() or np.isinf(right).any() or
            not np.array_equal(np.isnan(left), np.isnan(right))):
        raise ValueError("Native YASA feature infinity or NaN masks differ")
    valid = ~np.isnan(left)
    delta = np.abs(left[valid] - right[valid])
    maximum = float(delta.max()) if delta.size else 0.0
    if not np.allclose(left[valid], right[valid], rtol=FEATURE_RTOL, atol=FEATURE_ATOL):
        raise ValueError("Native YASA feature values differ beyond frozen tolerance")
    return {"feature_columns": list(reference_frame.columns),
            "column_dtypes": [str(dtype) for dtype in reference_frame.dtypes],
            "n_epochs": int(reference_frame.shape[0]),
            "native_nan_values": int(np.isnan(left).sum()),
            "feature_max_abs_error": maximum,
            "yasa_data_shape": list(reference_native.data.shape),
            "yasa_signal_unit": "uV", "yasa_rate_hz": 100}


def _native_yasa_branch(root: Path, signal: np.ndarray, group: str,
                        channel_types: list[str]):
    """Execute pinned packaged YASA branch up to its classifier call."""
    import mne
    source = (root / "vendor/usleepyland/utime/bin/predict_one.py").resolve()
    parsed = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                and node.name == "run_pred_on"]
    if len(selected) != 1 or group not in bridge.GROUPS or channel_types != ["EEG", "EOG"]:
        raise ValueError("Pinned packaged YASA function or channel inference changed")
    branch = [node for node in ast.walk(selected[0]) if isinstance(node, ast.Compare)
              and any(isinstance(item, ast.Constant) and item.value == "yasa"
                      for item in node.comparators)]
    if len(branch) != 1:
        raise ValueError("Pinned packaged YASA branch is ambiguous")
    namespace = {"mne": mne, "np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    yasa_source = (root / "vendor/yasa/src").resolve()
    if str(yasa_source) not in sys.path:
        sys.path.insert(0, str(yasa_source))
    import yasa
    if not Path(yasa.__file__).resolve().is_relative_to(yasa_source):
        raise ValueError("Packaged YASA branch resolved a non-pinned implementation")
    names = list(bridge.GROUPS[group])
    indices = [bridge.CHANNELS.index(name) for name in names]
    study = types.SimpleNamespace(psg=signal, sample_rate=128)
    channel_group = types.SimpleNamespace(channel_names=names, channel_indices=indices)
    fake_packages = {name: types.ModuleType(name) for name in
                     ("psg_utils", "psg_utils.io", "psg_utils.io.channels")}
    fake_packages["psg_utils"].__path__ = []
    fake_packages["psg_utils.io"].__path__ = []
    def pinned_types(requested):
        if list(requested) != names:
            raise ValueError("Packaged YASA branch changed selected channels")
        return channel_types
    fake_packages["psg_utils.io.channels"].infer_channel_types = pinned_types
    previous_modules = {name: sys.modules.get(name) for name in fake_packages}
    original_predict = yasa.SleepStaging.predict_proba
    class BeforeClassifier(Exception):
        def __init__(self, native):
            self.native = native
    def stop_before_classifier(native, *args, **kwargs):
        raise BeforeClassifier(native)
    try:
        sys.modules.update(fake_packages)
        yasa.SleepStaging.predict_proba = stop_before_classifier
        try:
            namespace["run_pred_on"](study, channel_group, None, "yasa", None, None)
        except BeforeClassifier as stop:
            native = stop.native
        else:
            raise ValueError("Packaged YASA branch did not reach classifier boundary")
    finally:
        yasa.SleepStaging.predict_proba = original_predict
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return native, native.get_features()


def _execute_record(root: Path, data_root: Path, snapshot: dict, record: dict,
                    attempt_dir: Path, started: float) -> tuple[dict, dict]:
    recording_id = record["recording_id"]
    item = snapshot["records"][recording_id]
    output = attempt_dir / (recording_id + ".native-signal.npz")
    request_path = attempt_dir / (recording_id + ".request.json")
    log_path = attempt_dir / (recording_id + ".reader.log")
    config = snapshot["feature_config"]
    request = {"schema_version": "1.0",
               "artifact_type": "sleepyland_yasa_native_reader_probe_request",
               "lease_run_id": RUN_ID, "recording_id": recording_id,
               "n_epochs": record["n_epochs"], "source_psg_sha256": record["psg_sha256"],
               "source_channels": record["channels"], "channel_names": list(bridge.CHANNELS),
               "data_root": str(data_root), "psg_path": item["psg_path"],
               "output_path": str(output.resolve()),
               "reader_sha256": config["implementation_sha256"]["tools/sleepyland_yasa_reader.py"],
               "runtime_lock_sha256": config["runtime_lock_sha256"]["usleepyland"],
               "usleepyland_source_manifest_sha256": config["source_manifest_sha256"]["usleepyland"],
               "sleepyland_source_manifest_sha256": config["source_manifest_sha256"]["sleepyland"],
               "probe_sha256": snapshot["probe_sha256"]}
    atomic_json(request_path, request, immutable=True)
    command = [str(root / ".venvs/usleepyland/Scripts/python.exe"), "-I",
               str(root / "tools/verify_sleepyland_yasa_reader.py"),
               "--request", str(request_path.resolve())]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       NUMEXPR_NUM_THREADS="4", NUMBA_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="-1")
    begin = time.monotonic()
    with log_path.open("xb") as log:
        child = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT)
        _supervise(child, started, begin)
    output_text = log_path.read_text(encoding="utf-8", errors="replace")
    if child.returncode != 0:
        raise RuntimeError("Unchanged native Sleepyland reader failed; log: " + str(log_path))
    worker = _parse_worker(output_text)
    if (worker.get("schema_version") != "1.0" or
            worker.get("artifact_type") != "sleepyland_yasa_native_reader_parity" or
            worker.get("recording_id") != recording_id or
            worker.get("n_epochs") != record["n_epochs"] or
            worker.get("source_psg_sha256") != record["psg_sha256"] or
            worker.get("exact_signal_equal") is not True or
            worker.get("max_signal_abs_error") != 0.0 or
            worker.get("signal_shape") != [record["n_epochs"] * 30 * 128, 3] or
            worker.get("signal_dtype") != "float32" or worker.get("signal_unit") != "V" or
            worker.get("signal_rate_hz") != 128 or
            worker.get("channel_names") != list(bridge.CHANNELS) or
            worker.get("groups") != {key: list(value) for key, value in bridge.GROUPS.items()} or
            worker.get("group_channel_types") != {key: ["EEG", "EOG"] for key in bridge.GROUPS} or
            worker.get("runtime") != {key: request[key] for key in (
                "reader_sha256", "runtime_lock_sha256", "usleepyland_source_manifest_sha256",
                "sleepyland_source_manifest_sha256")} or
            worker.get("probe_sha256") != snapshot["probe_sha256"] or
            worker.get("full_native_module_sha256") != snapshot["full_native_helper_sha256"] or
            worker.get("signal_payload_sha256") != file_sha256(output)):
        raise ValueError("Native Sleepyland reader parity report differs from frozen evidence")
    feature_request_path = attempt_dir / (recording_id + ".feature-request.json")
    feature_log_path = attempt_dir / (recording_id + ".feature.log")
    feature_request = {"schema_version": "1.0",
                       "artifact_type": "sleepyland_yasa_feature_parity_request",
                       "root": str(root), "lease_run_id": RUN_ID,
                       "recording_id": recording_id, "participant_id": record["participant_id"],
                       "source_psg_sha256": record["psg_sha256"],
                       "n_epochs": record["n_epochs"], "feature_config": config,
                       "group_channel_types": worker["group_channel_types"],
                       "input_path": str(output.resolve()), "input_sha256": file_sha256(output)}
    atomic_json(feature_request_path, feature_request, immutable=True)
    feature_command = [str(root / ".venvs/research/Scripts/python.exe"), "-I", "-c",
                       FEATURE_BOOTSTRAP, str(root), str(feature_request_path.resolve())]
    begin = time.monotonic()
    with feature_log_path.open("xb") as log:
        child = subprocess.Popen(feature_command, cwd=root, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT)
        _supervise(child, started, begin)
    if child.returncode != 0:
        raise RuntimeError("Sleepyland/YASA feature parity worker failed; log: " + str(feature_log_path))
    feature = _parse_worker(feature_log_path.read_text(encoding="utf-8", errors="replace"),
                            FEATURE_PREFIX)
    if (feature.get("schema_version") != "1.0" or
            feature.get("artifact_type") != "sleepyland_yasa_feature_parity" or
            feature.get("recording_id") != recording_id or
            feature.get("n_epochs") != record["n_epochs"] or
            feature.get("input_sha256") != feature_request["input_sha256"] or
            feature.get("config_id") != config["config_id"] or
            feature.get("native_branch") != "pinned_run_pred_on_until_classifier_boundary"):
        raise ValueError("Research YASA feature parity report differs from request")
    if time.monotonic() - started > MAX_SECONDS:
        raise TimeoutError("Sleepyland/YASA two-record parity exceeded 30 minutes")
    return {"reader": worker, "feature_worker": feature,
            "groups": feature["groups"], "bridge_evidence": feature["bridge_evidence"]}, {
                "native": log_path, "feature": feature_log_path}


def _execute(root: Path, data_root: Path, snapshot: dict,
             attempt_dir: Path, started: float) -> tuple[dict, dict]:
    _, _, records = bridge._development(root)
    selected = {record["recording_id"]: record for record in records
                if record["recording_id"] in RECORDINGS}
    results, logs = {}, {}
    for recording_id in RECORDINGS:
        try:
            result, record_logs = _execute_record(root, data_root, snapshot, selected[recording_id],
                                                   attempt_dir, started)
        finally:
            (attempt_dir / (recording_id + ".native-signal.npz")).unlink(missing_ok=True)
        results[recording_id] = result
        logs[recording_id] = {kind: {"path": str(path.resolve()),
                                    "sha256": file_sha256(path)}
                              for kind, path in record_logs.items()}
    return results, logs


def _check_result(results: dict, snapshot: dict, logs: dict) -> None:
    if set(results) != set(RECORDINGS) or set(logs) != set(RECORDINGS):
        raise ValueError("Native parity did not cover exactly the forced development nights")
    for recording_id in RECORDINGS:
        row = results[recording_id]
        worker = row["reader"]
        expected = snapshot["records"][recording_id]
        config = snapshot["feature_config"]
        expected_runtime = {
            "reader_sha256": config["implementation_sha256"]["tools/sleepyland_yasa_reader.py"],
            "runtime_lock_sha256": config["runtime_lock_sha256"]["usleepyland"],
            "usleepyland_source_manifest_sha256": config["source_manifest_sha256"]["usleepyland"],
            "sleepyland_source_manifest_sha256": config["source_manifest_sha256"]["sleepyland"]}
        bridge_evidence = row["bridge_evidence"]
        feature = row["feature_worker"]
        if (set(row) != {"reader", "feature_worker", "groups", "bridge_evidence"} or
                worker.get("schema_version") != "1.0" or
                worker.get("artifact_type") != "sleepyland_yasa_native_reader_parity" or
                worker.get("recording_id") != recording_id or
                worker.get("source_psg_sha256") != expected["source_psg_sha256"] or
                worker.get("exact_signal_equal") is not True or
                worker.get("max_signal_abs_error") != 0.0 or
                worker.get("signal_dtype") != "float32" or
                worker.get("signal_unit") != "V" or worker.get("signal_rate_hz") != 128 or
                worker.get("n_epochs") != expected["n_epochs"] or
                worker.get("signal_shape") != [expected["n_epochs"] * 30 * 128, 3] or
                worker.get("channel_names") != list(bridge.CHANNELS) or
                worker.get("groups") != {key: list(value) for key, value in bridge.GROUPS.items()} or
                worker.get("group_channel_types") != {key: ["EEG", "EOG"] for key in bridge.GROUPS} or
                worker.get("runtime") != expected_runtime or
                worker.get("probe_sha256") != snapshot["probe_sha256"] or
                worker.get("full_native_module_sha256") != snapshot["full_native_helper_sha256"] or
                not re.fullmatch(r"[0-9a-f]{64}", str(worker.get("signal_payload_sha256", ""))) or
                type(worker.get("seconds")) not in (int, float) or
                not math.isfinite(worker["seconds"]) or not 0 <= worker["seconds"] <= MAX_RECORD_SECONDS or
                feature.get("schema_version") != "1.0" or
                feature.get("artifact_type") != "sleepyland_yasa_feature_parity" or
                feature.get("recording_id") != recording_id or
                feature.get("n_epochs") != expected["n_epochs"] or
                feature.get("input_sha256") != worker.get("signal_payload_sha256") or
                feature.get("config_id") != config["config_id"] or
                feature.get("native_branch") != "pinned_run_pred_on_until_classifier_boundary" or
                feature.get("groups") != row["groups"] or
                feature.get("bridge_evidence") != bridge_evidence or
                set(row["groups"]) != set(bridge.GROUPS) or
                any(bridge_evidence.get(key) != expected[key] for key in
                    ("recording_id", "participant_id", "source_psg_sha256", "n_epochs")) or
                bridge_evidence.get("config_id") != config["config_id"] or
                bridge_evidence.get("unit") != "V" or
                bridge_evidence.get("sample_rate_hz") != 128 or
                bridge_evidence.get("feature_rate_hz") != 100 or
                set(bridge_evidence.get("groups", {})) != set(bridge.GROUPS)):
            raise ValueError("Native parity report scope or ancestry differs")
        for group in bridge.GROUPS:
            detail = row["groups"][group]
            produced = bridge_evidence["groups"][group]
            if (detail.get("n_epochs") != expected["n_epochs"] or
                    detail.get("yasa_rate_hz") != 100 or
                    detail.get("yasa_signal_unit") != "uV" or
                    detail.get("yasa_data_shape") != [2, expected["n_epochs"] * 30 * 100] or
                    not detail.get("feature_columns") or
                    len(detail["feature_columns"]) != len(detail.get("column_dtypes", [])) or
                    produced.get("columns") != detail["feature_columns"] or
                    produced.get("column_dtypes") != detail["column_dtypes"] or
                    produced.get("native_nan_values") != detail.get("native_nan_values") or
                    not re.fullmatch(r"[0-9a-f]{64}", str(produced.get("values_sha256", ""))) or
                    type(detail.get("native_nan_values")) is not int or
                    detail["native_nan_values"] < 0 or
                    detail["native_nan_values"] > expected["n_epochs"] * len(detail["feature_columns"]) or
                    type(detail.get("feature_max_abs_error")) not in (int, float) or
                    not math.isfinite(detail["feature_max_abs_error"]) or
                    detail["feature_max_abs_error"] < 0):
                raise ValueError("Native YASA feature parity evidence is incomplete")


def _attempt_path(root: Path) -> Path:
    return root / BASE / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
                                        "-" + uuid.uuid4().hex[:8])


def run(root: Path, data_root: Path) -> dict:
    root, data_root = root.resolve(), data_root.resolve()
    with compute_lease(root, RUN_ID):
        return _run_locked(root, data_root)


def _run_locked(root: Path, data_root: Path) -> dict:
    attempt_dir = _attempt_path(root)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    current = root / BASE / "current.json"
    atomic_json(current, {"schema_version": "1.0", "artifact_type": "sleepyland_yasa_parity_pending",
                          "attempt_path": str((attempt_dir / "attempt.json").resolve())})
    started = time.monotonic()
    before = after = results = logs = None
    error = None
    status = "FAILED"
    try:
        before = _snapshot(root, data_root)
        results, logs = _execute(root, data_root, before, attempt_dir, started)
        after = _snapshot(root, data_root)
        if before != after:
            raise ValueError("Native parity source/runtime/D bytes changed during probe")
        _check_result(results, before, logs)
        if time.monotonic() - started > MAX_SECONDS:
            raise TimeoutError("Native Sleepyland/YASA parity exceeded 30 minutes")
        status = "SUCCESS"
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    log_evidence = {}
    for recording_id in RECORDINGS:
        record_logs = {}
        for kind, suffix in (("native", ".reader.log"), ("feature", ".feature.log")):
            path = attempt_dir / (recording_id + suffix)
            if path.exists():
                record_logs[kind] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
        if record_logs:
            log_evidence[recording_id] = record_logs
    attempt = {"schema_version": "1.0", "artifact_type": "sleepyland_yasa_parity_attempt",
               "status": status, "before": before, "after": after,
               "results": results, "logs": log_evidence,
               "elapsed_seconds": time.monotonic() - started,
               "deadline_seconds": MAX_SECONDS, "error": error}
    atomic_json(attempt_dir / "attempt.json", attempt, immutable=True)
    if status != "SUCCESS":
        atomic_json(current, {"schema_version": "1.0", "artifact_type": "sleepyland_yasa_parity_failed",
                              "attempt_path": str((attempt_dir / "attempt.json").resolve()),
                              "status": status})
        raise RuntimeError("Sleepyland/YASA native parity failed; attempt: " + str(attempt_dir))
    verification_id = _identity(before)
    receipt = {"schema_version": "1.0", "artifact_type": "sleepyland_yasa_parity_receipt",
               "verification_id": verification_id, "inputs": before,
               "attempt_path": str((attempt_dir / "attempt.json").resolve()),
               "attempt_sha256": file_sha256(attempt_dir / "attempt.json")}
    receipt_path = root / BASE / "receipts" / (attempt_dir.name + ".json")
    atomic_json(receipt_path, receipt, immutable=True)
    atomic_json(current, {"schema_version": "1.0", "artifact_type": "sleepyland_yasa_parity_current",
                          "verification_id": verification_id,
                          "receipt_path": str(receipt_path.resolve()),
                          "receipt_sha256": file_sha256(receipt_path)})
    return require_verification(root)


def _recorded_path(value: object, parent: Path, name: str | None = None) -> Path:
    if type(value) is not str:
        raise ValueError("Native parity artifact path is not an absolute string")
    path = Path(value)
    if (not path.is_absolute() or path != path.resolve() or
            not path.is_relative_to(parent.resolve()) or
            (name is not None and path.name != name)):
        raise ValueError("Native parity artifact path escapes its run tree")
    return path


def require_verification(root: Path) -> dict:
    """Reject absent, stale, failed, altered, or wrong-scope real-native parity."""
    root = root.resolve()
    current = read_json(root / BASE / "current.json")
    if (current.get("schema_version") != "1.0" or
            current.get("artifact_type") != "sleepyland_yasa_parity_current"):
        raise ValueError("Sleepyland/YASA native parity has no current successful receipt")
    receipts = (root / BASE / "receipts").resolve()
    receipt_path = _recorded_path(current["receipt_path"], receipts)
    if receipt_path.parent != receipts or file_sha256(receipt_path) != current["receipt_sha256"]:
        raise ValueError("Sleepyland/YASA native parity receipt changed")
    receipt = read_json(receipt_path)
    if (receipt.get("schema_version") != "1.0" or
            receipt.get("artifact_type") != "sleepyland_yasa_parity_receipt"):
        raise ValueError("Sleepyland/YASA native parity receipt schema changed")
    snapshot = _snapshot(root, Path(receipt["inputs"]["data_root"]))
    if (receipt["inputs"] != snapshot or
            receipt.get("verification_id") != _identity(snapshot) or
            current.get("verification_id") != receipt["verification_id"]):
        raise ValueError("Sleepyland/YASA native parity is stale")
    attempts = (root / BASE / "attempts").resolve()
    attempt_path = _recorded_path(receipt["attempt_path"], attempts, "attempt.json")
    latest = max((path for path in attempts.iterdir() if path.is_dir()), default=None)
    if (attempt_path.parent.parent != attempts or latest != attempt_path.parent or
            file_sha256(attempt_path) != receipt["attempt_sha256"]):
        raise ValueError("Sleepyland/YASA native parity attempt changed or was superseded")
    attempt = read_json(attempt_path)
    if (attempt.get("status") != "SUCCESS" or attempt.get("before") != snapshot or
            attempt.get("after") != snapshot or attempt.get("error") is not None or
            attempt.get("deadline_seconds") != MAX_SECONDS or
            type(attempt.get("elapsed_seconds")) not in (int, float) or
            not math.isfinite(attempt["elapsed_seconds"]) or
            not 0 <= attempt["elapsed_seconds"] <= MAX_SECONDS):
        raise ValueError("Sleepyland/YASA native parity attempt did not complete")
    results, logs = attempt["results"], attempt["logs"]
    _check_result(results, snapshot, logs)
    for recording_id in RECORDINGS:
        if set(logs[recording_id]) != {"native", "feature"}:
            raise ValueError("Native parity logs are incomplete")
        for kind, suffix, prefix, result_key in (
                ("native", ".reader.log", PREFIX, "reader"),
                ("feature", ".feature.log", FEATURE_PREFIX, "feature_worker")):
            evidence = logs[recording_id][kind]
            log = _recorded_path(evidence["path"], attempt_path.parent,
                                 recording_id + suffix)
            if (file_sha256(log) != evidence["sha256"] or
                    _parse_worker(log.read_text(encoding="utf-8", errors="replace"), prefix) !=
                    results[recording_id][result_key]):
                raise ValueError("Native parity log no longer verifies its result")
    return {"artifact_type": "sleepyland_yasa_native_parity_contract",
            "verification_id": receipt["verification_id"],
            "protocol_hash": snapshot["protocol_hash"], "split_id": snapshot["split_id"],
            "feature_config_id": snapshot["feature_config"]["config_id"],
            "recordings": list(RECORDINGS), "verifier_sha256": snapshot["verifier_sha256"],
            "probe_sha256": snapshot["probe_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    print(json_text(run(args.root, args.data_root)))


if __name__ == "__main__":
    main()
