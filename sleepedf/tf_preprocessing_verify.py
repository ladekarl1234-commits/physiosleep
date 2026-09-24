"""Bounded, signal-only comparison of frozen U-Time preparation with native EDF loading.

This module imports no native model. A receipt is evidence only for the two named
development recordings; failure or a diagnostic helper never qualifies a fit.
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
import re
import shutil
import subprocess
import time
import uuid

import numpy as np

from .contracts import content_id, json_text, read_json
from .protocol import load_protocol
from .research import artifact_id, atomic_json, compute_lease, file_sha256
from .tf_native import locked_packages


RECORD_IDS = ("SC4362", "ST7011")
BASE = "runs/utime-preprocessing-parity"
PREFIX = "UTIME_PREPROCESSING_RESULT "
WORKER_PREFIX = "UTIME_PREPROCESSING_WORKER "
LEASE = "utime-preprocessing-parity"
MAX_SECONDS = 1800  # Per recording; a later invocation renews the bounded lease.
MAX_BYTES = 10 * 1024**3
MIN_FREE_BYTES = 4 * 1024**3
MIN_DISK_BYTES = 20 * 1024**3
THREADS = 4
# These are comparison limits on dimensionless RobustScaler output, not EDF uV.
OUTPUT_ATOL = 1e-6
OUTPUT_RTOL = 1e-6
RAW_ATOL_UV = 1e-6
RAW_RTOL = 1e-10
SOURCES = ("sleepedf/tf_preprocessing_verify.py", "tools/verify_tf_preprocessing.py",
           "docs/NATIVE_TF_PREPROCESSING.md",
           "sleepedf/tf_native.py", "sleepedf/dataset.py", "sleepedf/readers.py",
           "sleepedf/protocol.py", "sleepedf/research.py", "sleepedf/contracts.py",
           "vendor/utime/utime/bin/defaults/utime/dataset_configurations/dataset_1.yaml")
NATIVE_FILES = ("psg_utils/dataset/sleep_study/sleep_study.py",
                "psg_utils/dataset/sleep_study/subject_dir_sleep_study_base.py",
                "psg_utils/dataset/sleep_study/abc_sleep_study.py",
                "psg_utils/io/high_level_file_loaders.py",
                "psg_utils/io/channels/utils.py",
                "psg_utils/io/channels/channels.py",
                "psg_utils/io/psg/psg_extractors.py",
                "mne/io/edf/edf.py",
                "psg_utils/preprocessing/quality_control_funcs.py",
                "psg_utils/preprocessing/psg_sampling.py",
                "psg_utils/preprocessing/scaling.py")


def compare_arrays(reference: np.ndarray, candidate: np.ndarray, *, atol: float,
                   rtol: float) -> dict:
    """Return all-element finite comparison with reference-based normalization."""
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (atol, rtol)):
        raise ValueError("Invalid numerical tolerance")
    if atol <= 0:
        raise ValueError("Absolute tolerance must be positive for finite diagnostics")
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Comparison arrays differ in shape or contain non-finite samples")
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    bound = atol + rtol * np.abs(a.astype(np.float64))
    normalized = np.divide(diff, bound, out=np.full(diff.shape, np.inf), where=bound > 0)
    normalized[(bound == 0) & (diff == 0)] = 0
    index = tuple(int(i) for i in np.unravel_index(int(np.argmax(normalized)), normalized.shape))
    return {"shape": list(a.shape), "reference_dtype": str(a.dtype),
            "candidate_dtype": str(b.dtype), "max_abs_error": float(np.max(diff)),
            "max_normalized_error": float(np.max(normalized)),
            "failed_elements": int(np.count_nonzero(diff > bound)),
            "max_error_index": list(index),
            "abs_error_quantiles_50_95_99_9": [float(x) for x in np.quantile(diff, [0.5, 0.95, 0.999])],
            "atol": float(atol), "rtol": float(rtol)}


def _hex(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _bounded_psg_sha256(path: Path, root: Path, started: float) -> str:
    """Hash the original PSG while enforcing the same lease bounds as the worker."""
    import psutil
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if (time.monotonic() - started > MAX_SECONDS or
                    psutil.Process().memory_info().rss > MAX_BYTES or
                    psutil.virtual_memory().available < MIN_FREE_BYTES or
                    shutil.disk_usage(root).free < MIN_DISK_BYTES):
                raise TimeoutError("PSG hashing exceeded preprocessing resource bound")
            digest.update(chunk)
    return digest.hexdigest()


def _parent_resource_bound(root: Path, started: float) -> None:
    import psutil
    if (time.monotonic() - started > MAX_SECONDS or
            psutil.Process().memory_info().rss > MAX_BYTES or
            psutil.virtual_memory().available < MIN_FREE_BYTES or
            shutil.disk_usage(root).free < MIN_DISK_BYTES):
        raise TimeoutError("Preprocessing parent exceeded time, memory or disk bound")


def _record(root: Path, recording_id: str) -> tuple[dict, dict, dict]:
    if recording_id not in RECORD_IDS:
        raise ValueError("Parity probe is restricted to two previously exposed D recordings")
    protocol, split, readiness = load_protocol(root)
    matching = [r for r in readiness["records"] if r["recording_id"] == recording_id]
    if len(matching) != 1 or matching[0]["participant_id"] not in split["participants"]["development"]:
        raise ValueError("Probe record is not uniquely in the frozen development partition")
    record = matching[0]
    if (record.get("checks", {}).get("signals_verified") is not True or
            type(record.get("n_epochs")) is not int or record["n_epochs"] < 1 or
            type(record.get("duration_seconds")) not in (int, float) or
            not math.isfinite(record["duration_seconds"]) or
            record["n_epochs"] != int(record["duration_seconds"] // 30) or
            not _hex(record.get("psg_sha256"))):
        raise ValueError("Probe record lacks verified signal and complete-grid metadata")
    return protocol, split, record


def _snapshot(root: Path, data_root: Path, recording_id: str,
              *, started: float | None = None) -> dict:
    root, data_root = root.resolve(), data_root.resolve()
    protocol, split, record = _record(root, recording_id)
    psg = (data_root / record["psg"]).resolve()
    psg_sha = (file_sha256(psg) if started is None else
               _bounded_psg_sha256(psg, root, started)) if psg.is_relative_to(data_root) else None
    if psg_sha != record["psg_sha256"]:
        raise ValueError("Probe PSG path or bytes differ from frozen readiness")
    lock = root / "requirements/utime.lock.txt"
    packages = locked_packages(lock)
    site = root / ".venvs/utime/Lib/site-packages"
    observed = {}
    for dist in importlib.metadata.distributions(path=[str(site)]):
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        if name in observed:
            raise ValueError("Duplicate native runtime distribution")
        observed[name] = dist.version
    install = root / "research/runtimes/utime/install.json"
    if observed != packages or read_json(install).get("freeze_sha256") != file_sha256(lock):
        raise ValueError("Native runtime differs from exact frozen lock")
    decision_path = root / "research/utime-signal-domain-decision-v1.json"
    decision = read_json(decision_path)
    lineage_path = root / "research/utime-preprocessing-lineage-v2.json"
    lineage = read_json(lineage_path)
    old_hash = decision.get("source_bindings", {}).get("sleepedf/tf_native.py")
    archive = root / "runs/source-archives/utime-full-record-v2" / str(old_hash) / "tf_native.py"
    if (decision.get("decision_id") != artifact_id(decision, "decision_id") or
            decision.get("slot") != "utime_official" or decision.get("fit_authority") is not False or
            not _hex(old_hash) or not archive.is_file() or file_sha256(archive) != old_hash or
            decision.get("source_bindings", {}).get("reports/native-tf-preprocessing-order-review.md") !=
                file_sha256(root / "reports/native-tf-preprocessing-order-review.md") or
            decision.get("pre_data_tolerances") != {
                "grid_shape_channel_and_dtype": "exact",
                "raw_error_also_less_than_edf_lsb_fraction": 0.01,
                "raw_relative": RAW_RTOL, "raw_uv_absolute": RAW_ATOL_UV,
                "scaled_float32_absolute": OUTPUT_ATOL,
                "scaled_float32_relative": OUTPUT_RTOL}):
        raise ValueError("Frozen pre-measurement signal-domain decision differs")
    if (lineage.get("schema_version") != "1.0" or
            lineage.get("artifact_type") != "utime_full_record_implementation_lineage" or
            lineage.get("lineage_id") != artifact_id(lineage, "lineage_id") or
            lineage.get("decision_id") != decision["decision_id"] or
            lineage.get("decision_sha256") != file_sha256(decision_path) or
            lineage.get("archived_producer_path") != archive.relative_to(root).as_posix() or
            lineage.get("archived_producer_sha256") != old_hash or
            lineage.get("producer_sha256") != file_sha256(root / "sleepedf/tf_native.py") or
            lineage.get("compatibility_worker_sha256") != file_sha256(root / "tools/verify_tf_preprocessing.py") or
            lineage.get("verifier_sha256") != file_sha256(root / "sleepedf/tf_preprocessing_verify.py")):
        raise ValueError("Full-record implementation lineage differs")
    native_hashes = {name: file_sha256(site / name) for name in NATIVE_FILES}
    return {"schema_version": "1.0", "artifact_type": "utime_preprocessing_parity_request",
            "scope": "signal_only_two_development_records_no_hypnogram",
            "recording_id": recording_id, "participant_id": record["participant_id"],
            "n_epochs": record["n_epochs"], "duration_seconds": record["duration_seconds"],
            "psg_path": str(psg), "psg_sha256": record["psg_sha256"],
            "protocol_hash": protocol["protocol_hash"], "split_id": split["split_id"],
            "readiness_sha256": protocol["readiness_sha256"],
            "source_manifest_sha256": file_sha256(root / "research/sources/utime.json"),
            "signal_domain_decision_id": decision["decision_id"],
            "signal_domain_decision_sha256": file_sha256(decision_path),
            "implementation_lineage_sha256": file_sha256(lineage_path),
            "sources_sha256": {name: file_sha256(root / name) for name in SOURCES},
            "native_files_sha256": native_hashes,
            "runtime_lock_sha256": file_sha256(lock),
            "runtime_install_sha256": file_sha256(install),
            "native_executable_sha256": file_sha256(root / ".venvs/utime/Scripts/python.exe"),
            "packages": packages, "channel": "EEG Fpz-Cz", "source_hz": 100,
            "source_unit": "uV", "native_reader_unit": "V", "target_hz": 128,
            "samples_per_epoch": 3840, "output_atol": OUTPUT_ATOL,
            "output_rtol": OUTPUT_RTOL, "raw_atol_uv": RAW_ATOL_UV,
            "raw_rtol": RAW_RTOL, "threads": THREADS,
            "max_seconds": MAX_SECONDS, "max_combined_rss_bytes": MAX_BYTES,
            "min_free_ram_bytes": MIN_FREE_BYTES, "min_free_disk_bytes": MIN_DISK_BYTES}


def _worker_identity(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        lines = [line[len(WORKER_PREFIX):] for line in stream if line.startswith(WORKER_PREFIX)]
    if not lines:
        return None
    if len(lines) != 1:
        raise ValueError("Native worker emitted duplicate process identities")
    identity = json.loads(lines[0], parse_constant=lambda _: (_ for _ in ()).throw(
        ValueError("Non-finite worker identity")))
    if (type(identity) is not dict or set(identity) != {"pid", "process_start", "parent_pid"} or
            type(identity["pid"]) is not int or type(identity["parent_pid"]) is not int or
            type(identity["process_start"]) not in (int, float) or
            not math.isfinite(identity["process_start"])):
        raise ValueError("Invalid native worker process identity")
    return identity


def _monitor(child: subprocess.Popen, root: Path, started: float, log_path: Path) -> dict:
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
            if worker is None:
                identity = _worker_identity(log_path)
                if identity is not None:
                    if identity["parent_pid"] not in (child.pid, os.getpid()):
                        raise ValueError("Native worker is not associated with launcher")
                    process = psutil.Process(identity["pid"])
                    if process.create_time() != identity["process_start"]:
                        raise ValueError("Native worker PID/birth handshake differs")
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
            for pid, born in observed.items():
                try:
                    process = psutil.Process(pid)
                    if process.create_time() == born and process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                        live.append(process)
                except psutil.NoSuchProcess:
                    pass
            rss = psutil.Process().memory_info().rss
            for process in live:
                try:
                    rss += process.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, rss)
            elapsed = time.monotonic() - started
            if (rss > MAX_BYTES or psutil.virtual_memory().available < MIN_FREE_BYTES or
                    shutil.disk_usage(root).free < MIN_DISK_BYTES or elapsed > MAX_SECONDS):
                raise TimeoutError("Preprocessing probe exceeded memory, disk or time bound")
            if child.poll() is not None and worker is None and elapsed > 30:
                raise RuntimeError("Native worker launcher exited without PID/birth handshake")
            if child.poll() is not None and worker is not None and not live:
                return {"peak_combined_rss_bytes": peak, "elapsed_seconds": elapsed,
                        "worker_identity": worker}
            time.sleep(0.25)
    except BaseException as cause:
        survivors = []
        for pid, born in observed.items():
            try:
                process = psutil.Process(pid)
                if process.create_time() == born and process.is_running():
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
        _, remaining = psutil.wait_procs(remaining, timeout=3)
        if remaining:
            identities = [(process.pid, observed.get(process.pid)) for process in remaining]
            raise RuntimeError("Native probe cleanup left live PID/birth identities: " +
                               repr(identities)) from cause
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Native probe launcher remained live after bounded kill") from exc
        raise


def _result_from_log(path: Path) -> dict:
    lines = [line[len(PREFIX):] for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
             if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise ValueError("Native worker did not emit exactly one result")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate result metadata key")
            result[key] = value
        return result
    value = json.loads(lines[0], object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite metadata")))
    json_text(value)
    if type(value) is not dict:
        raise ValueError("Native result is not an object")
    return value


def _artifact(path_value: object, parent: Path, name: str, sha: object) -> Path:
    path = Path(path_value) if type(path_value) is str else Path(".")
    if (not path.is_absolute() or path != path.resolve() or path.parent != parent.resolve() or
            path.name != name or not _hex(sha) or file_sha256(path) != sha):
        raise ValueError("Parity comparison artifact path or content differs")
    return path


def _tail_diagnostic(result: dict, snapshot: dict, attempt_dir: Path,
                     native_raw_uv: np.ndarray, adapter_raw_uv: np.ndarray,
                     adapter: np.ndarray, raw_volts: np.ndarray) -> dict:
    """Check original 1s, reporting-only 30s and retained independent full chain."""
    from scipy.signal import resample_poly
    from sklearn.preprocessing import RobustScaler
    diagnostic = result.get("tail_safe_diagnostic")
    if (type(diagnostic) is not dict or
            diagnostic.get("route") != "SleepStudy_period_length_1s_full_record_diagnostic_only" or
            diagnostic.get("compat30_route") != "SleepStudy_period_length_30s_reporting_reshape_only"):
        raise ValueError("Missing labelled full-record compatibility diagnostic")
    expected_shape = (math.ceil(result["raw_sample_count"] * 128 / 100), 1)
    arrays = {}
    for key, filename in (("path", "tail-safe-one-second.npy"),
                          ("compat30_path", "native-compatible-30s.npy"),
                          ("independent_path", "independent-native-chain.npy")):
        stem = key.removesuffix("_path")
        path = _artifact(diagnostic.get(key), attempt_dir, filename,
                         diagnostic.get("sha256" if key == "path" else stem + "_sha256"))
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != expected_shape or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("Full-record native diagnostic shape, dtype or finite status differs")
        arrays[stem] = values
    original, compat, independent = arrays["path"], arrays["compat30"], arrays["independent"]
    if (diagnostic.get("shape") != list(expected_shape) or
            original.tobytes() != compat.tobytes() or
            original.tobytes() != independent.tobytes()):
        raise ValueError("Native 1s, compatible 30s and independent chain differ bitwise")
    if (raw_volts.shape != (result["raw_sample_count"], 1) or
            raw_volts.dtype != np.float64 or not np.isfinite(raw_volts).all() or
            not np.array_equal(raw_volts[:, 0] * 1e6, native_raw_uv)):
        raise ValueError("Original MNE volts artifact differs from calibrated native raw array")
    full_iqr_v = np.subtract(*np.percentile(raw_volts[:, 0], [75, 25]))
    threshold_v = 20 * full_iqr_v
    clipped = np.clip(raw_volts, -threshold_v, threshold_v)
    resampled = resample_poly(clipped, 128, 100, axis=0)
    scaler = RobustScaler(with_centering=True).fit(resampled.reshape(-1, 1))
    rebuilt = scaler.transform(resampled.reshape(-1, 1)).astype(np.float32)
    full_metric = compare_arrays(rebuilt, original, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
    if full_metric["failed_elements"]:
        raise ValueError("Full native chain differs from parent independent reconstruction")
    independent_scaler = diagnostic.get("independent_scaler")
    if type(independent_scaler) is not dict:
        raise ValueError("Independent native scaler metadata is absent")
    for key, bits_key in (("center_volts", "center_bits"), ("scale_volts", "scale_bits")):
        value = independent_scaler.get(key)
        if (type(value) not in (int, float) or not math.isfinite(value) or
                np.float64(value).tobytes().hex() != independent_scaler.get(bits_key)):
            raise ValueError("Independent scaler value and bit evidence disagree")
    for key, expected in (("center_volts", float(scaler.center_[0])),
                          ("scale_volts", float(scaler.scale_[0])),
                          ("clip_limit_volts", float(threshold_v))):
        observed = independent_scaler.get(key)
        if (type(observed) not in (int, float) or not math.isfinite(observed) or
                compare_arrays(np.asarray([expected]), np.asarray([observed]),
                               atol=1e-12, rtol=RAW_RTOL)["failed_elements"]):
            raise ValueError("Independent full-record scalar differs from parent reconstruction")
    for key in ("scaler", "compat30_scaler"):
        reported = diagnostic.get(key)
        if type(reported) is not dict:
            raise ValueError("Native scaler evidence is absent")
        for value_key, bits_key in (("center_volts", "center_bits"), ("scale_volts", "scale_bits")):
            value = reported.get(value_key)
            if (type(value) not in (int, float) or not math.isfinite(value) or
                    np.float64(value).tobytes().hex() != independent_scaler.get(bits_key)):
                raise ValueError("Native scaler differs bitwise from independent same-runtime chain")
    tail = result["partial_tail_samples"]
    observed_tail = np.asarray(raw_volts[-tail:, 0]) if tail else np.empty(0)
    count = int(np.count_nonzero(np.abs(observed_tail) > threshold_v))
    if diagnostic.get("tail_samples") != tail or diagnostic.get("tail_clipped_samples") != count:
        raise ValueError("Partial-tail QC reporting evidence differs")
    full_iqr_uv = float(np.subtract(*np.percentile(native_raw_uv, [75, 25])))
    prefix_iqr_uv = float(np.subtract(*np.percentile(
        adapter_raw_uv[:snapshot["n_epochs"] * 3000], [75, 25])))
    for key, expected in (("full_raw_iqr_uv", full_iqr_uv),
                          ("prefix_raw_iqr_uv", prefix_iqr_uv)):
        observed = diagnostic.get(key)
        if (type(observed) not in (int, float) or not math.isfinite(observed) or
                not math.isclose(observed, expected, rel_tol=RAW_RTOL, abs_tol=RAW_ATOL_UV)):
            raise ValueError("Raw IQR evidence differs")
    prefix = original[:snapshot["n_epochs"] * 3840].reshape(adapter.shape)
    prefix_metric = compare_arrays(prefix, adapter, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
    if prefix_metric != diagnostic.get("prefix_comparison"):
        raise ValueError("Native prefix comparison differs from retained arrays")
    source = np.asarray(adapter_raw_uv, dtype=np.float64)
    clip_limit_uv = 20 * float(np.subtract(*np.percentile(source, [75, 25])))
    adapted = resample_poly(np.clip(source, -clip_limit_uv, clip_limit_uv), 128, 100)
    adapted_scaler = RobustScaler().fit(adapted.reshape(-1, 1))
    adapted_full = adapted_scaler.transform(adapted.reshape(-1, 1)).astype(np.float32)
    adapted_prefix = adapted_full[:snapshot["n_epochs"] * 3840].reshape(adapter.shape)
    if compare_arrays(adapted_prefix, adapter, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)["failed_elements"]:
        raise ValueError("Adapter output differs from independent full-record reconstruction")
    transforms = result.get("adapter_transforms")
    if type(transforms) is not dict or transforms.get("statistics_domain") != "entire_continuous_record_including_partial_tail":
        raise ValueError("Adapter statistics domain differs from frozen full-record policy")
    if (transforms.get("source_samples") != result["raw_sample_count"] or
            transforms.get("partial_tail_samples") != tail or
            transforms.get("resampled_samples_before_epoch_export") != expected_shape[0]):
        raise ValueError("Adapter source or resampled sample count differs")
    for key, expected in (("clip_limit_uv", clip_limit_uv),
                          ("scaler_center_uv", float(adapted_scaler.center_[0])),
                          ("scaler_scale_uv", float(adapted_scaler.scale_[0]))):
        reported = transforms.get(key)
        if (type(reported) not in (int, float) or not math.isfinite(reported) or
                compare_arrays(np.asarray([expected]), np.asarray([reported]),
                               atol=RAW_ATOL_UV, rtol=RAW_RTOL)["failed_elements"]):
            raise ValueError("Adapter full-record transform metadata differs")
    return {"status": "COMPATIBILITY_VERIFIED", "native_chain_reconstruction": full_metric,
            "adapter_prefix_comparison": prefix_metric, "tail_samples": tail,
            "tail_clipped_samples": count, "full_clip_limit_volts": float(threshold_v),
            "native_scaler_center_volts": float(scaler.center_[0]),
            "native_scaler_scale_volts": float(scaler.scale_[0]),
            "adapter_clip_limit_uv": float(clip_limit_uv)}


def _check_result(result: dict, snapshot: dict, attempt_dir: Path) -> dict:
    if (result.get("schema_version") != "1.0" or
            result.get("artifact_type") != "utime_preprocessing_parity_result" or
            result.get("request_id") != content_id(snapshot) or
            result.get("recording_id") != snapshot["recording_id"] or
            result.get("native_route") != "SleepStudy_raw_EDF_no_hypnogram_full_record"):
        raise ValueError("Direct native SleepStudy route or evidence mismatched")
    n = result.get("raw_sample_count")
    if (type(n) is not int or n != round(snapshot["duration_seconds"] * 100) or
            result.get("partial_tail_samples") != n - snapshot["n_epochs"] * 3000 or
            result.get("excluded_tail_samples_before_adapter_transform") != 0 or
            result.get("statistics_domain_match") is not True or
            result.get("complete_grid_onsets_seconds") != [0, 30 * (snapshot["n_epochs"] - 1)] or
            result.get("source_unit") != "uV" or result.get("native_reader_unit") != "V" or
            result.get("target_hz") != 128):
        raise ValueError("Raw physical grid, tail, full-record domain or units differ")
    step = result.get("physical_step_uv")
    if type(step) not in (int, float) or not math.isfinite(step) or step <= 0:
        raise ValueError("EDF physical digital step is invalid")
    raw_native = _artifact(result.get("raw_native_path"), attempt_dir, "raw-native-uv.npy",
                           result.get("raw_native_sha256"))
    raw_adapter = _artifact(result.get("raw_adapter_path"), attempt_dir, "raw-adapter-uv.npy",
                            result.get("raw_adapter_sha256"))
    raw_volts = _artifact(result.get("raw_volts_path"), attempt_dir, "raw-native-v.npy",
                          result.get("raw_volts_sha256"))
    raw_a = np.load(raw_native, mmap_mode="r", allow_pickle=False)
    raw_b = np.load(raw_adapter, mmap_mode="r", allow_pickle=False)
    raw_v = np.load(raw_volts, mmap_mode="r", allow_pickle=False)
    if (raw_a.shape != (n,) or raw_b.shape != raw_a.shape or
            raw_a.dtype != np.float64 or raw_b.dtype != np.float64):
        raise ValueError("Independent raw reader arrays have wrong shape or dtype")
    raw = compare_arrays(raw_a, raw_b, atol=RAW_ATOL_UV, rtol=RAW_RTOL)
    if (raw != result.get("raw_calibration") or raw["failed_elements"] != 0 or
            raw["max_abs_error"] >= 0.01 * step):
        raise ValueError("Native MNE volts and pyedflib microvolts differ")
    adapter = _artifact(result.get("adapter_path"), attempt_dir, "adapter.npy",
                        result.get("adapter_sha256"))
    b = np.load(adapter, mmap_mode="r", allow_pickle=False)
    shape = (snapshot["n_epochs"], snapshot["samples_per_epoch"], 1)
    if b.shape != shape or b.dtype != np.float32 or not np.isfinite(b).all():
        raise ValueError("Adapter output does not share original complete grid")
    diagnostic = _tail_diagnostic(result, snapshot, attempt_dir, raw_a, raw_b, b, raw_v)
    d = result["tail_safe_diagnostic"]
    compat_path = _artifact(d.get("compat30_path"), attempt_dir, "native-compatible-30s.npy",
                            d.get("compat30_sha256"))
    compatible = np.load(compat_path, mmap_mode="r", allow_pickle=False)
    reference = compatible[:snapshot["n_epochs"] * 3840].reshape(shape)
    metric = compare_arrays(reference, b, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
    if metric["failed_elements"]:
        raise ValueError("Native full-record and adapter prefix differ beyond frozen tolerance")
    error = result.get("native_error")
    if error is not None:
        if (result["partial_tail_samples"] == 0 or type(error) is not dict or
                type(error.get("causes")) is not list or
                not any(c.get("type") == "ValueError" and "cannot reshape array" in c.get("message", "")
                        for c in error["causes"] if type(c) is dict) or
                result.get("native_path") is not None or result.get("native_sha256") is not None):
            raise ValueError("Unmodified native 30s failure is not the documented tail reshape")
    else:
        native = _artifact(result.get("native_path"), attempt_dir, "native.npy",
                           result.get("native_sha256"))
        direct = np.load(native, mmap_mode="r", allow_pickle=False)
        if direct.shape != shape or direct.dtype != np.float32 or direct.tobytes() != reference.tobytes():
            raise ValueError("Unmodified native 30s differs from compatible full-record prefix")
        direct_metric = compare_arrays(direct, b, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
        if direct_metric != result.get("output_comparison"):
            raise ValueError("Unmodified native comparison differs from retained arrays")
        direct_scaler = result.get("native_scaler")
        bits = d["independent_scaler"]
        if (type(direct_scaler) is not dict or
                any(type(direct_scaler.get(key)) not in (int, float) or
                    not math.isfinite(direct_scaler[key]) or
                    np.float64(direct_scaler[key]).tobytes().hex() != bits[bit_key]
                    for key, bit_key in (("center_volts", "center_bits"),
                                         ("scale_volts", "scale_bits")))):
            raise ValueError("Unmodified native scaler differs from full-record reference")
    return {"status": "PASS_COMPAT_30S" if error else "PASS",
            "recording_id": snapshot["recording_id"], "raw_calibration": raw,
            "output_comparison": metric, "native_direct_error": error,
            "full_record_diagnostic": diagnostic,
            "compat30_sha256": d["compat30_sha256"],
            "adapter_sha256": result["adapter_sha256"]}


def run(root: Path, data_root: Path, recording_id: str) -> dict:
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    with compute_lease(root, LEASE):
        return _run_locked(root, data_root, recording_id)


def _run_locked(root: Path, data_root: Path, recording_id: str) -> dict:
    if recording_id not in RECORD_IDS:
        raise ValueError("Unregistered preprocessing probe record")
    base = root / BASE / recording_id
    attempt = base / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
                                   "-" + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    current = base / "current.json"
    atomic_json(current, {"status": "PENDING", "attempt_path": str((attempt / "attempt.json").resolve())})
    start = time.monotonic()
    before = after = result = monitor = summary = invocation = None
    failure = None
    try:
        before = _snapshot(root, data_root, recording_id, started=start)
        request = {"schema_version": "1.0", "artifact_type": "utime_preprocessing_worker_request",
                   "root": str(root), "data_root": str(data_root), "attempt_dir": str(attempt),
                   "snapshot": before, "request_id": content_id(before), "lease_run_id": LEASE}
        atomic_json(attempt / "request.json", request, immutable=True)
        log_path = attempt / "worker.log"
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   NUMEXPR_NUM_THREADS="4", TF_NUM_INTRAOP_THREADS="4",
                   TF_NUM_INTEROP_THREADS="4", CUDA_VISIBLE_DEVICES="-1")
        lease_owner = read_json(root / "runs/compute.lock")
        env["UTIME_LEASE_PARENT_PID"] = str(lease_owner["pid"])
        env["UTIME_LEASE_PARENT_BIRTH"] = str(lease_owner["process_start"])
        command = [str(root / ".venvs/utime/Scripts/python.exe"), "-I",
                   str(root / "tools/verify_tf_preprocessing.py"), "--request",
                   str(attempt / "request.json")]
        invocation = {"command": command, "request_path": str((attempt / "request.json").resolve()),
                      "request_sha256": file_sha256(attempt / "request.json"),
                      "thread_environment": {name: env[name] for name in (
                          "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                          "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
                          "TF_NUM_INTEROP_THREADS", "CUDA_VISIBLE_DEVICES")},
                      "exit_code": None}
        with log_path.open("xb") as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT)
            monitor = _monitor(child, root, start, log_path)
        invocation["exit_code"] = child.returncode
        after = _snapshot(root, data_root, recording_id, started=start)
        if before != after or time.monotonic() - start > MAX_SECONDS:
            raise ValueError("Probe inputs changed or per-record deadline expired")
        if child.returncode != 0:
            raise RuntimeError("Native preprocessing worker failed; inspect immutable worker.log")
        result = _result_from_log(log_path)
        summary = _check_result(result, before, attempt)
        _parent_resource_bound(root, start)
        if summary["status"] not in ("PASS", "PASS_COMPAT_30S"):
            raise ValueError("Native preprocessing scientific comparison failed: " + summary["status"])
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    log = attempt / "worker.log"
    record = {"schema_version": "1.0", "artifact_type": "utime_preprocessing_attempt",
              "status": "SUCCESS" if failure is None else "FAILED",
              "before": before, "after": after, "result": result, "summary": summary,
              "monitor": monitor, "invocation": invocation, "failure": failure,
              "elapsed_seconds": time.monotonic() - start,
              "log_sha256": file_sha256(log) if log.exists() else None}
    atomic_json(attempt / "attempt.json", record, immutable=True)
    if failure is not None:
        atomic_json(current, {"status": "FAILED", "attempt_path": str((attempt / "attempt.json").resolve())})
        raise RuntimeError("U-Time preprocessing parity failed; immutable attempt: " + str(attempt))
    receipt = {"schema_version": "1.0", "artifact_type": "utime_preprocessing_receipt",
               "recording_id": recording_id, "request_id": content_id(before),
               "attempt_path": str((attempt / "attempt.json").resolve()),
               "attempt_sha256": file_sha256(attempt / "attempt.json")}
    receipt_path = base / "receipts" / (attempt.name + ".json")
    atomic_json(receipt_path, receipt, immutable=True)
    atomic_json(current, {"status": "SUCCESS", "receipt_path": str(receipt_path.resolve()),
                          "receipt_sha256": file_sha256(receipt_path)})
    try:
        return verify_record(root, data_root, recording_id, started=start)
    except BaseException:
        atomic_json(current, {"status": "FAILED", "attempt_path": str((attempt / "attempt.json").resolve())})
        raise


def verify_record(root: Path, data_root: Path, recording_id: str,
                  *, started: float | None = None) -> dict:
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    started = time.monotonic() if started is None else started
    snapshot = _snapshot(root, data_root, recording_id, started=started)
    base = root / BASE / recording_id
    current = read_json(base / "current.json")
    if type(current) is not dict or set(current) != {"status", "receipt_path", "receipt_sha256"} or current["status"] != "SUCCESS":
        raise ValueError("No successful current preprocessing receipt")
    receipt_path = Path(current["receipt_path"])
    if (not receipt_path.is_absolute() or receipt_path != receipt_path.resolve() or
            receipt_path.parent != (base / "receipts").resolve() or
            file_sha256(receipt_path) != current["receipt_sha256"]):
        raise ValueError("Current preprocessing receipt path or bytes changed")
    receipt = read_json(receipt_path)
    candidate = Path(receipt.get("attempt_path", ""))
    attempts = (base / "attempts").resolve()
    latest = max((p for p in attempts.iterdir() if p.is_dir()), default=None)
    if (receipt.get("recording_id") != recording_id or receipt.get("request_id") != content_id(snapshot) or
            not candidate.is_absolute() or candidate != candidate.resolve() or
            candidate.name != "attempt.json" or candidate.parent.parent != attempts or
            candidate.parent != latest or not _hex(receipt.get("attempt_sha256")) or
            file_sha256(candidate) != receipt["attempt_sha256"]):
        raise ValueError("Preprocessing attempt is stale, moved or superseded")
    attempt = read_json(candidate)
    expected_request = {"schema_version": "1.0", "artifact_type": "utime_preprocessing_worker_request",
                        "root": str(root), "data_root": str(data_root), "attempt_dir": str(candidate.parent),
                        "snapshot": snapshot, "request_id": content_id(snapshot), "lease_run_id": LEASE}
    invocation = attempt.get("invocation")
    expected_command = [str(root / ".venvs/utime/Scripts/python.exe"), "-I",
                        str(root / "tools/verify_tf_preprocessing.py"), "--request",
                        str(candidate.parent / "request.json")]
    if (read_json(candidate.parent / "request.json") != expected_request or
            type(invocation) is not dict or invocation.get("command") != expected_command or
            invocation.get("request_path") != str(candidate.parent / "request.json") or
            invocation.get("request_sha256") != file_sha256(candidate.parent / "request.json") or
            invocation.get("exit_code") != 0 or
            invocation.get("thread_environment") != {
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                "NUMEXPR_NUM_THREADS": "4", "TF_NUM_INTRAOP_THREADS": "4",
                "TF_NUM_INTEROP_THREADS": "4", "CUDA_VISIBLE_DEVICES": "-1"}):
        raise ValueError("Native preprocessing invocation or request differs")
    if (attempt.get("status") != "SUCCESS" or attempt.get("before") != snapshot or
            attempt.get("after") != snapshot or attempt.get("failure") is not None or
            type(attempt.get("elapsed_seconds")) not in (int, float) or
            not 0 < attempt["elapsed_seconds"] <= MAX_SECONDS or
            type(attempt.get("monitor")) is not dict or
            attempt["monitor"].get("worker_identity") != _worker_identity(candidate.parent / "worker.log") or
            type(attempt.get("summary")) is not dict or
            attempt["summary"].get("status") not in ("PASS", "PASS_COMPAT_30S") or
            type(attempt["monitor"].get("peak_combined_rss_bytes")) is not int or
            not 0 < attempt["monitor"]["peak_combined_rss_bytes"] <= MAX_BYTES or
            type(attempt["monitor"].get("elapsed_seconds")) not in (int, float) or
            not math.isfinite(attempt["monitor"]["elapsed_seconds"]) or
            not 0 < attempt["monitor"]["elapsed_seconds"] <= attempt["elapsed_seconds"] + 0.1):
        raise ValueError("Successful attempt lacks bounded execution evidence")
    log = candidate.parent / "worker.log"
    if (file_sha256(log) != attempt.get("log_sha256") or
            _result_from_log(log) != attempt.get("result") or
            _check_result(attempt["result"], snapshot, candidate.parent) != attempt.get("summary")):
        raise ValueError("Worker log or independently recomputed comparison differs")
    _parent_resource_bound(root, started)
    return {"recording_id": recording_id, "request_id": receipt["request_id"],
            "summary": attempt["summary"], "receipt_path": str(receipt_path)}


def require_verification(root: Path, data_root: Path) -> dict:
    """Both real D comparisons must pass; no synthetic or diagnostic route counts."""
    rows = [verify_record(root, data_root, rid) for rid in RECORD_IDS]
    return {"artifact_type": "utime_preprocessing_two_record_parity",
            "records": rows, "identity": content_id(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--recording-id", choices=RECORD_IDS, required=True)
    args = parser.parse_args()
    print(json_text(run(args.root, args.data_root, args.recording_id)))


if __name__ == "__main__":
    main()
