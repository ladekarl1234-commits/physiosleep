"""Development-only U-Time signal adapter and full-grid inference contracts.

This module constructs no TensorFlow model. The original U-Time source remains
in vendor/utime; native model execution is isolated in tools/tf_native_worker.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

CHANNEL = "EEG Fpz-Cz"
SOURCE_HZ = 100
TARGET_HZ = 128
EPOCH_SECONDS = 30
SOURCE_SAMPLES = 3000
TARGET_SAMPLES = 3840
CONTEXT = 35
CLASS_ORDER = ("W", "N1", "N2", "N3", "REM")
PRIMARY_POLICY = "all_overlap_count_normalized_v1"
TAIL_POLICY = "contiguous_35_plus_final_tail_v1"
SHORT_POLICY = "symmetric_nearest_boundary_repeat_v1"
PREPROCESS_RECIPE = {"version": "full_record_before_epoch_export_v2",
                     "statistics_domain": "entire_continuous_record_including_partial_tail",
                     "channel": CHANNEL, "source_unit": "uV", "source_hz": SOURCE_HZ,
                     "target_hz": TARGET_HZ, "clip": "whole_record_absolute_20_iqr",
                     "resample": "scipy.signal.resample_poly_continuous_128_100",
                     "scaler": "sklearn.RobustScaler_default_fit_whole_recording",
                     "output_dtype": "float32", "samples_per_epoch": TARGET_SAMPLES}


def inner_partition(split: dict, fold_id: int) -> dict:
    """Freeze cohort-stratified 40/8 within the original outer 48."""
    if type(fold_id) is not int or not 0 <= fold_id < 5:
        raise ValueError("U-Time fold must be 0..4")
    fold = split["folds"][fold_id]
    outer_train = set(fold["train"])
    outer_validation = set(fold["validation"])
    if len(outer_train) != 48 or len(outer_validation) != 12 or outer_train & outer_validation:
        raise ValueError("U-Time requires the frozen 48/12 participant fold")
    selected = []
    ranks = {}
    for cohort, count in (("SC", 6), ("ST", 2)):
        cohort_ids = [p for p in outer_train if p.startswith(cohort + ":")]
        if len(cohort_ids) <= count:
            raise ValueError("Insufficient participants for cohort-stratified inner split")
        ordered = sorted(cohort_ids,
                         key=lambda participant: (
                             hashlib.sha256(("native-inner-v1|" + split["split_id"] + "|" +
                                             str(fold_id) + "|" + participant).encode("utf-8")).hexdigest(),
                             participant))
        selected.extend(ordered[:count])
        ranks[cohort] = ordered
    if len(ranks["SC"]) + len(ranks["ST"]) != 48:
        raise ValueError("Unexpected cohort in the outer training fold")
    inner_validation = set(selected)
    inner_train = outer_train - inner_validation
    if len(inner_validation) != 8 or len(inner_train) != 40:
        raise ValueError("U-Time inner 40/8 split is incomplete")
    return {"fold_id": fold_id, "split_id": split["split_id"],
            "inner_train": sorted(inner_train), "inner_validation": sorted(inner_validation),
            "outer_train": sorted(outer_train), "outer_validation": sorted(outer_validation),
            "ranked_by_cohort": ranks, "method": "sha256_native-inner-v1_cohort_6SC_2ST"}


def steps_per_epoch(total_complete_epochs: int) -> int:
    if type(total_complete_epochs) is not int or total_complete_epochs < CONTEXT:
        raise ValueError("U-Time needs at least one complete 35-epoch training context")
    return math.ceil((min(total_complete_epochs, 500_000) // CONTEXT) / 12)


def continuous_blocks(n_epochs: int, blocks: list[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    if type(n_epochs) is not int or n_epochs < 1:
        raise ValueError("Full PSG grid must contain an epoch")
    if blocks is None:
        return [(0, n_epochs)]
    result = []
    end = 0
    for start, stop in blocks:
        if type(start) is not int or type(stop) is not int or start != end or stop <= start:
            raise ValueError("Signal blocks must partition the original grid in order")
        result.append((start, stop))
        end = stop
    if end != n_epochs:
        raise ValueError("Signal blocks omit original epochs")
    return result


def window_plan(n_epochs: int, *, policy: str = PRIMARY_POLICY,
                blocks: list[tuple[int, int]] | None = None) -> list[dict]:
    if policy not in (PRIMARY_POLICY, TAIL_POLICY):
        raise ValueError("Unregistered U-Time window policy")
    plans = []
    for first, stop in continuous_blocks(n_epochs, blocks):
        length = stop - first
        if length < CONTEXT:
            left = (CONTEXT - length) // 2
            plans.append({"start": first, "stop": stop, "block_start": first,
                          "block_stop": stop, "pad_left": left,
                          "pad_right": CONTEXT - length - left,
                          "context_padded": True})
            continue
        if policy == PRIMARY_POLICY:
            starts = range(first, stop - CONTEXT + 1)
        else:
            starts = list(range(first, stop - CONTEXT + 1, CONTEXT))
            if starts[-1] != stop - CONTEXT:
                starts.append(stop - CONTEXT)
        plans.extend({"start": start, "stop": start + CONTEXT,
                      "block_start": first, "block_stop": stop,
                      "pad_left": 0, "pad_right": 0,
                      "context_padded": False} for start in starts)
    return plans


def make_window_batch(signal: np.ndarray, plans: list[dict]) -> np.ndarray:
    """No labels or Hypnogram may enter the prediction window builder."""
    signal = np.asarray(signal)
    if signal.ndim != 3 or signal.shape[1:] != (TARGET_SAMPLES, 1) or signal.dtype != np.float32:
        raise ValueError("U-Time signal must be float32 [original epochs,3840,1]")
    windows = np.empty((len(plans), CONTEXT, TARGET_SAMPLES, 1), dtype=np.float32)
    for i, plan in enumerate(plans):
        start, stop = plan["start"], plan["stop"]
        if not 0 <= start < stop <= len(signal) or \
                start < plan["block_start"] or stop > plan["block_stop"]:
            raise ValueError("U-Time window crosses a recording or signal block")
        chunk = signal[start:stop]
        left, right = plan["pad_left"], plan["pad_right"]
        if left or right:
            chunk = np.pad(chunk, ((left, right), (0, 0), (0, 0)), mode="edge")
        if chunk.shape != (CONTEXT, TARGET_SAMPLES, 1) or not np.isfinite(chunk).all():
            raise ValueError("U-Time context width is not 35 epochs")
        windows[i] = chunk
    return windows


def paired_generator(signal: np.ndarray, labels: np.ndarray, plans: list[dict],
                     batch_size: int = 12):
    """Native-shaped pairs plus explicit invalid-loss mask for development fit."""
    labels = np.asarray(labels)
    if labels.shape != (len(signal),) or labels.dtype.kind not in "iu" or \
            np.any((labels < -1) | (labels > 4)) or batch_size < 1:
        raise ValueError("U-Time training labels must cover the full grid with -1 invalid")
    for offset in range(0, len(plans), batch_size):
        batch_plans = plans[offset:offset + batch_size]
        x = make_window_batch(signal, batch_plans)
        y = np.empty((len(batch_plans), CONTEXT, 1), dtype=np.int32)
        mask = np.empty_like(y, dtype=bool)
        for i, plan in enumerate(batch_plans):
            row = labels[plan["start"]:plan["stop"]]
            left, right = plan["pad_left"], plan["pad_right"]
            row = np.pad(row, (left, right), constant_values=-1) if left or right else row
            mask[i, :, 0] = row >= 0
            y[i, :, 0] = np.where(row >= 0, row, 0)
        yield x, y, mask


def aggregate_windows(predictions: np.ndarray, plans: list[dict], n_epochs: int,
                      *, policy: str = PRIMARY_POLICY) -> tuple[np.ndarray, np.ndarray]:
    """Average all overlaps, or retain only novel positions in the native tail route."""
    if policy not in (PRIMARY_POLICY, TAIL_POLICY):
        raise ValueError("Unregistered U-Time aggregation policy")
    predictions = np.asarray(predictions)
    if predictions.shape != (len(plans), CONTEXT, 5) or not np.isfinite(predictions).all() or \
            np.any((predictions < 0) | (predictions > 1)) or \
            np.any(np.abs(predictions.sum(axis=2) - 1) > 1e-6):
        raise ValueError("U-Time output must be finite five-class per-window probabilities")
    sums = np.zeros((n_epochs, 5), dtype=np.float64)
    counts = np.zeros(n_epochs, dtype=np.int64)
    assigned_to = 0
    for prediction, plan in zip(predictions, plans):
        start, stop = plan["start"], plan["stop"]
        novel_start = max(start, assigned_to) if policy == TAIL_POLICY else start
        left = plan["pad_left"] + novel_start - start
        real = prediction[left:left + stop - novel_start]
        if real.shape != (stop - novel_start, 5):
            raise ValueError("Padded U-Time window does not trim to the original block")
        sums[novel_start:stop] += real
        counts[novel_start:stop] += 1
        assigned_to = max(assigned_to, stop)
    if np.any(counts < 1):
        raise ValueError("U-Time predictions omit original PSG epochs")
    result = sums / counts[:, None]
    if not np.isfinite(result).all() or np.any(np.abs(result.sum(axis=1) - 1) > 1e-6):
        raise ValueError("Count-normalized U-Time probabilities are invalid")
    return result, counts


def recording_transform(source_uv: np.ndarray) -> tuple[np.ndarray, dict]:
    """Transform the full recording, then export its complete 30-second epochs."""
    from scipy.signal import resample_poly
    from sklearn.preprocessing import RobustScaler
    signal = np.asarray(source_uv, dtype=np.float64)
    if (signal.ndim not in (1, 2) or
            (signal.ndim == 2 and signal.shape[1] != SOURCE_SAMPLES) or
            signal.size < SOURCE_SAMPLES or not np.isfinite(signal).all()):
        raise ValueError("Physical EEG must contain a finite full recording in microvolts")
    flat = signal.reshape(-1)
    n_epochs = len(flat) // SOURCE_SAMPLES
    q25, q75 = np.percentile(flat, [25, 75])
    limit = 20.0 * (q75 - q25)
    if not np.isfinite(limit):
        raise ValueError("Recording-wide U-Time IQR is non-finite")
    clipped = np.clip(flat, -limit, limit)
    resampled = resample_poly(clipped, TARGET_HZ, SOURCE_HZ)
    expected_samples = (len(flat) * TARGET_HZ + SOURCE_HZ - 1) // SOURCE_HZ
    if resampled.shape != (expected_samples,):
        raise ValueError("Polyphase resampling changed the original epoch grid")
    scaler = RobustScaler()
    scaled = scaler.fit_transform(resampled.reshape(-1, 1))
    full_result = scaled.astype(np.float32)
    if not np.isfinite(full_result).all():
        raise ValueError("U-Time scaled signal is non-finite or overflows float32")
    result = full_result[:n_epochs * TARGET_SAMPLES].reshape(n_epochs, TARGET_SAMPLES, 1)
    return result, {"source_unit": "uV", "clip_global_iqr_multiplier": 20.0,
                    "statistics_domain": "entire_continuous_record_including_partial_tail",
                    "source_samples": len(flat),
                    "partial_tail_samples": len(flat) % SOURCE_SAMPLES,
                    "resampled_samples_before_epoch_export": expected_samples,
                    "clip_limit_uv": float(limit), "resample": "scipy.signal.resample_poly_128_100",
                    "scaler": "sklearn.preprocessing.RobustScaler_default_recording_local",
                    "scaler_center_uv": float(scaler.center_[0]),
                    "scaler_scale_uv": float(scaler.scale_[0])}


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked_packages(lock: Path) -> dict[str, str]:
    """Parse the complete isolated freeze with PEP 503-normalized names."""
    packages = {}
    for line in Path(lock).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9._+!-]*)", line)
        if match is None:
            raise ValueError("U-Time runtime lock contains an unpinned requirement")
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        if name in packages:
            raise ValueError("U-Time runtime lock contains a duplicate package")
        packages[name] = match.group(2)
    if not packages or packages.get("tensorflow") != "2.13.1":
        raise ValueError("U-Time runtime lock is incomplete")
    return packages


def verify_native_packages(root: Path) -> dict[str, str]:
    import importlib.metadata
    import sysconfig
    if Path(sys.executable).resolve() != (root / ".venvs" / "utime" / "Scripts" / "python.exe").resolve():
        raise ValueError("U-Time preprocessing requires the isolated native runtime")
    expected = locked_packages(root / "requirements" / "utime.lock.txt")
    observed = {}
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    for distribution in importlib.metadata.distributions():
        if Path(distribution.locate_file("")).resolve() != site_packages:
            continue
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        if name in observed:
            raise ValueError("U-Time runtime has duplicate normalized distributions")
        observed[name] = distribution.version
    if observed != expected:
        raise ValueError("U-Time installed package set differs from its exact runtime lock")
    return observed


def _preprocess_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    lock = root / "requirements" / "utime.lock.txt"
    packages = locked_packages(lock)
    return {"recipe": PREPROCESS_RECIPE,
            "implementation_sha256": {name: _fingerprint(root / name) for name in (
                "sleepedf/tf_native.py", "sleepedf/dataset.py", "sleepedf/readers.py")},
            "runtime_lock_sha256": _fingerprint(lock),
            "producer_dependencies": {name: packages[name] for name in (
                "numpy", "scipy", "scikit-learn", "psg-utils")},
            "native_source_manifest_sha256": _fingerprint(root / "research" / "sources" / "utime.json")}


def _safe_output(path: Path, data_root: Path | None = None) -> None:
    resolved = path.resolve()
    if (data_root is not None and resolved.is_relative_to(data_root.resolve())) or \
            any(parent.name == "sleep-edf-database-expanded-1.0.0" for parent in (resolved, *resolved.parents)):
        raise ValueError("U-Time artifacts cannot be written inside original data")


def prepare_signal_record(record: dict, data_root: Path, cache_dir: Path) -> dict:
    """D-only signal export, independent of labels at every transformation step."""
    from .contracts import read_json
    from .dataset import _source_path
    from .readers import read_signal_window, signal_metadata
    from .research import atomic_json
    verify_native_packages(Path(__file__).resolve().parents[1])
    _safe_output(cache_dir, data_root)
    if record.get("checks", {}).get("signals_verified") is not True:
        raise ValueError("U-Time signal access requires validated readiness")
    n = record["n_epochs"]
    if type(n) is not int or n < 1:
        raise ValueError("U-Time requires complete original PSG epochs")
    source = _source_path(data_root, record["psg"])
    if _fingerprint(source) != record["psg_sha256"]:
        raise ValueError("U-Time PSG differs from verified readiness")
    data_path = cache_dir / (record["recording_id"] + ".npy")
    meta_path = data_path.with_suffix(".json")
    if data_path.exists() or meta_path.exists():
        return load_signal_cache(record, cache_dir, metadata_only=True)
    header = signal_metadata(source)
    duration = header["duration_seconds"]
    channels = [item for item in header["channels"] if item["label"] == CHANNEL]
    if (header["continuity"] != "continuous_header_or_plain_edf" or len(channels) != 1 or
            channels[0]["unit"] not in ("uV", "µV", "μV") or
            channels[0]["sample_rate_hz"] != SOURCE_HZ or
            math.floor(duration / EPOCH_SECONDS) != n or
            ("duration_seconds" in record and
             not math.isclose(duration, record["duration_seconds"], rel_tol=0, abs_tol=1e-6))):
        raise ValueError("U-Time full recording duration, continuity, rate or calibration differs")
    window = read_signal_window(source, 0.0, duration, (CHANNEL,))
    channel = window["channels"][CHANNEL]
    raw = np.asarray(channel["samples_uv"])
    if (channel["unit"] != "uV" or channel["sample_rate_hz"] != SOURCE_HZ or
            raw.shape != (round(duration * SOURCE_HZ),) or
            len(raw) // SOURCE_SAMPLES != n or not np.isfinite(raw).all()):
        raise ValueError("U-Time EEG needs the full finite calibrated 100 Hz recording")
    transformed, details = recording_transform(raw)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_dir, prefix=".utime-", suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.save(temporary, transformed, allow_pickle=False)
        os.replace(temporary, data_path)
        meta = {"schema_version": "1.0", "artifact_type": "utime_signal_only_full_grid",
                "recording_id": record["recording_id"], "participant_id": record["participant_id"],
                "source_psg_sha256": record["psg_sha256"], "n_epochs": n,
                "source_channel": CHANNEL, "target_hz": TARGET_HZ,
                "samples_per_epoch": TARGET_SAMPLES, "output_unit": "robust_scaled",
                "continuous_blocks": [[0, n]], "transforms": details,
                "preprocess_identity": _preprocess_identity(),
                "payload_sha256": _fingerprint(data_path)}
        atomic_json(meta_path, meta, immutable=True)
        return meta
    finally:
        temporary.unlink(missing_ok=True)


def load_signal_cache(record: dict, cache_dir: Path, *, metadata_only: bool = False):
    from .contracts import read_json
    data_path = cache_dir / (record["recording_id"] + ".npy")
    meta_path = data_path.with_suffix(".json")
    if not data_path.is_file() or not meta_path.is_file():
        raise ValueError("U-Time signal cache is incomplete")
    meta = read_json(meta_path)
    if (meta.get("schema_version") != "1.0" or meta.get("artifact_type") != "utime_signal_only_full_grid" or
            meta.get("recording_id") != record["recording_id"] or
            meta.get("participant_id") != record["participant_id"] or
            meta.get("source_psg_sha256") != record["psg_sha256"] or
            meta.get("n_epochs") != record["n_epochs"] or
            meta.get("source_channel") != CHANNEL or meta.get("target_hz") != TARGET_HZ or
            meta.get("samples_per_epoch") != TARGET_SAMPLES or
            meta.get("output_unit") != "robust_scaled" or
            meta.get("continuous_blocks") != [[0, record["n_epochs"]]] or
            meta.get("preprocess_identity") != _preprocess_identity() or
            meta.get("payload_sha256") != _fingerprint(data_path)):
        raise ValueError("U-Time cache source, full grid, or content differs")
    transforms = meta.get("transforms")
    if (type(transforms) is not dict or
            transforms.get("source_unit") != "uV" or
            transforms.get("statistics_domain") != "entire_continuous_record_including_partial_tail" or
            type(transforms.get("source_samples")) is not int or
            type(transforms.get("partial_tail_samples")) is not int or
            type(transforms.get("resampled_samples_before_epoch_export")) is not int or
            transforms["source_samples"] // SOURCE_SAMPLES != record["n_epochs"] or
            transforms["partial_tail_samples"] != transforms["source_samples"] % SOURCE_SAMPLES or
            ("duration_seconds" in record and
             transforms["source_samples"] != round(record["duration_seconds"] * SOURCE_HZ)) or
            transforms["resampled_samples_before_epoch_export"] !=
            (transforms["source_samples"] * TARGET_HZ + SOURCE_HZ - 1) // SOURCE_HZ or
            transforms.get("clip_global_iqr_multiplier") != 20.0 or
            transforms.get("resample") != "scipy.signal.resample_poly_128_100" or
            transforms.get("scaler") != "sklearn.preprocessing.RobustScaler_default_recording_local" or
            any(type(transforms.get(key)) not in (int, float) or
                not math.isfinite(transforms[key]) for key in (
                    "clip_limit_uv", "scaler_center_uv", "scaler_scale_uv")) or
            transforms["clip_limit_uv"] < 0 or transforms["scaler_scale_uv"] <= 0):
        raise ValueError("U-Time cache preprocessing metadata differs")
    values = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if values.shape != (record["n_epochs"], TARGET_SAMPLES, 1) or values.dtype != np.float32:
        raise ValueError("U-Time cache shape or dtype differs")
    return meta if metadata_only else values


def prepare_development(root: Path, data_root: Path, cache_dir: Path, label_dir: Path) -> dict:
    """Stage D signals and private D labels; no audit participant can enter."""
    from .protocol import development_records, load_development_truth
    _safe_output(cache_dir, data_root)
    _safe_output(label_dir, data_root)
    _, split, records = development_records(root)
    if len(split["participants"]["development"]) != 60:
        raise ValueError("U-Time preparation requires frozen original 60 development participants")
    label_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        prepare_signal_record(record, data_root, cache_dir)
        truth = load_development_truth(record, split)
        labels = np.asarray(truth["reference_label"])
        mask = np.asarray(truth["valid_mask"], dtype=bool)
        if (labels.shape != (record["n_epochs"],) or mask.shape != labels.shape or
                not np.array_equal(truth["epoch_index"], np.arange(len(labels))) or
                not np.array_equal(truth["onset_seconds"], np.arange(len(labels)) * 30.0) or
                np.any(mask & ((labels < 0) | (labels > 4)))):
            raise ValueError("U-Time development truth differs from the original PSG grid")
        exported = np.where(mask, labels, -1).astype(np.int8)
        label_path = label_dir / (record["recording_id"] + ".npy")
        if label_path.exists():
            old = np.load(label_path, allow_pickle=False)
            if old.dtype != np.int8 or not np.array_equal(old, exported):
                raise ValueError("Existing U-Time development label export differs")
        else:
            with tempfile.NamedTemporaryFile(dir=label_dir, suffix=".npy", delete=False) as handle:
                temporary = Path(handle.name)
            try:
                np.save(temporary, exported, allow_pickle=False)
                os.replace(temporary, label_path)
            finally:
                temporary.unlink(missing_ok=True)
    return {"artifact_type": "utime_development_prepare", "split_id": split["split_id"],
            "recordings": len(records), "cache_dir": str(cache_dir.resolve()),
            "label_dir": str(label_dir.resolve())}


def _build_inner_manifest(root: Path, fold_id: int, protocol: dict,
                          split: dict, records: list[dict]) -> dict:
    from .contracts import content_id
    partition = inner_partition(split, fold_id)
    counts = {r["recording_id"]: {"participant_id": r["participant_id"],
                                   "n_epochs": r["n_epochs"],
                                   "source_psg_sha256": r["psg_sha256"]}
              for r in records if r["participant_id"] in partition["outer_train"] + partition["outer_validation"]}
    if {r["participant_id"] for r in records if r["participant_id"] in partition["outer_train"]} != \
            set(partition["outer_train"]):
        raise ValueError("U-Time outer training recordings are incomplete")
    def total(ids):
        return sum(r["n_epochs"] for r in records if r["participant_id"] in ids)
    inner_t, outer_t = total(partition["inner_train"]), total(partition["outer_train"])
    recipe = {"architecture": "vendor/utime UTime", "class_order": list(CLASS_ORDER),
              "channel": CHANNEL, "source_hz": SOURCE_HZ, "model_hz": TARGET_HZ,
              "context_epochs": CONTEXT, "margin": 17,
              "batch_size": 12, "optimizer": {"type": "Adam", "learning_rate": 5e-6,
                                                "decay": 0.0, "beta_1": 0.9,
                                                "beta_2": 0.999, "epsilon": 1e-8},
              "loss": "SparseCategoricalCrossentropy_mask_invalid_original_grid",
              "balanced_sampling": True, "max_training_epochs": 2000,
              "patience": 80, "min_delta": 0, "stopping_metric": "inner_pooled_fixed5_macro_f1_rounded4",
              "selection_tie": "earliest_best_epoch", "outer_refit": "all48_for_selected_inner_epochs",
              "inner_steps_per_epoch": steps_per_epoch(inner_t),
              "outer_steps_per_epoch": steps_per_epoch(outer_t),
              "inference_policy": PRIMARY_POLICY, "short_block_policy": SHORT_POLICY}
    source_manifest = root / "research" / "sources" / "utime.json"
    hparams = root / "vendor" / "utime" / "utime" / "bin" / "defaults" / "utime" / "hparams.yaml"
    item = {"schema_version": "1.0", "artifact_type": "utime_development_inner_freeze",
            "protocol_hash": protocol["protocol_hash"], "split_id": split["split_id"],
            "registry_hash": protocol["registry_hash"], "partition": partition,
            "recordings": counts, "recipe": recipe,
            "preprocess_identity": _preprocess_identity(),
            "source_manifest_sha256": _fingerprint(source_manifest),
            "native_hparams_sha256": _fingerprint(hparams)}
    item["manifest_id"] = content_id(item)
    return item


def freeze_inner_manifest(root: Path, fold_id: int, output: Path) -> dict:
    """Persist split/recipe before fitting; never opens audit labels or weights."""
    from .protocol import development_records
    from .research import atomic_json
    protocol, split, records = development_records(root)
    item = _build_inner_manifest(root, fold_id, protocol, split, records)
    _safe_output(output)
    atomic_json(output, item, immutable=True)
    return item


def _validate_outer_checkpoint(checkpoint: dict, frozen: dict, record: dict) -> None:
    from .contracts import content_id
    partition = frozen["partition"]
    if (checkpoint.get("artifact_type") != "utime_outer48_refit_complete" or
            checkpoint.get("selection_rule") != "frozen_inner40_8_earliest_best" or
            checkpoint.get("inner_manifest_id") != frozen["manifest_id"] or
            checkpoint.get("fold_id") != partition["fold_id"] or
            checkpoint.get("protocol_hash") != frozen["protocol_hash"] or
            checkpoint.get("split_id") != frozen["split_id"] or
            checkpoint.get("train_participants") != partition["outer_train"] or
            checkpoint.get("validation_participants") != partition["outer_validation"] or
            checkpoint.get("source_manifest_sha256") != frozen["source_manifest_sha256"] or
            checkpoint.get("native_hparams_sha256") != frozen["native_hparams_sha256"] or
            checkpoint.get("runtime_lock_sha256") != frozen["preprocess_identity"]["runtime_lock_sha256"] or
            checkpoint.get("adapter_sha256") != frozen["preprocess_identity"]["implementation_sha256"]["sleepedf/tf_native.py"] or
            checkpoint.get("recipe_id") != content_id(frozen["recipe"]) or
            checkpoint.get("preprocess_identity") != frozen["preprocess_identity"] or
            checkpoint.get("selected_inner_epoch") != checkpoint.get("completed_epochs") or
            type(checkpoint.get("completed_epochs")) is not int or
            not 1 <= checkpoint["completed_epochs"] <= 2000 or
            record["participant_id"] in checkpoint["train_participants"]):
        raise ValueError("U-Time prediction requires completed outer-48 selected lineage")
    weights = Path(checkpoint["weights_path"]).resolve()
    if not weights.is_file() or _fingerprint(weights) != checkpoint.get("weights_sha256"):
        raise ValueError("U-Time outer refit weights differ from the checkpoint manifest")
    # This adapter has no native fit runner or independent selection/refit
    # verifier. A self-declared epoch, history, or weights hash cannot establish
    # that the inner 40/8 selection and outer 48 refit actually occurred.
    raise ValueError("U-Time prediction awaits independently verified inner selection and outer refit")


def predict_development(root: Path, fold_id: int, inner_manifest_path: Path,
                        checkpoint_meta_path: Path, cache_dir: Path,
                        recording_id: str, output: Path) -> dict:
    """Signal-only OOF prediction request; the worker receives no labels."""
    from .contracts import read_json
    from .predictions import save_prediction
    from .protocol import development_records
    from .research import compute_lease
    _safe_output(output)
    protocol, split, records = development_records(root)
    frozen = read_json(inner_manifest_path)
    expected_freeze = _build_inner_manifest(root, fold_id, protocol, split, records)
    if frozen != expected_freeze:
        raise ValueError("Frozen U-Time inner selection differs from protocol")
    matches = [r for r in records if r["recording_id"] == recording_id and
               r["participant_id"] in frozen["partition"]["outer_validation"]]
    if len(matches) != 1:
        raise ValueError("U-Time OOF prediction requires one outer-12 recording")
    record = matches[0]
    checkpoint = read_json(checkpoint_meta_path)
    _validate_outer_checkpoint(checkpoint, frozen, record)
    cache_meta = load_signal_cache(record, cache_dir, metadata_only=True)
    lock = root / "requirements" / "utime.lock.txt"
    installation = read_json(root / "research" / "runtimes" / "utime" / "install.json")
    if _fingerprint(lock) != installation["freeze_sha256"]:
        raise ValueError("U-Time isolated runtime lock changed")
    source_manifest = root / "research" / "sources" / "utime.json"
    hparams = root / "vendor" / "utime" / "utime" / "bin" / "defaults" / "utime" / "hparams.yaml"
    if (_fingerprint(source_manifest) != frozen["source_manifest_sha256"] or
            _fingerprint(hparams) != frozen["native_hparams_sha256"]):
        raise ValueError("U-Time pinned source or native hparams changed")
    signal_path = cache_dir / (recording_id + ".npy")
    request = {"schema_version": "1.0", "artifact_type": "utime_signal_only_predict_request",
               "policy": PRIMARY_POLICY, "short_block_policy": SHORT_POLICY,
               "class_order": list(CLASS_ORDER),
               "runtime_lock_sha256": _fingerprint(lock),
               "source_manifest_sha256": frozen["source_manifest_sha256"],
               "native_hparams_sha256": frozen["native_hparams_sha256"],
               "signal": {"recording_id": recording_id,
                          "participant_id": record["participant_id"],
                          "path": str(signal_path.resolve()),
                          "payload_sha256": cache_meta["payload_sha256"],
                          "meta_path": str(signal_path.with_suffix(".json").resolve()),
                          "meta_sha256": _fingerprint(signal_path.with_suffix(".json")),
                          "source_psg_sha256": record["psg_sha256"],
                          "channel": CHANNEL, "rate_hz": TARGET_HZ,
                          "samples_per_epoch": TARGET_SAMPLES,
                          "n_epochs": record["n_epochs"],
                          "continuous_blocks": cache_meta["continuous_blocks"]},
               "checkpoint": {key: checkpoint[key] for key in (
                   "artifact_type", "selection_rule", "inner_manifest_id", "fold_id",
                   "protocol_hash", "split_id", "train_participants",
                   "validation_participants", "source_manifest_sha256",
                   "native_hparams_sha256", "selected_inner_epoch", "completed_epochs",
                   "weights_path", "weights_sha256")}}
    environment = os.environ.copy()
    environment.update(TF_NUM_INTRAOP_THREADS="4", TF_NUM_INTEROP_THREADS="4",
                       OMP_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="-1",
                       WANDB_MODE="disabled")
    with tempfile.TemporaryDirectory(dir=root / "runs", prefix="utime-predict-") as scratch:
        scratch = Path(scratch)
        request_path, raw_path = scratch / "request.json", scratch / "raw.npz"
        request_path.write_text(json.dumps(request, sort_keys=True, allow_nan=False), encoding="utf-8")
        command = [str(root / ".venvs" / "utime" / "Scripts" / "python.exe"), "-I",
                   str(root / "tools" / "tf_native_worker.py"), "predict",
                   "--request", str(request_path), "--output", str(raw_path)]
        with compute_lease(root, "utime-predict-fold-" + str(fold_id)):
            completed = subprocess.run(command, cwd=root, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, timeout=4 * 3600 + 300)
        if completed.returncode:
            raise RuntimeError("Isolated U-Time prediction failed: " + completed.stdout[-2000:])
        lines = [line for line in completed.stdout.splitlines() if line.startswith("UTIME_RESULT ")]
        if len(lines) != 1:
            raise ValueError("U-Time worker returned no unique prediction result")
        result = json.loads(lines[0][len("UTIME_RESULT "):])
        if (result.get("recording_id") != recording_id or result.get("n_epochs") != record["n_epochs"] or
                result.get("raw_sha256") != _fingerprint(raw_path) or
                result.get("weights_sha256") != checkpoint["weights_sha256"]):
            raise ValueError("U-Time native output coverage or checkpoint differs")
        with np.load(raw_path, allow_pickle=False) as raw:
            if set(raw.files) != {"probabilities", "hard_label", "coverage_count"}:
                raise ValueError("Unexpected native U-Time output fields")
            prob = raw["probabilities"].copy()
            hard = raw["hard_label"].copy()
            coverage = raw["coverage_count"].copy()
        if coverage.shape != (record["n_epochs"],) or np.any(coverage < 1):
            raise ValueError("U-Time native output omitted original epochs")
        provenance = {"candidate": "original_utime_native_v1", "fold_id": fold_id,
                      "training_participants": frozen["partition"]["outer_train"],
                      "inner_manifest_id": frozen["manifest_id"],
                      "checkpoint_sha256": checkpoint["weights_sha256"],
                      "source_psg_sha256": record["psg_sha256"],
                      "source_channel": CHANNEL, "sampling_hz": TARGET_HZ,
                      "policy": PRIMARY_POLICY, "short_block_policy": SHORT_POLICY,
                      "context_padded": result["context_padded"],
                      "coverage_min": int(coverage.min()), "coverage_max": int(coverage.max()),
                      "native_window_count": result["windows"]}
        return save_prediction(output, participant_id=record["participant_id"],
                               recording_id=recording_id,
                               epoch_index=np.arange(record["n_epochs"]),
                               onset_seconds=np.arange(record["n_epochs"]) * 30.0,
                               hard_label=hard, probabilities=prob,
                               model_id="native-utime-v1", protocol_hash=protocol["protocol_hash"],
                               registry_hash=protocol["registry_hash"], provenance=provenance)


def main(argv: list[str] | None = None) -> None:
    from .contracts import json_text
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze_inner", "prepare", "predict"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/utime/eeg128-cache"))
    parser.add_argument("--label-dir", type=Path, default=Path("runs/utime/labels"))
    parser.add_argument("--inner-manifest", type=Path)
    parser.add_argument("--checkpoint-meta", type=Path)
    parser.add_argument("--recording-id")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    cache = args.cache_dir if args.cache_dir.is_absolute() else root / args.cache_dir
    labels = args.label_dir if args.label_dir.is_absolute() else root / args.label_dir
    if args.action == "freeze_inner":
        if args.output is None:
            parser.error("freeze_inner requires --output")
        print(json_text(freeze_inner_manifest(root, args.fold, args.output.resolve())))
    elif args.action == "prepare":
        if args.data_root is None:
            parser.error("prepare requires --data-root")
        from .research import compute_lease
        with compute_lease(root, "utime-development-prepare"):
            print(json_text(prepare_development(root, args.data_root.resolve(), cache, labels)))
    else:
        if args.output is None or args.inner_manifest is None or \
                args.checkpoint_meta is None or not args.recording_id:
            parser.error("predict requires --output, --inner-manifest, --checkpoint-meta, --recording-id")
        print(json_text(predict_development(root, args.fold, args.inner_manifest,
                                            args.checkpoint_meta, cache, args.recording_id,
                                            args.output.resolve())))


if __name__ == "__main__":
    main()
