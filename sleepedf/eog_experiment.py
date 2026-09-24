"""Provisional D-only EOG staging: separate feature preparation and fresh OOF fit.

This is an experimental single-sensor route, not native YASA SleepStaging
classification and not an Audit A/B or sensor-superiority result.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import sys
import time
from uuid import uuid4

import numpy as np

from . import eog_features
from .contracts import content_id, read_json
from .evaluation import evaluate_saved_records
from .predictions import load_prediction, save_prediction
from .protocol import development_records, load_development_truth
from .readers import read_signal_window, signal_metadata
from .research import atomic_json, compute_lease, file_sha256


CHANNEL = eog_features.CHANNEL
SEED = 17
MAX_SECONDS = 8 * 3600
MAX_RSS = 10 * 1024**3
MIN_FREE = 4 * 1024**3
MIN_DISK = 20 * 1024**3
MODEL_ID = "provisional-eog-only-lightgbm-v1"
SOURCES = (
    "sleepedf/eog_experiment.py", "sleepedf/eog_features.py", "sleepedf/yasa_baseline.py",
    "sleepedf/readers.py", "sleepedf/protocol.py", "sleepedf/splits.py",
    "sleepedf/dataset.py", "sleepedf/research.py", "sleepedf/contracts.py",
    "sleepedf/predictions.py", "sleepedf/evaluation.py",
    "research/sources/yasa.json", "research/provisional-phase2-authorization-v1.json",
)


def _resources(root: Path, started: float) -> int:
    import psutil

    if time.monotonic() - started > MAX_SECONDS:
        raise RuntimeError("Provisional EOG invocation exceeded eight hours")
    own = psutil.Process()
    rss = own.memory_info().rss
    for child in own.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    if rss > MAX_RSS or psutil.virtual_memory().available < MIN_FREE or shutil.disk_usage(root).free < MIN_DISK:
        raise RuntimeError("Provisional EOG resource bound reached; retain completed artifacts")
    return rss


def _paths(root: Path, path: Path, area: str) -> Path:
    root = root.resolve()
    base = root / area
    if base.resolve() != base or not path.is_relative_to(base) or path.resolve() != path:
        raise ValueError("Provisional EOG output redirects outside its canonical area")
    return path


def _source_identity(root: Path) -> dict:
    return {name: file_sha256(root / name) for name in SOURCES}


def _runtime(root: Path) -> dict:
    executable = (root / ".venvs/research/Scripts/python.exe").resolve()
    if Path(sys.executable).resolve() != executable:
        raise ValueError("Provisional EOG route requires the locked research Python")
    def normalized(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()
    pinned = {}
    for line in (root / "requirements/research.lock.txt").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line.strip())
        if match:
            pinned[normalized(match[1])] = match[2]
    installed = {normalized(dist.metadata["Name"]): dist.version
                 for dist in importlib.metadata.distributions()}
    if not pinned or installed != pinned:
        raise ValueError("Installed research package set differs from the pinned lock")
    return {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
            "executable_sha256": file_sha256(executable),
            "lock_sha256": file_sha256(root / "requirements/research.hashed.txt"),
            "pin_sha256": file_sha256(root / "requirements/research.lock.txt"),
            "packages": installed}


def _context(root: Path) -> tuple[dict, dict, list[dict], dict]:
    protocol, split, records = development_records(root)
    records = sorted(records, key=lambda row: row["recording_id"])
    if (len(records) != 119 or len(split["participants"]["development"]) != 60 or
            len({r["recording_id"] for r in records}) != 119 or
            any(r["participant_id"] not in split["participants"]["development"] or
                r.get("checks", {}).get("signals_verified") is not True or
                type(r.get("n_epochs")) is not int or r["n_epochs"] < 2 or
                type(r.get("duration_seconds")) not in (int, float) or
                not np.isfinite(r["duration_seconds"]) for r in records)):
        raise ValueError("Provisional EOG route requires original verified D60/119 records")
    authorization = read_json(root / "research/provisional-phase2-authorization-v1.json")
    if (authorization.get("artifact_type") != "owner_authorized_provisional_phase_revision" or
            authorization.get("authorization_id") != content_id({
                key: value for key, value in authorization.items() if key != "authorization_id"}) or
            authorization.get("original_protocol_sha256") != file_sha256(root / "research/protocol-v1.json") or
            authorization.get("gate_A") != "NOT_RUN" or authorization.get("gate_B") != "NOT_RUN" or
            authorization.get("audit_access") is not False or
            "development_sensor_reduction" not in authorization.get("allowed_work", [])):
        raise ValueError("Current provisional D-only authorization is missing or changed")
    return protocol, split, records, authorization


def _safe_psg(data_root: Path, rec: dict) -> Path:
    relative = Path(rec["psg"])
    path = (data_root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(data_root.resolve()) or file_sha256(path) != rec["psg_sha256"]:
        raise ValueError("Original PSG differs from frozen development identity")
    return path


def _read_eog(data_root: Path, rec: dict) -> np.ndarray:
    psg = _safe_psg(data_root, rec)
    header = signal_metadata(psg)
    matches = [item for item in header["channels"] if item["label"] == CHANNEL]
    if (len(matches) != 1 or matches[0]["sample_rate_hz"] != 100 or
            matches[0]["unit"] not in ("uV", "µV", "μV") or
            abs(header["duration_seconds"] - rec["duration_seconds"]) > 1e-6):
        raise ValueError("EOG channel, native 100-Hz grid or duration changed")
    result = read_signal_window(psg, 0., rec["duration_seconds"], channel_names=(CHANNEL,))
    if set(result["channels"]) != {CHANNEL}:
        raise ValueError("EOG reader returned undeclared channels")
    one = result["channels"][CHANNEL]
    samples = one["samples_uv"]
    if (one["unit"] != "uV" or one["sample_rate_hz"] != 100 or
            type(samples) is not np.ndarray or samples.ndim != 1 or
            len(samples) != round(rec["duration_seconds"] * 100) or
            len(samples) // eog_features.EPOCH_SAMPLES != rec["n_epochs"] or
            file_sha256(psg) != rec["psg_sha256"]):
        raise ValueError("Single EOG physical samples changed their original full grid")
    return samples


def _feature_config(root: Path, protocol: dict, split: dict) -> dict:
    return {"model_id": MODEL_ID, "channel": CHANNEL, "sample_rate_hz": 100, "unit": "uV",
            "epoch_seconds": 30, "minimum_complete_epochs": 2,
            "tail_policy": "full signal enters native EOG filter; only complete epochs exported",
            "missing_policy": "raw nonfinite rejected; native undefined features retained as NaN with mask",
            "context": "offline whole-record unlabeled centered 15-epoch and past 4-epoch native features",
            "protocol_hash": protocol["protocol_hash"], "split_id": split["split_id"],
            "source_sha256": _source_identity(root), "runtime": _runtime(root)}


def _feature_paths(root: Path, rec: dict) -> tuple[Path, Path]:
    base = root / "derived/eog_features"
    stem = rec["recording_id"]
    if not stem.isalnum():
        raise ValueError("Unsafe original recording ID")
    return (_paths(root, base / f"{stem}.npz", "derived/eog_features"),
            _paths(root, base / f"{stem}.json", "derived/eog_features"))


def _feature_expected(rec: dict, config_id: str) -> dict:
    return {"recording_id": rec["recording_id"], "participant_id": rec["participant_id"],
            "source_psg_sha256": rec["psg_sha256"], "n_epochs": rec["n_epochs"],
            "config_id": config_id, "channel": CHANNEL}


def _feature_load(root: Path, rec: dict, config_id: str):
    import pandas as pd

    path, sidecar = _feature_paths(root, rec)
    meta = read_json(sidecar)
    if (any(meta.get(key) != value for key, value in _feature_expected(rec, config_id).items()) or
            meta.get("payload_sha256") != file_sha256(path) or
            type(meta.get("columns")) is not list or len(meta["columns"]) != 53 or
            len(set(meta["columns"])) != 53 or
            any(not isinstance(name, str) or not (name.startswith("eog_") or
                name in ("time_hour", "time_norm")) for name in meta["columns"]) or
            type(meta.get("trailing_samples")) is not int or not 0 <= meta["trailing_samples"] < 3000):
        raise ValueError("EOG feature payload/identity differs from frozen preparation")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"features", "missing_mask", "epoch_index"}:
            raise ValueError("EOG cache fields changed")
        values = archive["features"]
        missing = archive["missing_mask"]
        epoch_index = archive["epoch_index"]
        if (values.shape != (rec["n_epochs"], 53) or values.dtype != np.float32 or
                missing.shape != values.shape or missing.dtype != np.bool_ or
                not np.array_equal(missing, np.isnan(values)) or np.isinf(values).any() or
                not np.array_equal(epoch_index, np.arange(rec["n_epochs"], dtype=np.int32)) or
                int(missing.sum()) != meta.get("missing_feature_values")):
            raise ValueError("EOG cache lost complete epochs, dtypes or missingness")
        return pd.DataFrame(values.copy(), columns=meta["columns"]), meta


def _save_features(root: Path, rec: dict, config_id: str, result: dict) -> dict:
    path, sidecar = _feature_paths(root, rec)
    frame = result["features"]
    if (len(frame) != rec["n_epochs"] or len(frame.columns) != 53 or
            len(set(frame.columns)) != 53 or
            any(not isinstance(name, str) or not (name.startswith("eog_") or
                name in ("time_hour", "time_norm")) for name in frame.columns) or
            not np.array_equal(result["epoch_index"], np.arange(rec["n_epochs"]))):
        raise ValueError("EOG extractor omitted original complete epochs")
    values = frame.to_numpy(dtype=np.float32)
    missing = result["missing_mask"]
    if (missing.shape != values.shape or not np.array_equal(missing, np.isnan(values)) or
            np.isinf(values).any()):
        raise ValueError("EOG extractor changed explicit native missingness")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _paths(root, path.with_name(path.stem + ".partial-" + uuid4().hex + ".npz"),
                       "derived/eog_features")
    try:
        np.savez_compressed(temporary, features=values, missing_mask=missing,
                            epoch_index=np.arange(rec["n_epochs"], dtype=np.int32))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    meta = dict(_feature_expected(rec, config_id), columns=list(frame.columns),
                trailing_samples=result["metadata"]["trailing_samples"],
                missing_feature_values=int(missing.sum()), payload_sha256=file_sha256(path))
    atomic_json(sidecar, meta, immutable=True)
    return {"payload_sha256": meta["payload_sha256"], "sidecar_sha256": file_sha256(sidecar)}


def _existing_or_quarantine(root: Path, rec: dict, config_id: str) -> dict | None:
    path, sidecar = _feature_paths(root, rec)
    if path.exists() != sidecar.exists():
        orphan = path if path.exists() else sidecar
        archived = _paths(root, orphan.with_name("unattested-" + file_sha256(orphan) +
                            "-" + uuid4().hex + orphan.suffix), "derived/eog_features")
        orphan.rename(archived)
    if path.exists():
        _feature_load(root, rec, config_id)
        return {"payload_sha256": file_sha256(path), "sidecar_sha256": file_sha256(sidecar)}
    return None


def _verify_manifest(root: Path, manifest: dict, protocol: dict, split: dict,
                     records: list[dict], config: dict) -> None:
    if (manifest.get("status") != "DEVELOPMENT_EOG_FEATURES_COMPLETE" or
            manifest.get("manifest_id") != content_id({k: v for k, v in manifest.items() if k != "manifest_id"}) or
            manifest.get("protocol_hash") != protocol["protocol_hash"] or
            manifest.get("split_id") != split["split_id"] or manifest.get("config") != config or
            set(manifest.get("artifacts", {})) != {r["recording_id"] for r in records}):
        raise ValueError("EOG feature manifest differs from current D60 protocol/config")
    config_id = content_id(config)
    for rec in records:
        path, sidecar = _feature_paths(root, rec)
        frozen = manifest["artifacts"][rec["recording_id"]]
        if (frozen != {"payload_sha256": file_sha256(path),
                       "sidecar_sha256": file_sha256(sidecar)}):
            raise ValueError("EOG feature bytes changed after manifest publication")
        _feature_load(root, rec, config_id)


PROBE_IDS = ("SC4362", "ST7011")


def _probe_identity(root: Path, records: list[dict], config: dict) -> dict:
    by_id = {rec["recording_id"]: rec for rec in records}
    if not set(PROBE_IDS) <= set(by_id):
        raise ValueError("The two forced EOG profiling records are not both in D")
    artifacts = {}
    for record_id in PROBE_IDS:
        rec = by_id[record_id]
        path, sidecar = _feature_paths(root, rec)
        _feature_load(root, rec, content_id(config))
        artifacts[record_id] = {"payload_sha256": file_sha256(path),
                                "sidecar_sha256": file_sha256(sidecar)}
    return {"schema_version": "1.0", "status": "TWO_RECORD_EOG_PROFILE_ONLY",
            "recording_ids": list(PROBE_IDS), "config_id": content_id(config),
            "artifacts": artifacts, "full_development_prepared": False,
            "confirmatory": False, "gate_A": "NOT_RUN", "gate_B": "NOT_RUN"}


def _checked_probe(receipt: dict, expected: dict) -> None:
    if (any(receipt.get(key) != value for key, value in expected.items()) or
            set(receipt.get("timings", {})) != set(PROBE_IDS)):
        raise ValueError("Two forced D-record EOG profile is absent or stale")
    for row in receipt["timings"].values():
        if (type(row) is not dict or type(row.get("elapsed_seconds")) not in (int, float) or
                not np.isfinite(row["elapsed_seconds"]) or row["elapsed_seconds"] < 0 or
                row["elapsed_seconds"] > MAX_SECONDS or
                type(row.get("sampled_max_rss_bytes")) is not int or
                not 0 < row["sampled_max_rss_bytes"] <= MAX_RSS):
            raise ValueError("Two-record EOG profile contains invalid resource evidence")


def probe(root: Path, data_root: Path) -> dict:
    """Profile exactly SC4362 and ST7011 before authorizing a full D119 pass."""
    from threadpoolctl import threadpool_limits

    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    started = time.monotonic()
    _paths(root, root / "runs/compute.lock", "runs")
    _paths(root, root / "derived/eog_features/probe.json", "derived/eog_features")
    with compute_lease(root, "provisional-eog-two-record-profile"), threadpool_limits(limits=4):
        protocol, split, records, _ = _context(root)
        config = _feature_config(root, protocol, split)
        path = _paths(root, root / "derived/eog_features/probe.json", "derived/eog_features")
        if path.exists():
            receipt = read_json(path)
            _checked_probe(receipt, _probe_identity(root, records, config))
            return receipt
        by_id = {rec["recording_id"]: rec for rec in records}
        timing = {}
        for record_id in PROBE_IDS:
            rec = by_id[record_id]
            rss_before = _resources(root, started)
            record_started = time.monotonic()
            old = _existing_or_quarantine(root, rec, content_id(config))
            signal = _read_eog(data_root, rec)
            result = eog_features.extract_eog_features(root, signal, channel_name=CHANNEL,
                                                       sample_rate_hz=100, unit="uV")
            if old is None:
                _save_features(root, rec, content_id(config), result)
            else:
                cached, meta = _feature_load(root, rec, content_id(config))
                if (list(cached.columns) != list(result["features"].columns) or
                        not np.array_equal(cached.to_numpy(dtype=np.float32),
                                           result["features"].to_numpy(dtype=np.float32), equal_nan=True) or
                        not np.array_equal(cached.isna().to_numpy(), result["missing_mask"]) or
                        meta["trailing_samples"] != result["metadata"]["trailing_samples"]):
                    raise ValueError("Forced EOG profile disagrees with retained cached features")
            timing[record_id] = {"elapsed_seconds": time.monotonic() - record_started,
                                 "sampled_max_rss_bytes": max(rss_before, _resources(root, started))}
        if _source_identity(root) != config["source_sha256"] or _runtime(root) != config["runtime"]:
            raise ValueError("EOG source/runtime changed during two-record profile")
        receipt = dict(_probe_identity(root, records, config), timings=timing)
        atomic_json(path, receipt, immutable=True)
        return receipt


def prepare(root: Path, data_root: Path) -> dict:
    """Acquire one lease and export all canonical D119 EOG features, never truth."""
    from threadpoolctl import threadpool_limits

    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    started = time.monotonic()
    _paths(root, root / "runs/compute.lock", "runs")
    _paths(root, root / "derived/eog_features/manifest.json", "derived/eog_features")
    with compute_lease(root, "provisional-eog-prepare"), threadpool_limits(limits=4):
        _resources(root, started)
        protocol, split, records, _ = _context(root)
        config = _feature_config(root, protocol, split)
        config_id = content_id(config)
        probe_path = _paths(root, root / "derived/eog_features/probe.json", "derived/eog_features")
        prior = read_json(probe_path)
        expected_probe = _probe_identity(root, records, config)
        _checked_probe(prior, expected_probe)
        manifest_path = _paths(root, root / "derived/eog_features/manifest.json", "derived/eog_features")
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            _verify_manifest(root, manifest, protocol, split, records, config)
            return manifest
        artifacts = {}
        for rec in records:
            _resources(root, started)
            old = _existing_or_quarantine(root, rec, config_id)
            if old is None:
                signal = _read_eog(data_root, rec)
                result = eog_features.extract_eog_features(
                    root, signal, channel_name=CHANNEL, sample_rate_hz=100, unit="uV")
                old = _save_features(root, rec, config_id, result)
            artifacts[rec["recording_id"]] = old
            _resources(root, started)
        if _source_identity(root) != config["source_sha256"] or _runtime(root) != config["runtime"]:
            raise ValueError("EOG source/runtime changed during D119 preparation")
        manifest = {"schema_version": "1.0", "status": "DEVELOPMENT_EOG_FEATURES_COMPLETE",
                    "protocol_hash": protocol["protocol_hash"], "split_id": split["split_id"],
                    "config": config, "artifacts": artifacts, "records": 119, "participants": 60,
                    "confirmatory": False, "gate_A": "NOT_RUN", "gate_B": "NOT_RUN"}
        manifest["manifest_id"] = content_id(manifest)
        atomic_json(manifest_path, manifest, immutable=True)
        _verify_manifest(root, manifest, protocol, split, records, config)
        return manifest


def _fit_config(root: Path, protocol: dict, split: dict, manifest_path: Path,
                manifest: dict) -> dict:
    params = {"boosting_type": "gbdt", "n_estimators": 400, "max_depth": 5,
              "num_leaves": 90, "colsample_bytree": .5, "importance_type": "gain",
              "learning_rate": .1, "class_weight": {"0": 1, "1": 2.2, "2": 1, "3": 1.2, "4": 1.4},
              "n_jobs": 4, "random_state": SEED, "deterministic": True,
              "force_col_wise": True, "verbosity": -1}
    return {"model_id": MODEL_ID, "seed": SEED, "params": params,
            "protocol_hash": protocol["protocol_hash"], "registry_hash": protocol["registry_hash"],
            "split_id": split["split_id"], "feature_manifest_sha256": file_sha256(manifest_path),
            "feature_manifest_id": manifest["manifest_id"],
            "truth_manifest_sha256": file_sha256(root / "research/development-truth-v1.json"),
            "source_sha256": _source_identity(root), "runtime": _runtime(root),
            "provisional_authorization_sha256": file_sha256(root / "research/provisional-phase2-authorization-v1.json"),
            "confirmatory": False, "gate_A": "NOT_RUN", "gate_B": "NOT_RUN"}


def _model_path(root: Path, run: Path, fold: dict) -> Path:
    return _paths(root, run / f"fold-{fold['fold_id']}" / "model.joblib", "runs/provisional-eog")


def _fit_receipt(path: Path, config: dict, fold: dict) -> dict:
    fit = read_json(path.with_name("fit.json"))
    required = {"status", "seed", "fold_id", "config_hash", "training_participants",
                "validation_participants", "checkpoint_sha256", "training_epochs",
                "valid_epochs_by_training_record", "rounds_requested", "trees"}
    counts = fit.get("valid_epochs_by_training_record")
    if (set(fit) != required or type(counts) is not dict or
            not counts or any(type(v) is not int or v < 0 for v in counts.values()) or
            type(fit.get("training_epochs")) is not int or
            fit["training_epochs"] != sum(counts.values()) or
            fit.get("trees") != 2000 or
            fit.get("status") != "FIT_COMPLETE" or fit.get("config_hash") != content_id(config) or
            fit.get("fold_id") != fold["fold_id"] or fit.get("checkpoint_sha256") != file_sha256(path) or
            fit.get("training_participants") != sorted(fold["train"]) or
            fit.get("validation_participants") != sorted(fold["validation"]) or
            fit.get("seed") != SEED or fit.get("rounds_requested") != 400 or
            len(set(fold["train"])) != 48 or len(set(fold["validation"])) != 12 or
            set(fold["train"]) & set(fold["validation"])):
        raise ValueError("EOG fold checkpoint or participant ancestry changed")
    return fit


def _verify_frozen_truth(records: list[dict]) -> None:
    """Bind D evaluation bytes without decoding labels into the fit controller."""
    for rec in records:
        path = Path(rec["truth_path"])
        frozen = rec.get("frozen_truth")
        if (type(frozen) is not dict or
                file_sha256(path) != frozen.get("payload_sha256") or
                file_sha256(path.with_suffix(".json")) != frozen.get("sidecar_sha256")):
            raise ValueError("EOG evaluation truth differs from pre-training D freeze")


def train_or_load(root: Path) -> dict:
    """Fresh five-fold D60 OOF model; reference labels enter only training/evaluation."""
    import joblib
    import lightgbm as lgb
    import pandas as pd
    from threadpoolctl import threadpool_limits

    root = Path(root).resolve()
    started = time.monotonic()
    _paths(root, root / "runs/compute.lock", "runs")
    _paths(root, root / "runs/provisional-eog", "runs/provisional-eog")
    with compute_lease(root, "provisional-eog-fit"), threadpool_limits(limits=4):
        _resources(root, started)
        protocol, split, records, _ = _context(root)
        manifest_path = root / "derived/eog_features/manifest.json"
        manifest = read_json(manifest_path)
        _verify_manifest(root, manifest, protocol, split, records,
                         _feature_config(root, protocol, split))
        config = _fit_config(root, protocol, split, manifest_path, manifest)
        run = root / "runs/provisional-eog" / content_id(config)[7:23]
        _paths(root, run / "config.json", "runs/provisional-eog")
        atomic_json(run / "config.json", config, immutable=True)
        predictions, truth_paths = [], []
        for fold in split["folds"]:
            _resources(root, started)
            path = _model_path(root, run, fold)
            fit_path = _paths(root, path.with_name("fit.json"), "runs/provisional-eog")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not fit_path.exists():
                archived = _paths(root, path.with_name("unattested-" + file_sha256(path) +
                              "-" + uuid4().hex + ".joblib"), "runs/provisional-eog")
                path.rename(archived)
            if fit_path.exists() and not path.exists():
                archived = _paths(root, fit_path.with_name("unattested-" + file_sha256(fit_path) +
                              "-" + uuid4().hex + ".json"), "runs/provisional-eog")
                fit_path.rename(archived)
            if not path.exists():
                frames, labels, counts = [], [], {}
                for rec in records:
                    if rec["participant_id"] not in fold["train"]:
                        continue
                    truth = load_development_truth(rec, split)
                    frame, _ = _feature_load(root, rec, content_id(manifest["config"]))
                    valid = truth["valid_mask"]
                    if len(frame) != len(valid):
                        raise ValueError("EOG training features and frozen truth grid differ")
                    frames.append(frame.loc[valid])
                    labels.append(truth["reference_label"][valid].astype(np.int32))
                    counts[rec["recording_id"]] = int(valid.sum())
                    _resources(root, started)
                X, y = pd.concat(frames, ignore_index=True), np.concatenate(labels)
                if set(y) != set(range(5)):
                    raise ValueError("EOG training fold lacks a required five-class label")
                parameters = dict(config["params"])
                parameters["class_weight"] = {int(key): value for key, value in parameters["class_weight"].items()}
                model = lgb.LGBMClassifier(**parameters)
                def check_iteration(_):
                    _resources(root, started)
                check_iteration.before_iteration = True
                check_iteration.order = 0
                model.fit(X, y, callbacks=[check_iteration])
                _resources(root, started)
                temporary = _paths(root, path.with_suffix(".partial.joblib"), "runs/provisional-eog")
                joblib.dump(model, temporary, compress=3)
                temporary.replace(path)
                atomic_json(fit_path, {"status": "FIT_COMPLETE", "seed": SEED,
                    "fold_id": fold["fold_id"], "config_hash": content_id(config),
                    "training_participants": sorted(fold["train"]),
                    "validation_participants": sorted(fold["validation"]),
                    "checkpoint_sha256": file_sha256(path),
                    "training_epochs": len(y), "valid_epochs_by_training_record": counts,
                    "rounds_requested": 400, "trees": model.booster_.num_trees()}, immutable=True)
                del X, y, frames, labels, model
            fit = _fit_receipt(path, config, fold)
            model = joblib.load(path)
            if set(map(int, model.classes_)) != set(range(5)):
                raise ValueError("EOG classifier lacks the fixed five classes")
            indices = [list(model.classes_).index(k) for k in range(5)]
            for rec in records:
                if rec["participant_id"] not in fold["validation"]:
                    continue
                _resources(root, started)
                frame, _ = _feature_load(root, rec, content_id(manifest["config"]))
                if list(frame.columns) != list(model.feature_name_):
                    raise ValueError("EOG-only feature schema differs from clean classifier")
                probability = np.asarray(model.predict_proba(frame), dtype=np.float64)[:, indices]
                if (probability.shape != (rec["n_epochs"], 5) or
                        not np.isfinite(probability).all() or np.any(probability < 0) or
                        np.any(probability > 1) or
                        not np.allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-6)):
                    raise ValueError("EOG-only prediction omitted epochs or changed class simplex")
                target = _paths(root, path.parent / "predictions" / (rec["recording_id"] + ".npz"),
                                "runs/provisional-eog")
                sidecar = _paths(root, target.with_suffix(".json"), "runs/provisional-eog")
                if target.exists() != sidecar.exists():
                    orphan = target if target.exists() else sidecar
                    archived = _paths(root, orphan.with_name("unattested-" + file_sha256(orphan) +
                                      "-" + uuid4().hex + orphan.suffix), "runs/provisional-eog")
                    orphan.rename(archived)
                provenance = {"config_hash": content_id(config),
                              "checkpoint_sha256": fit["checkpoint_sha256"],
                              "fold_id": fold["fold_id"],
                              "training_participants": sorted(fold["train"]),
                              "source_psg_sha256": rec["psg_sha256"],
                              "feature_manifest_sha256": config["feature_manifest_sha256"],
                              "channels": [CHANNEL], "hypnogram_required": False,
                              "provisional": True}
                if not target.exists():
                    save_prediction(target, participant_id=rec["participant_id"],
                        recording_id=rec["recording_id"],
                        epoch_index=np.arange(rec["n_epochs"]),
                        onset_seconds=np.arange(rec["n_epochs"]) * 30.,
                        hard_label=probability.argmax(axis=1), probabilities=probability,
                        model_id=MODEL_ID, protocol_hash=protocol["protocol_hash"],
                        registry_hash=protocol["registry_hash"], provenance=provenance)
                meta, saved = load_prediction(target)
                if (meta["provenance"] != provenance or meta["model_id"] != MODEL_ID or
                        saved["participant_id"] != rec["participant_id"] or
                        saved["recording_id"] != rec["recording_id"] or
                        not np.array_equal(saved["probabilities"], probability) or
                        not np.array_equal(saved["hard_label"], probability.argmax(axis=1))):
                    raise ValueError("Resumed EOG prediction differs from checkpoint or original grid")
                predictions.append(target)
                truth_paths.append(Path(rec["truth_path"]))
            _resources(root, started)
        if len(predictions) != 119 or len(set(predictions)) != 119:
            raise ValueError("EOG OOF run lacks unique complete D119 prediction coverage")
        if (_source_identity(root) != config["source_sha256"] or _runtime(root) != config["runtime"] or
                file_sha256(manifest_path) != config["feature_manifest_sha256"] or
                file_sha256(root / "research/development-truth-v1.json") != config["truth_manifest_sha256"]):
            raise ValueError("EOG source/runtime/features/truth changed during fit")
        _verify_manifest(root, manifest, protocol, split, records, manifest["config"])
        _verify_frozen_truth(records)
        metrics = evaluate_saved_records(truth_paths, predictions,
                protocol_hash=protocol["protocol_hash"], registry_hash=protocol["registry_hash"],
                model_id=MODEL_ID)
        _verify_frozen_truth(records)
        _verify_manifest(root, manifest, protocol, split, records, manifest["config"])
        for fold in split["folds"]:
            _fit_receipt(_model_path(root, run, fold), config, fold)
        if (_source_identity(root) != config["source_sha256"] or _runtime(root) != config["runtime"] or
                file_sha256(manifest_path) != config["feature_manifest_sha256"] or
                file_sha256(root / "research/development-truth-v1.json") != config["truth_manifest_sha256"]):
            raise ValueError("EOG fit or evaluator inputs changed before result publication")
        result = {"status": "DEVELOPMENT_OOF_COMPLETE", "confirmatory": False,
                  "run_id": run.name, "config_hash": content_id(config),
                  "protocol_hash": protocol["protocol_hash"], "split_id": split["split_id"],
                  "metrics": metrics, "prediction_paths": [str(p.resolve()) for p in predictions],
                  "truth_paths": [str(p.resolve()) for p in truth_paths],
                  "channel_names": [CHANNEL], "gate_A": "NOT_RUN", "gate_B": "NOT_RUN"}
        _resources(root, started)
        atomic_json(_paths(root, run / "result.json", "runs/provisional-eog"), result, immutable=True)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "prepare", "train_or_load"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=Path("sleep-edf-database-expanded-1.0.0"))
    args = parser.parse_args()
    result = (probe(args.root, args.data_root) if args.action == "probe" else
              prepare(args.root, args.data_root) if args.action == "prepare" else train_or_load(args.root))
    print(result["status"])


if __name__ == "__main__":
    main()
