"""Pinned native YASA features/prediction with clean participant-disjoint fitting.

No released classifier is downloaded or implicitly selected. The public predict
interface accepts a PSG, calibrated channel mapping and a local trained model.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
import os
import shutil
import sys
import time

import numpy as np

from .contracts import content_id, read_json
from .evaluation import evaluate_saved_records
from .predictions import save_prediction, load_prediction
from .protocol import development_records, load_development_truth
from .research import atomic_json, append_event, compute_lease, file_sha256, utc_now

SOURCE_SHA = "e581bf452097e01b493b3072964e206ae9b01dc4"
RECIPE_SHA = "42d8300a3f1f5b57333c74a0c815710bafbe00c2"
CHANNELS = {"eeg": ("EEG Fpz-Cz",), "eeg_eog": ("EEG Fpz-Cz", "EOG horizontal")}
LABEL_NAMES = np.asarray(["W", "N1", "N2", "N3", "R"], dtype=object)


def native_hard_labels(labels) -> np.ndarray:
    mapping = {"W": 0, "WAKE": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "REM": 4}
    try:
        return np.asarray([mapping[str(label)] for label in labels], dtype=np.int8)
    except KeyError as exc:
        raise ValueError("Native classifier emitted an unsupported stage") from exc


@lru_cache(maxsize=2)
def _verify_source(root: Path):
    manifest = read_json(root / "research" / "sources" / "yasa.json")
    if manifest["source_sha"] != SOURCE_SHA or manifest["weights_downloaded"] is not False:
        raise ValueError("YASA source provenance differs from approved source-only acquisition")
    for name, record in manifest["files"].items():
        if file_sha256(root / "vendor" / "yasa" / name) != record["sha256"]:
            raise ValueError("Pinned native source content changed")


def _feature_config(protocol: dict, variant: str) -> dict:
    return {"source_sha": SOURCE_SHA, "channels": list(CHANNELS[variant]),
            "normalization": "native unsupervised recording-local robust feature scaling; offline whole-record inference",
            "metadata_inputs": [], "native_rate_hz": 100, "protocol_hash": protocol["protocol_hash"]}


def _native(root: Path):
    _verify_source(root)
    source = root / "vendor" / "yasa" / "src"
    if not source.is_dir():
        raise ValueError("Fetch the pinned source-only YASA distribution first")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import yasa
    if not Path(yasa.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError("Unexpected YASA installation would change the pinned implementation")
    return yasa


def prepare(root: Path, psg_path: Path, channel_names: tuple[str, ...], *, n_epochs: int):
    """Build native SleepStaging from signal-only EDF reads in physical volts."""
    import mne
    import pyedflib
    if channel_names not in CHANNELS.values() or n_epochs <= 0:
        raise ValueError("Unsupported declared channel configuration or interval")
    with pyedflib.EdfReader(str(psg_path), annotations_mode=pyedflib.DO_NOT_READ_ANNOTATIONS) as reader:
        labels = reader.getSignalLabels()
        if len(set(labels)) != len(labels) or not set(channel_names) <= set(labels):
            raise ValueError("Ambiguous or missing named physiological channel")
        data = []
        for name in channel_names:
            index = labels.index(name)
            if reader.getPhysicalDimension(index).strip() not in ("uV", "µV", "μV"):
                raise ValueError("Unsupported physical unit")
            if reader.getSampleFrequency(index) != 100:
                raise ValueError("Native YASA adapter currently requires verified 100 Hz EEG/EOG")
            samples = reader.readSignal(index, start=0, n=n_epochs * 3000)
            if samples.size != n_epochs * 3000 or not np.isfinite(samples).all():
                raise ValueError("Incomplete or non-finite physiological samples")
            data.append(samples * 1e-6)
    info = mne.create_info(list(channel_names), 100, ["eeg"] + (["eog"] if len(channel_names) == 2 else []))
    raw = mne.io.RawArray(np.asarray(data), info, verbose="ERROR")
    return _native(root).SleepStaging(raw, eeg_name=channel_names[0],
                eog_name=channel_names[1] if len(channel_names) == 2 else None)


def predict(root: Path, psg_path: Path, channel_names: tuple[str, ...], *,
            n_epochs: int, model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    native = prepare(root, psg_path, channel_names, n_epochs=n_epochs)
    hypno = native.predict(path_to_model=str(model_path))
    probabilities = hypno.proba[["WAKE", "N1", "N2", "N3", "REM"]].to_numpy(dtype=np.float64)
    if probabilities.shape != (n_epochs, 5):
        raise ValueError("Native YASA prediction does not cover the complete interval")
    return native_hard_labels(hypno.hypno), probabilities


def _safe_psg(data_root: Path, record: dict) -> Path:
    path = (data_root / record["psg"]).resolve()
    if not path.is_relative_to(data_root.resolve()) or file_sha256(path) != record["psg_sha256"]:
        raise ValueError("PSG identity differs from frozen readiness")
    return path


def prepare_features(root: Path, data_root: Path, variant: str) -> dict:
    import psutil
    protocol, split, records = development_records(root)
    directory = root / "derived" / "yasa_features" / variant
    directory.mkdir(parents=True, exist_ok=True)
    config = _feature_config(protocol, variant)
    config_hash = content_id(config)
    for i, rec in enumerate(records):
        path = directory / (rec["recording_id"] + ".npz")
        sidecar = path.with_suffix(".json")
        expected = {"config_hash": config_hash, "source_psg_sha256": rec["psg_sha256"],
                    "participant_id": rec["participant_id"], "recording_id": rec["recording_id"],
                    "n_epochs": rec["n_epochs"]}
        if path.exists() and sidecar.exists():
            meta = read_json(sidecar)
            if any(meta.get(k) != v for k, v in expected.items()) or meta.get("payload_sha256") != file_sha256(path):
                raise ValueError("Cached features differ from pinned data/configuration")
            continue
        started = time.perf_counter()
        native = prepare(root, _safe_psg(data_root, rec), CHANNELS[variant], n_epochs=rec["n_epochs"])
        features = native.get_features()
        values = features.to_numpy(dtype=np.float32)
        if len(values) != rec["n_epochs"] or np.isinf(values).any():
            raise ValueError("Native feature extraction returned wrong epoch count or infinity")
        temporary = path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, features=values, columns=np.asarray(features.columns, dtype="U"))
        os.replace(temporary, path)
        meta = dict(expected, payload_sha256=file_sha256(path), feature_count=values.shape[1],
                    native_nan_values=int(np.isnan(values).sum()), nan_policy="native LightGBM missing-value support",
                    elapsed_seconds=time.perf_counter()-started,
                    process_rss_bytes=psutil.Process().memory_info().rss)
        atomic_json(sidecar, meta)
        if meta["process_rss_bytes"] > 10 * 1024**3 or psutil.virtual_memory().available < 4 * 1024**3:
            raise RuntimeError("Feature checkpoint saved; memory safeguard pauses the run")
        append_event(root / "runs" / "yasa-features.jsonl", dict(event="record_features_complete", variant=variant, **meta))
        print(f"features {variant} {i+1}/{len(records)} {rec['recording_id']}", flush=True)
    artifacts = {r["recording_id"]: {
        "payload_sha256": file_sha256(directory / (r["recording_id"] + ".npz")),
        "sidecar_sha256": file_sha256(directory / (r["recording_id"] + ".json"))} for r in records}
    report = {"status": "DEVELOPMENT_FEATURES_COMPLETE", "config": config, "config_hash": config_hash,
              "recordings": len(records), "participants": len(split["participants"]["development"]),
              "artifacts": artifacts}
    atomic_json(directory / "manifest.json", report, immutable=True)
    return report


def _features(root: Path, variant: str, record: dict, feature_manifest: dict):
    import pandas as pd
    path = root / "derived" / "yasa_features" / variant / (record["recording_id"] + ".npz")
    meta = read_json(path.with_suffix(".json"))
    frozen = feature_manifest["artifacts"][record["recording_id"]]
    if (meta["payload_sha256"] != file_sha256(path) or meta["source_psg_sha256"] != record["psg_sha256"] or
        meta["payload_sha256"] != frozen["payload_sha256"] or
        file_sha256(path.with_suffix(".json")) != frozen["sidecar_sha256"] or
        meta["config_hash"] != feature_manifest["config_hash"] or
        any(meta[k] != record[k] for k in ("participant_id", "recording_id", "n_epochs"))):
        raise ValueError("Feature payload or source changed")
    with np.load(path, allow_pickle=False) as cache:
        values, columns = cache["features"], cache["columns"]
        if (set(cache.files) != {"features", "columns"} or values.dtype != np.float32 or
            values.shape != (record["n_epochs"], meta["feature_count"]) or
            columns.ndim != 1 or len(columns) != values.shape[1] or len(set(columns)) != len(columns) or
            np.isinf(values).any() or int(np.isnan(values).sum()) != meta["native_nan_values"]):
            raise ValueError("Feature values or columns violate the frozen native schema")
        return pd.DataFrame(values, columns=columns)


def _verify_feature_cache(root: Path, variant: str, records: list[dict], manifest: dict) -> None:
    """Check all frozen feature bytes before fitting, without decoding or regenerating them."""
    directory = root / "derived" / "yasa_features" / variant
    for rec in records:
        path = directory / (rec["recording_id"] + ".npz")
        sidecar = path.with_suffix(".json")
        frozen = manifest["artifacts"][rec["recording_id"]]
        meta = read_json(sidecar)
        if (file_sha256(path) != frozen["payload_sha256"] or
            file_sha256(sidecar) != frozen["sidecar_sha256"] or
            meta.get("payload_sha256") != frozen["payload_sha256"] or
            meta.get("config_hash") != manifest["config_hash"] or
            meta.get("source_psg_sha256") != rec["psg_sha256"] or
            any(meta.get(k) != rec[k] for k in ("participant_id", "recording_id", "n_epochs")) or
            type(meta.get("feature_count")) is not int or meta["feature_count"] <= 0 or
            type(meta.get("native_nan_values")) is not int or meta["native_nan_values"] < 0):
            raise ValueError("Frozen YASA feature cache differs from source, grid or manifest")


def _selection_evidence(root: Path, path: Path, protocol: dict, split: dict,
                        records: list[dict], variant: str, seed: int, lr_multiplier: float) -> dict:
    path = (root / path).resolve()
    if not path.is_relative_to((root / "research").resolve()) or not path.is_file():
        raise ValueError("Selection evidence must be an existing frozen research artifact")
    selection_sha = file_sha256(path)
    selection = read_json(path)
    if file_sha256(path) != selection_sha:
        raise ValueError("Selection evidence changed while being read")
    expected_keys = {"schema_version", "artifact_type", "protocol_hash", "split_id", "chosen",
                     "selection_participants", "reviewed_results"}
    chosen = selection.get("chosen") if type(selection) is dict else None
    if (type(selection) is not dict or set(selection) != expected_keys or
        selection["schema_version"] != "1.0" or
        selection["artifact_type"] != "yasa_development_selection_v1" or
        selection["protocol_hash"] != protocol["protocol_hash"] or
        selection["split_id"] != split["split_id"] or
        type(chosen) is not dict or type(chosen.get("seed")) is not int or
        type(chosen.get("lr_multiplier")) not in (int, float) or
        chosen != {"variant": variant, "seed": seed, "lr_multiplier": lr_multiplier}):
        raise ValueError("Selection decision differs from this YASA development run")
    participants = selection["selection_participants"]
    development = set(split["participants"]["development"])
    if (type(participants) is not list or not participants or
        any(type(p) is not str for p in participants) or
        len(participants) != len(set(participants)) or not set(participants) <= development):
        raise ValueError("Selection participants must be unique members of frozen development")
    reviewed = selection["reviewed_results"]
    if type(reviewed) is not list or not reviewed:
        raise ValueError("Selection decision needs its reviewed development results")
    truth_participant = {Path(rec["truth_path"]).resolve(): rec["participant_id"] for rec in records}
    viewed, paths = set(), set()
    for item in reviewed:
        if type(item) is not dict or set(item) != {"path", "sha256"} or type(item["path"]) is not str:
            raise ValueError("Malformed reviewed YASA development result")
        result_path = (root / item["path"]).resolve()
        if (not result_path.is_relative_to((root / "runs").resolve()) or
            result_path.name != "result.json" or result_path in paths or
            not result_path.is_file() or type(item["sha256"]) is not str or
            file_sha256(result_path) != item["sha256"]):
            raise ValueError("Reviewed development result path or bytes changed")
        paths.add(result_path)
        result = read_json(result_path)
        if file_sha256(result_path) != item["sha256"]:
            raise ValueError("Reviewed development result changed while being read")
        truths = result.get("truth_paths")
        if (result.get("status") != "DEVELOPMENT_OOF_COMPLETE" or
            result.get("confirmatory") is not False or
            result.get("protocol_hash") != protocol["protocol_hash"] or
            result.get("gate_A") != "NOT_RUN" or result.get("gate_B") != "NOT_RUN" or
            type(truths) is not list or not truths or
            any(type(p) is not str for p in truths)):
            raise ValueError("Reviewed result is not a frozen D-only OOF result")
        resolved = [Path(p).resolve() for p in truths]
        if len(resolved) != len(set(resolved)) or not set(resolved) <= set(truth_participant):
            raise ValueError("Reviewed result includes non-development truth")
        viewed.update(truth_participant[p] for p in resolved)
    if set(participants) != viewed:
        raise ValueError("Declared selection participants differ from reviewed results")
    return {"path": path.relative_to(root.resolve()).as_posix(), "sha256": selection_sha,
            "content": selection}


def _fold_roles(fold: dict, selection: dict, split: dict, feature_manifest: dict) -> dict:
    return {"schema_version": "1.0", "fitted_participants": fold["train"],
            "selection_participants": selection["content"]["selection_participants"],
            "pretraining_participants": [], "scaler_participants": [],
            "calibration_participants": [], "teacher_participants": [], "pseudolabel_participants": [],
            "recording_local_normalization": {
                "scope": "unlabeled, per recording, complete night, native robust feature scaling",
                "participants": split["participants"]["development"],
                "recordings": feature_manifest["recordings"],
                "feature_config_hash": feature_manifest["config_hash"]},
            "role_basis": {"pretraining": "fresh local LGBMClassifier; no pretrained model is loaded for fit",
                           "selection": "frozen D-only reviewed result decision, separate from each fold's fixed 400 boosting rounds",
                           "scaler": "native feature normalization is recording-local, not a fitted cross-record scaler",
                           "calibration": "raw local LightGBM predict_proba; no posthoc calibrator",
                           "teacher": "fit uses frozen development reference labels directly",
                           "pseudolabel": "fit uses frozen development reference labels directly"},
            "selection_evidence_sha256": selection["sha256"]}


def _check_fold_roles(fit: dict, roles: dict) -> None:
    if fit.get("roles") != roles or fit.get("roles_id") != content_id(roles):
        raise ValueError("Resumed YASA learned-role ancestry changed")


def train_or_load(root: Path, data_root: Path, variant: str, seed: int, *,
                  lr_multiplier: float = 1.0, selection_path: Path) -> dict:
    import joblib
    import pandas as pd
    import lightgbm as lgb
    import psutil
    protocol, split, records = development_records(root)
    manifest_path = root / "derived" / "yasa_features" / variant / "manifest.json"
    feature_manifest = read_json(manifest_path)
    expected_config = _feature_config(protocol, variant)
    if (feature_manifest["config"] != expected_config or feature_manifest["config_hash"] != content_id(expected_config) or
        set(feature_manifest["artifacts"]) != {r["recording_id"] for r in records} or
        feature_manifest["recordings"] != len(records) or
        feature_manifest["participants"] != len(split["participants"]["development"])):
        raise ValueError("Features came from another frozen protocol")
    _verify_feature_cache(root, variant, records, feature_manifest)
    _verify_source(root)
    if seed not in (17, 43, 101) or lr_multiplier not in (0.3, 1, 3):
        raise ValueError("Use the registered seeds and tuning menu")
    selection = _selection_evidence(root, selection_path, protocol, split, records,
                                    variant, seed, lr_multiplier)
    params = dict(boosting_type="gbdt", n_estimators=400, max_depth=5, num_leaves=90,
                  colsample_bytree=0.5, importance_type="gain", learning_rate=0.1*lr_multiplier,
                  class_weight={"N1": 2.2, "N2": 1, "N3": 1.2, "R": 1.4, "W": 1},
                  n_jobs=4, random_state=seed, deterministic=True, force_col_wise=True, verbosity=-1)
    import importlib.metadata
    module_names = ("yasa_baseline.py", "protocol.py", "research.py", "splits.py",
                    "contracts.py", "predictions.py", "evaluation.py")
    implementation = {name: file_sha256(root / "sleepedf" / name) for name in module_names}
    config = {"variant": variant, "seed": seed, "params": params, "source_sha": SOURCE_SHA,
              "recipe_sha": RECIPE_SHA, "split_id": split["split_id"],
              "feature_config_hash": feature_manifest["config_hash"],
              "feature_manifest_sha256": file_sha256(manifest_path),
              "selection_evidence": selection,
              "truth_manifest_sha256": file_sha256(root / "research" / "development-truth-v1.json"),
              "runtime_lock_sha256": file_sha256(root / "requirements" / "research.hashed.txt"),
              "dependency_sources": implementation,
              "runtime": {"python": sys.version, "executable": sys.executable,
                          "packages": dict(sorted((dist.metadata["Name"].lower(), dist.version)
                                                 for dist in importlib.metadata.distributions()))},
              "implementation_sha": file_sha256(Path(__file__))}
    config_hash = content_id(config)
    run_id = f"yasa-{variant}-s{seed}-lr{lr_multiplier:g}-{config_hash[-10:]}"
    out = root / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "config.json", config, immutable=True)
    snapshot = out / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name, expected in implementation.items():
        target = snapshot / name
        if target.exists() and file_sha256(target) != expected:
            raise ValueError("Frozen training implementation changed")
        if not target.exists():
            shutil.copyfile(root / "sleepedf" / name, target)
    atomic_json(snapshot / "manifest.json", {"capture_scope": "before fitting any fold",
                                            "files": implementation}, immutable=True)
    all_predictions, all_truth = [], []
    for fold in split["folds"]:
        fold_dir = out / f"fold-{fold['fold_id']}"
        fold_dir.mkdir(exist_ok=True)
        model_path = fold_dir / "model.joblib"
        train_records = [r for r in records if r["participant_id"] in fold["train"]]
        val_records = [r for r in records if r["participant_id"] in fold["validation"]]
        roles = _fold_roles(fold, selection, split, feature_manifest)
        start = time.perf_counter()
        if model_path.exists() and not (fold_dir / "fit.json").exists():
            orphan = model_path.with_name(f"unattested-{file_sha256(model_path)}.joblib")
            os.replace(model_path, orphan)
        if not model_path.exists():
            xs, ys = [], []
            for rec in train_records:
                truth = load_development_truth(rec, split)
                frame = _features(root, variant, rec, feature_manifest)
                if len(frame) != len(truth["epoch_index"]):
                    raise ValueError("Feature/truth grid mismatch")
                xs.append(frame.loc[truth["valid_mask"]])
                ys.append(LABEL_NAMES[truth["reference_label"][truth["valid_mask"]]])
            X = pd.concat(xs, ignore_index=True)
            y = np.concatenate(ys)
            if set(y) != set(LABEL_NAMES):
                raise ValueError("Training fold lacks at least one class")
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X, y)
            temporary = model_path.with_suffix(".partial.joblib")
            joblib.dump(clf, temporary, compress=3)
            os.replace(temporary, model_path)
            atomic_json(fold_dir / "fit.json", {"status": "FIT_COMPLETE", "fitted_participants": fold["train"],
                        "excluded_validation_participants": fold["validation"], "protocol_hash": protocol["protocol_hash"],
                        "roles": roles, "roles_id": content_id(roles),
                        "config_hash": config_hash, "training_epochs": len(y), "trees": clf.booster_.num_trees(),
                        "checkpoint_sha": file_sha256(model_path), "elapsed_seconds": time.perf_counter()-start,
                        "rss_bytes": psutil.Process().memory_info().rss,
                        "resume_granularity": "completed fold; interrupted tree fit restarts the same deterministic fold"})
            del X, y, xs, ys, clf
        fit = read_json(fold_dir / "fit.json")
        _check_fold_roles(fit, roles)
        if (fit["config_hash"] != config_hash or fit["checkpoint_sha"] != file_sha256(model_path) or
            fit["fitted_participants"] != fold["train"] or
            fit.get("excluded_validation_participants") != fold["validation"] or
            fit.get("protocol_hash") != protocol["protocol_hash"]):
            raise ValueError("Resumed checkpoint or training ancestry changed")
        model_id = f"{run_id}/fold-{fold['fold_id']}"
        native_parity = []
        clf = joblib.load(model_path)
        indices = [list(clf.classes_).index(label) for label in LABEL_NAMES]
        for j, rec in enumerate(val_records):
            pred_path = fold_dir / "predictions" / (rec["recording_id"] + ".npz")
            if not pred_path.exists():
                frame = _features(root, variant, rec, feature_manifest)
                probabilities = clf.predict_proba(frame[clf.feature_name_])[:, indices]
                labels = native_hard_labels(clf.predict(frame[clf.feature_name_]))
                # Execute the full pinned native signal-to-prediction route on each fold's first night.
                if j == 0:
                    native_labels, native_prob = predict(root, _safe_psg(data_root, rec), CHANNELS[variant],
                                                         n_epochs=rec["n_epochs"], model_path=model_path)
                    error = float(np.max(np.abs(probabilities-native_prob)))
                    if not np.array_equal(labels, native_labels) or error > 1e-12:
                        raise ValueError("Cached native feature route fails native end-to-end parity")
                    native_parity.append({"recording_id": rec["recording_id"], "maximum_probability_error": error})
                save_prediction(pred_path, participant_id=rec["participant_id"], recording_id=rec["recording_id"],
                    epoch_index=np.arange(rec["n_epochs"], dtype=np.int64),
                    onset_seconds=np.arange(rec["n_epochs"], dtype=np.float64)*30,
                    hard_label=labels, probabilities=probabilities, model_id=model_id,
                    protocol_hash=protocol["protocol_hash"], registry_hash=protocol["registry_hash"],
                    provenance={"source_sha": SOURCE_SHA, "config_hash": config_hash,
                                "checkpoint_sha": fit["checkpoint_sha"], "channels": list(CHANNELS[variant]),
                                "implementation_sha": config["implementation_sha"],
                                "roles_id": fit["roles_id"], "selection_evidence_sha256": selection["sha256"],
                                "inference_route": "native YASA features and local native LightGBM classifier",
                                "hypnogram_required": False, "fitted_participants": fold["train"]})
            meta, _ = load_prediction(pred_path)
            if (meta["model_id"] != model_id or meta["provenance"]["checkpoint_sha"] != fit["checkpoint_sha"] or
                meta["provenance"].get("config_hash") != config_hash or
                meta["provenance"].get("implementation_sha") != config["implementation_sha"] or
                meta["provenance"].get("roles_id") != fit["roles_id"] or
                meta["provenance"].get("selection_evidence_sha256") != selection["sha256"]):
                raise ValueError("Resumed prediction belongs to another checkpoint")
            if j == 0 and not native_parity and not (fold_dir / "native-parity.json").exists():
                _, cached = load_prediction(pred_path)
                native_labels, native_prob = predict(root, _safe_psg(data_root, rec), CHANNELS[variant],
                                                     n_epochs=rec["n_epochs"], model_path=model_path)
                error = float(np.max(np.abs(cached["probabilities"]-native_prob)))
                if not np.array_equal(cached["hard_label"], native_labels) or error > 1e-12:
                    raise ValueError("Resumed predictions fail full native inference parity")
                native_parity.append({"recording_id": rec["recording_id"], "maximum_probability_error": error})
            all_predictions.append(pred_path)
            all_truth.append(Path(rec["truth_path"]))
        if native_parity:
            atomic_json(fold_dir / "native-parity.json", {"checks": native_parity})
        append_event(out / "events.jsonl", {"event": "fold_complete", "fold": fold["fold_id"],
                    "elapsed_seconds": time.perf_counter()-start, "checkpoint_sha": fit["checkpoint_sha"]})
        print(f"trained {run_id} fold {fold['fold_id']}", flush=True)
    for rec in records:
        load_development_truth(rec, split)
    metrics = evaluate_saved_records(all_truth, all_predictions, protocol_hash=protocol["protocol_hash"],
                                      registry_hash=protocol["registry_hash"])
    report = {"status": "DEVELOPMENT_OOF_COMPLETE", "confirmatory": False, "run_id": run_id,
              "config_hash": config_hash, "protocol_hash": protocol["protocol_hash"],
              "selection_evidence_sha256": selection["sha256"],
              "fold_roles": {str(f["fold_id"]): content_id(_fold_roles(f, selection, split, feature_manifest))
                             for f in split["folds"]},
              "metrics": metrics, "prediction_paths": [str(p.resolve()) for p in all_predictions],
              "truth_paths": [str(p.resolve()) for p in all_truth], "gate_A": "NOT_RUN", "gate_B": "NOT_RUN"}
    atomic_json(out / "result.json", report)
    append_event(root / "runs" / "experiments.jsonl", {"event": "development_run_complete", "run_id": run_id,
                "report_sha256": file_sha256(out / "result.json"), "macro_f1": metrics["macro_f1"],
                "confirmatory": False})
    return report


def export_provenance(root: Path) -> dict:
    return {"source_sha": SOURCE_SHA, "code_license": "BSD-3-Clause", "weights": "clean local training only",
            "recipe_source": f"https://github.com/raphaelvallat/yasa_classifier/blob/{RECIPE_SHA}/02_train_export_classifier.ipynb",
            "implementation_sha": file_sha256(Path(__file__)), "released_weights_loaded": False}


def validate_output(path: Path):
    return load_prediction(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train_or_load", "predict", "export_provenance", "validate_output"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=Path("sleep-edf-database-expanded-1.0.0"))
    parser.add_argument("--variant", choices=CHANNELS, default="eeg_eog")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr-multiplier", type=float, default=1)
    parser.add_argument("--selection-evidence", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--psg", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--recording-id")
    parser.add_argument("--participant-id")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("NUMBA_NUM_THREADS", "4")
    if args.action == "export_provenance":
        from .contracts import json_text
        print(json_text(export_provenance(args.root)))
        return
    if args.action == "validate_output":
        validate_output(args.prediction)
        return
    if args.action == "predict":
        if any(v is None for v in (args.psg, args.model, args.n_epochs, args.recording_id, args.participant_id, args.prediction)):
            parser.error("predict requires --psg --model --n-epochs --recording-id --participant-id --prediction")
        from .protocol import load_protocol
        protocol, _, _ = load_protocol(args.root)
        fit = read_json(args.model.with_name("fit.json"))
        config = read_json(args.model.parent.parent / "config.json")
        if file_sha256(args.model) != fit["checkpoint_sha"] or content_id(config) != fit["config_hash"]:
            raise ValueError("Model/config artifact integrity failure")
        if config["variant"] != args.variant or fit["protocol_hash"] != protocol["protocol_hash"]:
            raise ValueError("Model channels/protocol differ from inference declaration")
        role_provenance = {}
        if "selection_evidence" in config:
            selection = config["selection_evidence"]
            selection_path = (args.root / selection["path"]).resolve()
            if (not selection_path.is_relative_to((args.root / "research").resolve()) or
                file_sha256(selection_path) != selection["sha256"] or
                read_json(selection_path) != selection["content"] or
                type(fit.get("roles")) is not dict or
                fit.get("roles_id") != content_id(fit["roles"]) or
                fit["roles"].get("selection_evidence_sha256") != selection["sha256"]):
                raise ValueError("YASA inference learned-role ancestry changed")
            role_provenance = {"roles_id": fit["roles_id"],
                               "selection_evidence_sha256": selection["sha256"]}
        with compute_lease(args.root, "yasa-signal-only-predict"):
            labels, proba = predict(args.root, args.psg, CHANNELS[args.variant], n_epochs=args.n_epochs, model_path=args.model)
            save_prediction(args.prediction, participant_id=args.participant_id, recording_id=args.recording_id,
                epoch_index=np.arange(args.n_epochs), onset_seconds=np.arange(args.n_epochs)*30.,
                hard_label=labels, probabilities=proba,
                model_id=f"{args.model.parent.parent.name}/{args.model.parent.name}",
                protocol_hash=protocol["protocol_hash"], registry_hash=protocol["registry_hash"],
                provenance={"source_sha": SOURCE_SHA, "checkpoint_sha": fit["checkpoint_sha"],
                            "config_hash": fit["config_hash"], "implementation_sha": config["implementation_sha"],
                            "channels": list(CHANNELS[args.variant]), "hypnogram_required": False,
                            **role_provenance})
        return
    if args.action == "train_or_load" and args.selection_evidence is None:
        parser.error("train_or_load requires --selection-evidence")
    with compute_lease(args.root, f"yasa-{args.action}-{args.variant}-{args.seed}"):
        if args.action == "prepare":
            result = prepare_features(args.root, args.data_root.resolve(), args.variant)
        else:
            result = train_or_load(args.root, args.data_root.resolve(), args.variant, args.seed,
                                   lr_multiplier=args.lr_multiplier, selection_path=args.selection_evidence)
        print(result["status"], flush=True)


if __name__ == "__main__":
    main()
