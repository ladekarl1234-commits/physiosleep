"""Clean pooled-channel YASA fitting for the packaged Sleepyland signal route."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time
import uuid

import numpy as np

from . import sleepyland_yasa_bridge as bridge
from . import sleepyland_yasa_classifier_route as route
from .contracts import content_id, read_json
from .evaluation import evaluate_saved_records
from .predictions import load_prediction, save_prediction
from .protocol import development_records, load_development_truth
from .research import atomic_json, append_event, compute_lease, file_sha256

LABELS = ("W", "N1", "N2", "N3", "R")
MODEL_ID = "sleepyland-yasa-clean-pool-both-v1"
SOURCES = ("sleepedf/sleepyland_yasa_experiment.py", "sleepedf/sleepyland_yasa_bridge.py",
           "sleepedf/sleepyland_yasa_classifier_route.py",
           "sleepedf/sleepyland_yasa_verify.py", "tools/verify_sleepyland_yasa_reader.py",
           "tools/sleepyland_yasa_reader.py", "sleepedf/protocol.py", "sleepedf/contracts.py",
           "sleepedf/research.py", "sleepedf/predictions.py", "sleepedf/evaluation.py",
           "sleepedf/splits.py", "sleepedf/dataset.py", "sleepedf/readers.py", "sleepedf/timing.py",
           "research/variants/sleepyland-yasa-clean-v1.json",
           "research/sleepyland-yasa-probability-contract-v2.json")


def resource_guard(started: float) -> int:
    import psutil
    rss = psutil.Process().memory_info().rss
    available = psutil.virtual_memory().available
    elapsed = time.monotonic() - started
    if rss > 10 * 1024**3 or available < 4 * 1024**3 or elapsed > 4 * 3600:
        raise RuntimeError(f"Packaged YASA fold resource pause: rss={rss}, available={available}, seconds={elapsed}")
    return rss


def require_same_schema(frame, expected: tuple | None) -> tuple:
    schema = (tuple(frame.columns), tuple(str(dtype) for dtype in frame.dtypes))
    if len(set(schema[0])) != len(schema[0]) or (expected is not None and schema != expected):
        raise ValueError("Packaged YASA group/training feature schemas differ before fitting")
    return schema


def ordered_probabilities(frame, n_epochs: int) -> np.ndarray:
    aliases = {"WAKE": "W", "REM": "R"}
    columns = [aliases.get(str(name), str(name)) for name in frame.columns]
    if len(columns) != 5 or set(columns) != set(LABELS):
        raise ValueError("Native YASA probability columns differ or aliases collide")
    values = frame.to_numpy(dtype=np.float64)[:, [columns.index(name) for name in LABELS]]
    if (values.shape != (n_epochs, 5) or not np.isfinite(values).all() or
            np.any(values < 0) or np.any(values > 1) or
            not np.allclose(values.sum(axis=1), 1., rtol=0, atol=1e-6)):
        raise ValueError("Native YASA probabilities omit epochs or violate the five-class simplex")
    return values


def group_mean(probabilities: dict[str, np.ndarray]) -> np.ndarray:
    if tuple(probabilities) != route.ORDER:
        raise ValueError("Packaged YASA needs both declared channel groups in native order")
    try:
        return route._native_mean(probabilities, len(probabilities[route.ORDER[0]]))[1]
    except ValueError as error:
        raise ValueError("Individual packaged YASA group probabilities are invalid") from error


def _sources(root: Path) -> dict:
    return {name: file_sha256(root / name) for name in SOURCES}


def _local_output(root: Path, path: Path) -> Path:
    runs = root / "runs"
    if runs.resolve() != runs or path.resolve() != path or not path.is_relative_to(runs):
        raise ValueError("Packaged YASA output path redirects outside canonical runs")
    return path


def _snapshot(root: Path, run: Path, sources: dict) -> None:
    for name, digest in sources.items():
        target = _local_output(root, run / "source_snapshot" / name)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        if file_sha256(target) != digest or file_sha256(root / name) != digest:
            raise ValueError("Packaged YASA frozen source snapshot changed")


def _model_path(run: Path, fold: dict) -> Path:
    return run / f"fold-{fold['fold_id']}" / "model.joblib"


def validate_fit(model_path: Path, config: dict, fold: dict) -> dict:
    expected_run = f"sleepyland-yasa-pool-s{config['seed']}-{content_id(config)[-10:]}"
    if (model_path.name != "model.joblib" or model_path.parent.name != f"fold-{fold['fold_id']}" or
            model_path.parent.parent.name != expected_run or
            read_json(model_path.parent.parent / "config.json") != config):
        raise ValueError("Packaged YASA classifier path is not an explicit fold model")
    fit = read_json(model_path.with_name("fit.json"))
    if (fit.get("status") != "FIT_COMPLETE" or fit.get("config_hash") != content_id(config) or
            fit.get("fold_id") != fold["fold_id"] or fit.get("checkpoint_sha256") != file_sha256(model_path) or
            fit.get("training_participants") != sorted(fold["train"]) or
            fit.get("validation_participants") != sorted(fold["validation"]) or
            fit.get("seed") != config["seed"] or fit.get("rounds_requested") != 400 or
            fit.get("per_group_sample_weight") != 0.5 or
            len(fold["train"]) != 48 or len(set(fold["train"])) != 48 or
            len(fold["validation"]) != 12 or len(set(fold["validation"])) != 12 or
            set(fold["train"]) & set(fold["validation"])):
        raise ValueError("Packaged YASA classifier bytes or participant ancestry changed")
    return fit


def _cached_probabilities(root: Path, rec: dict, manifest: dict, model) -> dict:
    import pandas as pd
    if len(model.classes_) != 5 or set(model.classes_) != set(LABELS):
        raise ValueError("Clean YASA classifier lacks the fixed five classes")
    probabilities = {}
    for group in bridge.GROUPS:
        frame = bridge.load_features(root, rec, manifest, group)
        if list(frame.columns) != list(model.feature_name_):
            raise ValueError("Native YASA feature schema differs from the fitted classifier")
        probabilities[group] = ordered_probabilities(
            pd.DataFrame(model.predict_proba(frame), columns=model.classes_), rec["n_epochs"])
    return probabilities


def _parity_artifact(root: Path, data_root: Path, rec: dict, path: Path,
                     config: dict, fold: dict, fit: dict, manifest: dict,
                     cached_groups: dict) -> dict:
    """Retain and recheck both original-grid float32 routes before serialization."""
    array_path = _local_output(root, path.parent / "native-parity-float32.npz")
    meta_path = _local_output(root, path.parent / "native-parity.json")
    n_epochs = rec["n_epochs"]
    expected_groups, expected_majority = route._native_mean(cached_groups, n_epochs)
    identity = {"status": "PASS_EXACT_NATIVE_FLOAT32", "recording_id": rec["recording_id"],
                "fold_id": fold["fold_id"], "config_hash": content_id(config),
                "checkpoint_sha256": fit["checkpoint_sha256"],
                "source_psg_sha256": rec["psg_sha256"],
                "feature_manifest_sha256": config["feature_manifest_sha256"],
                "class_order": list(LABELS), "group_order": list(route.ORDER),
                "n_epochs": n_epochs, "array_path": str(array_path.resolve())}
    arrays = {"epoch_index": np.arange(n_epochs, dtype=np.int32)}
    for group in route.ORDER:
        arrays["cached_" + group] = expected_groups[group]
    arrays["cached_majority"] = expected_majority
    if array_path.exists() != meta_path.exists():
        orphan = array_path if array_path.exists() else meta_path
        if not orphan.is_file():
            raise ValueError("Interrupted parity artifact is not a regular file")
        archived = _local_output(root, orphan.with_name(
            "unattested-" + orphan.name + "-" + file_sha256(orphan) + "-" + uuid.uuid4().hex))
        orphan.rename(archived)
    if array_path.exists() or meta_path.exists():
        if not array_path.is_file() or not meta_path.is_file():
            raise ValueError("Native parity arrays or immutable sidecar are incomplete")
        parity = read_json(meta_path)
        if (set(parity) != set(identity) | {"array_sha256", "native_evidence",
                                             "maximum_probability_error", "group_probability_errors"} or
                any(parity.get(key) != value for key, value in identity.items()) or
                parity.get("array_sha256") != file_sha256(array_path) or
                parity.get("maximum_probability_error") != 0. or
                parity.get("group_probability_errors") != {group: 0. for group in route.ORDER}):
            raise ValueError("Resumed native parity identity or byte hash changed")
        evidence = parity["native_evidence"]
        if (not isinstance(evidence, dict) or evidence.get("recording_id") != rec["recording_id"] or
                evidence.get("n_epochs") != n_epochs or evidence.get("fold_id") != fold["fold_id"] or
                evidence.get("checkpoint_sha256") != fit["checkpoint_sha256"] or
                evidence.get("source_psg_sha256") != rec["psg_sha256"] or
                evidence.get("feature_config_id") != config["feature_config"]["config_id"] or
                evidence.get("classifier_route_sha256") != config["sources"]["sleepedf/sleepyland_yasa_classifier_route.py"] or
                evidence.get("probability_contract_sha256") != config["sources"]["research/sleepyland-yasa-probability-contract-v2.json"] or
                evidence.get("label_storage_compatibility") != "copied_object_dtype_classes_facade" or
                evidence.get("vendor_sha256") != route.PINNED or
                evidence.get("fitted_participants") != sorted(fold["train"]) or
                evidence.get("excluded_participants") != sorted(fold["validation"]) or
                evidence.get("group_order") != list(route.ORDER) or
                evidence.get("classifier_calls") != 2):
            raise ValueError("Resumed native classifier evidence is stale")
        with np.load(array_path, allow_pickle=False) as retained:
            if set(retained.files) != set(arrays) | {"native_" + group for group in route.ORDER} | {"native_majority"}:
                raise ValueError("Resumed native parity arrays omit a required route")
            for key, expected in arrays.items():
                actual = retained[key]
                if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
                    raise ValueError("Resumed cached route or original epoch grid changed")
            for group in route.ORDER:
                native = route._five(retained["native_" + group], n_epochs, dtype=np.float32)
                if native.tobytes() != expected_groups[group].tobytes():
                    raise ValueError("Resumed native group differs from cached float32 bytes")
            majority = route._five(retained["native_majority"], n_epochs, dtype=np.float32)
            if majority.tobytes() != expected_majority.tobytes():
                raise ValueError("Resumed native majority differs from cached float32 bytes")
        return parity
    compared = route.compare_record(root, data_root, rec, path, config, fold, manifest)
    for group in route.ORDER:
        if compared["cached_groups"][group].tobytes() != expected_groups[group].tobytes():
            raise ValueError("Independent cache route changed across classifier calls")
        native = route._five(compared["groups"][group], n_epochs, dtype=np.float32)
        if native.tobytes() != expected_groups[group].tobytes():
            raise ValueError("Native group differs from independent cache bytes")
        arrays["native_" + group] = native
    native_majority = route._five(compared["majority"], n_epochs, dtype=np.float32)
    if (compared["cached_majority"].tobytes() != expected_majority.tobytes() or
            native_majority.tobytes() != expected_majority.tobytes()):
        raise ValueError("Independent cache aggregation changed across classifier calls")
    arrays["native_majority"] = native_majority
    with array_path.open("xb") as output:
        np.savez_compressed(output, **arrays)
    parity = dict(identity, array_sha256=file_sha256(array_path),
                  native_evidence=compared["evidence"], maximum_probability_error=0.,
                  group_probability_errors={group: 0. for group in route.ORDER})
    atomic_json(meta_path, parity, immutable=True)
    return parity


def predict(root: Path, data_root: Path, record: dict, model_path: Path, config: dict,
            fold: dict) -> tuple[np.ndarray, dict]:
    """Signal-only full-grid native inference under the caller's verified fit lease."""
    protocol, split, records = development_records(root)
    if (fold not in split["folds"] or record not in records or
            config["protocol_hash"] != protocol["protocol_hash"] or config["split_id"] != split["split_id"] or
            model_path.resolve().parent.parent.parent != (root / "runs").resolve()):
        raise ValueError("Packaged YASA inference is not a canonical development fold and model")
    validate_fit(model_path, config, fold)
    if (record["participant_id"] not in fold["validation"] or
            record["participant_id"] in fold["train"] or _sources(root) != config["sources"]):
        raise ValueError("Packaged YASA inference participant or frozen implementation differs")
    prepared = bridge.prepare_record_locked(root, data_root, record)
    result = {}
    for group, item in prepared["groups"].items():
        hypno = item["native"].predict(path_to_model=str(model_path))
        result[group] = ordered_probabilities(hypno.proba, record["n_epochs"])
    return group_mean(result), {"groups": result, "signal": prepared["evidence"]}


def train_or_load(root: Path, data_root: Path, seed: int, lr_multiplier: float = 1.) -> dict:
    if seed not in (17, 43, 101) or lr_multiplier not in (.3, 1., 3.):
        raise ValueError("Use registered packaged YASA seeds and learning-rate menu")
    lease_id = content_id({"variant": MODEL_ID, "seed": seed, "lr_multiplier": lr_multiplier})[7:23]
    with compute_lease(root, f"sleepyland-yasa-fit-{lease_id}"):
        return _fit_locked(root.resolve(), data_root.resolve(), seed, lr_multiplier)


def _fit_locked(root: Path, data_root: Path, seed: int, lr_multiplier: float) -> dict:
    import joblib
    import lightgbm as lgb
    import pandas as pd
    import psutil
    from .sleepyland_yasa_verify import require_verification
    resource_started = time.monotonic()
    resource_guard(resource_started)
    verification = require_verification(root)
    protocol, split, records = development_records(root)
    records = sorted(records, key=lambda rec: rec["recording_id"])
    manifest_path = root / "derived/sleepyland_yasa/manifest.json"
    manifest = read_json(manifest_path)
    bridge.verify_manifest(root, manifest, protocol, records)
    resource_guard(resource_started)
    registration_path = root / "research/variants/sleepyland-yasa-clean-v1.json"
    registration = read_json(registration_path)
    if (registration["registration_id"] != content_id({k: v for k, v in registration.items() if k != "registration_id"}) or
            registration["protocol_hash"] != protocol["protocol_hash"] or
            registration["split_id"] != split["split_id"]):
        raise ValueError("Packaged YASA recipe registration differs from the frozen protocol")
    params = dict(n_estimators=400, max_depth=5, num_leaves=90, colsample_bytree=.5,
                  importance_type="gain", learning_rate=.1 * lr_multiplier,
                  class_weight={"N1": 2.2, "N2": 1, "N3": 1.2, "R": 1.4, "W": 1},
                  n_jobs=4, random_state=seed, deterministic=True, force_col_wise=True, verbosity=-1)
    config = {"variant": MODEL_ID, "seed": seed, "params": params, "sample_weight": .5,
              "bridge_verification": verification,
              "protocol_hash": protocol["protocol_hash"], "split_id": split["split_id"],
              "registration_id": registration["registration_id"], "sources": _sources(root),
              "feature_manifest_sha256": file_sha256(manifest_path), "feature_manifest_id": manifest["manifest_id"],
              "feature_config": manifest["config"],
              "truth_manifest_sha256": file_sha256(root / "research/development-truth-v1.json")}
    run_id = f"sleepyland-yasa-pool-s{seed}-{content_id(config)[-10:]}"
    run = root / "runs" / run_id
    _local_output(root, run / "config.json")
    atomic_json(run / "config.json", config, immutable=True)
    _snapshot(root, run, config["sources"])
    paths, truths = [], []
    for fold in split["folds"]:
        started = time.monotonic()
        peak_rss = resource_guard(started)
        if (_sources(root) != config["sources"] or file_sha256(manifest_path) != config["feature_manifest_sha256"] or
                require_verification(root) != verification):
            raise ValueError("Packaged YASA inputs changed before a fold")
        path = _model_path(run, fold)
        _local_output(root, path)
        fit_path = _local_output(root, path.with_name("fit.json"))
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not fit_path.exists():
            archived = _local_output(root, path.with_name(
                "unattested-" + file_sha256(path) + "-" + uuid.uuid4().hex + ".joblib"))
            path.rename(archived)
        if not path.exists():
            frames, labels = [], []
            counts = {}
            schema = None
            for rec in records:
                if rec["participant_id"] not in fold["train"]:
                    continue
                truth = load_development_truth(rec, split)
                valid = truth["valid_mask"]
                counts[rec["recording_id"]] = int(valid.sum())
                for group in bridge.GROUPS:
                    peak_rss = max(peak_rss, resource_guard(started))
                    frame = bridge.load_features(root, rec, manifest, group)
                    schema = require_same_schema(frame, schema)
                    if len(frame) != len(valid):
                        raise ValueError("Packaged YASA features differ from original truth grid")
                    frames.append(frame.loc[valid])
                    labels.append(np.asarray(LABELS)[truth["reference_label"][valid]])
                    peak_rss = max(peak_rss, resource_guard(started))
            X, y = pd.concat(frames, ignore_index=True), np.concatenate(labels)
            peak_rss = max(peak_rss, resource_guard(started))
            if set(y) != set(LABELS):
                raise ValueError("Packaged YASA training fold lacks a required class")
            model = lgb.LGBMClassifier(**params)
            def check_iteration(_):
                nonlocal peak_rss
                peak_rss = max(peak_rss, resource_guard(started))
            check_iteration.before_iteration = True
            check_iteration.order = 0
            model.fit(X, y, sample_weight=np.full(len(y), .5), callbacks=[check_iteration])
            peak_rss = max(peak_rss, resource_guard(started))
            temporary = _local_output(root, path.with_suffix(".partial.joblib"))
            joblib.dump(model, temporary, compress=3)
            temporary.replace(path)
            atomic_json(fit_path, {"status": "FIT_COMPLETE", "seed": seed,
                "fold_id": fold["fold_id"], "config_hash": content_id(config),
                "training_participants": sorted(fold["train"]), "validation_participants": sorted(fold["validation"]),
                "checkpoint_sha256": file_sha256(path), "per_group_sample_weight": .5,
                "valid_epochs_by_training_record": counts, "training_rows": len(y),
                "rounds_requested": 400, "trees": model.booster_.num_trees(),
                "peak_observed_rss_bytes": peak_rss, "fold_invocation_limit_seconds": 4 * 3600,
                "elapsed_seconds": time.monotonic() - started}, immutable=True)
            del frames, labels, X, y, model
        fit = validate_fit(path, config, fold)
        model = joblib.load(path)
        held = [rec for rec in records if rec["participant_id"] in fold["validation"]]
        parity = None
        for index, rec in enumerate(held):
            peak_rss = max(peak_rss, resource_guard(started))
            cached_groups = _cached_probabilities(root, rec, manifest, model)
            probability = group_mean(cached_groups)
            if index == 0:
                parity = _parity_artifact(root, data_root, rec, path, config, fold,
                                          fit, manifest, cached_groups)
            target = path.parent / "predictions" / (rec["recording_id"] + ".npz")
            _local_output(root, target)
            provenance = {"config_hash": content_id(config), "checkpoint_sha256": fit["checkpoint_sha256"],
                          "fold_id": fold["fold_id"], "seed": seed, "source_psg_sha256": rec["psg_sha256"],
                          "training_participants": sorted(fold["train"]),
                          "groups": {k: list(v) for k, v in bridge.GROUPS.items()},
                          "group_combination": "float32_native_copy_add_divide_2",
                          "native_parity_array_sha256": parity["array_sha256"],
                          "hypnogram_required": False}
            if not target.exists():
                save_prediction(target, participant_id=rec["participant_id"], recording_id=rec["recording_id"],
                    epoch_index=np.arange(rec["n_epochs"]), onset_seconds=np.arange(rec["n_epochs"]) * 30.,
                    hard_label=probability.argmax(1), probabilities=probability, model_id=MODEL_ID,
                    protocol_hash=protocol["protocol_hash"], registry_hash=protocol["registry_hash"], provenance=provenance)
            meta, saved = load_prediction(target)
            if (meta["provenance"] != provenance or meta["model_id"] != MODEL_ID or
                    saved["recording_id"] != rec["recording_id"] or saved["participant_id"] != rec["participant_id"] or
                    not np.array_equal(saved["probabilities"], probability) or
                    not np.array_equal(saved["hard_label"], probability.argmax(1))):
                raise ValueError("Packaged YASA resumed predictions differ from frozen clean model")
            load_development_truth(rec, split)
            paths.append(target); truths.append(Path(rec["truth_path"]))
            peak_rss = max(peak_rss, resource_guard(started))
        if parity is None:
            raise ValueError("Packaged YASA fold has no held parity recording")
        if file_sha256(Path(parity["array_path"])) != parity["array_sha256"]:
            raise ValueError("Native float32 parity arrays changed during fold publication")
        atomic_json(path.parent / "native-parity.json", parity, immutable=True)
        print(f"completed {run_id} fold {fold['fold_id']}", flush=True)
    if len(paths) != len(records) or len(set(paths)) != len(records) or _sources(root) != config["sources"]:
        raise ValueError("Packaged YASA complete OOF coverage or source identity changed")
    bridge.verify_manifest(root, manifest, protocol, records)
    metrics = evaluate_saved_records(truths, paths, protocol_hash=protocol["protocol_hash"],
                                     registry_hash=protocol["registry_hash"], model_id=MODEL_ID)
    result = {"status": "DEVELOPMENT_OOF_COMPLETE", "confirmatory": False, "run_id": run_id,
              "config_hash": content_id(config), "metrics": metrics,
              "prediction_paths": [str(path.resolve()) for path in paths],
              "truth_paths": [str(path.resolve()) for path in truths], "gate_A": "NOT_RUN", "gate_B": "NOT_RUN"}
    _local_output(root, run / "result.json")
    atomic_json(run / "result.json", result, immutable=True)
    events_path = _local_output(root, root / "runs/experiments.jsonl")
    append_event(events_path, {"event": "development_run_complete", "run_id": run_id,
                 "result_sha256": file_sha256(run / "result.json"), "confirmatory": False})
    return result


def prepare(root: Path, data_root: Path) -> dict:
    return bridge.prepare_features(root, data_root)


def export_provenance(root: Path) -> dict:
    return {"variant": MODEL_ID, "released_weights_loaded": False, "exact_duplicate": False,
            "registration": read_json(root / "research/variants/sleepyland-yasa-clean-v1.json"),
            "sources": _sources(root)}


def validate_output(path: Path):
    return load_prediction(path)


def main():
    from .contracts import json_text
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train_or_load", "export_provenance", "validate_output"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=Path("sleep-edf-database-expanded-1.0.0"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr-multiplier", type=float, default=1.)
    parser.add_argument("--prediction", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.root.resolve(), args.data_root.resolve())
    elif args.action == "train_or_load":
        result = train_or_load(args.root.resolve(), args.data_root.resolve(), args.seed, args.lr_multiplier)
    elif args.action == "export_provenance":
        result = export_provenance(args.root.resolve())
    else:
        result, _ = validate_output(args.prediction)
    print(json_text(result))


if __name__ == "__main__":
    main()
