"""Index completed provisional EOG-only D60 evidence without granting a gate."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import eog_experiment as experiment
from .contracts import CLASS_ORDER, content_id, read_json
from .predictions import load_prediction
from .research import file_sha256


def _bound(path: Path, expected: str | None = None) -> dict:
    actual = file_sha256(path)
    if expected is not None and actual != expected:
        raise ValueError("EOG evidence artifact hash changed")
    return {"path": str(path.resolve()), "sha256": actual}


def _prediction_entry(root: Path, run: Path, rec: dict, fold: dict,
                      config: dict, fit: dict) -> dict:
    if (rec["participant_id"] not in fold["validation"] or
            rec["participant_id"] in fold["train"] or
            len(set(fold["train"])) != 48 or len(set(fold["validation"])) != 12):
        raise ValueError("EOG held prediction leaks its fitted participant")
    path = experiment._paths(root, run / f"fold-{fold['fold_id']}" / "predictions" /
                             (rec["recording_id"] + ".npz"), "runs/provisional-eog")
    sidecar = experiment._paths(root, path.with_suffix(".json"), "runs/provisional-eog")
    sidecar_sha = file_sha256(sidecar)
    meta, arrays = load_prediction(path)
    provenance = meta.get("provenance", {})
    expected_provenance = {"config_hash": content_id(config),
                           "checkpoint_sha256": fit["checkpoint_sha256"],
                           "fold_id": fold["fold_id"],
                           "training_participants": sorted(fold["train"]),
                           "source_psg_sha256": rec["psg_sha256"],
                           "feature_manifest_sha256": config["feature_manifest_sha256"],
                           "channels": [experiment.CHANNEL], "hypnogram_required": False,
                           "provisional": True}
    if (meta.get("model_id") != experiment.MODEL_ID or
            meta.get("protocol_hash") != config["protocol_hash"] or
            meta.get("registry_hash") != config["registry_hash"] or
            meta.get("class_order") != list(CLASS_ORDER) or
            provenance != expected_provenance or
            arrays["participant_id"] != rec["participant_id"] or
            arrays["recording_id"] != rec["recording_id"] or
            not np.array_equal(arrays["epoch_index"], np.arange(rec["n_epochs"])) or
            not np.array_equal(arrays["onset_seconds"], np.arange(rec["n_epochs"]) * 30.) or
            arrays["probabilities"].shape != (rec["n_epochs"], 5) or
            not np.array_equal(arrays["hard_label"], arrays["probabilities"].argmax(axis=1))):
        raise ValueError("EOG prediction changed its source, held fold, class or original grid")
    return {"recording_id": rec["recording_id"], "participant_id": rec["participant_id"],
            "n_epochs": rec["n_epochs"], "fold_id": fold["fold_id"],
            "prediction": _bound(path, meta["payload_sha256"]),
            "sidecar": _bound(sidecar, sidecar_sha),
            "checkpoint_sha256": fit["checkpoint_sha256"],
            "source_psg_sha256": rec["psg_sha256"], "truth_binding": rec["frozen_truth"]}


def index_eog(root: Path, run_dir: Path) -> dict:
    """Index saved predictions without EDF/model access or decoding truth labels."""
    root = Path(root).resolve()
    run = Path(run_dir).resolve()
    if run.parent != root / "runs/provisional-eog":
        raise ValueError("EOG index requires a canonical provisional development run")
    protocol, split, records, authorization = experiment._context(root)
    manifest_path = experiment._paths(root, root / "derived/eog_features/manifest.json",
                                      "derived/eog_features")
    manifest = read_json(manifest_path)
    experiment._verify_manifest(root, manifest, protocol, split, records,
                                experiment._feature_config(root, protocol, split))
    config_path = experiment._paths(root, run / "config.json", "runs/provisional-eog")
    result_path = experiment._paths(root, run / "result.json", "runs/provisional-eog")
    config, result = read_json(config_path), read_json(result_path)
    expected_config = experiment._fit_config(root, protocol, split, manifest_path, manifest)
    if config != expected_config or run.name != content_id(config)[7:23]:
        raise ValueError("EOG run config/source/runtime differs from current fixed D60 recipe")
    if (result.get("status") != "DEVELOPMENT_OOF_COMPLETE" or
            result.get("confirmatory") is not False or result.get("run_id") != run.name or
            result.get("config_hash") != content_id(config) or
            result.get("protocol_hash") != protocol["protocol_hash"] or
            result.get("split_id") != split["split_id"] or
            result.get("channel_names") != [experiment.CHANNEL] or
            result.get("gate_A") != "NOT_RUN" or result.get("gate_B") != "NOT_RUN"):
        raise ValueError("EOG result is not a complete nonconfirmatory D-only run")
    metrics = result.get("metrics", {})
    if (metrics.get("recording_count") != 119 or
            metrics.get("complete_psg_epochs") != 276133 or
            metrics.get("evaluated_epochs") != 274271 or
            type(metrics.get("macro_f1")) not in (int, float) or
            not math.isfinite(metrics["macro_f1"])):
        raise ValueError("EOG result lacks the frozen full-development metric grid")
    folds = []
    by_fold = {}
    for fold in split["folds"]:
        path = experiment._model_path(root, run, fold)
        fit_path = experiment._paths(root, path.with_name("fit.json"), "runs/provisional-eog")
        fit = experiment._fit_receipt(path, config, fold)
        by_fold[fold["fold_id"]] = fit
        folds.append({"fold_id": fold["fold_id"], "train_participants": sorted(fold["train"]),
                      "held_participants": sorted(fold["validation"]),
                      "fit_record": _bound(fit_path),
                      "checkpoint": _bound(path, fit["checkpoint_sha256"]),
                      "training_epochs": fit["training_epochs"], "rounds": 400,
                      "class_expanded_trees": fit["trees"]})
    fold_for = {pid: fold for fold in split["folds"] for pid in fold["validation"]}
    if (len(fold_for) != 60 or len(by_fold) != 5 or
            sum(len(fold["validation"]) for fold in split["folds"]) != 60):
        raise ValueError("EOG index needs the frozen five participant-disjoint folds")
    entries = []
    expected_predictions = []
    expected_truth = []
    for rec in sorted(records, key=lambda row: row["recording_id"]):
        fold = fold_for[rec["participant_id"]]
        entry = _prediction_entry(root, run, rec, fold, config, by_fold[fold["fold_id"]])
        entries.append(entry)
        expected_predictions.append(entry["prediction"]["path"])
        expected_truth.append(str(Path(rec["truth_path"]).resolve()))
    for key, expected in (("prediction_paths", expected_predictions), ("truth_paths", expected_truth)):
        paths = result.get(key)
        if (type(paths) is not list or len(paths) != 119 or set(paths) != set(expected)):
            raise ValueError("EOG result differs from exactly one original-grid output per D night")
    experiment._verify_frozen_truth(records)
    result_binding = _bound(result_path)
    config_binding = _bound(config_path)
    if (experiment._source_identity(root) != config["source_sha256"] or
            experiment._runtime(root) != config["runtime"] or
            file_sha256(manifest_path) != config["feature_manifest_sha256"]):
        raise ValueError("EOG source/runtime/features changed during evidence indexing")
    index = {"schema_version": "1.0", "artifact_type": "provisional_eog_development_evidence_index",
             "run_id": run.name, "model_id": experiment.MODEL_ID,
             "confirmatory": False, "eligibility_status": "NOT_EVALUATED",
             "gate_A": "NOT_RUN", "gate_B": "NOT_RUN",
             "protocol_hash": protocol["protocol_hash"], "split_id": split["split_id"],
             "authorization_id": authorization["authorization_id"],
             "config_id": content_id(config), "result": result_binding, "config": config_binding,
             "feature_manifest": _bound(manifest_path, config["feature_manifest_sha256"]),
             "source_and_runtime": {"sources": config["source_sha256"], "runtime": config["runtime"]},
             "folds": folds, "records": entries,
             "checks_not_performed": ["EDF_access", "model_deserialization_or_inference",
                                      "new_training", "audit_truth_access", "confirmatory_gate_testing"],
             "producer": _bound(Path(__file__))}
    index["index_id"] = content_id(index)
    return index
