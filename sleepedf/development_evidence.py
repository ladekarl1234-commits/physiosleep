"""Index existing development evidence without granting precision eligibility.

This first producer handles the six completed local YASA runs. It hashes opaque
prediction/model files but neither deserializes them nor reads reference arrays.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .contracts import CLASS_ORDER, content_id, read_json
from .protocol import development_records
from .research import atomic_json, file_sha256


def _bound(path: Path, expected: str | None = None) -> dict:
    actual = file_sha256(path)
    if expected is not None and actual != expected:
        raise ValueError("Development artifact hash mismatch: " + str(path))
    return {"path": str(path.resolve()), "sha256": actual}


def validate_yasa_fold(config: dict, fit: dict, fold: dict, *, protocol_hash: str) -> None:
    """Validate recorded fixed-round fit metadata; this does not inspect the model."""
    rounds = config["params"].get("n_estimators")
    if type(rounds) is not int or rounds < 400:
        raise ValueError("YASA requires at least 400 boosting rounds")
    if (fit.get("status") != "FIT_COMPLETE" or fit.get("config_hash") != content_id(config) or
            fit.get("protocol_hash") != protocol_hash or
            fit.get("fitted_participants") != sorted(fold["train"]) or
            fit.get("excluded_validation_participants") != sorted(fold["validation"]) or
            type(fit.get("trees")) is not int or fit["trees"] != 5 * rounds or
            type(fit.get("training_epochs")) is not int or fit["training_epochs"] < 1):
        raise ValueError("YASA fit metadata, class-expanded trees or held-participant ancestry differs")


def index_yasa(root: Path, run_dir: Path) -> dict:
    root = Path(root).resolve(); run_dir = Path(run_dir).resolve()
    if run_dir.parent != root / "runs":
        raise ValueError("YASA development run must be directly inside this workspace's runs directory")
    protocol, split, records = development_records(root)
    result_path, config_path = run_dir / "result.json", run_dir / "config.json"
    result, config = read_json(result_path), read_json(config_path)
    config_id = content_id(config)
    if (result.get("status") != "DEVELOPMENT_OOF_COMPLETE" or result.get("confirmatory") is not False or
            result.get("run_id") != run_dir.name or not run_dir.name.startswith("yasa-") or
            result.get("protocol_hash") != protocol["protocol_hash"] or
            result.get("config_hash") != config_id or config.get("split_id") != split["split_id"] or
            config.get("variant") not in ("eeg", "eeg_eog") or type(config.get("seed")) is not int or
            config["seed"] not in (17, 43, 101) or
            config.get("source_sha") != "e581bf452097e01b493b3072964e206ae9b01dc4" or
            config.get("truth_manifest_sha256") != file_sha256(root / "research/development-truth-v1.json")):
        raise ValueError("YASA development result/config differs from the frozen protocol")
    fold_for = {pid: fold["fold_id"] for fold in split["folds"] for pid in fold["validation"]}
    expected_predictions = {str((run_dir / f"fold-{fold_for[record['participant_id']]}" /
                                  "predictions" / (record["recording_id"] + ".npz")).resolve())
                            for record in records}
    expected_truth = {str(Path(record["truth_path"]).resolve()) for record in records}
    for key, expected in (("prediction_paths", expected_predictions), ("truth_paths", expected_truth)):
        paths = result.get(key)
        if (type(paths) is not list or len(paths) != len(expected) or
                any(type(path) is not str for path in paths) or
                {str(Path(path).resolve()) for path in paths} != expected):
            raise ValueError("YASA result lacks exactly one path per original development recording")
    folds = []
    for fold in split["folds"]:
        directory = run_dir / f"fold-{fold['fold_id']}"
        fit_path = directory / "fit.json"; fit = read_json(fit_path)
        validate_yasa_fold(config, fit, fold, protocol_hash=protocol["protocol_hash"])
        parity_path = directory / "native-parity.json"
        folds.append({
            "fold_id": fold["fold_id"], "train_participants": sorted(fold["train"]),
            "held_participants": sorted(fold["validation"]), "fit_record": _bound(fit_path),
            "checkpoint": _bound(directory / "model.joblib", fit["checkpoint_sha"]),
            "native_parity_record": _bound(parity_path),
            "boosting_rounds_recorded": config["params"]["n_estimators"],
            "class_expanded_trees_recorded": fit["trees"],
            "training_valid_epochs_recorded": fit["training_epochs"],
            "recorded_learned_roles": {"fitted_participants": fit["fitted_participants"]},
            "unresolved_role_attestations": ["selection_participants", "pretraining_participants",
                "scaler_participants", "calibration_participants", "teacher_participants", "pseudolabel_participants"],
        })
    entries = []
    for record in sorted(records, key=lambda item: item["recording_id"]):
        rid, pid = record["recording_id"], record["participant_id"]
        fold_id = fold_for[pid]; fold = folds[fold_id]
        path = run_dir / f"fold-{fold_id}" / "predictions" / (rid + ".npz")
        sidecar_path = path.with_suffix(".json"); sidecar = read_json(sidecar_path)
        provenance = sidecar.get("provenance", {})
        if (sidecar.get("schema_version") != "1.0" or sidecar.get("artifact_type") != "signal_only_prediction" or
                sidecar.get("protocol_hash") != protocol["protocol_hash"] or
                sidecar.get("registry_hash") != protocol["registry_hash"] or
                sidecar.get("class_order") != list(CLASS_ORDER) or
                sidecar.get("model_id") != run_dir.name + f"/fold-{fold_id}" or
                provenance.get("config_hash") != config_id or
                provenance.get("checkpoint_sha") != fold["checkpoint"]["sha256"] or
                provenance.get("source_sha") != config["source_sha"] or
                provenance.get("implementation_sha") != config["implementation_sha"] or
                provenance.get("fitted_participants") != fold["train_participants"] or
                provenance.get("hypnogram_required") is not False):
            raise ValueError("YASA prediction sidecar does not match its held fold/checkpoint")
        channels = ["EEG Fpz-Cz"] + (["EOG horizontal"] if config["variant"] == "eeg_eog" else [])
        if provenance.get("channels") != channels:
            raise ValueError("YASA channel declaration differs from the recorded variant")
        entries.append({"recording_id": rid, "participant_id": pid, "n_epochs": record["n_epochs"],
                        "fold_id": fold_id, "prediction": _bound(path, sidecar["payload_sha256"]),
                        "sidecar": _bound(sidecar_path), "checkpoint_sha256": fold["checkpoint"]["sha256"],
                        "truth_binding": record["frozen_truth"],
                        "identity_and_grid_source": "frozen_readiness_and_expected_path_not_payload_validation"})
    index = {
        "schema_version": "1.0", "artifact_type": "development_evidence_index_not_eligibility",
        "slot_id": "yasa_native", "run_id": run_dir.name, "variant": config["variant"], "seed": config["seed"],
        "confirmatory": False, "eligibility_status": "NOT_EVALUATED",
        "protocol_hash": protocol["protocol_hash"], "registry_hash": protocol["registry_hash"],
        "split_id": split["split_id"], "config_id": config_id,
        "result": _bound(result_path), "config": _bound(config_path),
        "bindings": {"protocol": _bound(root / "research/protocol-v1.json"),
                     "registry": _bound(root / "research/baseline_registry.json"),
                     "split": _bound(Path(protocol["split_path"])),
                     "readiness": _bound(Path(protocol["readiness_path"])),
                     "development_truth": _bound(root / "research/development-truth-v1.json"),
                     "index_producer": _bound(Path(__file__))},
        "recorded_source_and_runtime": {key: config[key] for key in
            ("source_sha", "implementation_sha", "recipe_sha", "runtime_lock_sha256", "feature_config_hash", "feature_manifest_sha256")},
        "folds": folds, "records": entries,
        "configuration_selection": {"allowed_pool": sorted(split["participants"]["development"]),
                                    "frozen_selection_evidence": None,
                                    "scope": "adaptive_development_D60_if_selected_later"},
        "unresolved_eligibility": ["payload_record_identity_full_grid_probabilities_and_common_mask_recomputation",
                                   "complete_fold_learned_role_and_native_adequacy_attestations",
                                   "rights_and_external_lineage_evidence",
                                   "global_configuration_selection_freeze_and_adaptive_selection_limits"],
        "checks_not_performed": ["prediction_or_truth_array_deserialization", "model_loading_or_inference",
                                 "EDF_access", "audit_truth_access", "new_metric_or_precision_calculation"],
    }
    index["index_id"] = content_id(index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = index_yasa(args.root, args.run_dir)
    output = args.root.resolve() / "research/development-evidence" / (result["run_id"] + ".json")
    atomic_json(output, result, immutable=True)
    print(str(output))


if __name__ == "__main__":
    main()
