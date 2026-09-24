"""Equal-probability ensemble of complete, prospective YASA development predictions.

The input is saved signal-only predictions. Checkpoints are hashed but never
deserialized; no feature, EDF, or audit file is opened.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import time

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "4"

import numpy as np

from .contracts import CLASS_ORDER, content_id, read_json
from .evaluation import evaluate_saved_records
from .predictions import load_prediction, save_prediction
from .protocol import development_records
from .research import atomic_json, compute_lease, file_sha256


SECONDS = 900
HOST_BYTES = 10 * 1024**3
FREE_RAM = 4 * 1024**3
FREE_DISK = 20 * 1024**3
SNAPSHOT_FILES = ("sleepedf/development_ensemble.py", "sleepedf/predictions.py",
                  "sleepedf/evaluation.py", "sleepedf/protocol.py", "sleepedf/contracts.py")
YASA_SOURCES = {"yasa_baseline.py", "protocol.py", "research.py", "splits.py",
                "contracts.py", "predictions.py", "evaluation.py"}


def _limits(root: Path, deadline: float) -> None:
    import psutil
    if (time.monotonic() >= deadline or psutil.Process().memory_info().rss > HOST_BYTES or
        psutil.virtual_memory().available < FREE_RAM or shutil.disk_usage(root).free < FREE_DISK):
        raise RuntimeError("Development ensemble exceeded time, RAM, or disk bound")


def _hash(path: Path, root: Path, deadline: float) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            _limits(root, deadline)
    return digest.hexdigest()


def _inside(path: Path, directory: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Development ensemble input path escapes its declared root")
    return path


def _member(root: Path, result_path: Path, protocol: dict, split: dict,
            records: list[dict], deadline: float) -> dict:
    runs = root / "runs"
    result_path = _inside(result_path, runs)
    run_dir = result_path.parent
    if result_path.name != "result.json" or run_dir.parent != runs or not run_dir.name.startswith("yasa-"):
        raise ValueError("Only complete prospective YASA D result manifests are accepted")
    result_sha = _hash(result_path, root, deadline)
    result = read_json(result_path)
    config_path = _inside(run_dir / "config.json", run_dir)
    config_sha = _hash(config_path, root, deadline)
    config = read_json(config_path)
    selection = config.get("selection_evidence")
    expected_truth = {str(Path(rec["truth_path"]).resolve()) for rec in records}
    fold_for = {pid: fold for fold in split["folds"] for pid in fold["validation"]}
    expected_predictions = {str((run_dir / f"fold-{fold_for[rec['participant_id']]['fold_id']}" /
                                  "predictions" / (rec["recording_id"] + ".npz")).resolve())
                            for rec in records}
    if (result.get("status") != "DEVELOPMENT_OOF_COMPLETE" or result.get("confirmatory") is not False or
        result.get("gate_A") != "NOT_RUN" or result.get("gate_B") != "NOT_RUN" or
        result.get("run_id") != run_dir.name or result.get("protocol_hash") != protocol["protocol_hash"] or
        result.get("config_hash") != content_id(config) or
        config.get("split_id") != split["split_id"] or
        config.get("variant") not in ("eeg", "eeg_eog") or config.get("seed") not in (17, 43, 101) or
        type(config.get("source_sha")) is not str or not config["source_sha"] or
        type(config.get("implementation_sha")) is not str or not config["implementation_sha"] or
        type(selection) is not dict or
        result.get("selection_evidence_sha256") != selection.get("sha256") or
        type(result.get("prediction_paths")) is not list or
        type(result.get("truth_paths")) is not list or
        len(result["prediction_paths"]) != len(records) or
        len(result["truth_paths"]) != len(records) or
        set(result["prediction_paths"]) != expected_predictions or
        set(result["truth_paths"]) != expected_truth or
        len(set(result["prediction_paths"])) != len(records) or
        len(set(result["truth_paths"])) != len(records)):
        raise ValueError("Member result is not a complete frozen D-only OOF result")
    selection_path = _inside(root / selection["path"], root / "research")
    if (_hash(selection_path, root, deadline) != selection["sha256"] or
        read_json(selection_path) != selection["content"] or
        selection["content"].get("protocol_hash") != protocol["protocol_hash"] or
        selection["content"].get("split_id") != split["split_id"] or
        type(selection["content"].get("chosen")) is not dict or
        selection["content"]["chosen"].get("variant") != config.get("variant") or
        selection["content"]["chosen"].get("seed") != config.get("seed") or
        selection["content"]["chosen"].get("lr_multiplier") not in (.3, 1, 3) or
        config.get("params", {}).get("learning_rate") !=
            .1 * selection["content"]["chosen"]["lr_multiplier"] or
        type(selection["content"].get("selection_participants")) is not list or
        not selection["content"]["selection_participants"] or
        len(selection["content"]["selection_participants"]) !=
            len(set(selection["content"]["selection_participants"])) or
        not set(selection["content"].get("selection_participants", [])) <=
        set(split["participants"]["development"])):
        raise ValueError("Member selection evidence changed or escapes D")
    source_manifest_path = _inside(run_dir / "source_snapshot/manifest.json", run_dir)
    source_manifest = read_json(source_manifest_path)
    if (source_manifest.get("files") != config.get("dependency_sources") or
        type(source_manifest.get("files")) is not dict or not source_manifest["files"] or
        set(source_manifest["files"]) != YASA_SOURCES or
        source_manifest["files"].get("yasa_baseline.py") != config["implementation_sha"]):
        raise ValueError("Member source snapshot differs from its config")
    source_hashes = {"manifest": _hash(source_manifest_path, root, deadline)}
    for name, digest in source_manifest["files"].items():
        if (type(name) is not str or Path(name).name != name or type(digest) is not str or
            _hash(_inside(run_dir / "source_snapshot" / name, run_dir), root, deadline) != digest):
            raise ValueError("Member source snapshot bytes differ")
        source_hashes[name] = digest
    fold_evidence = {}
    for fold in split["folds"]:
        _limits(root, deadline)
        fid = fold["fold_id"]
        folder = run_dir / f"fold-{fid}"
        fit_path = _inside(folder / "fit.json", run_dir)
        fit = read_json(fit_path)
        roles = fit.get("roles")
        if (fit.get("status") != "FIT_COMPLETE" or fit.get("config_hash") != content_id(config) or
            fit.get("protocol_hash") != protocol["protocol_hash"] or
            fit.get("fitted_participants") != sorted(fold["train"]) or
            fit.get("excluded_validation_participants") != sorted(fold["validation"]) or
            set(fit["fitted_participants"]) & set(fold["validation"]) or
            type(roles) is not dict or fit.get("roles_id") != content_id(roles) or
            roles.get("fitted_participants") != sorted(fold["train"]) or
            roles.get("selection_participants") != selection["content"]["selection_participants"] or
            roles.get("selection_evidence_sha256") != selection["sha256"] or
            any(roles.get(key) != [] for key in ("pretraining_participants", "scaler_participants",
                "calibration_participants", "teacher_participants", "pseudolabel_participants")) or
            roles.get("recording_local_normalization") != {
                "scope": "unlabeled, per recording, complete night, native robust feature scaling",
                "participants": split["participants"]["development"],
                "recordings": len(records),
                "feature_config_hash": config.get("feature_config_hash")} or
            result.get("fold_roles", {}).get(str(fid)) != fit["roles_id"] or
            _hash(_inside(folder / "model.joblib", run_dir), root, deadline) != fit.get("checkpoint_sha")):
            raise ValueError("Member fold training, held exclusion, roles, or checkpoint differs")
        parity_path = _inside(folder / "native-parity.json", run_dir)
        parity = read_json(parity_path)
        first_held = next(rec["recording_id"] for rec in records
                          if rec["participant_id"] in fold["validation"])
        checks = parity.get("checks") if type(parity) is dict else None
        if (type(checks) is not list or len(checks) != 1 or type(checks[0]) is not dict or
            set(checks[0]) != {"recording_id", "maximum_probability_error"} or
            checks[0]["recording_id"] != first_held or
            type(checks[0]["maximum_probability_error"]) not in (int, float) or
            not math.isfinite(checks[0]["maximum_probability_error"]) or
            not 0 <= checks[0]["maximum_probability_error"] <= 1e-12):
            raise ValueError("Member native first-night parity record differs")
        fold_evidence[str(fid)] = {"fit_sha256": _hash(fit_path, root, deadline),
                                   "roles_id": fit["roles_id"],
                                   "checkpoint_sha256": fit["checkpoint_sha"],
                                   "native_parity_sha256": _hash(parity_path, root, deadline)}
    if _hash(result_path, root, deadline) != result_sha or _hash(config_path, root, deadline) != config_sha:
        raise ValueError("Member result or config changed during inspection")
    return {"run_id": run_dir.name, "variant": config["variant"],
            "result_path": str(result_path), "result_sha256": result_sha,
            "config_path": str(config_path), "config_sha256": config_sha,
            "config_id": content_id(config), "source_sha": config.get("source_sha"),
            "implementation_sha": config.get("implementation_sha"),
            "selection_path": str(selection_path), "selection_sha256": selection["sha256"],
            "selection_participants": selection["content"]["selection_participants"],
            "source_snapshot": source_hashes, "folds": fold_evidence,
            "prediction_paths": {rec["recording_id"]: str((run_dir /
                f"fold-{fold_for[rec['participant_id']]['fold_id']}" / "predictions" /
                (rec["recording_id"] + ".npz")).resolve()) for rec in records}}


def _combine(records: list[tuple[dict, dict]], *, participant: str, recording: str,
             protocol_hash: str, registry_hash: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(records) not in (2, 3):
        raise ValueError("Equal-probability ensemble requires two or three members")
    first_meta, first = records[0]
    probabilities = []
    for meta, data in records:
        if (meta["protocol_hash"] != protocol_hash or meta["registry_hash"] != registry_hash or
            meta["class_order"] != list(CLASS_ORDER) or
            data["participant_id"] != participant or data["recording_id"] != recording or
            "probabilities" not in data or
            not np.array_equal(data["epoch_index"], first["epoch_index"]) or
            not np.array_equal(data["onset_seconds"], first["onset_seconds"])):
            raise ValueError("Ensemble member identity, class order, or full grid differs")
        probabilities.append(data["probabilities"].astype(np.float64, copy=False))
    mean = np.mean(np.stack(probabilities, axis=0), axis=0, dtype=np.float64)
    if mean.shape != probabilities[0].shape or not np.isfinite(mean).all():
        raise ValueError("Equal-probability mean is invalid")
    return first["epoch_index"], first["onset_seconds"], mean, np.argmax(mean, axis=1).astype(np.int8)


def _frozen_truth(records: list[dict], root: Path, deadline: float) -> list[Path]:
    paths = []
    for rec in records:
        _limits(root, deadline)
        path = Path(rec["truth_path"])
        frozen = rec.get("frozen_truth")
        if (type(frozen) is not dict or
            _hash(path, root, deadline) != frozen.get("payload_sha256") or
            _hash(path.with_suffix(".json"), root, deadline) != frozen.get("sidecar_sha256")):
            raise ValueError("Development truth bytes differ from frozen D manifest")
        paths.append(path)
    return paths


def run(root: Path, result_paths: list[Path]) -> dict:
    root = root.resolve()
    runs = root / "runs"
    if runs.resolve() != runs or (runs / "compute.lock").resolve() != runs / "compute.lock":
        raise ValueError("Ensemble output redirects outside canonical runs")
    started = time.monotonic()
    deadline = started + SECONDS
    if len(result_paths) not in (2, 3) or len({str(path.resolve()) for path in result_paths}) != len(result_paths):
        raise ValueError("Specify two or three distinct complete member results")
    with compute_lease(root, "development-equal-probability-ensemble"):
        protocol, split, records = development_records(root)
        if (len(records) != 119 or len(split["participants"]["development"]) != 60 or
            len({r["recording_id"] for r in records}) != 119):
            raise ValueError("Frozen development D60/119 membership differs")
        members = [_member(root, path, protocol, split, records, deadline) for path in result_paths]
        selection_union = sorted(set().union(*(set(m["selection_participants"]) for m in members)))
        identity = {"member_results": [{key: item[key] for key in
                                         ("run_id", "result_sha256", "config_sha256", "selection_sha256")}
                                        for item in members],
                    "protocol_hash": protocol["protocol_hash"], "registry_hash": protocol["registry_hash"],
                    "split_id": split["split_id"], "rule": "unweighted_mean_float64_probabilities"}
        run_id = "development-equal-probability-" + content_id(identity)[-12:]
        out = runs / run_id
        if out.resolve() != out or (out / "predictions").resolve() != out / "predictions" or out.exists():
            raise ValueError("Ensemble output already exists or redirects")
        out.mkdir(parents=True, exist_ok=False)
        snapshot = out / "source_snapshot"
        snapshot.mkdir()
        sources = {}
        for name in SNAPSHOT_FILES:
            _limits(root, deadline)
            source = root / name
            target = snapshot / Path(name).name
            shutil.copyfile(source, target)
            sources[name] = _hash(source, root, deadline)
            if _hash(target, root, deadline) != sources[name]:
                raise ValueError("Ensemble source snapshot changed")
        atomic_json(snapshot / "manifest.json", {"sources": sources}, immutable=True)
        atomic_json(out / "members.json", {"identity": identity, "members": members,
                    "selection_union": selection_union,
                    "source_snapshot": sources, "confirmatory": False}, immutable=True)
        prediction_paths = []
        for rec in sorted(records, key=lambda value: value["recording_id"]):
            _limits(root, deadline)
            rid, pid = rec["recording_id"], rec["participant_id"]
            loaded = []
            ancestry = []
            channels = set()
            fold = next(f for f in split["folds"] if pid in f["validation"])
            for member in members:
                path = _inside(Path(member["prediction_paths"][rid]),
                               Path(member["result_path"]).parent)
                meta, data = load_prediction(path)
                evidence = member["folds"][str(fold["fold_id"])]
                provenance = meta["provenance"]
                expected_channels = ["EEG Fpz-Cz"] + (["EOG horizontal"] if member["variant"] == "eeg_eog" else [])
                if (meta["model_id"] != member["run_id"] + f"/fold-{fold['fold_id']}" or
                    provenance.get("config_hash") != member["config_id"] or
                    provenance.get("source_sha") != member["source_sha"] or
                    provenance.get("implementation_sha") != member["implementation_sha"] or
                    provenance.get("checkpoint_sha") != evidence["checkpoint_sha256"] or
                    provenance.get("roles_id") != evidence["roles_id"] or
                    provenance.get("selection_evidence_sha256") != member["selection_sha256"] or
                    provenance.get("fitted_participants") != sorted(fold["train"]) or
                    provenance.get("channels") != expected_channels or
                    pid in provenance["fitted_participants"] or
                    len(data["epoch_index"]) != rec["n_epochs"]):
                    raise ValueError("Member prediction lacks held-participant ancestry or complete grid")
                payload_sha = _hash(path, root, deadline)
                if meta["payload_sha256"] not in (payload_sha, "sha256:" + payload_sha):
                    raise ValueError("Member prediction changed after loading")
                loaded.append((meta, data))
                channels.update(expected_channels)
                ancestry.append({"run_id": member["run_id"], "prediction_path": str(path),
                                 "prediction_sha256": payload_sha,
                                 "sidecar_sha256": _hash(path.with_suffix(".json"), root, deadline),
                                 "checkpoint_sha256": evidence["checkpoint_sha256"],
                                 "roles_id": evidence["roles_id"],
                                 "selection_sha256": member["selection_sha256"]})
            epochs, onsets, mean, hard = _combine(loaded, participant=pid, recording=rid,
                protocol_hash=protocol["protocol_hash"], registry_hash=protocol["registry_hash"])
            target = out / "predictions" / (rid + ".npz")
            save_prediction(target, participant_id=pid, recording_id=rid,
                epoch_index=epochs, onset_seconds=onsets, hard_label=hard, probabilities=mean,
                model_id=run_id, protocol_hash=protocol["protocol_hash"],
                registry_hash=protocol["registry_hash"],
                provenance={"ensemble_rule": "unweighted_mean_float64_probabilities",
                            "members": ancestry, "hypnogram_required": False,
                            "confirmatory": False, "config_hash": content_id(identity),
                            "fitted_participants": sorted(fold["train"]),
                            "selection_participants": selection_union,
                            "channels": sorted(channels)})
            prediction_paths.append(target)
        truth_paths = _frozen_truth(sorted(records, key=lambda value: value["recording_id"]),
                                    root, deadline)
        _limits(root, deadline)
        metrics = evaluate_saved_records(truth_paths, prediction_paths,
            protocol_hash=protocol["protocol_hash"], registry_hash=protocol["registry_hash"],
            model_id=run_id)
        _limits(root, deadline)
        result = {"schema_version": "1.0", "artifact_type": "development_equal_probability_ensemble",
                  "status": "DEVELOPMENT_OOF_COMPLETE", "confirmatory": False,
                  "gate_A": "NOT_RUN", "gate_B": "NOT_RUN", "run_id": run_id,
                  "identity": identity, "config_hash": content_id(identity),
                  "protocol_hash": protocol["protocol_hash"],
                  "registry_hash": protocol["registry_hash"], "split_id": split["split_id"],
                  "members_sha256": _hash(out / "members.json", root, deadline),
                  "source_snapshot_sha256": _hash(snapshot / "manifest.json", root, deadline),
                  "metrics": metrics, "prediction_paths": [str(p.resolve()) for p in prediction_paths],
                  "truth_paths": [str(p.resolve()) for p in truth_paths],
                  "elapsed_seconds": time.monotonic() - started}
        _limits(root, deadline)
        atomic_json(out / "result.json", result, immutable=True)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--member-result", type=Path, action="append", required=True)
    args = parser.parse_args()
    result = run(args.root, args.member_result)
    print(result["run_id"], flush=True)


if __name__ == "__main__":
    main()
