"""Recompute indexed development payloads; never grant model or gate eligibility."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np

from .contracts import CLASS_ORDER, content_id, read_json
from .development_evidence import index_yasa
from .evaluation import _metrics, macro_f1_fraction, score_aligned_record
from .predictions import load_prediction
from .protocol import development_records, load_development_truth
from .research import atomic_json, file_sha256


LEGACY_METRIC_KEYS = frozenset(("accuracy", "confusion", "evaluated_epochs", "invalid_epochs",
    "kappa", "kappa_unavailable_reason", "macro_f1", "macro_f1_exact", "participant_count",
    "participant_macro_f1_mean", "per_class_f1", "per_participant_confusion", "recording_count", "support"))


def compare_saved_metrics(recomputed: dict, saved: dict) -> list[str]:
    """Older runs predate supplemental metric fields; every recorded value must match."""
    if (type(saved) is not dict or not LEGACY_METRIC_KEYS <= set(saved) or
            not set(saved) <= set(recomputed)):
        raise ValueError("Saved development metric fields are missing or unknown")
    differing = [key for key in saved if saved[key] != recomputed[key]]
    if differing:
        raise ValueError("Recomputed development metrics differ: " + ", ".join(sorted(differing)))
    return sorted(set(recomputed) - set(saved))


def _binding(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _array_id(value: np.ndarray, dtype: str) -> str:
    """Hash canonical little-endian contiguous data with shape and dtype."""
    array = np.ascontiguousarray(value, dtype=np.dtype(dtype))
    return content_id({"dtype": dtype, "shape": list(array.shape),
                       "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest()})


def check_record(entry: dict, record: dict, split: dict, *, protocol_hash: str,
                 registry_hash: str, run_id: str, config_id: str) -> dict:
    """Read only the supplied frozen development truth and indexed prediction."""
    rid, pid = record["recording_id"], record["participant_id"]
    held = [fold for fold in split["folds"] if pid in fold["validation"]]
    if (pid not in split["participants"]["development"] or len(held) != 1 or
            entry.get("recording_id") != rid or entry.get("participant_id") != pid or
            entry.get("n_epochs") != record["n_epochs"] or
            entry.get("fold_id") != held[0]["fold_id"] or
            entry.get("truth_binding") != record["frozen_truth"]):
        raise ValueError("Payload entry differs from the original held development record")
    path = Path(entry["prediction"]["path"])
    if (file_sha256(path) != entry["prediction"]["sha256"] or
            Path(entry["sidecar"]["path"]) != path.with_suffix(".json") or
            file_sha256(path.with_suffix(".json")) != entry["sidecar"]["sha256"]):
        raise ValueError("Indexed prediction or sidecar bytes changed")
    metadata, prediction = load_prediction(path)
    provenance = metadata["provenance"]
    if (metadata["protocol_hash"] != protocol_hash or
            metadata["registry_hash"] != registry_hash or
            metadata["model_id"] != run_id + f"/fold-{held[0]['fold_id']}" or
            provenance.get("config_hash") != config_id or
            provenance.get("checkpoint_sha") != entry["checkpoint_sha256"] or
            provenance.get("fitted_participants") != sorted(held[0]["train"]) or
            provenance.get("hypnogram_required") is not False):
        raise ValueError("Payload provenance differs from its original training fold")
    truth = load_development_truth(record, split)
    confusion, invalid = score_aligned_record(truth, prediction)
    probabilities = prediction.get("probabilities")
    if probabilities is None:
        raise ValueError("This YASA payload requires its recorded native probabilities")
    chosen = probabilities[np.arange(len(probabilities)), prediction["hard_label"]]
    if np.any(chosen != probabilities.max(axis=1)):
        raise ValueError("YASA hard class is not a probability maximizer")
    if len(prediction["epoch_index"]) != record["n_epochs"]:
        raise ValueError("Prediction does not cover the full frozen PSG grid")
    return {"recording_id": rid, "participant_id": pid, "fold_id": entry["fold_id"],
            "complete_epochs": record["n_epochs"], "valid_epochs": int(confusion.sum()),
            "invalid_epochs": invalid, "confusion": confusion.tolist(),
            "epoch_index_id": _array_id(truth["epoch_index"], "<i8"),
            "onset_seconds_id": _array_id(truth["onset_seconds"], "<f8"),
            "valid_mask_id": _array_id(truth["valid_mask"], "|u1"),
            "reference_label_id": _array_id(truth["reference_label"], "|i1"),
            "prediction": entry["prediction"], "sidecar": entry["sidecar"],
            "truth_binding": entry["truth_binding"], "probabilities_validated": True}


def recompute_yasa(root: Path, index_path: Path) -> dict:
    """Normalize actual saved arrays without fitting, inference or bootstrap work."""
    import psutil

    root, index_path = Path(root).resolve(), Path(index_path).resolve()
    index = read_json(index_path)
    run_dir = Path(index["result"]["path"]).parent
    if index_path != root / "research/development-evidence" / (run_dir.name + ".json"):
        raise ValueError("Expected an immutable local YASA development evidence index")
    index_binding = _binding(index_path)
    if index_yasa(root, run_dir) != index:
        raise ValueError("YASA evidence index no longer matches its bound artifacts")
    protocol, split, records = development_records(root)
    records = {record["recording_id"]: record for record in records}
    source_paths = [Path(__file__), *(root / "sleepedf" / name for name in (
        "development_evidence.py", "evaluation.py", "predictions.py", "protocol.py",
        "research.py", "contracts.py", "splits.py"))]
    sources = [_binding(path) for path in source_paths]
    started = time.monotonic()
    checked, by_participant = [], {}
    for entry in index["records"]:
        rss = psutil.Process().memory_info().rss
        if (rss > 1024**3 or psutil.virtual_memory().available < 4 * 1024**3 or
                time.monotonic() - started > 120):
            raise RuntimeError("Read-only development artifact check exceeded its resource bound")
        row = check_record(entry, records[entry["recording_id"]], split,
                           protocol_hash=protocol["protocol_hash"],
                           registry_hash=protocol["registry_hash"], run_id=index["run_id"],
                           config_id=index["config_id"])
        checked.append(row)
        by_participant.setdefault(row["participant_id"], np.zeros((5, 5), dtype=np.int64))
        by_participant[row["participant_id"]] += np.asarray(row["confusion"], dtype=np.int64)
    if (set(by_participant) != set(split["participants"]["development"]) or
            any(not matrix.sum() for matrix in by_participant.values())):
        raise ValueError("Development participants lack complete positive-support payloads")
    metrics = _metrics(sum(by_participant.values(), np.zeros((5, 5), dtype=np.int64)))
    invalid = sum(row["invalid_epochs"] for row in checked)
    metrics.update(invalid_epochs=invalid,
                   complete_psg_epochs=metrics["evaluated_epochs"] + invalid,
                   reference_valid_fraction=metrics["evaluated_epochs"] / (metrics["evaluated_epochs"] + invalid),
                   prediction_coverage_fraction=1.0, participant_count=len(by_participant),
                   recording_count=len(checked),
                   participant_macro_f1_mean=float(np.mean([
                       float(macro_f1_fraction(cm)) for _, cm in sorted(by_participant.items())])),
                   per_participant_confusion={pid: cm.tolist() for pid, cm in sorted(by_participant.items())})
    previous = read_json(Path(index["result"]["path"]))["metrics"]
    supplemental = compare_saved_metrics(metrics, previous)
    if (index_yasa(root, run_dir) != index or _binding(index_path) != index_binding or
            [_binding(path) for path in source_paths] != sources):
        raise ValueError("Evidence or verifier source changed during payload verification")
    result = {
        "schema_version": "1.0", "artifact_type": "development_payload_recomputation_not_eligibility",
        "run_id": index["run_id"], "slot_id": index["slot_id"], "variant": index["variant"],
        "seed": index["seed"], "confirmatory": False, "eligibility_status": "NOT_EVALUATED",
        "protocol_hash": protocol["protocol_hash"], "split_id": split["split_id"],
        "registry_hash": protocol["registry_hash"], "class_order": list(CLASS_ORDER),
        "index": index_binding, "index_id": index["index_id"], "sources": sources,
        "records": checked, "metrics": metrics,
        "supplemental_recomputed_fields_absent_from_historical_result": supplemental,
        "canonical_array_hash_format": "dtype and shape plus SHA256 of contiguous little-endian data",
        "resource_scope": "read_only_saved_development_artifacts_one_record_at_a_time_1GiB_120seconds",
        "unresolved_eligibility": [item for item in index["unresolved_eligibility"]
                                   if not item.startswith("payload_")],
        "checks_not_performed": ["model_loading_or_inference", "new_fit", "EDF_access",
                                 "audit_truth_access", "precision_or_bootstrap_calculation",
                                 "native_adequacy_or_rights_attestation"],
    }
    result["recomputation_id"] = content_id(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    result = recompute_yasa(args.root, args.index)
    output = args.root.resolve() / "research/development-payloads" / (result["run_id"] + ".json")
    atomic_json(output, result, immutable=True)
    print(str(output))


if __name__ == "__main__":
    main()
