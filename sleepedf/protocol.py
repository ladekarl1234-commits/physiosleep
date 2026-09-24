"""Freeze the common benchmark before results and restrict fitting to development."""
from __future__ import annotations

from pathlib import Path

from .contracts import CLASS_ORDER, content_id, read_json
from .research import artifact_id, atomic_json, append_event, file_sha256
from .splits import freeze_split, validate_split_v2


def freeze_protocol(root: Path, readiness_path: Path, data_root: Path) -> dict:
    readiness = read_json(readiness_path)
    split_path = root / "research" / "split-v2.json"
    split = freeze_split(readiness, data_root, split_path)
    registry_path = root / "research" / "baseline_registry.json"
    registry = read_json(registry_path)
    if registry.get("registry_hash") != artifact_id(registry, "registry_hash"):
        raise ValueError("Registry identity mismatch")
    audit_recordings = {}
    for phase, group in (("A", "audit_a"), ("B", "audit_b")):
        records = {}
        for rec in readiness["records"]:
            if rec["participant_id"] not in split["participants"][group]:
                continue
            truth = Path(rec["truth_path"])
            meta = read_json(truth.with_suffix(".json"))
            if meta["payload_sha256"] != file_sha256(truth):
                raise ValueError("Private truth payload identity mismatch")
            records[rec["recording_id"]] = {
                "participant_id": rec["participant_id"], "n_epochs": rec["n_epochs"],
                "psg_sha256": rec["psg_sha256"], "hypnogram_sha256": rec["hypnogram_sha256"],
                "truth_payload_sha256": meta["payload_sha256"],
                "truth_sidecar_sha256": file_sha256(truth.with_suffix(".json"))}
        audit_recordings[phase] = records
    protocol = {
        "schema_version": "1.0", "protocol_id": "physiosleep-v1-approved",
        "split_id": split["split_id"], "readiness_id": readiness["manifest_id"],
        "split_path": str(split_path.resolve()), "readiness_path": str(readiness_path.resolve()),
        "split_sha256": file_sha256(split_path), "readiness_sha256": file_sha256(readiness_path),
        "registry_hash": registry["registry_hash"], "registry_sha256": file_sha256(registry_path),
        "audit_recordings": audit_recordings,
        "class_order": list(CLASS_ORDER), "epoch_seconds": 30,
        "interval": "all complete half-open 30-second PSG epochs; no reference-guided cropping",
        "mask": "fixed reference-valid mask; invalid positions remain; missing prediction fails",
        "metric": "pooled_epoch_fixed_five_class_macro_f1_zero_denominator_zero",
        "point_margin": {"numerator": 1, "denominator": 50, "comparison": ">="},
        "bootstrap": {"draws": 10000, "rng": "PCG64", "seeds": {"A": 2026092301, "B": 2026092302},
                      "strata": ["SC", "ST"], "cluster": "participant_all_nights",
                      "multiplicity": "two_sided_Bonferroni_all_frozen_comparators", "quantile": "linear"},
        "gate_requires": ["integrity", "all_11_slots_executed_clean_or_proven_duplicate",
                          "every_exact_point_delta_at_least_0.02", "every_simultaneous_lower_bound_positive",
                          "independent_recomputation_attestation"],
        "audit_policy": "one preregistered A batch; B reserved for phase2; failed audit never reused",
        "access_boundary": "procedural_shared_account; metadata-only hashes read by modeling controller",
        "development_seeds": [17, 43, 101], "phase2_training_pool": "original_development_only",
        "resources": {"gpu_jobs": 1, "cpu_threads": 4, "memory_limit_gib": 10,
                      "minimum_free_disk_gib": 20, "minimum_available_memory_gib": 4},
    }
    protocol["protocol_hash"] = content_id(protocol)
    atomic_json(root / "research" / "protocol-v1.json", protocol, immutable=True)
    append_event(root / "runs" / "exposure.jsonl", {
        "event": "data_protocol_frozen", "protocol_hash": protocol["protocol_hash"],
        "audit_A_model_selection_exposed": False, "audit_B_model_selection_exposed": False,
        "structural_validation": "all annotations validated by data role; no audit labels supplied to model selection"})
    return protocol


def load_protocol(root: Path) -> tuple[dict, dict, dict]:
    protocol = read_json(root / "research" / "protocol-v1.json")
    if protocol.get("protocol_hash") != artifact_id(protocol, "protocol_hash"):
        raise ValueError("Protocol identity mismatch")
    for key in ("split", "readiness"):
        if file_sha256(Path(protocol[key + "_path"])) != protocol[key + "_sha256"]:
            raise ValueError("Frozen data or split changed")
    registry_path = root / "research" / "baseline_registry.json"
    if file_sha256(registry_path) != protocol["registry_sha256"]:
        raise ValueError("Frozen baseline registry changed")
    readiness = read_json(Path(protocol["readiness_path"]))
    split = read_json(Path(protocol["split_path"]))
    validate_split_v2(split, readiness)
    return protocol, split, readiness


def development_records(root: Path) -> tuple[dict, dict, list[dict]]:
    protocol, split, readiness = load_protocol(root)
    ids = set(split["participants"]["development"])
    records = [dict(rec) for rec in readiness["records"] if rec["participant_id"] in ids]
    truth_manifest_path = root / "research" / "development-truth-v1.json"
    if truth_manifest_path.exists():
        frozen = read_json(truth_manifest_path)
        if (frozen.get("manifest_id") != artifact_id(frozen, "manifest_id") or
            frozen.get("protocol_hash") != protocol["protocol_hash"] or frozen.get("split_id") != split["split_id"] or
            set(frozen["records"]) != {r["recording_id"] for r in records}):
            raise ValueError("Development truth freeze differs from protocol/split")
        for rec in records:
            rec["frozen_truth"] = frozen["records"][rec["recording_id"]]
            rec["truth_manifest_id"] = frozen["manifest_id"]
    return protocol, split, records


def freeze_development_truth(root: Path) -> dict:
    protocol, split, readiness = load_protocol(root)
    records = {}
    for rec in readiness["records"]:
        if rec["participant_id"] not in split["participants"]["development"]:
            continue
        path = Path(rec["truth_path"])
        meta = read_json(path.with_suffix(".json"))
        if (meta["source_psg_sha256"] != rec["psg_sha256"] or
            meta["source_hypnogram_sha256"] != rec["hypnogram_sha256"] or
            meta["n_epochs"] != rec["n_epochs"] or meta["recording_id"] != rec["recording_id"] or
            meta["participant_id"] != rec["participant_id"] or meta["payload_sha256"] != file_sha256(path)):
            raise ValueError("Development truth disagrees with source/grid evidence")
        records[rec["recording_id"]] = {"payload_sha256": file_sha256(path),
                                      "sidecar_sha256": file_sha256(path.with_suffix(".json"))}
    result = {"schema_version": "1.0", "protocol_hash": protocol["protocol_hash"],
              "split_id": split["split_id"], "records": records}
    result["manifest_id"] = content_id(result)
    atomic_json(root / "research" / "development-truth-v1.json", result, immutable=True)
    return result


def load_development_truth(record: dict, split: dict) -> dict:
    from .predictions import load_truth
    if record["participant_id"] not in split["participants"]["development"]:
        raise ValueError("Training process cannot load audit truth")
    path = Path(record["truth_path"])
    frozen = record.get("frozen_truth")
    if (not isinstance(frozen, dict) or file_sha256(path) != frozen.get("payload_sha256") or
        file_sha256(path.with_suffix(".json")) != frozen.get("sidecar_sha256")):
        raise ValueError("Development truth requires the pre-training immutable hash binding")
    meta, arrays = load_truth(path)
    if arrays["participant_id"] != record["participant_id"] or arrays["recording_id"] != record["recording_id"]:
        raise ValueError("Development truth identity mismatch")
    if (meta["source_psg_sha256"] != record["psg_sha256"] or
        meta["source_hypnogram_sha256"] != record["hypnogram_sha256"] or
        len(arrays["epoch_index"]) != record["n_epochs"]):
        raise ValueError("Development truth source or full grid differs from readiness")
    return arrays


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--readiness", type=Path, default=Path("reports/data-readiness.json"))
    parser.add_argument("--data-root", type=Path, default=Path("sleep-edf-database-expanded-1.0.0"))
    args = parser.parse_args()
    result = freeze_protocol(args.root.resolve(), args.readiness.resolve(), args.data_root.resolve())
    print(result["protocol_hash"])
