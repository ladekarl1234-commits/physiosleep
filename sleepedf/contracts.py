"""Label, serialization, identity, and participant split contracts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import re

CLASS_ORDER = ("W", "N1", "N2", "N3", "REM")
EPOCH_SECONDS = 30
_STAGES = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}


def map_annotation(description: str) -> dict:
    """Map a label only; callers must retain its original onset and duration."""
    if not isinstance(description, str):
        raise ValueError("Annotation description must be a string")
    if description in _STAGES:
        return {"label": _STAGES[description], "valid": True, "reason": None}
    reason = {
        "Movement time": "movement",
        "Sleep stage M": "movement",
        "Sleep stage ?": "unscored",
    }.get(description, "unrecognized_annotation")
    return {"label": None, "valid": False, "reason": reason}


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _reject_constant(value: str):
    raise ValueError("Non-finite JSON number is forbidden: " + value)


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def read_json(path: Path) -> object:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant,
                       object_pairs_hook=_unique_object)
    json_text(value)  # Also rejects exponent overflow such as 1e999.
    return value


def content_id(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def inventory_id(report: dict) -> str:
    """Hash inventory metadata, never claim verification of raw EDF bytes."""
    canonical = {k: v for k, v in report.items() if k != "manifest_id"}
    canonical["scope"] = {k: v for k, v in report.get("scope", {}).items() if k != "data_root"}
    return content_id(canonical)


def code_identity(root: Path) -> dict:
    paths = sorted(set(root.glob("sleepedf/*.py")) | set(root.glob("tests/*.py")) |
                   {p for p in (root / "pyproject.toml", root / "uv.lock") if p.exists()})
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in paths if p.is_file()}
    return {"kind": "source_content_sha256", "value": content_id(files), "files": files,
            "git_revision": None, "includes_raw_data": False}


def _finite_number(value: object, positive: bool = False) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and (not positive or value > 0)
    except OverflowError:
        return False


def _valid_header(header: object, role: str) -> bool:
    if not isinstance(header, dict) or type(header.get("data_records")) is not int:
        return False
    if role == "hypnogram":
        return (header.get("kind") == "annotation_only" and header.get("channels") == [] and
                header.get("duration_seconds") is None and
                _finite_number(header.get("data_record_seconds")) and
                header["data_record_seconds"] >= 0 and header["data_records"] >= 0)
    duration = header.get("duration_seconds")
    record_seconds = header.get("data_record_seconds")
    channels = header.get("channels")
    if (header.get("kind") != "sampled_signals" or
        not _finite_number(duration, positive=True) or
        not _finite_number(record_seconds, positive=True) or
        header["data_records"] <= 0 or not isinstance(channels, list) or not channels):
        return False
    try:
        if not math.isclose(duration, record_seconds * header["data_records"], rel_tol=1e-7):
            return False
    except OverflowError:
        return False
    for channel in channels:
        if not isinstance(channel, dict):
            return False
        samples = channel.get("samples_per_data_record")
        rate = channel.get("sample_rate_hz")
        pmin, pmax = channel.get("physical_min"), channel.get("physical_max")
        dmin, dmax = channel.get("digital_min"), channel.get("digital_max")
        if (not isinstance(channel.get("label"), str) or not channel["label"].strip() or
            not isinstance(channel.get("unit"), str) or
            type(samples) is not int or samples <= 0 or
            not _finite_number(rate, positive=True) or
            not _finite_number(pmin) or not _finite_number(pmax) or pmin == pmax or
            type(dmin) is not int or type(dmax) is not int or dmin >= dmax):
            return False
        if not math.isclose(rate, samples / record_seconds, rel_tol=1e-7):
            return False
    return True


def validate_split(split: dict, inventory: dict) -> dict:
    if not isinstance(split, dict) or not isinstance(inventory, dict):
        raise ValueError("Split and inventory must be JSON objects")
    required = {"schema_version", "protocol_id", "inventory_id", "seed", "method", "participants"}
    if set(split) != required or split.get("schema_version") != "1.0":
        raise ValueError("Split must use the documented schema_version 1.0 and exact fields")
    for key in ("protocol_id", "method"):
        if not isinstance(split[key], str) or not split[key].strip():
            raise ValueError(f"Split {key} must be a nonempty string")
    if type(split["seed"]) is not int:
        raise ValueError("Split seed must be an integer")
    if inventory.get("report_kind") != "audit" or inventory.get("schema_version") != "1.0":
        raise ValueError("Split requires a schema 1.0 audit report")
    scope = inventory.get("scope")
    records = inventory.get("recordings")
    if not isinstance(scope, dict) or scope.get("headers_requested") is not True or (
        type(scope.get("headers_completed")) is not int or
        not isinstance(records, list) or scope["headers_completed"] != 2 * len(records)
    ) or not isinstance(scope.get("reader_version"), str) or not scope["reader_version"]:
        raise ValueError("Split requires completed EDF header checks")
    if inventory.get("manifest_id") != inventory_id(inventory):
        raise ValueError("Inventory metadata digest does not match its content; rerun audit")
    if split["inventory_id"] != inventory["manifest_id"]:
        raise ValueError("Split refers to a different inventory; never regenerate it silently")
    if not isinstance(inventory.get("errors"), list) or inventory["errors"]:
        raise ValueError("Inventory has unresolved audit errors")
    if not records:
        raise ValueError("Inventory contains no recordings")
    files = inventory.get("files")
    if not isinstance(files, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("path"), str) or
        type(item.get("size_bytes")) is not int or item["size_bytes"] < 0
        for item in files
    ):
        raise ValueError("Inventory requires a file size inventory")
    file_paths = {item["path"] for item in files}
    if len(file_paths) != len(files):
        raise ValueError("Inventory has duplicate file paths")
    known = set()
    record_ids = set()
    pair_paths = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Inventory recording must be an object")
        rid = record.get("recording_id")
        match = re.fullmatch(r"(SC4|ST7)(\d{2})([12])", rid) if isinstance(rid, str) else None
        if match is None or rid in record_ids:
            raise ValueError("Inventory has an invalid or duplicate recording ID")
        record_ids.add(rid)
        participant = record.get("participant_id")
        if (record.get("identity_status") != "resolved" or
            participant != f"{match.group(1)[:2]}:{match.group(2)}" or
            record.get("cohort") != match.group(1)[:2] or
            type(record.get("night")) is not int or record["night"] != int(match.group(3))):
            raise ValueError("Inventory contains unresolved participant identity")
        evidence = record.get("evidence")
        metadata = record.get("metadata")
        if (not isinstance(evidence, dict) or any(
            evidence.get(key) is not True
            for key in ("pair_unique", "metadata_verified", "headers_verified")
        ) or not isinstance(metadata, dict) or type(metadata.get("lights_off_available")) is not bool or
            "condition" not in metadata or (
                metadata.get("condition") is not None if record["cohort"] == "SC" else
                metadata.get("condition") not in ("placebo", "temazepam")
            )):
            raise ValueError("Inventory lacks verified pair, metadata, or header evidence")
        alignment = record.get("start_alignment")
        if (not isinstance(alignment, dict) or
            type(alignment.get("equal")) is not bool or
            not _finite_number(alignment.get("offset_seconds")) or
            alignment["equal"] != (alignment["offset_seconds"] == 0)):
            raise ValueError("Inventory lacks a resolved PSG/Hypnogram start offset")
        equipment = None
        for role in ("psg", "hypnogram"):
            path = record.get(role)
            if (not isinstance(path, str) or not path or
                PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts or
                "\\" in path or ":" in path or path in pair_paths or path not in file_paths or
                not _valid_header(record.get(f"{role}_header"), role)):
                raise ValueError("Inventory has a missing, reused, or unverified EDF pair")
            suffix = r"0-PSG\.edf" if role == "psg" else r"[A-Z]-Hypnogram\.edf"
            path_match = re.fullmatch(re.escape(rid) + r"([EFGJ])" + suffix,
                                      PurePosixPath(path).name)
            if (path_match is None or path_match.group(1) not in (
                "EFG" if record["cohort"] == "SC" else "J") or
                (equipment is not None and path_match.group(1) != equipment)):
                raise ValueError("Inventory EDF paths disagree with recording identity")
            equipment = path_match.group(1)
            pair_paths.add(path)
        known.add(participant)
    groups = split["participants"]
    if not isinstance(groups, dict) or set(groups) != {"train", "validation", "test"}:
        raise ValueError("Split participants must contain train, validation, and test")
    assigned = set()
    for name in ("train", "validation", "test"):
        ids = groups[name]
        if not isinstance(ids, list) or any(not isinstance(p, str) for p in ids):
            raise ValueError(f"{name} must be a list of participant IDs")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate participant in {name}")
        if assigned.intersection(ids):
            raise ValueError("Participant overlap across splits")
        if set(ids) - known:
            raise ValueError(f"Unknown participant in {name}")
        assigned.update(ids)
    if assigned != known:
        raise ValueError("Unassigned participants: inventory changed or split is incomplete")
    if not groups["train"] or not groups["validation"] or not groups["test"]:
        raise ValueError("Train, validation, and test must each contain participants")
    return {"valid": True, "split_id": content_id(split), "inventory_id": split["inventory_id"],
            "protocol_id": split["protocol_id"],
            "participant_counts": {k: len(v) for k, v in groups.items()}}
