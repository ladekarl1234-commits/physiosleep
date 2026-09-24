"""Approved metadata-only participant allocation and explicit sequential audits."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re

from .contracts import content_id
from .research import artifact_id, atomic_json, file_sha256

SEED = 20260923
EXPOSED = ("SC:36", "ST:01")
CHECKS = ("identity_verified", "content_verified", "signals_verified", "timing_verified")


def demographics(data_root: Path) -> dict:
    import xlrd
    people = {}
    sources = {}
    for cohort, first, age_col, sex_col in (("SC", 1, 2, 3), ("ST", 2, 1, 2)):
        path = data_root / f"{cohort}-subjects.xls"
        sources[path.name] = file_sha256(path)
        sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
        for i in range(first, sheet.nrows):
            row = sheet.row_values(i)
            if not any(v != "" for v in row):
                continue
            if row[0] != int(row[0]):
                raise ValueError("Noninteger metadata participant")
            pid = f"{cohort}:{int(row[0]):02d}"
            age = row[age_col]
            if type(age) not in (int, float) or not math.isfinite(age) or not 0 <= age <= 120:
                raise ValueError("Missing/invalid age requires an explicit protocol amendment")
            sex = row[sex_col]
            if sex not in (1, 2):
                raise ValueError("Unknown source sex code")
            sex = ("F" if sex == 1 else "M") if cohort == "SC" else ("M" if sex == 1 else "F")
            entry = {"participant_id": pid, "cohort": cohort, "age": float(age), "sex": sex}
            if pid in people and people[pid] != entry:
                raise ValueError("Conflicting participant demographics across nights")
            people[pid] = entry
    value = {"participants": people, "source_sha256": sources}
    value["metadata_id"] = content_id(value)
    return value


def validate_readiness(manifest: dict) -> set[str]:
    if (manifest.get("schema_version") != "2.0" or
        manifest.get("report_kind") != "data-readiness" or
        manifest.get("manifest_id") != artifact_id(manifest, "manifest_id") or
        manifest.get("errors") != []):
        raise ValueError("A complete, consistent data-readiness v2 artifact is required")
    records = manifest.get("records")
    if (not isinstance(records, list) or len(records) != 197 or
        manifest.get("complete_dataset_scope") is not True or
        len(set(manifest.get("expected_recording_ids", []))) != 197):
        raise ValueError("Readiness must cover all 197 original recordings")
    people, record_ids, paths = set(), set(), set()
    for rec in records:
        rid = rec.get("recording_id")
        match = re.fullmatch(r"(SC4|ST7)(\d{2})([12])", rid or "")
        if not match or rid in record_ids:
            raise ValueError("Invalid or duplicate recording identity")
        pid = f"{match[1][:2]}:{match[2]}"
        if rec.get("participant_id") != pid or rec.get("cohort") != match[1][:2] or rec.get("night") != int(match[3]):
            raise ValueError("Recording and participant identity disagree")
        if rec.get("errors") != [] or any(rec.get("checks", {}).get(k) is not True for k in CHECKS):
            raise ValueError("Unresolved recording evidence")
        if not rec.get("reader_evidence") or not rec.get("channels"):
            raise ValueError("Missing reader/channel evidence")
        for key in ("psg_sha256", "hypnogram_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", rec.get(key, "")):
                raise ValueError("Missing source content digest")
        if type(rec.get("n_epochs")) is not int or rec["n_epochs"] <= 0:
            raise ValueError("Missing complete epoch grid")
        duration = rec.get("duration_seconds", 0)
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or rec["n_epochs"] != math.floor(duration / 30):
            raise ValueError("Epoch grid must cover every complete PSG epoch")
        psg = rec.get("psg")
        if not isinstance(psg, str) or Path(psg).name[:6] != rid or psg in paths:
            raise ValueError("Reused or mismatched PSG path")
        paths.add(psg)
        record_ids.add(rid)
        people.add(pid)
    if record_ids != set(manifest["expected_recording_ids"]):
        raise ValueError("Missing or extra recording against complete source inventory")
    return people


def _rank(pid: str, prefix: str = "physiosleep-split-v1") -> bytes:
    return hashlib.sha256(f"{prefix}|{SEED}|{pid}".encode()).digest()


def _age_sorted(ids: list[str], people: dict) -> list[str]:
    return sorted(ids, key=lambda p: (people[p]["age"], p))


def _folds(dev: list[str], people: dict) -> list[dict]:
    validation = [[] for _ in range(5)]
    for cohort, residual_folds in (("SC", [0]), ("ST", [1, 2, 3, 4])):
        ids = _age_sorted([p for p in dev if p.startswith(cohort + ":")], people)
        full, remainder = divmod(len(ids), 5)
        if remainder != len(residual_folds):
            raise ValueError("Approved fold quotas require 46 SC and 14 ST development participants")
        for block in range(full):
            ranked = sorted(ids[block*5:(block+1)*5], key=lambda p: _rank(p, f"fold-{cohort}-{block}"))
            fold_order = sorted(range(5), key=lambda f: _rank(str(f), f"fold-order-{cohort}-{block}"))
            for fold_id, pid in zip(fold_order, ranked):
                validation[fold_id].append(pid)
        remaining = sorted(ids[full*5:], key=lambda p: _rank(p, f"fold-tail-{cohort}"))
        for fold_id, pid in zip(residual_folds, remaining):
            validation[fold_id].append(pid)
    return [{"fold_id": i, "train": sorted(set(dev) - set(v)), "validation": sorted(v)}
            for i, v in enumerate(validation)]


def make_split(manifest: dict, metadata: dict) -> dict:
    known = validate_readiness(manifest)
    if metadata.get("metadata_id") != artifact_id(metadata, "metadata_id"):
        raise ValueError("Demographic metadata identity mismatch")
    people = metadata["participants"]
    if known != set(people) or (sum(p.startswith("SC:") for p in known), sum(p.startswith("ST:") for p in known)) != (78, 22):
        raise ValueError("Approved cohort changed; review eligibility before any split amendment")
    partitions = {"development": [], "audit_a": [], "audit_b": []}
    for cohort, nblocks in (("SC", 16), ("ST", 4)):
        ids = _age_sorted([p for p in known if p.startswith(cohort + ":")], people)
        size, extra = divmod(len(ids), nblocks)
        offset = 0
        for block in range(nblocks):
            count = size + (block < extra)
            members = ids[offset:offset+count]
            offset += count
            eligible = sorted(set(members) - set(EXPOSED), key=_rank)
            if len(eligible) < 2:
                raise ValueError("Exposure history prevents approved audit allocation")
            a, b = eligible[:2]
            partitions["audit_a"].append(a)
            partitions["audit_b"].append(b)
            partitions["development"].extend(p for p in members if p not in (a, b))
    partitions = {k: sorted(v) for k, v in partitions.items()}
    result = {"schema_version": "2.0", "protocol_id": "physiosleep-v1-approved",
              "readiness_id": manifest["manifest_id"], "metadata_id": metadata["metadata_id"],
              "seed": SEED, "method": "cohort_age_blocks_sha256_v1",
              "forced_development": list(EXPOSED), "participants": partitions,
              "folds": _folds(partitions["development"], people),
              "access_boundary": "procedural_shared_account",
              "phase2_training_pool": "original_development_only"}
    result["split_id"] = artifact_id(result, "split_id")
    validate_split_v2(result, manifest)
    return result


def validate_split_v2(split: dict, manifest: dict) -> dict:
    known = validate_readiness(manifest)
    if (split.get("schema_version") != "2.0" or
        split.get("split_id") != artifact_id(split, "split_id") or
        split.get("readiness_id") != manifest["manifest_id"]):
        raise ValueError("Split version or artifact identity mismatch")
    if (split.get("protocol_id") != "physiosleep-v1-approved" or
        split.get("seed") != SEED or split.get("method") != "cohort_age_blocks_sha256_v1" or
        split.get("forced_development") != list(EXPOSED) or
        split.get("phase2_training_pool") != "original_development_only" or
        split.get("access_boundary") != "procedural_shared_account"):
        raise ValueError("Approved allocation or access contract changed")
    groups = split.get("participants", {})
    if set(groups) != {"development", "audit_a", "audit_b"}:
        raise ValueError("Explicit development/audit_a/audit_b groups required")
    seen = set()
    for group, expected in (("development", (46, 14)), ("audit_a", (16, 4)), ("audit_b", (16, 4))):
        ids = groups[group]
        if not isinstance(ids, list) or any(not isinstance(p, str) for p in ids):
            raise ValueError("Participant groups must be lists of IDs")
        if len(set(ids)) != len(ids) or seen.intersection(ids) or set(ids) - known:
            raise ValueError("Duplicate, overlapping, or unknown participant")
        if tuple(sum(p.startswith(c + ":") for p in ids) for c in ("SC", "ST")) != expected:
            raise ValueError("Approved cohort quotas changed")
        seen.update(ids)
    if seen != known or not set(EXPOSED) <= set(groups["development"]):
        raise ValueError("Incomplete split or previously exposed audit participant")
    folds = split.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("Five development folds required")
    held = []
    for i, fold in enumerate(folds):
        train, val = fold.get("train"), fold.get("validation")
        if not isinstance(train, list) or not isinstance(val, list):
            raise ValueError("Invalid fold membership")
        if (fold.get("fold_id") != i or len(train) != 48 or len(val) != 12 or
            len(set(train)) != 48 or len(set(val)) != 12 or set(train) & set(val) or
            set(train) | set(val) != set(groups["development"])):
            raise ValueError("Fold leakage or incomplete development coverage")
        if tuple(sum(p.startswith(c + ":") for p in val) for c in ("SC", "ST")) != ((10, 2) if i == 0 else (9, 3)):
            raise ValueError("Fixed development fold cohort quotas changed")
        held.extend(val)
    if len(set(held)) != 60 or set(held) != set(groups["development"]):
        raise ValueError("Each development participant must be held out exactly once")
    return {"valid": True, "split_id": split["split_id"],
            "participant_counts": {k: len(v) for k, v in groups.items()}}


def freeze_split(manifest: dict, data_root: Path, output: Path) -> dict:
    result = make_split(manifest, demographics(data_root))
    atomic_json(output, result, immutable=True)
    return result
