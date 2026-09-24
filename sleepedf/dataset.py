"""Private evaluator truth preparation and signal-only Sleep-EDF access.

Only ``build_readiness`` opens Hypnograms.  Training/inference callers use
``iter_signal_epochs`` with the public readiness record and PSG alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyedflib

from .audit import _identity
from .contracts import content_id, inventory_id, json_text, read_json
from .readers import (PHYSIOLOGICAL_CHANNELS, read_annotation_pair,
                      signal_metadata, verify_physiological_signals)
from .timing import EPOCH_US, project_annotations


TRUTH_SCHEMA = "sleepedf-truth-v1"
READINESS_SCHEMA = "2.0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Invalid source path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Source path unavailable or outside data root")
    return path


def _checksum_entries(root: Path, trusted_sha256: str) -> dict[str, str]:
    manifest = root / "SHA256SUMS.txt"
    if _sha256_file(manifest) != trusted_sha256.lower():
        raise ValueError("Local checksum manifest differs from independently verified source")
    entries = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError("Malformed source checksum entry")
        if relative in entries:
            raise ValueError("Duplicate source checksum entry")
        _source_path(root, relative)
        entries[relative] = digest.lower()
    if len(entries) != 398:
        raise ValueError("Expected complete 398-entry official Sleep-EDF manifest")
    return entries


def _truth_arrays(record: dict, timing: dict) -> dict[str, np.ndarray]:
    epochs = timing["epochs"]
    indices = np.asarray([e["index"] for e in epochs], dtype=np.int64)
    onsets = np.asarray([e["onset_us"] / 1_000_000 for e in epochs], dtype=np.float64)
    valid = np.asarray([e["valid"] for e in epochs], dtype=np.bool_)
    labels = np.asarray([e["label"] if e["valid"] else -1 for e in epochs], dtype=np.int8)
    if not (len(indices) == len(onsets) == len(valid) == len(labels)):
        raise ValueError("Truth arrays disagree in length")
    if not np.array_equal(valid, labels != -1):
        raise ValueError("Invalid mask and labels disagree")
    return {"participant_id": np.asarray(record["participant_id"]),
            "recording_id": np.asarray(record["recording_id"]),
            "epoch_index": indices, "onset_seconds": onsets,
            "reference_label": labels, "valid_mask": valid}


def _manifest_identity(report: dict) -> str:
    data = {key: value for key, value in report.items() if key != "manifest_id"}
    return content_id(data)


def build_readiness(inventory_path: Path, data_root: Path, truth_root: Path,
                    trusted_manifest_sha256: str, *,
                    recording_ids: set[str] | None = None,
                    verify_signals: bool = True) -> dict:
    """Validate source hashes, timed labels and optionally all signal samples.

    Caller writes the returned public report separately.  Truth NPZ files are
    created only in ``truth_root``; the public report contains no stage labels.
    A failed record remains in the report with explicit failed checks.
    """
    root = data_root.resolve()
    private = truth_root.resolve()
    if private.is_relative_to(root):
        raise ValueError("Evaluator truth cannot be saved inside source data")
    inventory = read_json(inventory_path)
    if not isinstance(inventory, dict) or inventory.get("report_kind") != "audit":
        raise ValueError("A structured Sleep-EDF inventory is required")
    if inventory.get("manifest_id") != inventory_id(inventory):
        raise ValueError("Inventory metadata identity differs from its content")
    entries = _checksum_entries(root, trusted_manifest_sha256)
    selected = [r for r in inventory["recordings"]
                if recording_ids is None or r["recording_id"] in recording_ids]
    if recording_ids is not None and {r["recording_id"] for r in selected} != recording_ids:
        raise ValueError("Requested recording identity absent from inventory")
    all_ids = [r["recording_id"] for r in inventory["recordings"]]
    publisher_edfs = {name for name in entries if name.lower().endswith(".edf")}
    inventory_edfs = {name for r in inventory["recordings"]
                      for name in (r.get("psg"), r.get("hypnogram")) if isinstance(name, str)}
    complete_scope = (recording_ids is None and len(all_ids) == 197
                      and len(set(all_ids)) == 197 and len(publisher_edfs) == 394
                      and publisher_edfs == inventory_edfs)
    report = {"schema_version": READINESS_SCHEMA, "producer": "sleepedf",
              "report_kind": "data-readiness", "data_root": str(root),
              "source_inventory_id": inventory.get("manifest_id"),
              "source_checksum_manifest_sha256": trusted_manifest_sha256.lower(),
              "complete_dataset_scope": complete_scope,
              "expected_recording_ids": sorted(all_ids),
              "content_hash_scope": "Selected PSG/Hypnogram EDF files compared bytewise to independently verified official SHA256SUMS.txt",
              "signal_check_scope": "All samples of four named physiological channels, chunked to at most 300 s per read" if verify_signals else "not_requested",
              "records": [], "errors": [], "warnings": [],
              "participant_count": 0}
    if recording_ids is None and not complete_scope:
        report["errors"].append({"code": "incomplete_inventory", "count": len(selected)})
    for record in selected:
        rid = record["recording_id"]
        public = {"recording_id": rid, "participant_id": record["participant_id"],
                  "cohort": record["cohort"], "night": record["night"],
                  "psg": record.get("psg"), "hypnogram": record.get("hypnogram"),
                  "truth_path": None,
                  "psg_sha256": None, "hypnogram_sha256": None,
                  "duration_seconds": None, "n_epochs": None, "channels": [],
                  "checks": {"identity_verified": False, "content_verified": False,
                             "signals_verified": False, "timing_verified": False},
                  "reader_evidence": None, "errors": [], "warnings": []}
        report["records"].append(public)
        try:
            evidence = record.get("evidence", {})
            if not evidence.get("pair_unique") or not evidence.get("metadata_verified") or record.get("identity_status") != "resolved":
                raise ValueError("Identity or pair not metadata verified")
            public["checks"]["identity_verified"] = True
            psg_relative, hyp_relative = record["psg"], record["hypnogram"]
            psg_identity, hyp_identity = _identity(Path(psg_relative)), _identity(Path(hyp_relative))
            if (psg_identity is None or hyp_identity is None
                    or psg_identity["role"] != "PSG" or hyp_identity["role"] != "Hypnogram"
                    or psg_identity["equipment"] != hyp_identity["equipment"]
                    or any(parsed[key] != record[key] for parsed in (psg_identity, hyp_identity)
                           for key in ("recording_id", "participant_id", "cohort", "night"))):
                raise ValueError("Source filenames conflict with verified recording identity")
            psg = _source_path(root, psg_relative)
            hyp = _source_path(root, hyp_relative)
            psg_hash, hyp_hash = _sha256_file(psg), _sha256_file(hyp)
            public["psg_sha256"], public["hypnogram_sha256"] = psg_hash, hyp_hash
            if entries.get(psg_relative) != psg_hash or entries.get(hyp_relative) != hyp_hash:
                raise ValueError("EDF content differs from verified source checksum")
            public["checks"]["content_verified"] = True
            metadata = signal_metadata(psg)
            public["duration_seconds"] = metadata["duration_seconds"]
            public["channels"] = metadata["channels"]
            annotations = read_annotation_pair(psg, hyp)
            timing = project_annotations(rid, metadata["duration_seconds"],
                                         annotations["intervals"],
                                         offset_seconds=annotations["offset_seconds"])
            public["n_epochs"] = len(timing["epochs"])
            public["reader_evidence"] = {"signal_reader": metadata["reader"],
                                         "annotation_reader": annotations["reader"],
                                         "compatibility_fallback": annotations["compatibility_fallback"],
                                         "offset_seconds": annotations["offset_seconds"],
                                         "offset_evidence": annotations.get("offset_evidence")}
            public["checks"]["timing_verified"] = True
            reasons = Counter(reason for epoch in timing["epochs"]
                              for reason in epoch["invalid_reasons"])
            public["timing_summary"] = {"valid_epochs": sum(e["valid"] for e in timing["epochs"]),
                                        "invalid_reason_counts": dict(sorted(reasons.items())),
                                        "off_record_intervals": len(timing["off_record_intervals"]),
                                        "trailing_microseconds": (timing["incomplete_trailing_epoch"] or {}).get("duration_us", 0),
                                        "timing_tolerance_us": timing["timing_tolerance_us"]}
            if reasons or timing["off_record_intervals"] or timing["incomplete_trailing_epoch"]:
                public["warnings"].append("timing_masks_or_boundaries_present")
            if verify_signals:
                verified = verify_physiological_signals(psg)
                public["signals_verified"] = verified
                public["checks"]["signals_verified"] = True
            arrays = _truth_arrays(record, timing)
            private.mkdir(parents=True, exist_ok=True)
            truth_path = private / f"{rid}.npz"
            sidecar_path = private / f"{rid}.json"
            if truth_path.exists() or sidecar_path.exists():
                if not truth_path.is_file() or not sidecar_path.is_file():
                    raise ValueError("Partial existing truth artifact requires review")
                previous = read_json(sidecar_path)
                if (previous.get("source_psg_sha256") != psg_hash
                        or previous.get("source_hypnogram_sha256") != hyp_hash
                        or previous.get("payload_sha256") != _sha256_file(truth_path)):
                    raise ValueError("Existing truth provenance differs")
                with np.load(truth_path, allow_pickle=False) as saved:
                    if set(saved.files) != set(arrays) or any(
                        not np.array_equal(saved[key], value) for key, value in arrays.items()
                    ):
                        raise ValueError("Existing truth values differ")
            else:
                with tempfile.NamedTemporaryFile(dir=private, prefix=".truth-",
                                                 suffix=".npz", delete=False) as handle:
                    temporary = Path(handle.name)
                try:
                    np.savez_compressed(temporary, **arrays)
                    os.replace(temporary, truth_path)
                finally:
                    temporary.unlink(missing_ok=True)
            payload_hash = _sha256_file(truth_path)
            sidecar = {"schema_version": TRUTH_SCHEMA, "producer": "sleepedf",
                       "recording_id": rid, "participant_id": record["participant_id"],
                       "payload_sha256": payload_hash,
                       "source_psg_sha256": psg_hash,
                       "source_hypnogram_sha256": hyp_hash,
                       "timing_tolerance_us": timing["timing_tolerance_us"],
                       "epoch_seconds": 30,
                       "reader_evidence": public["reader_evidence"],
                       "valid_epochs": public["timing_summary"]["valid_epochs"],
                       "n_epochs": len(timing["epochs"])}
            if sidecar_path.exists():
                if read_json(sidecar_path) != sidecar:
                    raise ValueError("Existing truth sidecar differs")
            else:
                sidecar_path.write_text(json_text(sidecar), encoding="utf-8")
            public["truth_path"] = str(truth_path)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            # Never copy exception text: established readers can include EDF header fields.
            public["errors"].append({"code": "data_validation_failed",
                                     "stage": next((name for name, done in public["checks"].items() if not done), "truth_write")})
            report["errors"].append({"recording_id": rid, "code": "data_validation_failed"})
    report["participant_count"] = len({r["participant_id"] for r in report["records"]
                                       if all(r["checks"].values()) and not r["errors"]})
    report["summary"] = {"recordings": len(report["records"]),
                         "ready_recordings": sum(all(r["checks"].values()) and not r["errors"] for r in report["records"]),
                         "participants_ready": report["participant_count"],
                         "compatibility_reader_recordings": sum(bool((r["reader_evidence"] or {}).get("compatibility_fallback")) for r in report["records"]),
                         "truth_labels_public": False}
    report["manifest_id"] = _manifest_identity(report)
    return report


def iter_signal_epochs(record: dict, data_root: Path,
                       channel_names: tuple[str, ...] = PHYSIOLOGICAL_CHANNELS):
    """Yield all complete native-rate physical PSG epochs; never open labels."""
    root = data_root.resolve()
    psg = _source_path(root, record["psg"])
    if not record.get("checks", {}).get("signals_verified"):
        raise ValueError("PSG signals have not passed readiness checks")
    with pyedflib.EdfReader(str(psg), annotations_mode=pyedflib.DO_NOT_READ_ANNOTATIONS) as reader:
        labels = reader.getSignalLabels()
        if len(labels) != len(set(labels)) or not set(channel_names).issubset(labels):
            raise ValueError("Required physiological channel unavailable")
        n_epochs = int(record["n_epochs"])
        for epoch in range(n_epochs):
            channels = {}
            for name in channel_names:
                index = labels.index(name)
                unit = reader.getPhysicalDimension(index).strip()
                if unit not in ("uV", "µV", "μV"):
                    raise ValueError("Physiological unit absent or unsupported")
                rate = float(reader.getSampleFrequency(index))
                first, count = round(epoch * 30 * rate), round(30 * rate)
                if abs(count / rate - 30) > 1e-6:
                    raise ValueError("Epoch does not align with native sampling rate")
                values = reader.readSignal(index, start=first, n=count)
                if len(values) != count or not np.isfinite(values).all():
                    raise ValueError("PSG epoch incomplete or non-finite")
                channels[name] = {"samples_uv": values, "sample_rate_hz": rate, "unit": "uV"}
            yield {"participant_id": record["participant_id"],
                   "recording_id": record["recording_id"],
                   "epoch_index": epoch, "onset_seconds": epoch * 30.0,
                   "channels": channels}
