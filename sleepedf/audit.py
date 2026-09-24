"""Deterministic Sleep-EDF inventory; waveform and annotation payloads are not read."""

from __future__ import annotations

import importlib
import hashlib
import math
import re
from collections import defaultdict
from importlib import metadata as package_metadata
from pathlib import Path, PurePosixPath


_NAME = re.compile(
    r"(?P<recording>(?P<prefix>SC4|ST7)(?P<subject>\d{2})(?P<night>[12]))"
    r"(?P<equipment>[A-Z])(?P<scorer>[A-Z0-9])-(?P<role>PSG|Hypnogram)\.edf"
)


def _issue(report: dict, level: str, code: str, path: str, message: str) -> None:
    report[level].append({"code": code, "path": path, "message": message})


def _identity(path: Path) -> dict | None:
    match = _NAME.fullmatch(path.name)
    if not match:
        return None
    item = match.groupdict()
    cohort = item["prefix"][:2]
    if item["equipment"] not in ("EFG" if cohort == "SC" else "J"):
        return None
    if (item["role"] == "PSG" and item["scorer"] != "0") or (
        item["role"] == "Hypnogram" and not item["scorer"].isalpha()
    ):
        return None
    return {
        "recording_id": item["recording"],
        "participant_id": f"{cohort}:{item['subject']}",
        "cohort": cohort,
        "night": int(item["night"]),
        "equipment": item["equipment"],
        "role": item["role"],
    }


def _dependency(name: str, report: dict):
    try:
        return importlib.import_module(name)
    except ImportError:
        _issue(
            report, "errors", "missing_dependency", "",
            f"Install the locked project audit dependencies to use {name}: uv sync --frozen --extra audit --group dev.",
        )
        return None


def _header(path: Path, relative: str, role: str, reader_module, report: dict):
    try:
        with reader_module.EdfReader(
            str(path), annotations_mode=reader_module.DO_NOT_READ_ANNOTATIONS
        ) as reader:
            channels = []
            record_duration = float(reader.datarecord_duration)
            for index, signal in enumerate(reader.getSignalHeaders()):
                samples = int(reader.samples_in_datarecord(index))
                channel = {
                    "label": signal["label"].strip(),
                    "samples_per_data_record": samples,
                    "sample_rate_hz": samples / record_duration if record_duration > 0 else 0,
                    "unit": signal["dimension"].strip(),
                    "physical_min": float(signal["physical_min"]),
                    "physical_max": float(signal["physical_max"]),
                    "digital_min": int(signal["digital_min"]),
                    "digital_max": int(signal["digital_max"]),
                }
                if not all(math.isfinite(channel[key]) for key in (
                    "sample_rate_hz", "physical_min", "physical_max"
                )) or channel["sample_rate_hz"] <= 0 or samples <= 0:
                    raise ValueError("Non-finite or non-positive native channel rate/calibration")
                if channel["physical_min"] == channel["physical_max"] or (
                    channel["digital_min"] >= channel["digital_max"]
                ):
                    raise ValueError("Invalid calibration range")
                channels.append(channel)
                if not channel["unit"]:
                    _issue(report, "warnings", "missing_physical_unit", relative,
                           f"Channel {channel['label']} has no declared physical unit; do not infer one.")
            duration = float(reader.file_duration)
            if not math.isfinite(duration) or not math.isfinite(record_duration):
                raise ValueError("Non-finite duration")
            annotation_only = not channels and reader.filetype in (1, 3)
            if role == "PSG" and (not channels or record_duration <= 0):
                raise ValueError("PSG requires sampled channels and positive record duration")
            if role == "Hypnogram" and not annotation_only:
                raise ValueError("Expected an annotation-only EDF+ Hypnogram")
            result = {
                "kind": "annotation_only" if annotation_only else "sampled_signals",
                "channels": channels,
                "duration_seconds": None if annotation_only else duration,
                "data_record_seconds": record_duration,
                "data_records": int(reader.datarecords_in_file),
            }
            return result, reader.getStartdatetime()
    except (OSError, ValueError, RuntimeError, IndexError, OverflowError) as exc:
        # Reader exception messages can contain raw EDF fields; keep them out of reports.
        incompatible = "recordingfield" in str(exc).lower()
        _issue(report, "errors", "reader_incompatible_header" if incompatible else "invalid_edf_header", relative,
               "Established EDF reader rejected the recording field; review with another established reader."
               if incompatible else "EDF reader rejected header or file size; check format and truncation.")
        return None, None


def _metadata(root: Path, report: dict) -> None:
    workbooks = [(cohort, root / f"{cohort}-subjects.xls") for cohort in ("SC", "ST")]
    present = [(cohort, path) for cohort, path in workbooks if path.is_file()]
    report["metadata"] = {}
    for cohort, path in workbooks:
        if not path.is_file():
            _issue(report, "warnings", "metadata_unavailable", path.name,
                   "Participant/night metadata is absent; identity is based on standard filenames only.")
    if not present:
        return
    xlrd = _dependency("xlrd", report)
    if xlrd is None:
        return
    for cohort, path in present:
        try:
            book = xlrd.open_workbook(path)
            sheet = book.sheet_by_index(0)
            expected = {}
            first_row = 1 if cohort == "SC" else 2
            for index in range(first_row, sheet.nrows):
                row = sheet.row_values(index)
                if not any(value != "" for value in row):
                    continue
                participant = int(row[0])
                if row[0] != participant or not 0 <= participant <= 99:
                    raise ValueError("Invalid participant identifier")
                for night_column, lights_column, condition in (
                    [(1, 4, None)] if cohort == "SC" else [(3, 4, "placebo"), (5, 6, "temazepam")]
                ):
                    night = int(row[night_column])
                    if row[night_column] != night or night not in (1, 2):
                        raise ValueError("Invalid night identifier")
                    key = (f"{cohort}:{participant:02}", night)
                    if key in expected:
                        raise ValueError("Duplicate participant/night metadata")
                    expected[key] = {
                        "lights_off_available": row[lights_column] != "",
                        "condition": condition,
                    }
            matched = set()
            for recording in report["recordings"]:
                if recording["cohort"] != cohort:
                    continue
                key = (recording["participant_id"], recording["night"])
                if key not in expected:
                    recording["identity_status"] = "unresolved"
                    _issue(report, "errors", "identity_not_in_metadata", recording["recording_id"],
                           "Filename participant/night identity is absent from the supplied cohort metadata.")
                else:
                    matched.add(key)
                    recording["metadata"] = expected[key]
                    recording["evidence"]["metadata_verified"] = True
                    if recording["evidence"]["pair_unique"]:
                        recording["identity_status"] = "resolved"
            report["metadata"][cohort] = {
                "path": path.name,
                "recordings": len(expected),
                "participants": len({participant for participant, _ in expected}),
                "matched_recordings": len(matched),
                "lights_off_available": sum(item["lights_off_available"] for item in expected.values()),
                "lights_on_available": False,
            }
            if expected.keys() - matched:
                _issue(report, "warnings", "metadata_recordings_not_present", path.name,
                       f"{len(expected.keys() - matched)} metadata recording identities are not present locally.")
        except (ValueError, IndexError, OSError, xlrd.XLRDError):
            _issue(report, "errors", "invalid_subject_metadata", path.name,
                   "Cannot validate workbook participant/night rows; inspect its schema and duplicate identities.")
            for recording in report["recordings"]:
                if recording["cohort"] == cohort:
                    recording["identity_status"] = "unresolved"
                    recording["evidence"]["metadata_verified"] = False


def _manifests(root: Path, present: dict[str, int], report: dict,
               verify_checksums: bool, max_hash_bytes: int | None) -> None:
    report["source_manifests"] = {}
    for name in ("RECORDS", "SHA256SUMS.txt"):
        path = root / name
        if not path.is_file():
            _issue(report, "errors" if name == "SHA256SUMS.txt" and verify_checksums else "warnings",
                   "checksum_manifest_unavailable" if name == "SHA256SUMS.txt" and verify_checksums
                   else "source_manifest_unavailable", name,
                   "Local manifest absent; completeness and content could not be checked.")
            continue
        try:
            if not path.resolve().is_relative_to(root) or path.stat().st_size > 8_000_000:
                raise ValueError("Unsafe or oversized manifest")
            entries = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                if name == "SHA256SUMS.txt":
                    match = re.fullmatch(r"[a-fA-F0-9]{64}\s+\*?(.+)", line)
                    if not match:
                        raise ValueError("Invalid checksum manifest line")
                    digest, entry = line[:64].lower(), match.group(1)
                else:
                    entry = line.strip()
                item = PurePosixPath(entry)
                if (item.is_absolute() or item.as_posix() != entry or entry == "."
                        or ".." in item.parts or "\\" in entry or ":" in entry
                        or not (root / entry).resolve().is_relative_to(root)):
                    raise ValueError("Unsafe manifest path")
                entries.append((item.as_posix(), digest if name == "SHA256SUMS.txt" else None))
            paths = [entry for entry, _ in entries]
            duplicate = len(set(paths)) != len(paths)
            if duplicate:
                _issue(report, "errors", "duplicate_manifest_entry", name,
                       "Manifest contains repeated paths; inspect the source manifest.")
            missing = sorted(set(paths) - present.keys())
            manifest = {"entries": len(entries), "missing": missing,
                        "presence_verified": bool(entries) and not missing and not duplicate,
                        "content_hash_verified": False}
            report["source_manifests"][name] = manifest
            for entry in missing:
                _issue(report, "errors", "manifest_file_missing", entry,
                       f"Path listed in {name} is absent locally.")
            if name != "SHA256SUMS.txt":
                continue
            manifest["unlisted_local_files"] = sorted(set(present) - {name} - set(paths))
            checksums = []
            total_size = sum(present.get(entry, 0) for entry in set(paths))
            over_budget = max_hash_bytes is not None and total_size > max_hash_bytes
            checked_bytes = 0
            for entry, expected in entries:
                item = {"path": entry, "expected_sha256": expected,
                        "observed_sha256": None, "status": "not_requested"}
                if entry not in present:
                    item["status"] = "missing"
                elif verify_checksums and over_budget:
                    item["status"] = "omitted_budget"
                elif verify_checksums and not duplicate:
                    digest = hashlib.sha256()
                    try:
                        with (root / entry).open("rb") as stream:
                            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                                digest.update(chunk)
                                checked_bytes += len(chunk)
                        item["observed_sha256"] = digest.hexdigest()
                        item["status"] = "match_local_manifest" if digest.hexdigest() == expected else "mismatch"
                        if item["status"] == "mismatch":
                            _issue(report, "errors", "checksum_mismatch", entry,
                                   "File SHA-256 differs from the local checksum manifest.")
                    except OSError:
                        item["status"] = "read_error"
                        _issue(report, "errors", "checksum_read_error", entry,
                               "Could not read file for SHA-256 comparison.")
                checksums.append(item)
            if over_budget and verify_checksums:
                _issue(report, "warnings", "hash_budget_exceeded", name,
                       "Preflight total exceeds max_hash_bytes; no manifest files were hashed.")
            manifest["checksums"] = checksums
            manifest["hashes_checked"] = sum(item["observed_sha256"] is not None for item in checksums)
            manifest["hashes_matched"] = sum(item["status"] == "match_local_manifest" for item in checksums)
            manifest["hash_bytes_read"] = checked_bytes
            manifest["content_hash_verified"] = (
                verify_checksums and not duplicate and bool(entries)
                and all(item["status"] == "match_local_manifest" for item in checksums)
                and not manifest["unlisted_local_files"]
            )
            report["scope"]["content_hash_verified"] = manifest["content_hash_verified"]
            report["scope"]["hashes_checked"] = manifest["hashes_checked"]
            report["scope"]["hash_bytes_read"] = checked_bytes
            report["scope"]["checksum_entries"] = len(entries)
            report["scope"]["checksum_covered_edf_files"] = len({entry for entry, _ in entries
                                                                   if entry.lower().endswith(".edf")
                                                                   and entry in present})
        except (OSError, UnicodeError, ValueError):
            _issue(report, "errors", "invalid_source_manifest", name,
                   "Cannot read a valid local path manifest; inspect its formatting.")


def audit_dataset(root: Path, headers: bool = True, verify_checksums: bool = False,
                  max_hash_bytes: int | None = None) -> dict:
    """Inventory local files without reading waveform samples or EDF annotations.

    Native headers use pyedflib with annotation loading disabled and its default
    file-size validation. Filename-only inventory remains usable without it.
    """
    if max_hash_bytes is not None and (type(max_hash_bytes) is not int or max_hash_bytes < 0):
        raise ValueError("max_hash_bytes must be a non-negative integer or None")
    root = Path(root).resolve()
    versions = {}
    for name in ("pyedflib", "xlrd"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    report = {
        "report_kind": "audit", "schema_version": "1.0", "tool_version": "0.1.0",
        "scope": {"data_root": str(root), "headers_requested": headers,
                  "headers_completed": 0, "reader_version": versions["pyedflib"] if headers else None,
                  "dependency_versions": versions,
                  "verify_checksums_requested": verify_checksums,
                  "max_hash_bytes": max_hash_bytes, "hashes_checked": 0,
                  "hash_bytes_read": 0, "checksum_entries": 0,
                  "checksum_covered_edf_files": 0,
                  "checksum_manifest_trusted": False,
                  "waveform_reads": 0, "annotations_read": False, "content_hash_verified": False},
        "summary": {"files": 0, "edf_files": 0, "recordings": 0, "participants": 0, "cohorts": {}},
        "files": [], "recordings": [], "errors": [], "warnings": [],
    }
    if not root.is_dir():
        _issue(report, "errors", "dataset_not_found", "", "Dataset root is not a directory; pass an existing data root.")
        return report
    try:
        candidates = sorted((path for path in root.rglob("*") if path.is_file()),
                            key=lambda p: p.relative_to(root).as_posix())
        files = []
        for path in candidates:
            relative = path.relative_to(root).as_posix()
            if not path.resolve().is_relative_to(root):
                _issue(report, "errors", "unsafe_dataset_path", relative,
                       "File symlink escapes the dataset root and was excluded.")
                continue
            report["files"].append({"path": relative, "size_bytes": path.stat().st_size})
            files.append(path)
    except OSError:
        _issue(report, "errors", "dataset_unreadable", "", "Cannot enumerate dataset root; check local access.")
        return report
    edfs = [path for path in files if path.suffix.lower() == ".edf"]
    report["summary"].update(files=len(files), edf_files=len(edfs))
    if not edfs:
        _issue(report, "errors", "no_edf_files", "", "No EDF files found; check the selected dataset root.")
    groups = defaultdict(list)
    for path in edfs:
        identity = _identity(path)
        relative = path.relative_to(root).as_posix()
        if identity is None:
            _issue(report, "errors", "unresolved_identity", relative,
                   "Filename is not a supported standard Sleep-EDF identity; restore provenance or supply a reviewed mapping.")
            report["recordings"].append({
                "recording_id": None, "participant_id": None, "cohort": None,
                "night": None, "identity_status": "unresolved", "input_path": relative,
                "psg": None, "hypnogram": None, "psg_header": None, "hypnogram_header": None,
                "evidence": {"pair_unique": False, "metadata_verified": False, "headers_verified": False},
            })
        else:
            groups[identity["recording_id"]].append((path, identity))
    reader_module = _dependency("pyedflib", report) if headers and edfs else None
    for recording_id, members in sorted(groups.items()):
        identity = members[0][1]
        recording = {key: identity[key] for key in ("recording_id", "participant_id", "cohort", "night")}
        recording.update(identity_status="unresolved", psg=None, hypnogram=None,
                         psg_header=None, hypnogram_header=None,
                         evidence={"pair_unique": False, "metadata_verified": False,
                                   "headers_verified": False})
        if len({item["equipment"] for _, item in members}) > 1:
            recording["identity_status"] = "unresolved"
            _issue(report, "errors", "inconsistent_pair_identity", recording_id,
                   "PSG/Hypnogram equipment identifiers disagree; do not guess a pair.")
        starts = {}
        for role in ("PSG", "Hypnogram"):
            paths = [path for path, item in members if item["role"] == role]
            key = role.lower()
            if len(paths) != 1:
                _issue(report, "errors", "missing_pair" if not paths else "duplicate_pair", recording_id,
                       f"Expected one {role} file; found {len(paths)}.")
                if len(paths) > 1:
                    recording["identity_status"] = "unresolved"
                    recording[f"{key}_candidates"] = [path.relative_to(root).as_posix() for path in paths]
                continue
            path = paths[0]
            relative = path.relative_to(root).as_posix()
            recording[key] = relative
            if reader_module is not None:
                recording[f"{key}_header"], starts[key] = _header(path, relative, role, reader_module, report)
                if recording[f"{key}_header"] is not None:
                    report["scope"]["headers_completed"] += 1
        recording["evidence"]["pair_unique"] = (
            len({item["equipment"] for _, item in members}) == 1
            and recording["psg"] is not None and recording["hypnogram"] is not None
        )
        recording["evidence"]["headers_verified"] = (
            recording["evidence"]["pair_unique"] and recording["psg_header"] is not None
            and recording["hypnogram_header"] is not None
        )
        if starts.get("psg") is not None and starts.get("hypnogram") is not None:
            offset = (starts["hypnogram"] - starts["psg"]).total_seconds()
            recording["start_alignment"] = {"equal": offset == 0, "offset_seconds": offset}
            if offset:
                _issue(report, "warnings", "pair_start_offset", recording_id,
                       f"Hypnogram header starts {offset:g} seconds after PSG; timing must account for this offset.")
        report["recordings"].append(recording)
    report["recordings"].sort(key=lambda item: (item["recording_id"] or "~", item.get("input_path", "")))
    _metadata(root, report)
    _manifests(root, {item["path"]: item["size_bytes"] for item in report["files"]},
               report, verify_checksums, max_hash_bytes)
    for cohort in ("SC", "ST"):
        cohort_records = [item for item in report["recordings"] if item["cohort"] == cohort]
        report["summary"]["cohorts"][cohort] = {
            "recordings": len(cohort_records),
            "participants": len({item["participant_id"] for item in cohort_records if item["identity_status"] == "resolved"}),
        }
        if cohort == "SC" and cohort_records:
            _issue(report, "warnings", "sc_emg_processed_envelope", "sleep-cassette",
                   "SC submental EMG is a low-rate processed envelope, not raw high-frequency EMG.")
        markers = [channel for item in cohort_records for channel in (item["psg_header"] or {}).get("channels", []) if channel["label"] == "Marker"]
        if cohort == "ST" and any(channel["sample_rate_hz"] != 1 for channel in markers):
            _issue(report, "warnings", "st_marker_documentation_discrepancy", "sleep-telemetry",
                   "Native ST Marker rates differ from the official page's 1 Hz description; use audited header rates and retain this discrepancy.")
    report["summary"]["recordings"] = len(groups)
    report["summary"]["participants"] = sum(item["participants"] for item in report["summary"]["cohorts"].values())
    for level in ("errors", "warnings"):
        report[level].sort(key=lambda item: (item["code"], item["path"], item["message"]))
    return report
