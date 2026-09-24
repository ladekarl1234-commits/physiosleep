"""Bounded reader checks; this module does not generate training epochs."""

from __future__ import annotations

import math
from importlib import metadata
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from .contracts import EPOCH_SECONDS, map_annotation


def inspect_intervals(intervals: list[tuple], recording_seconds: float, offset_seconds: float = 0) -> dict:
    """Inspect annotation intervals in PSG time without rounding or concatenating."""
    if type(recording_seconds) not in (int, float) or type(offset_seconds) not in (int, float):
        raise ValueError("Recording duration must be positive and timing offset must be finite")
    try:
        finite = math.isfinite(recording_seconds) and math.isfinite(offset_seconds)
    except OverflowError:
        finite = False
    if not finite or recording_seconds <= 0:
        raise ValueError("Recording duration must be positive and timing offset must be finite")
    counts = Counter()
    labels = Counter()
    positions = []
    coverage = []
    covered_until = 0.0
    gap_seconds = 0.0
    previous_onset = -math.inf
    for interval in intervals:
        if not isinstance(interval, (tuple, list)) or len(interval) != 3:
            raise ValueError("Each annotation interval must contain onset, duration, and description")
        onset, duration, description = interval
        try:
            original_onset, length = float(onset), float(duration)
        except (TypeError, ValueError, OverflowError):
            original_onset, length = math.nan, math.nan
        start = original_onset + offset_seconds
        mapped = map_annotation(description) if isinstance(description, str) else {
            "label": None, "valid": False, "reason": "unrecognized_annotation"}
        labels[str(mapped["label"]) if mapped["valid"] else mapped["reason"]] += 1
        position = {"onset_seconds": original_onset if math.isfinite(original_onset) else None,
                    "duration_seconds": length if math.isfinite(length) else None,
                    "psg_start_seconds": start if math.isfinite(start) else None,
                    "psg_end_seconds": None, "label": mapped["label"],
                    "valid_label": mapped["valid"], "invalid_label_reason": mapped["reason"],
                    "source_out_of_order": False}
        positions.append(position)
        if not math.isfinite(start) or not math.isfinite(length) or length <= 0:
            counts["invalid_intervals"] += 1
            continue
        end = start + length
        if not math.isfinite(end):
            counts["invalid_intervals"] += 1
            continue
        position["psg_end_seconds"] = end
        if start < previous_onset:
            counts["out_of_order_intervals"] += 1
            position["source_out_of_order"] = True
        previous_onset = start
        if abs(start / EPOCH_SECONDS - round(start / EPOCH_SECONDS)) > 1e-7 or abs(end / EPOCH_SECONDS - round(end / EPOCH_SECONDS)) > 1e-7:
            counts["non_aligned_intervals"] += 1
        if start < 0 or end > recording_seconds:
            counts["outside_psg_intervals"] += 1
        left, right = max(0.0, start), min(recording_seconds, end)
        if left < right:
            coverage.append((left, right))
    for left, right in sorted(coverage):
        if left < right:
            if left > covered_until:
                gap_seconds += left - covered_until
                counts["gaps"] += 1
            if left < covered_until:
                counts["overlaps"] += 1
            covered_until = max(covered_until, right)
    if covered_until < recording_seconds:
        gap_seconds += recording_seconds - covered_until
        counts["gaps"] += 1
    return {
        "annotation_intervals": len(intervals),
        "interval_positions": positions,
        "annotation_label_counts": dict(sorted(labels.items())),
        "issues": {key: counts[key] for key in (
            "gaps", "overlaps", "non_aligned_intervals", "outside_psg_intervals",
            "invalid_intervals", "out_of_order_intervals")},
        "uncovered_seconds": gap_seconds,
        "hypnogram_start_offset_seconds": offset_seconds,
        "psg_duration_seconds": recording_seconds,
        "complete_grid_epochs": math.floor(recording_seconds / EPOCH_SECONDS),
        "trailing_seconds": recording_seconds % EPOCH_SECONDS,
        "epochs_generated": False,
        "coverage_includes_invalid_labels": True,
    }


def smoke_recordings(root: Path, records: list[dict], seconds: float) -> dict:
    if (not 0 < len(records) <= 2 or any(not isinstance(r, dict) or
        not isinstance(r.get("recording_id"), str) for r in records) or
        len({r["recording_id"] for r in records}) != len(records)):
        raise ValueError("Smoke requires one or two distinct recordings")
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 < seconds <= 60:
        raise ValueError("Smoke waveform read duration must be greater than 0 and at most 60 seconds")
    try:
        import numpy as np
        import pyedflib
    except ImportError as exc:
        raise RuntimeError("Reader unavailable; run uv sync --frozen --extra audit --group dev") from exc
    versions = {}
    for name in ("pyedflib", "numpy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    result = {"scope": {"data_root": str(root.resolve()), "recording_limit": 2,
                         "waveform_seconds_per_recording": seconds,
                         "annotation_scope": "All intervals in selected hypnograms only",
                         "raw_samples_saved": False, "waveform_unit": "uV",
                         "reader_version": versions["pyedflib"],
                         "dependency_versions": versions,
                         "not_performed": ["epoch_projection", "full_signal_quality_assessment",
                                           "full_dataset_validation"]},
              "recordings": [], "errors": [], "warnings": []}
    allowed = {"EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"}
    for record in records:
        rid = record["recording_id"]
        inspected = {"recording_id": rid, "participant_id": record.get("participant_id"),
                     "channels": [], "timing": None, "status": "failed"}
        result["recordings"].append(inspected)
        if (record.get("identity_status") != "resolved" or
            not isinstance(record.get("psg"), str) or
            not isinstance(record.get("hypnogram"), str)):
            result["errors"].append({"code": "unresolved_pair", "path": rid,
                                      "message": "Selected recording lacks a resolved PSG/Hypnogram pair"})
            continue
        psg, hyp = root / record["psg"], root / record["hypnogram"]
        if not psg.resolve().is_relative_to(root.resolve()) or not hyp.resolve().is_relative_to(root.resolve()):
            result["errors"].append({"code": "unsafe_pair_path", "path": rid,
                                      "message": "Selected EDF pair resolves outside the dataset root"})
            continue
        stage = "hypnogram_size"
        try:
            if hyp.stat().st_size > 16 * 1024 * 1024:
                result["errors"].append({"code": "hypnogram_size_limit", "path": record["hypnogram"],
                                          "message": "Hypnogram exceeds the 16 MiB smoke annotation limit"})
                continue
            # Both headers must open before any waveform samples are read.
            with ExitStack() as stack:
                stage = "psg_header"
                reader = stack.enter_context(pyedflib.EdfReader(
                    str(psg), annotations_mode=pyedflib.DO_NOT_READ_ANNOTATIONS))
                stage = "hypnogram_header"
                annotations = stack.enter_context(pyedflib.EdfReader(str(hyp)))
                stage = "waveform"
                start = reader.getStartdatetime()
                duration = float(reader.getFileDuration())
                offset = (annotations.getStartdatetime() - start).total_seconds()
                if not math.isfinite(duration) or duration <= 0 or not math.isfinite(offset):
                    raise ValueError("Invalid PSG duration or annotation offset")
                for i, label in enumerate(reader.getSignalLabels()):
                    if label not in allowed:
                        continue
                    rate = float(reader.getSampleFrequency(i))
                    samples_available = int(reader.getNSamples()[i])
                    if not math.isfinite(rate) or rate <= 0 or samples_available < 0:
                        raise ValueError("Invalid channel sampling metadata")
                    n = min(int(seconds * rate), samples_available)
                    pmin, pmax = float(reader.getPhysicalMinimum(i)), float(reader.getPhysicalMaximum(i))
                    dmin, dmax = float(reader.getDigitalMinimum(i)), float(reader.getDigitalMaximum(i))
                    unit = str(reader.getPhysicalDimension(i)).strip()
                    if (not all(math.isfinite(x) for x in (pmin, pmax, dmin, dmax)) or
                        pmin == pmax or dmin >= dmax):
                        raise ValueError("Invalid channel calibration range")
                    if n <= 0:
                        result["errors"].append({"code": "waveform_check_failed", "path": record["psg"],
                                                  "message": "No bounded waveform samples available"})
                        continue
                    physical = reader.readSignal(i, start=0, n=n)
                    digital = reader.readSignal(i, start=0, n=n, digital=True)
                    expected = (digital.astype(float) - dmin) * (pmax - pmin) / (dmax - dmin) + pmin
                    finite = bool(np.isfinite(physical).all() and np.isfinite(digital).all())
                    calibrated = bool(finite and np.allclose(physical, expected, rtol=1e-7, atol=1e-6))
                    unit_valid = unit in ("uV", "µV", "μV")
                    inspected["channels"].append({"label": label, "sample_rate_hz": rate,
                                                   "unit": unit, "samples_read": len(physical),
                                                   "finite": finite, "calibration_matches_header": calibrated,
                                                   "unit_is_microvolts": unit_valid})
                    if len(physical) != n or len(digital) != n or not finite or not calibrated or not unit_valid:
                        result["errors"].append({"code": "waveform_check_failed", "path": record["psg"],
                                                  "message": "Bounded waveform count, unit, finiteness, or calibration failed"})
                if not inspected["channels"]:
                    result["errors"].append({"code": "no_expected_channels", "path": record["psg"],
                                              "message": "No expected EEG/EOG/EMG channels found"})
                stage = "annotations"
                onset, durations, descriptions = annotations.readAnnotations()
                if not len(onset) == len(durations) == len(descriptions):
                    raise ValueError("Annotation arrays have different lengths")
                inspected["timing"] = inspect_intervals(list(zip(onset, durations, descriptions)), duration, offset)
            timing = inspected["timing"]
            for issue, count in timing["issues"].items():
                if count:
                    result["warnings"].append({"code": issue, "path": record["hypnogram"],
                                                "message": f"{count} intervals/events require timing review"})
            if timing["trailing_seconds"]:
                result["warnings"].append({"code": "partial_trailing_epoch", "path": record["psg"],
                                            "message": "PSG ends before the next complete 30-second boundary"})
            inspected["status"] = "passed" if not any(
                issue.get("path") in (record["psg"], record["hypnogram"])
                for issue in result["errors"]) else "failed"
        except (OSError, ValueError, RuntimeError, IndexError, OverflowError, TypeError) as exc:
            path = record["hypnogram"] if stage in ("hypnogram_size", "hypnogram_header", "annotations") else record["psg"]
            result["errors"].append({"code": "reader_failure", "path": path,
                                      "message": f"Selected EDF pair failed during {stage} ({type(exc).__name__}); inspect locally."})
    return result
