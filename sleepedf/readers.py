"""Established EDF readers with separate signal-only and evaluator-label paths."""

from __future__ import annotations

import math
import re
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pyedflib


PHYSIOLOGICAL_CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental")


def compatibility_offsets(psg_path: Path, hypnogram_path: Path) -> dict:
    """Compare two fixed-header date interpretations without exporting dates.

    This bounded diagnostic reads 256 bytes per file. It never parses TALs or
    establishes absolute clinical timing by itself.
    """
    clocks = []
    for path in (psg_path, hypnogram_path):
        with path.open("rb") as stream:
            header = stream.read(256)
        if len(header) != 256:
            raise ValueError("Truncated fixed EDF header")
        try:
            fixed = datetime.strptime((header[168:176] + b" " + header[176:184]).decode("ascii"),
                                      "%d.%m.%y %H.%M.%S")
            field = header[88:168].decode("ascii")
            token = re.search(r"Startdate\s+(\d{2}-[A-Za-z]{3}-\d{4})", field)
            if token is None:
                raise ValueError("EDF+ Recordingfield date absent")
            field_date = datetime.strptime(token.group(1), "%d-%b-%Y")
            field_clock = field_date.replace(hour=fixed.hour, minute=fixed.minute,
                                             second=fixed.second)
        except (UnicodeError, ValueError) as exc:
            raise ValueError("EDF date diagnostic cannot parse both date fields") from None
        clocks.append((fixed, field_clock))
    fixed_offset = (clocks[1][0] - clocks[0][0]).total_seconds()
    field_offset = (clocks[1][1] - clocks[0][1]).total_seconds()
    return {"fixed_header_pair_offset_seconds": fixed_offset,
            "recordingfield_pair_offset_seconds": field_offset,
            "pair_offsets_equal": abs(fixed_offset - field_offset) <= 1e-6,
            "recordingfield_minus_fixed_days_psg": (clocks[0][1].date() - clocks[0][0].date()).days,
            "recordingfield_minus_fixed_days_hypnogram": (clocks[1][1].date() - clocks[1][0].date()).days,
            "scope": "256 bytes per EDF; absolute dates omitted"}


def first_tal_origin_microseconds(hypnogram_path: Path) -> int:
    """Bounded structural check of the first EDF+ timing marker only.

    MNE normalizes the first empty TAL's fractional onset away.  We therefore
    accept its compatibility path only when this marker is exactly zero.
    Stage descriptions and subsequent TAL payloads are never inspected here.
    """
    with hypnogram_path.open("rb") as stream:
        fixed = stream.read(256)
        if len(fixed) != 256:
            raise ValueError("Truncated Hypnogram header")
        try:
            header_bytes = int(fixed[184:192].decode("ascii").strip())
            signals = int(fixed[252:256].decode("ascii").strip())
        except (UnicodeError, ValueError):
            raise ValueError("Invalid Hypnogram header size") from None
        if signals != 1 or header_bytes != 512:
            raise ValueError("Unexpected annotation-only header structure")
        label = stream.read(16).decode("ascii", errors="ignore").strip()
        if label != "EDF Annotations":
            raise ValueError("Expected annotation-only EDF+ signal")
        stream.seek(header_bytes)
        prefix = stream.read(64)
    token = prefix.split(b"\x14", 1)[0]
    if b"\x14" not in prefix or not re.fullmatch(rb"[+-]\d+(?:\.\d+)?", token):
        raise ValueError("First EDF+ timing marker unavailable")
    from .timing import to_microseconds
    return to_microseconds(token.decode("ascii"))


def signal_metadata(psg_path: Path) -> dict:
    """Return native channel metadata; no annotation reader is opened."""
    with psg_path.open("rb") as stream:
        fixed_header = stream.read(256)
    if len(fixed_header) != 256:
        raise ValueError("Truncated fixed EDF header")
    reserved = fixed_header[192:236]
    if b"EDF+D" in reserved or b"BDF+D" in reserved:
        raise ValueError("Discontinuous PSG requires explicit segment boundaries")
    with pyedflib.EdfReader(str(psg_path), annotations_mode=pyedflib.DO_NOT_READ_ANNOTATIONS) as reader:
        seconds = float(reader.getFileDuration())
        record_seconds = float(reader.datarecord_duration)
        if not math.isfinite(seconds) or seconds <= 0 or record_seconds <= 0:
            raise ValueError("Invalid PSG duration")
        channels = []
        for index, item in enumerate(reader.getSignalHeaders()):
            samples = int(reader.samples_in_datarecord(index))
            rate = samples / record_seconds
            channel = {"label": item["label"].strip(), "sample_rate_hz": rate,
                       "samples_per_data_record": samples,
                       "unit": item["dimension"].strip(),
                       "physical_min": float(item["physical_min"]),
                       "physical_max": float(item["physical_max"]),
                       "digital_min": int(item["digital_min"]),
                       "digital_max": int(item["digital_max"])}
            if (not math.isfinite(rate) or rate <= 0
                    or channel["physical_min"] == channel["physical_max"]
                    or channel["digital_min"] >= channel["digital_max"]):
                raise ValueError("Invalid native channel calibration")
            channels.append(channel)
        if not channels:
            raise ValueError("PSG has no signals")
        return {"duration_seconds": seconds, "data_record_seconds": record_seconds,
                "filetype": int(reader.filetype), "channels": channels,
                "continuity": "continuous_header_or_plain_edf",
                "reader": {"name": "pyedflib", "version": metadata.version("pyedflib")},
                "unit_policy": "physical values in EDF-declared units; physiological channels require uV"}


def read_signal_window(psg_path: Path, start_seconds: float, duration_seconds: float,
                       channel_names: tuple[str, ...] = PHYSIOLOGICAL_CHANNELS) -> dict:
    """Read one bounded PSG window in EDF physical units, without a Hypnogram."""
    if (not math.isfinite(start_seconds) or not math.isfinite(duration_seconds)
            or start_seconds < 0 or duration_seconds <= 0):
        raise ValueError("Invalid signal window")
    with pyedflib.EdfReader(str(psg_path), annotations_mode=pyedflib.DO_NOT_READ_ANNOTATIONS) as reader:
        labels = reader.getSignalLabels()
        if len(labels) != len(set(labels)):
            raise ValueError("Duplicate PSG channel labels")
        if not set(channel_names).issubset(labels):
            raise ValueError("Required physiological channel is absent")
        if start_seconds + duration_seconds > float(reader.getFileDuration()) + 1e-6:
            raise ValueError("Signal window exceeds PSG duration")
        output = {}
        for name in channel_names:
            index = labels.index(name)
            unit = reader.getPhysicalDimension(index).strip()
            if unit not in ("uV", "µV", "μV"):
                raise ValueError("Physiological channel unit is missing or unsupported")
            rate = float(reader.getSampleFrequency(index))
            first, count = round(start_seconds * rate), round(duration_seconds * rate)
            if abs(first / rate - start_seconds) > 1e-6 or abs(count / rate - duration_seconds) > 1e-6:
                raise ValueError("Signal window does not align with native samples")
            values = reader.readSignal(index, start=first, n=count)
            if len(values) != count or not np.isfinite(values).all():
                raise ValueError("Signal window has missing or non-finite samples")
            output[name] = {"samples_uv": values, "sample_rate_hz": rate, "unit": "uV"}
        return {"start_seconds": start_seconds, "duration_seconds": duration_seconds,
                "channels": output}


def verify_physiological_signals(psg_path: Path, chunk_seconds: int = 300) -> dict:
    """Stream four channels, checking finiteness and EDF digital calibration."""
    if type(chunk_seconds) is not int or not 1 <= chunk_seconds <= 300:
        raise ValueError("Chunk size must be 1..300 seconds")
    with pyedflib.EdfReader(str(psg_path), annotations_mode=pyedflib.DO_NOT_READ_ANNOTATIONS) as reader:
        labels = reader.getSignalLabels()
        if len(labels) != len(set(labels)) or not set(PHYSIOLOGICAL_CHANNELS).issubset(labels):
            raise ValueError("Missing or duplicate physiological channel")
        result = []
        for name in PHYSIOLOGICAL_CHANNELS:
            index = labels.index(name)
            unit = reader.getPhysicalDimension(index).strip()
            if unit not in ("uV", "µV", "μV"):
                raise ValueError("Physiological channel unit is missing or unsupported")
            rate = float(reader.getSampleFrequency(index))
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("Invalid native rate")
            n_total = int(reader.getNSamples()[index])
            n_chunk = max(1, int(chunk_seconds * rate))
            pmin, pmax = reader.getPhysicalMinimum(index), reader.getPhysicalMaximum(index)
            dmin, dmax = reader.getDigitalMinimum(index), reader.getDigitalMaximum(index)
            if pmin == pmax or dmin >= dmax:
                raise ValueError("Invalid calibration range")
            for first in range(0, n_total, n_chunk):
                count = min(n_chunk, n_total - first)
                physical = reader.readSignal(index, start=first, n=count)
                digital = reader.readSignal(index, start=first, n=count, digital=True)
                if len(physical) != count or len(digital) != count:
                    raise ValueError("Truncated signal read")
                expected = (digital.astype(np.float64) - dmin) * (pmax - pmin) / (dmax - dmin) + pmin
                if (not np.isfinite(physical).all() or not np.isfinite(digital).all()
                        or not np.allclose(physical, expected, rtol=1e-7, atol=1e-6)):
                    raise ValueError("Signal has non-finite or uncalibrated samples")
            result.append({"label": name, "sample_rate_hz": rate,
                           "unit": "uV", "samples_verified": n_total})
        return {"channels": result, "chunk_seconds": chunk_seconds,
                "reader": {"name": "pyedflib", "version": metadata.version("pyedflib")}}


def read_annotation_pair(psg_path: Path, hypnogram_path: Path) -> dict:
    """Read all timed labels for evaluator preparation, with MNE compatibility.

    MNE is used only when pyEDFlib rejects an EDF+ Recordingfield.  Both
    readers are cross-checked on normal files by the dataset validation pass.
    Absolute header dates are kept in memory and never returned.
    """
    psg_start = None
    try:
        with pyedflib.EdfReader(str(psg_path), annotations_mode=pyedflib.DO_NOT_READ_ANNOTATIONS) as psg:
            psg_start = psg.getStartdatetime()
        with pyedflib.EdfReader(str(hypnogram_path)) as hyp:
            offset = (hyp.getStartdatetime() - psg_start).total_seconds()
            onset, duration, description = hyp.readAnnotations()
        if not math.isfinite(offset):
            raise ValueError("Non-finite annotation offset")
        return {"intervals": list(zip(onset.tolist(), duration.tolist(), description.tolist())),
                "offset_seconds": offset,
                "reader": {"name": "pyedflib", "version": metadata.version("pyedflib")},
                "compatibility_fallback": False}
    except (OSError, ValueError, RuntimeError) as exc:
        if "recordingfield" not in str(exc).lower():
            raise ValueError("EDF annotation reader failed outside known compatibility case") from None
    try:
        if (psg_start is None or psg_start.microsecond != 0
                or first_tal_origin_microseconds(hypnogram_path) != 0):
            raise ValueError("Fractional EDF+ origin needs a verified reader path")
        import mne
        psg = mne.io.read_raw_edf(str(psg_path), preload=False, verbose="ERROR")
        hyp = mne.io.read_raw_edf(str(hypnogram_path), preload=False, verbose="ERROR")
        if psg.info["meas_date"] is None or hyp.info["meas_date"] is None:
            raise ValueError("Independent reader lacks timing origin")
        offset = (hyp.info["meas_date"] - psg.info["meas_date"]).total_seconds()
        annotations = mne.read_annotations(str(hypnogram_path))
        if not math.isfinite(offset) or not len(annotations):
            raise ValueError("Independent reader lacks usable annotations")
        diagnostic = compatibility_offsets(psg_path, hypnogram_path)
        if (not diagnostic["pair_offsets_equal"] or
                abs(offset - diagnostic["fixed_header_pair_offset_seconds"]) > 1e-6):
            raise ValueError("Independent EDF readers disagree on pair-relative origin")
        return {"intervals": list(zip(annotations.onset.tolist(),
                                      annotations.duration.tolist(),
                                      annotations.description.tolist())),
                "offset_seconds": offset,
                "reader": {"name": "mne", "version": metadata.version("mne")},
                "compatibility_fallback": True,
                "offset_evidence": diagnostic}
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise ValueError("Both established EDF annotation readers failed") from None
