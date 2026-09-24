"""Pinned YASA EOG arithmetic on one declared, calibrated EOG signal.

This is a research feature route, not the YASA SleepStaging classifier route.
It uses no EEG input, reference labels, metadata, or pretrained weights.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .contracts import read_json
from .research import file_sha256
from . import yasa_baseline


CHANNEL = "EOG horizontal"
RATE_HZ = 100
EPOCH_SAMPLES = 30 * RATE_HZ


def extract_eog_features(root: Path, eog_uv: np.ndarray, *, channel_name: str,
                         sample_rate_hz: int, unit: str) -> dict:
    """Return every complete-epoch feature row and an explicit missing-value mask.

    The entire signal, including any trailing incomplete epoch, enters the
    pinned native filter. Native centered context and robust normalization use
    the complete unlabelled recording, so this is offline whole-record input.
    """
    if channel_name != CHANNEL or type(sample_rate_hz) is not int or sample_rate_hz != RATE_HZ or unit != "uV":
        raise ValueError("Expected only calibrated EOG horizontal at exactly 100 Hz in uV")
    if (not isinstance(eog_uv, np.ndarray) or eog_uv.ndim != 1 or
            eog_uv.dtype not in (np.dtype("float32"), np.dtype("float64"))):
        raise ValueError("EOG input must be one float32/float64 channel, not an EEG or multichannel array")
    if not np.isfinite(eog_uv).all():
        raise ValueError("Missing or non-finite EOG samples require explicit upstream rejection")
    epochs, tail = divmod(eog_uv.size, EPOCH_SAMPLES)
    if epochs < 2:
        raise ValueError("Pinned EOG feature route requires at least two complete 30-second epochs")

    root = Path(root).resolve()
    manifest_path = root / "research/sources/yasa.json"
    manifest_sha = file_sha256(manifest_path)
    manifest = read_json(manifest_path)
    staging_path = root / "vendor/yasa/src/yasa/staging.py"
    staging_sha = manifest["files"]["src/yasa/staging.py"]["sha256"]
    if (manifest.get("source_sha") != yasa_baseline.SOURCE_SHA or
            manifest.get("weights_downloaded") is not False or
            file_sha256(staging_path) != staging_sha):
        raise ValueError("Pinned source-only YASA feature implementation changed")
    yasa = yasa_baseline._native(root)

    # SleepStaging.__init__ requires EEG. Its unchanged fit() can run the EOG
    # branch with only the four fields it reads; no surrogate EEG is supplied.
    native = object.__new__(yasa.SleepStaging)
    native.sf = RATE_HZ
    native.ch_types = ["eog"]
    native.data = np.asarray(eog_uv, dtype=np.float64)[None, :]
    native.metadata = None
    frame = native.get_features()
    if (len(frame) != epochs or frame.columns.has_duplicates or
            any(not (name.startswith("eog_") or name in ("time_hour", "time_norm"))
                for name in frame.columns) or
            not all(np.issubdtype(dtype, np.number) for dtype in frame.dtypes)):
        raise ValueError("Pinned EOG features changed their full grid or schema")
    values = frame.to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("Pinned EOG features contain infinity; no value is silently imputed")
    missing = np.isnan(values)
    if file_sha256(staging_path) != staging_sha or file_sha256(manifest_path) != manifest_sha:
        raise ValueError("Pinned YASA source changed during EOG feature extraction")
    return {
        "features": frame,
        "missing_mask": missing,
        "epoch_index": np.arange(epochs, dtype=np.int64),
        "onset_seconds": np.arange(epochs, dtype=np.float64) * 30.,
        "metadata": {
            "channel_name": CHANNEL, "unit": "uV", "sample_rate_hz": RATE_HZ,
            "complete_epochs": epochs, "trailing_samples": tail,
            "trailing_seconds": tail / RATE_HZ,
            "feature_count": len(frame.columns), "missing_feature_values": int(missing.sum()),
            "source_manifest_sha256": manifest_sha,
            "staging_sha256": staging_sha,
            "context": "offline whole-record EOG; centered 15-epoch and past 4-epoch native rolling features",
            "normalization": "unlabelled recording-local native robust 5-95% feature scaling",
            "reference_labels_used": False,
        },
    }
