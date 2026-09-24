"""Small, signal-only prediction artifacts and private evaluator truth readers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .contracts import CLASS_ORDER, json_text, read_json

PROBABILITY_TOLERANCE = 1e-6
_PREDICTION_KEYS = {"participant_id", "recording_id", "epoch_index", "onset_seconds", "hard_label"}
_TRUTH_KEYS = {"participant_id", "recording_id", "epoch_index", "onset_seconds", "reference_label", "valid_mask"}
_FORBIDDEN_METADATA_KEYS = {"truth", "reference_label", "valid_mask", "hypnogram", "scorer", "labels"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty trimmed string")
    return value


def _vector(value: object, name: str, kind: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or (length is not None and len(array) != length):
        raise ValueError(f"{name} must be a one-dimensional array of the expected length")
    if array.dtype.kind not in kind:
        raise ValueError(f"{name} has the wrong dtype")
    return array


def _identity(participant_id: object, recording_id: object, epoch_index: object,
              onset_seconds: object) -> tuple[str, str, np.ndarray, np.ndarray]:
    participant = _text(participant_id, "participant_id")
    recording = _text(recording_id, "recording_id")
    epochs = _vector(epoch_index, "epoch_index", "iu")
    onsets = _vector(onset_seconds, "onset_seconds", "fi", len(epochs))
    if (len(epochs) == 0 or np.any(epochs < 0)
            or np.any(epochs > np.iinfo(np.int64).max)
            or np.any(np.diff(epochs.astype(np.int64)) <= 0)):
        raise ValueError("epoch_index must be nonempty, nonnegative, and strictly increasing")
    onsets = onsets.astype(np.float64)
    if not np.all(np.isfinite(onsets)) or np.any(onsets < 0) or np.any(np.diff(onsets) <= 0):
        raise ValueError("onset_seconds must be finite, nonnegative, and strictly increasing")
    if not np.array_equal(epochs, np.arange(len(epochs), dtype=np.int64)) or \
            not np.array_equal(onsets, np.arange(len(epochs), dtype=np.float64) * 30):
        raise ValueError("epoch_index and onset_seconds must preserve the complete 30-second PSG grid")
    return participant, recording, epochs.astype(np.int64), onsets.astype(np.float64)


def _metadata_is_signal_only(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _FORBIDDEN_METADATA_KEYS:
                raise ValueError(f"prediction metadata contains forbidden field: {key}")
            _metadata_is_signal_only(child)
    elif isinstance(value, list):
        for child in value:
            _metadata_is_signal_only(child)


def save_prediction(path: Path, *, participant_id: str, recording_id: str,
                    epoch_index: object, onset_seconds: object, hard_label: object,
                    model_id: str, protocol_hash: str, registry_hash: str,
                    provenance: dict, probabilities: object | None = None) -> dict:
    """Write one recording's predictions; the sidecar has no reference labels."""
    path = Path(path)
    if path.suffix != ".npz" or path.exists() or path.with_suffix(".json").exists():
        raise ValueError("prediction path must be a new .npz artifact")
    participant, recording, epochs, onsets = _identity(
        participant_id, recording_id, epoch_index, onset_seconds)
    labels = _vector(hard_label, "hard_label", "iu", len(epochs))
    if np.any((labels < 0) | (labels > 4)):
        raise ValueError("hard_label must use fixed class IDs 0..4")
    arrays = dict(participant_id=np.asarray(participant), recording_id=np.asarray(recording),
                  epoch_index=epochs, onset_seconds=onsets, hard_label=labels.astype(np.int8))
    if probabilities is not None:
        prob = np.asarray(probabilities)
        if prob.shape != (len(epochs), 5) or prob.dtype.kind != "f":
            raise ValueError("probabilities must have shape [epochs, 5] and floating dtype")
        if not np.all(np.isfinite(prob)) or np.any((prob < 0) | (prob > 1)):
            raise ValueError("probabilities must be finite and within [0, 1]")
        if np.any(np.abs(prob.sum(axis=1) - 1) > PROBABILITY_TOLERANCE):
            raise ValueError("probability rows must sum to one within tolerance")
        arrays["probabilities"] = prob.astype(np.float64)
    if type(provenance) is not dict:
        raise ValueError("provenance must be a JSON object")
    _metadata_is_signal_only(provenance)
    sidecar = {"schema_version": "1.0", "artifact_type": "signal_only_prediction",
               "model_id": _text(model_id, "model_id"),
               "protocol_hash": _text(protocol_hash, "protocol_hash"),
               "registry_hash": _text(registry_hash, "registry_hash"),
               "class_order": list(CLASS_ORDER),
               "probability_tolerance": PROBABILITY_TOLERANCE,
               "provenance": provenance}
    json_text(sidecar)  # Reject non-finite metadata before writing either file.
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    sidecar["payload_sha256"] = _sha256(path)
    path.with_suffix(".json").write_text(json_text(sidecar), encoding="utf-8")
    return sidecar


def _read_npz(path: Path, required: set[str], optional: set[str] = frozenset()) -> dict:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if not required <= keys or keys - required - optional:
            raise ValueError("NPZ keys differ from the declared artifact schema")
        return {key: data[key].copy() for key in keys}


def _scalar_string(value: np.ndarray, name: str) -> str:
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"{name} must be a scalar Unicode string")
    return _text(value.item(), name)


def _hash_matches(expected: str, actual: str) -> bool:
    return expected in (actual, "sha256:" + actual)


def load_prediction(path: Path) -> tuple[dict, dict]:
    path = Path(path)
    meta = read_json(path.with_suffix(".json"))
    if type(meta) is not dict or set(meta) != {
        "schema_version", "artifact_type", "model_id", "protocol_hash", "registry_hash",
        "class_order", "probability_tolerance", "provenance", "payload_sha256"
    }:
        raise ValueError("prediction sidecar keys differ from schema")
    if meta["schema_version"] != "1.0" or meta["artifact_type"] != "signal_only_prediction":
        raise ValueError("wrong prediction artifact type")
    if meta["class_order"] != list(CLASS_ORDER) or meta["probability_tolerance"] != PROBABILITY_TOLERANCE:
        raise ValueError("wrong class order or probability tolerance")
    if not _hash_matches(meta["payload_sha256"], _sha256(path)):
        raise ValueError("prediction payload hash mismatch")
    for key in ("model_id", "protocol_hash", "registry_hash"):
        _text(meta[key], key)
    if type(meta["provenance"]) is not dict:
        raise ValueError("provenance must be a JSON object")
    _metadata_is_signal_only(meta["provenance"])
    data = _read_npz(path, _PREDICTION_KEYS, {"probabilities"})
    participant, recording, epochs, onsets = _identity(
        _scalar_string(data["participant_id"], "participant_id"),
        _scalar_string(data["recording_id"], "recording_id"),
        data["epoch_index"], data["onset_seconds"])
    labels = _vector(data["hard_label"], "hard_label", "iu", len(epochs))
    if np.any((labels < 0) | (labels > 4)):
        raise ValueError("invalid hard_label")
    if "probabilities" in data:
        prob = data["probabilities"]
        if prob.shape != (len(epochs), 5) or prob.dtype.kind != "f" or not np.all(np.isfinite(prob)):
            raise ValueError("invalid probabilities")
        if np.any((prob < 0) | (prob > 1)) or np.any(np.abs(prob.sum(axis=1) - 1) > PROBABILITY_TOLERANCE):
            raise ValueError("invalid probability values")
    data.update(participant_id=participant, recording_id=recording,
                epoch_index=epochs, onset_seconds=onsets, hard_label=labels.astype(np.int8))
    return meta, data


def load_truth(path: Path) -> tuple[dict, dict]:
    """Read evaluator-only truth; never pass this object into model inference."""
    path = Path(path)
    meta = read_json(path.with_suffix(".json"))
    required_meta = {"schema_version", "payload_sha256", "source_psg_sha256",
                     "source_hypnogram_sha256", "recording_id", "participant_id",
                     "timing_tolerance_us", "epoch_seconds", "reader_evidence",
                     "valid_epochs", "n_epochs"}
    if type(meta) is not dict or not required_meta <= set(meta) or \
            meta["schema_version"] != "sleepedf-truth-v1":
        raise ValueError("private truth requires a hash-bound JSON sidecar")
    if not _hash_matches(meta["payload_sha256"], _sha256(path)):
        raise ValueError("truth payload hash mismatch")
    data = _read_npz(path, _TRUTH_KEYS)
    participant, recording, epochs, onsets = _identity(
        _scalar_string(data["participant_id"], "participant_id"),
        _scalar_string(data["recording_id"], "recording_id"),
        data["epoch_index"], data["onset_seconds"])
    labels = _vector(data["reference_label"], "reference_label", "iu", len(epochs))
    mask = _vector(data["valid_mask"], "valid_mask", "b", len(epochs))
    if np.any(mask & ((labels < 0) | (labels > 4))) or np.any(~mask & (labels != -1)):
        raise ValueError("reference_label and valid_mask disagree")
    if (meta["participant_id"] != participant or meta["recording_id"] != recording
            or type(meta["n_epochs"]) is not int or meta["n_epochs"] != len(epochs)
            or type(meta["valid_epochs"]) is not int or meta["valid_epochs"] != int(mask.sum())
            or meta["epoch_seconds"] != 30):
        raise ValueError("truth sidecar identity or grid counts disagree")
    for key in ("source_psg_sha256", "source_hypnogram_sha256"):
        value = meta[key]
        if not isinstance(value, str) or len(value.removeprefix("sha256:")) != 64 or \
                any(c not in "0123456789abcdef" for c in value.removeprefix("sha256:")):
            raise ValueError(f"truth sidecar {key} must be a SHA256 digest")
    data.update(participant_id=participant, recording_id=recording,
                epoch_index=epochs, onset_seconds=onsets,
                reference_label=labels.astype(np.int8), valid_mask=mask.astype(bool))
    return meta, data
