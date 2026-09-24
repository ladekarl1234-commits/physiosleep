"""Small, development-only EEG/EOG waveform candidate (offline 30-second epochs).

Physical input is calibrated microvolts from the PSG reader: EEG Fpz-Cz and
horizontal EOG, each at 100 Hz. Fold statistics are fitted on training people
only. This intentionally simple epoch model has no between-epoch context.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np

from .contracts import CLASS_ORDER, content_id, json_text, read_json
from .dataset import iter_signal_epochs
from .predictions import load_prediction, save_prediction
from .protocol import development_records, load_development_truth
from .research import append_event, atomic_json, compute_lease, file_sha256

CHANNELS = ("EEG Fpz-Cz", "EOG horizontal")
RATE_HZ = 100
SAMPLES = 30 * RATE_HZ
SEED = 17
EPOCHS = 20
BATCH = 32
MAX_THREADS = 4
MAX_HOST_BYTES = 10 * 1024**3
MAX_GPU_BYTES = 4 * 1024**3


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch 2.3.1 is required in .venvs/waveform") from exc
    if torch.__version__.split("+")[0] != "2.3.1":
        raise RuntimeError("The waveform candidate requires PyTorch 2.3.1")
    return torch


def _id(record: dict) -> str:
    rid = record.get("recording_id")
    if not isinstance(rid, str) or re.fullmatch(r"(?:SC4|ST7)\d{2}[12]", rid) is None:
        raise ValueError("Invalid recording identity")
    return rid


def _cache_paths(cache_dir: Path, record: dict) -> tuple[Path, Path]:
    rid = _id(record)
    return cache_dir / f"{rid}.npy", cache_dir / f"{rid}.json"


def _writable_path(path: Path, data_root: Path | None = None) -> None:
    target = Path(path).resolve()
    if (data_root is not None and target.is_relative_to(Path(data_root).resolve())) or \
            any(parent.name == "sleep-edf-database-expanded-1.0.0" for parent in (target, *target.parents)):
        raise ValueError("Research artifacts cannot be written inside original data")


def prepare_record(record: dict, data_root: Path, cache_dir: Path) -> dict:
    """Materialize one validated development PSG in bounded host memory."""
    import numpy.lib.format as fmt
    _writable_path(cache_dir, data_root)
    data_path, meta_path = _cache_paths(cache_dir, record)
    n = record.get("n_epochs")
    if type(n) is not int or n < 1:
        raise ValueError("Invalid complete-epoch count")
    source = (Path(data_root) / record["psg"]).resolve()
    if not source.is_relative_to(Path(data_root).resolve()) or not source.is_file():
        raise ValueError("PSG source is unavailable or outside data root")
    if file_sha256(source) != record["psg_sha256"]:
        raise ValueError("PSG content differs from verified readiness")
    if data_path.exists() or meta_path.exists():
        if not (data_path.exists() and meta_path.exists()):
            raise ValueError("Incomplete waveform cache")
        meta = read_json(meta_path)
        if (meta.get("recording_id") != _id(record) or
                meta.get("participant_id") != record["participant_id"] or
                meta.get("source_psg_sha256") != record["psg_sha256"] or
                meta.get("n_epochs") != n or meta.get("channels") != list(CHANNELS) or
                meta.get("unit") != "uV" or meta.get("rate_hz") != RATE_HZ or
                meta.get("samples_per_epoch") != SAMPLES or
                meta.get("payload_sha256") != file_sha256(data_path)):
            raise ValueError("Waveform cache identity/content mismatch")
        return meta
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_dir, prefix=".wave-", suffix=".npy", delete=False) as out:
        temporary = Path(out.name)
    array = None
    try:
        array = fmt.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(n, 2, SAMPLES))
        seen = 0
        for epoch in iter_signal_epochs(record, data_root, CHANNELS):
            if seen >= n or epoch["epoch_index"] != seen or epoch["onset_seconds"] != seen * 30.0 or \
                    epoch["participant_id"] != record["participant_id"] or \
                    epoch["recording_id"] != _id(record):
                raise ValueError("Signal epoch differs from the full PSG grid")
            for channel, name in enumerate(CHANNELS):
                signal = epoch["channels"][name]
                raw = np.asarray(signal["samples_uv"])
                if signal["unit"] != "uV" or signal["sample_rate_hz"] != RATE_HZ or \
                        raw.shape != (SAMPLES,) or not np.isfinite(raw).all():
                    raise ValueError("Waveform channel, unit, rate, or sample count differs")
                array[seen, channel] = raw
                if not np.isfinite(array[seen, channel]).all():
                    raise ValueError("Microvolt values exceed float32 range")
            seen += 1
        if seen != n:
            raise ValueError("Signal iterator omitted a complete PSG epoch")
        array.flush()
        array._mmap.close()
        array = None
        os.replace(temporary, data_path)
        meta = {"schema_version": "1.0", "recording_id": _id(record),
                "participant_id": record["participant_id"], "source_psg_sha256": record["psg_sha256"],
                "n_epochs": n, "channels": list(CHANNELS), "unit": "uV",
                "rate_hz": RATE_HZ, "samples_per_epoch": SAMPLES,
                "payload_sha256": file_sha256(data_path)}
        atomic_json(meta_path, meta, immutable=True)
        return meta
    finally:
        if array is not None:
            array._mmap.close()
        temporary.unlink(missing_ok=True)


def _load_cache(cache_dir: Path, record: dict) -> np.ndarray:
    data_path, meta_path = _cache_paths(cache_dir, record)
    meta = read_json(meta_path)
    if (meta.get("recording_id") != _id(record) or meta.get("participant_id") != record["participant_id"] or
            meta.get("source_psg_sha256") != record["psg_sha256"] or
            meta.get("channels") != list(CHANNELS) or meta.get("unit") != "uV" or
            meta.get("rate_hz") != RATE_HZ or meta.get("samples_per_epoch") != SAMPLES or
            meta.get("n_epochs") != record["n_epochs"] or
            meta.get("payload_sha256") != file_sha256(data_path)):
        raise ValueError("Waveform cache identity/content mismatch")
    array = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if array.shape != (record["n_epochs"], 2, SAMPLES) or array.dtype != np.float32:
        raise ValueError("Waveform cache shape/dtype mismatch")
    return array


def fit_scaler(records: list[dict], cache_dir: Path) -> dict:
    """Training-only microvolt mean/std; invertible and fixed at inference."""
    sums = np.zeros(2, np.float64)
    squares = np.zeros(2, np.float64)
    count = 0
    for record in records:
        values = _load_cache(cache_dir, record)
        for start in range(0, len(values), 32):
            block = np.asarray(values[start:start+32], dtype=np.float64)
            if not np.isfinite(block).all():
                raise ValueError("Non-finite cached waveform")
            sums += block.sum(axis=(0, 2))
            squares += np.square(block).sum(axis=(0, 2))
            count += block.shape[0] * SAMPLES
    if count == 0:
        raise ValueError("No training signal samples")
    mean = sums / count
    variance = np.maximum(0.0, squares / count - mean * mean)
    std = np.sqrt(variance)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 1e-6):
        raise ValueError("Undefined or near-flat training channel scale")
    return {"mean_uv": mean.tolist(), "std_uv": std.tolist(), "count_per_channel": count,
            "method": "train_participants_all_complete_epochs_global_mean_std"}


def _scale(values: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["mean_uv"], dtype=np.float64)
    std = np.asarray(stats["std_uv"], dtype=np.float64)
    if mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all() or \
            not np.isfinite(std).all() or np.any(std < 1e-6):
        raise ValueError("Invalid training-only microvolt scaler")
    result = (np.asarray(values, dtype=np.float64) - mean[None, :, None]) / std[None, :, None]
    if not np.isfinite(result).all() or np.any(np.abs(result) > np.finfo(np.float32).max):
        raise ValueError("Non-finite normalized waveform")
    return result.astype(np.float32)


def build_model(torch):
    nn = torch.nn
    return nn.Sequential(
        nn.Conv1d(2, 16, 15, stride=4, padding=7), nn.GroupNorm(4, 16), nn.GELU(),
        nn.Conv1d(16, 32, 9, stride=4, padding=4), nn.GroupNorm(4, 32), nn.GELU(),
        nn.Conv1d(32, 64, 7, stride=4, padding=3), nn.GroupNorm(8, 64), nn.GELU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, len(CLASS_ORDER)))


def masked_loss(torch, logits, labels, valid):
    if logits.ndim != 2 or logits.shape[1] != 5 or labels.shape != logits.shape[:1] or \
            valid.shape != labels.shape or valid.dtype != torch.bool or \
            bool(torch.any(valid & ((labels < 0) | (labels > 4)))) or \
            bool(torch.any(~valid & (labels != -1))):
        raise ValueError("Five-class logits and original-grid label mask disagree")
    if not bool(valid.any()):
        raise ValueError("Loss batch has no valid reference epochs")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Non-finite waveform logits")
    loss = torch.nn.functional.cross_entropy(logits[valid], labels[valid])
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite waveform loss")
    return loss


def _f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    if len(truth) == 0 or truth.shape != predicted.shape or \
            np.any((truth < 0) | (truth > 4)) or np.any((predicted < 0) | (predicted > 4)):
        raise ValueError("Invalid five-class development evaluation")
    cm = np.bincount(5 * truth + predicted, minlength=25).reshape(5, 5)
    support = cm.sum(axis=1) + cm.sum(axis=0)
    terms = np.divide(2 * np.diag(cm), support, out=np.zeros(5, np.float64), where=support != 0)
    return float(terms.mean())


def _samples(records: list[dict], split: dict, cache_dir: Path):
    arrays, indices, labels = [], [], []
    for record in records:
        signal = _load_cache(cache_dir, record)
        truth = load_development_truth(record, split)
        if len(signal) != len(truth["epoch_index"]) or not np.array_equal(
                truth["epoch_index"], np.arange(len(signal))) or not np.array_equal(
                truth["onset_seconds"], np.arange(len(signal)) * 30.0):
            raise ValueError("Development signal and truth grids disagree")
        arrays.append(signal)
        for index in range(len(signal)):
            indices.append((len(arrays)-1, int(index)))
            labels.append(int(truth["reference_label"][index]))
    if not indices or not np.any(np.asarray(labels) >= 0):
        raise ValueError("No valid development epochs")
    return arrays, indices, np.asarray(labels, np.int64)


def _truth_fingerprints(records: list[dict]) -> dict:
    """Bind training and validation label artifacts without storing their labels."""
    result = {}
    for record in records:
        path = Path(record["truth_path"])
        sidecar = path.with_suffix(".json")
        meta = read_json(sidecar)
        payload_hash = file_sha256(path)
        if (meta.get("participant_id") != record["participant_id"] or
                meta.get("recording_id") != _id(record) or
                meta.get("source_psg_sha256") != record["psg_sha256"] or
                meta.get("source_hypnogram_sha256") != record["hypnogram_sha256"] or
                meta.get("n_epochs") != record["n_epochs"] or
                meta.get("payload_sha256") != payload_hash):
            raise ValueError("Development truth differs from readiness source identity")
        result[_id(record)] = {"payload_sha256": payload_hash,
                               "sidecar_sha256": file_sha256(sidecar)}
    return result


def _implementation_fingerprint(root: Path) -> dict:
    import importlib.metadata
    import platform
    import sys
    modules = ("waveform", "protocol", "splits", "research", "contracts",
               "predictions", "dataset", "readers", "timing", "audit")
    sources = {f"sleepedf/{name}.py": file_sha256(root / "sleepedf" / f"{name}.py")
               for name in modules}
    environment_name = Path(sys.prefix).resolve().name
    lock_name = {"waveform-cpu": "waveform-cpu.lock.txt", "waveform": "waveform.lock.txt"}.get(environment_name)
    if lock_name is None or not (lock := root / "requirements" / lock_name).is_file():
        raise ValueError("The active waveform environment needs its matching installed runtime lock")
    packages = sorted((distribution.metadata["Name"].lower(), distribution.version)
                      for distribution in importlib.metadata.distributions())
    return {"source_sha256": sources, "runtime_lock_path": lock.relative_to(root).as_posix(),
            "runtime_lock_sha256": file_sha256(lock), "installed_packages": packages,
            "python_version": platform.python_version(), "python_implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(),
            "python_executable": str(Path(sys.executable).resolve())}


def _cache_fingerprints(records: list[dict], cache_dir: Path) -> dict:
    return {_id(record): read_json(_cache_paths(cache_dir, record)[1])["payload_sha256"]
            for record in records}


def _device(torch):
    torch.set_num_threads(MAX_THREADS)
    if torch.get_num_interop_threads() > MAX_THREADS:
        torch.set_num_interop_threads(MAX_THREADS)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(min(MAX_GPU_BYTES, props.total_memory) * 0.9 / props.total_memory, 0)
        return torch.device("cuda:0")
    return torch.device("cpu")


def _check_host_memory():
    import psutil
    rss = psutil.Process().memory_info().rss
    if rss > MAX_HOST_BYTES or psutil.virtual_memory().available < 4 * 1024**3:
        raise RuntimeError("Waveform job exceeds host memory guard")
    return rss


def _check_gradients(torch, model):
    if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
           for parameter in model.parameters()):
        raise FloatingPointError("Non-finite waveform gradient")


def _check_parameters(torch, model):
    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("Non-finite waveform parameter")


def _training_log(path: Path, identity_hash: str, completed: int, last_epoch: dict | None) -> None:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            json_text(row)
            rows.append(row)
    if any(row.get("event") != "waveform_epoch" or row.get("identity_hash") != identity_hash or
           row.get("epoch") != i + 1 for i, row in enumerate(rows)):
        raise ValueError("Waveform training log differs from checkpoint identity/order")
    if len(rows) == completed - 1 and last_epoch is not None and last_epoch.get("epoch") == completed:
        append_event(path, last_epoch)
    elif len(rows) != completed:
        raise ValueError("Waveform checkpoint and append-only epoch log disagree")


def _save_checkpoint(torch, path: Path, state: dict):
    _writable_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", suffix=".pt", delete=False) as out:
        temporary = Path(out.name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fit_fold(root: Path, fold_id: int, cache_dir: Path, checkpoint: Path, *,
             seed: int = SEED, max_epochs: int = EPOCHS, stop_after: int | None = None) -> dict:
    torch = _torch()
    if not 0 <= fold_id < 5 or seed not in (17, 43, 101) or not 1 <= max_epochs <= EPOCHS:
        raise ValueError("Unapproved fold, seed, or epoch budget")
    protocol, split, records = development_records(root)
    _writable_path(cache_dir)
    _writable_path(checkpoint)
    fold = split["folds"][fold_id]
    train_ids, val_ids = set(fold["train"]), set(fold["validation"])
    if train_ids & val_ids or len(train_ids) != 48 or len(val_ids) != 12:
        raise ValueError("Development fold identity changed")
    train = [r for r in records if r["participant_id"] in train_ids]
    val = [r for r in records if r["participant_id"] in val_ids]
    if {r["participant_id"] for r in train} != train_ids or {r["participant_id"] for r in val} != val_ids:
        raise ValueError("Development recordings incomplete")
    stats = fit_scaler(train, cache_dir)
    tr_arrays, tr_indices, tr_labels = _samples(train, split, cache_dir)
    for rec in val:
        _load_cache(cache_dir, rec)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = _device(torch)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    identity = {"protocol_hash": protocol["protocol_hash"], "split_id": split["split_id"],
                "fold_id": fold_id, "seed": seed, "max_epochs": max_epochs,
                "train_participants": sorted(train_ids), "validation_participants": sorted(val_ids),
                "development_truth_sha256": _truth_fingerprints(train + val),
                "cache_sha256": _cache_fingerprints(train + val, cache_dir),
                "implementation": _implementation_fingerprint(root),
                "channels": list(CHANNELS), "rate_hz": RATE_HZ, "samples_per_epoch": SAMPLES,
                "architecture": "compact_epoch_cnn_v1", "batch_size": BATCH,
                "optimizer": "Adam", "learning_rate": 1e-3,
                "loss": "five_class_masked_cross_entropy", "selection_rule": f"fixed_final_epoch_{max_epochs}",
                "class_order": list(CLASS_ORDER), "scaler": stats, "torch_version": str(torch.__version__),
                "device": str(device), "cudnn_version": torch.backends.cudnn.version(),
                "deterministic_algorithms": True, "cudnn_deterministic": True,
                "cudnn_benchmark": False, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    sampler = torch.Generator().manual_seed(seed)
    model = build_model(torch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    identity_hash = content_id(identity)
    log_path = checkpoint.with_suffix(".jsonl")
    start = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("identity") != identity:
            raise ValueError("Checkpoint protocol, data, scaler, or configuration changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["completed_epochs"]
        torch.set_rng_state(state["torch_rng"])
        sampler.set_state(state["sampler_rng"])
        if device.type == "cuda":
            if state["cuda_rng"] is None:
                raise ValueError("Checkpoint device RNG missing")
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        _training_log(log_path, identity_hash, start, state.get("last_epoch"))
    else:
        _training_log(log_path, identity_hash, 0, None)
    for epoch in range(start, max_epochs):
        started = time.monotonic()
        peak_rss = _check_host_memory()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        order = torch.randperm(len(tr_indices), generator=sampler).tolist()
        loss_total, valid_total, batches, skipped = 0.0, 0, 0, 0
        for batch_number, offset in enumerate(range(0, len(order), BATCH)):
            if batch_number % 100 == 0:
                peak_rss = max(peak_rss, _check_host_memory())
            chosen = order[offset:offset+BATCH]
            raw = np.stack([tr_arrays[tr_indices[i][0]][tr_indices[i][1]] for i in chosen])
            x = torch.from_numpy(_scale(raw, stats)).to(device)
            y = torch.as_tensor(tr_labels[chosen], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            valid = y >= 0
            if not bool(valid.any()):
                skipped += 1
                continue
            loss = masked_loss(torch, model(x), y, valid)
            loss.backward()
            _check_gradients(torch, model)
            optimizer.step()
            _check_parameters(torch, model)
            count = int(valid.sum().item())
            loss_total += float(loss.detach().cpu()) * count
            valid_total += count
            batches += 1
        peak_rss = max(peak_rss, _check_host_memory())
        if valid_total == 0 or not np.isfinite(loss_total):
            raise FloatingPointError("Waveform epoch has no finite supervised loss")
        gpu_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        if gpu_peak > MAX_GPU_BYTES:
            raise RuntimeError("Waveform job exceeds GPU memory guard")
        epoch_row = {"event": "waveform_epoch", "identity_hash": identity_hash, "epoch": epoch+1,
                     "train_loss": loss_total / valid_total, "valid_examples": valid_total,
                     "trained_batches": batches, "all_invalid_batches": skipped,
                     "wall_seconds": time.monotonic()-started, "peak_host_bytes": peak_rss,
                     "peak_gpu_bytes": gpu_peak, "cpu_threads": MAX_THREADS, "device": str(device)}
        state = {"identity": identity, "completed_epochs": epoch+1,
                 "complete": epoch+1 >= max_epochs, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                 "sampler_rng": sampler.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                 "last_epoch": epoch_row}
        _save_checkpoint(torch, checkpoint, state)
        append_event(log_path, epoch_row)
        print(json_text(epoch_row).strip(), flush=True)
        if stop_after is not None and epoch+1 >= stop_after:
            break
    completed = state["completed_epochs"] if "state" in locals() else start
    score = None
    if completed == max_epochs:
        va_arrays, va_indices, va_labels = _samples(val, split, cache_dir)
        model.eval()
        guesses = []
        with torch.no_grad():
            for offset in range(0, len(va_indices), BATCH):
                selected = va_indices[offset:offset+BATCH]
                raw = np.stack([va_arrays[r][i] for r, i in selected])
                x = torch.from_numpy(_scale(raw, stats)).to(device)
                guesses.extend(model(x).argmax(dim=1).cpu().numpy().tolist())
        valid = va_labels >= 0
        score = _f1(va_labels[valid], np.asarray(guesses, np.int64)[valid])
    return {"checkpoint": str(checkpoint.resolve()), "completed_epochs": completed,
            "complete": completed == max_epochs, "outer_fold_macro_f1_descriptive": score,
            "selection_rule": f"fixed_final_epoch_{max_epochs}", "training_log": str(log_path.resolve())}


def predict_signal_epochs(torch, model, epochs, stats: dict, device, expected_n: int) -> dict:
    """Signal-only inference over every complete original epoch, including invalid ones."""
    if type(expected_n) is not int or expected_n < 1:
        raise ValueError("Invalid full PSG epoch count")
    indices, onsets, labels, probabilities, batch = [], [], [], [], []
    def flush():
        if not batch:
            return
        raw = np.stack(batch)
        x = torch.from_numpy(_scale(raw, stats)).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(x), dim=1).cpu().numpy()
        if prob.shape != (len(batch), 5) or not np.isfinite(prob).all():
            raise ValueError("Invalid waveform probabilities")
        probabilities.extend(prob.tolist())
        labels.extend(prob.argmax(axis=1).tolist())
        batch.clear()
    model.eval()
    for epoch in epochs:
        if set(epoch) & {"reference_label", "valid_mask", "hypnogram", "truth"}:
            raise ValueError("Reference data cannot enter waveform inference")
        index = epoch["epoch_index"]
        if index != len(indices) or epoch["onset_seconds"] != index * 30.0 or index >= expected_n:
            raise ValueError("Signal inference requires the full original PSG grid")
        waveform = []
        for name in CHANNELS:
            item = epoch["channels"][name]
            sample = np.asarray(item["samples_uv"])
            if item["unit"] != "uV" or item["sample_rate_hz"] != RATE_HZ or \
                    sample.shape != (SAMPLES,) or not np.isfinite(sample).all():
                raise ValueError("Waveform inference channel or units differ")
            waveform.append(sample)
        batch.append(np.stack(waveform))
        indices.append(index)
        onsets.append(index * 30.0)
        if len(batch) == BATCH:
            flush()
    flush()
    if len(indices) != expected_n:
        raise ValueError("Signal inference omitted complete PSG epochs")
    return {"epoch_index": np.asarray(indices, np.int64), "onset_seconds": np.asarray(onsets, np.float64),
            "hard_label": np.asarray(labels, np.int8), "probabilities": np.asarray(probabilities, np.float64)}


def predict_fold(root: Path, fold_id: int, checkpoint: Path, record_id: str,
                 data_root: Path, output: Path) -> dict:
    torch = _torch()
    if not 0 <= fold_id < 5:
        raise ValueError("Unapproved development fold")
    _writable_path(output, data_root)
    protocol, split, records = development_records(root)
    fold = split["folds"][fold_id]
    record = next((r for r in records if r["recording_id"] == record_id), None)
    if record is None or record["participant_id"] not in fold["validation"]:
        raise ValueError("OOF prediction requires a validation participant of this fold")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    identity = state["identity"]
    if identity["protocol_hash"] != protocol["protocol_hash"] or identity["split_id"] != split["split_id"] or \
            identity["fold_id"] != fold_id or identity["validation_participants"] != sorted(fold["validation"]) or \
            record["participant_id"] in identity["train_participants"] or \
            identity["channels"] != list(CHANNELS) or identity["class_order"] != list(CLASS_ORDER) or \
            identity["train_participants"] != sorted(fold["train"]) or \
            identity["rate_hz"] != RATE_HZ or identity["samples_per_epoch"] != SAMPLES or \
            identity["architecture"] != "compact_epoch_cnn_v1" or identity["batch_size"] != BATCH or \
            identity["optimizer"] != "Adam" or identity["learning_rate"] != 1e-3 or \
            identity["loss"] != "five_class_masked_cross_entropy" or \
            identity["selection_rule"] != f"fixed_final_epoch_{EPOCHS}" or \
            identity["seed"] not in (17, 43, 101) or \
            identity["max_epochs"] != EPOCHS or state["completed_epochs"] != EPOCHS or \
            state.get("complete") is not True:
        raise ValueError("OOF checkpoint identity or separation differs")
    source = (Path(data_root) / record["psg"]).resolve()
    if not source.is_relative_to(Path(data_root).resolve()) or file_sha256(source) != record["psg_sha256"]:
        raise ValueError("Prediction PSG differs from verified readiness")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = _device(torch)
    if (str(device) != identity["device"] or
            str(torch.__version__) != identity["torch_version"] or
            torch.backends.cudnn.version() != identity["cudnn_version"] or
            os.environ["CUBLAS_WORKSPACE_CONFIG"] != identity["cublas_workspace_config"] or
            _implementation_fingerprint(root) != identity["implementation"]):
        raise ValueError("OOF runtime or implementation differs from trained checkpoint")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model = build_model(torch).to(device)
    model.load_state_dict(state["model"])
    arrays = predict_signal_epochs(torch, model, iter_signal_epochs(record, data_root, CHANNELS),
                                   identity["scaler"], device, record["n_epochs"])
    provenance = {"candidate": "compact_epoch_cnn_v1", "checkpoint_sha256": file_sha256(checkpoint),
                  "fold_id": fold_id, "seed": identity["seed"], "training_participants": identity["train_participants"],
                  "source_psg_sha256": record["psg_sha256"], "input_channels": list(CHANNELS),
                  "input_unit": "uV", "sample_rate_hz": RATE_HZ,
                  "normalization": identity["scaler"], "torch_version": identity["torch_version"]}
    return save_prediction(output, participant_id=record["participant_id"], recording_id=record_id,
                           model_id="compact-waveform-v1", protocol_hash=protocol["protocol_hash"],
                           registry_hash=protocol["registry_hash"], provenance=provenance, **arrays)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare", "train_or_load", "predict", "export_provenance", "validate_output"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, default=Path.cwd())
        if name in ("prepare", "predict"):
            p.add_argument("--data-root", type=Path, required=True)
        if name in ("prepare", "train_or_load"):
            p.add_argument("--cache-dir", type=Path, required=True)
        if name in ("train_or_load", "predict", "export_provenance"):
            p.add_argument("--fold", type=int, required=True)
            p.add_argument("--checkpoint", type=Path, required=True)
        if name == "train_or_load":
            p.add_argument("--seed", type=int, default=SEED)
        if name == "predict":
            p.add_argument("--recording-id", required=True)
            p.add_argument("--output", type=Path, required=True)
        if name == "validate_output":
            p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.action == "prepare":
        _, _, records = development_records(root)
        with compute_lease(root, "waveform-prepare"):
            for rec in records:
                prepare_record(rec, args.data_root, args.cache_dir)
        print(json_text({"prepared_recordings": len(records), "cache_dir": str(args.cache_dir.resolve())}))
    elif args.action == "train_or_load":
        with compute_lease(root, f"waveform-fold-{args.fold}-seed-{args.seed}"):
            print(json_text(fit_fold(root, args.fold, args.cache_dir, args.checkpoint, seed=args.seed)))
    elif args.action == "predict":
        with compute_lease(root, f"waveform-predict-fold-{args.fold}"):
            print(json_text(predict_fold(root, args.fold, args.checkpoint, args.recording_id,
                                         args.data_root, args.output)))
    elif args.action == "export_provenance":
        torch = _torch()
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if state["identity"]["fold_id"] != args.fold:
            raise ValueError("Checkpoint fold identity mismatch")
        print(json_text({"checkpoint_sha256": file_sha256(args.checkpoint),
                         "identity": state["identity"], "completed_epochs": state["completed_epochs"],
                         "selection_rule": f"fixed_final_epoch_{state['identity']['max_epochs']}"}))
    else:
        meta, arrays = load_prediction(args.output)
        print(json_text({"recording_id": arrays["recording_id"], "epochs": len(arrays["epoch_index"]),
                         "model_id": meta["model_id"], "payload_sha256": meta["payload_sha256"]}))


if __name__ == "__main__":
    main()
