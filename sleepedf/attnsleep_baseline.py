"""Development-only native AttnSleep adapter (isolated CPython 3.7 worker)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

import numpy as np

from .contracts import CLASS_ORDER, content_id, json_text, read_json
from .dataset import iter_signal_epochs
from .predictions import load_prediction, save_prediction
from .protocol import development_records, load_development_truth
from .research import atomic_json, compute_lease, file_sha256
from .waveform import _writable_path

SEED = 17
SEEDS = (17, 43, 101)
EPOCHS = 100
CHANNEL = "EEG Fpz-Cz"
SAMPLES = 3000


def _signal_paths(cache_dir: Path, record: dict) -> tuple[Path, Path]:
    rid = record["recording_id"]
    if not isinstance(rid, str) or not rid or any(ch in rid for ch in ("/", "\\", ":", ".")):
        raise ValueError("Invalid AttnSleep recording identity")
    return cache_dir / (rid + ".npy"), cache_dir / (rid + ".json")


def prepare_signal_record(record: dict, data_root: Path, cache_dir: Path) -> dict:
    """Export only calibrated Fpz-Cz microvolts on the original PSG grid."""
    import numpy.lib.format as fmt
    _writable_path(cache_dir, data_root)
    data_path, meta_path = _signal_paths(cache_dir, record)
    n = record.get("n_epochs")
    if type(n) is not int or n < 1:
        raise ValueError("Invalid complete-epoch count")
    source = (data_root / record["psg"]).resolve()
    if not source.is_relative_to(data_root.resolve()) or not source.is_file() or \
            file_sha256(source) != record["psg_sha256"]:
        raise ValueError("AttnSleep PSG differs from verified readiness")
    if data_path.exists() or meta_path.exists():
        return _load_signal_cache(cache_dir, record, return_meta=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_dir, prefix=".attn-eeg-", suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
    signal = None
    try:
        signal = fmt.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(n, 1, SAMPLES))
        seen = 0
        for epoch in iter_signal_epochs(record, data_root, (CHANNEL,)):
            if (seen >= n or epoch["recording_id"] != record["recording_id"] or
                    epoch["participant_id"] != record["participant_id"] or
                    epoch["epoch_index"] != seen or epoch["onset_seconds"] != seen * 30.0 or
                    set(epoch["channels"]) != {CHANNEL}):
                raise ValueError("AttnSleep EEG epoch differs from original PSG grid")
            channel = epoch["channels"][CHANNEL]
            values = np.asarray(channel["samples_uv"])
            if (channel["unit"] != "uV" or channel["sample_rate_hz"] != 100 or
                    values.shape != (SAMPLES,) or not np.isfinite(values).all()):
                raise ValueError("AttnSleep EEG requires 100 Hz finite physical microvolts")
            signal[seen, 0] = values
            if not np.isfinite(signal[seen, 0]).all():
                raise ValueError("AttnSleep EEG exceeds float32 range")
            seen += 1
        if seen != n:
            raise ValueError("AttnSleep EEG iterator omitted a PSG epoch")
        signal.flush()
        signal._mmap.close()
        signal = None
        os.replace(temporary, data_path)
        meta = {"schema_version": "1.0", "artifact_type": "attnsleep_single_eeg_cache",
                "recording_id": record["recording_id"], "participant_id": record["participant_id"],
                "source_psg_sha256": record["psg_sha256"], "n_epochs": n,
                "channels": [CHANNEL], "unit": "uV", "rate_hz": 100,
                "samples_per_epoch": SAMPLES, "payload_sha256": file_sha256(data_path)}
        atomic_json(meta_path, meta, immutable=True)
        return meta
    finally:
        if signal is not None:
            signal._mmap.close()
        temporary.unlink(missing_ok=True)


def _load_signal_cache(cache_dir: Path, record: dict, *, return_meta: bool = False):
    data_path, meta_path = _signal_paths(cache_dir, record)
    if not data_path.is_file() or not meta_path.is_file():
        raise ValueError("Incomplete AttnSleep EEG cache")
    meta = read_json(meta_path)
    if (meta.get("schema_version") != "1.0" or
            meta.get("artifact_type") != "attnsleep_single_eeg_cache" or
            meta.get("recording_id") != record["recording_id"] or
            meta.get("participant_id") != record["participant_id"] or
            meta.get("source_psg_sha256") != record["psg_sha256"] or
            meta.get("n_epochs") != record["n_epochs"] or
            meta.get("channels") != [CHANNEL] or meta.get("unit") != "uV" or
            meta.get("rate_hz") != 100 or meta.get("samples_per_epoch") != SAMPLES or
            meta.get("payload_sha256") != file_sha256(data_path)):
        raise ValueError("AttnSleep single-channel cache provenance differs")
    signal = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if signal.dtype != np.float32 or signal.shape != (record["n_epochs"], 1, SAMPLES):
        raise ValueError("AttnSleep single-channel cache shape/dtype differs")
    return meta if return_meta else signal


def _runtime(root: Path) -> dict:
    executable = root / ".venvs" / "attn37" / "python.exe"
    if not executable.is_file():
        raise ValueError("Isolated AttnSleep Python 3.7 runtime is absent")
    evidence = root / "research" / "runtimes" / "attn37"
    freeze = evidence / "freeze.txt"
    provenance = evidence / "provenance.json"
    if not freeze.is_file() or not provenance.is_file():
        raise ValueError("AttnSleep runtime freeze is absent")
    runtime = read_json(provenance)
    installation = runtime.get("install")
    verification = runtime.get("verification")
    if (not isinstance(installation, dict) or not isinstance(verification, dict) or
            verification.get("freeze_sha256") != file_sha256(freeze) or
            runtime["extraction"].get("python_exe_sha256") != file_sha256(executable) or
            runtime["requirements_sha256"] != file_sha256(root / "requirements" / "attn37.in") or
            installation["pip_report_sha256"] != file_sha256(Path(installation["pip_report_path"]))):
        raise ValueError("Isolated AttnSleep runtime verification differs")
    wheel = installation.get("local_verified_torch_wheel")
    if wheel is not None and (wheel["wheel"]["sha256"] != file_sha256(Path(wheel["wheel"]["path"])) or
                              wheel["wheel"]["manifest_sha256"] != file_sha256(Path(wheel["wheel"]["manifest_path"])) or
                              wheel["pip_report_sha256"] != file_sha256(Path(wheel["pip_report_path"]))):
        raise ValueError("Verified native Torch wheel provenance differs")
    return {"python": str(executable.resolve()), "python_sha256": file_sha256(executable),
            "freeze_sha256": file_sha256(freeze),
            "requirements_sha256": runtime["requirements_sha256"],
            "python_archive_sha256": runtime["python_archive"]["sha256"],
            "get_pip_sha256": runtime["get_pip"]["sha256"],
            "package_install_report_sha256": installation["pip_report_sha256"],
            "torch_wheel_sha256": wheel["wheel"]["sha256"] if wheel else None,
            "torch_wheel_manifest_sha256": wheel["wheel"]["manifest_sha256"] if wheel else None,
            "torch_install_report_sha256": wheel["pip_report_sha256"] if wheel else None,
            "installed_versions": verification["versions"]}


def _source(root: Path) -> dict:
    registry = read_json(root / "research" / "sources" / "attnsleep.json")
    vendor = root / "vendor" / "attnsleep"
    hashes = {}
    for name in sorted(registry["files"]):
        actual = file_sha256(vendor / name)
        if actual != registry["files"][name]["sha256"]:
            raise ValueError("Pinned AttnSleep vendor source changed: " + name)
        hashes[name] = actual
    return {"source_manifest_sha256": file_sha256(root / "research" / "sources" / "attnsleep.json"),
            "files": hashes}


def _label_path(label_dir: Path, record: dict) -> Path:
    return label_dir / (record["recording_id"] + ".npy")


def prepare(root: Path, data_root: Path, cache_dir: Path, label_dir: Path) -> dict:
    """Cache full PSG grids and private development-only labels."""
    _writable_path(cache_dir, data_root)
    _writable_path(label_dir, data_root)
    protocol, split, records = development_records(root)
    if len(split["participants"]["development"]) != 60:
        raise ValueError("Original 60-participant development split required")
    label_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        prepare_signal_record(record, data_root, cache_dir)
        truth = load_development_truth(record, split)
        labels = np.asarray(truth["reference_label"])
        valid = np.asarray(truth["valid_mask"], dtype=bool)
        if (labels.shape != (record["n_epochs"],) or valid.shape != labels.shape or
                not np.array_equal(truth["epoch_index"], np.arange(len(labels))) or
                not np.array_equal(truth["onset_seconds"], np.arange(len(labels)) * 30.0) or
                np.any(valid & ((labels < 0) | (labels > 4)))):
            raise ValueError("Development truth does not match the full PSG grid")
        exported = np.where(valid, labels, -1).astype(np.int8)
        path = _label_path(label_dir, record)
        if path.exists():
            old = np.load(path, allow_pickle=False)
            if old.dtype != np.int8 or not np.array_equal(old, exported):
                raise ValueError("Existing development label export changed")
        else:
            with tempfile.NamedTemporaryFile(dir=label_dir, suffix=".npy", delete=False) as handle:
                temporary = Path(handle.name)
            try:
                np.save(temporary, exported, allow_pickle=False)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    return {"artifact_type": "attnsleep_development_prepare", "protocol_hash": protocol["protocol_hash"],
            "split_id": split["split_id"], "recordings": len(records), "label_dir": str(label_dir.resolve())}


def _record_entry(record: dict, cache_dir: Path, label_dir: Path, *, training: bool,
                  split: dict | None = None) -> dict:
    signal = _load_signal_cache(cache_dir, record)
    if signal.shape != (record["n_epochs"], 1, SAMPLES):
        raise ValueError("AttnSleep requires the verified 100 Hz full EEG grid")
    cache_path, meta_path = _signal_paths(cache_dir, record)
    entry = {"recording_id": record["recording_id"], "participant_id": record["participant_id"],
             "n_epochs": record["n_epochs"], "source_psg_sha256": record["psg_sha256"],
             "signal_path": str(cache_path.resolve()), "signal_sha256": file_sha256(cache_path),
             "signal_meta_sha256": file_sha256(meta_path)}
    if training:
        if split is None:
            raise ValueError("Training label export requires the canonical development split")
        truth = load_development_truth(record, split)
        truth_path = Path(record["truth_path"])
        frozen = record.get("frozen_truth")
        if (not isinstance(frozen, dict) or
                file_sha256(truth_path) != frozen.get("payload_sha256") or
                file_sha256(truth_path.with_suffix(".json")) != frozen.get("sidecar_sha256")):
            raise ValueError("AttnSleep training truth differs from immutable development freeze")
        label_path = _label_path(label_dir, record)
        label = np.load(label_path, mmap_mode="r", allow_pickle=False)
        if label.dtype != np.int8 or label.shape != (record["n_epochs"],) or \
                np.any((label < -1) | (label > 4)):
            raise ValueError("Training-only masked label export is invalid")
        expected_labels = np.where(truth["valid_mask"], truth["reference_label"], -1).astype(np.int8)
        if (not np.array_equal(truth["epoch_index"], np.arange(record["n_epochs"])) or
                not np.array_equal(truth["onset_seconds"], np.arange(record["n_epochs"]) * 30.0) or
                not np.array_equal(label, expected_labels)):
            raise ValueError("Training label export differs from the frozen truth transformation")
        entry.update(label_path=str(label_path.resolve()), label_sha256=file_sha256(label_path),
                     truth_sha256=frozen["payload_sha256"],
                     truth_sidecar_sha256=frozen["sidecar_sha256"])
    return entry


def _training_ancestry(record: dict, cache_dir: Path, label_dir: Path, saved: dict) -> dict:
    """Verify stored training references without opening training inputs at inference."""
    cache_path, _ = _signal_paths(cache_dir, record)
    frozen = record.get("frozen_truth", {})
    expected = {"recording_id": record["recording_id"], "participant_id": record["participant_id"],
                "n_epochs": record["n_epochs"], "source_psg_sha256": record["psg_sha256"],
                "signal_path": str(cache_path.resolve()),
                "label_path": str(_label_path(label_dir, record).resolve()),
                "truth_sha256": frozen.get("payload_sha256"),
                "truth_sidecar_sha256": frozen.get("sidecar_sha256")}
    digests = {"signal_sha256", "signal_meta_sha256", "label_sha256"}
    if (set(saved) != set(expected) | digests or
            any(saved[name] != value for name, value in expected.items()) or
            any(not isinstance(saved[name], str) or len(saved[name]) != 64 or
                any(char not in "0123456789abcdef" for char in saved[name]) for name in digests)):
        raise ValueError("Saved training ancestry differs from canonical development metadata")
    return saved


def _manifest(root: Path, fold_id: int, cache_dir: Path, label_dir: Path, seed: int = SEED,
              *, frozen_training: list[dict] | None = None) -> dict:
    if not 0 <= fold_id < 5 or seed not in SEEDS:
        raise ValueError("AttnSleep fold must be 0..4 and seed must be 17, 43, or 101")
    protocol, split, records = development_records(root)
    fold = split["folds"][fold_id]
    train_ids, val_ids = set(fold["train"]), set(fold["validation"])
    if len(train_ids) != 48 or len(val_ids) != 12 or train_ids & val_ids:
        raise ValueError("AttnSleep requires a 48/12 participant fold")
    train = [r for r in records if r["participant_id"] in train_ids]
    val = [r for r in records if r["participant_id"] in val_ids]
    if {r["participant_id"] for r in train} != train_ids or {r["participant_id"] for r in val} != val_ids:
        raise ValueError("Development fold recordings are incomplete")
    if frozen_training is None:
        train_entries = [_record_entry(r, cache_dir, label_dir, training=True, split=split) for r in train]
    else:
        if [entry["recording_id"] for entry in frozen_training] != [r["recording_id"] for r in train]:
            raise ValueError("Saved training recording set or order differs")
        train_entries = [_training_ancestry(r, cache_dir, label_dir, entry)
                         for r, entry in zip(train, frozen_training)]
    identity = {"artifact_type": "attnsleep_native_fit_manifest", "schema_version": "1.0",
                "protocol_hash": protocol["protocol_hash"], "split_id": split["split_id"],
                "registry_hash": protocol["registry_hash"], "fold_id": fold_id,
                "seed": seed, "epochs": EPOCHS, "train_participants": sorted(train_ids),
                "validation_participants": sorted(val_ids),
                "train": train_entries,
                "validation": [_record_entry(r, cache_dir, label_dir, training=False) for r in val],
                "channel": CHANNEL, "channel_index": 0, "unit": "uV", "rate_hz": 100,
                "samples_per_epoch": 3000, "class_order": list(CLASS_ORDER),
                "loss": "native_weighted_CrossEntropyLoss_train_only_class_weights",
                "optimizer": {"type": "Adam", "lr": 0.001, "weight_decay": 0.001,
                              "amsgrad": True, "lr_after_epoch_10": 0.0001},
                "selection_rule": "fixed_final_epoch_100", "batch_size_requested": 128,
                "source": _source(root), "runtime": _runtime(root),
                "adapter_sha256": file_sha256(Path(__file__)),
                "worker_sha256": file_sha256(root / "tools" / "attn37_worker.py")}
    identity["manifest_id"] = content_id(identity)
    return identity


def _inference_manifest(root: Path, fold_id: int, cache_dir: Path, label_dir: Path,
                        run_dir: Path, seed: int) -> dict:
    saved = read_json(run_dir / "manifest.json")
    if saved.get("manifest_id") != content_id({key: value for key, value in saved.items() if key != "manifest_id"}):
        raise ValueError("Saved native fit manifest identity differs")
    expected = _manifest(root, fold_id, cache_dir, label_dir, seed, frozen_training=saved["train"])
    if saved != expected:
        raise ValueError("AttnSleep inference ancestry changed since fit")
    return saved


def _worker(root: Path, action: str, manifest_path: Path, run_dir: Path,
            *extra: str) -> dict:
    executable = _runtime(root)["python"]
    cmd = [executable, "-I", str(root / "tools" / "attn37_worker.py"), action,
           "--manifest", str(manifest_path), "--run-dir", str(run_dir), *extra]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               NUMEXPR_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="")
    log = run_dir / (action + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    with subprocess.Popen(cmd, cwd=root, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        timeout = threading.Timer(24 * 3600 + 300, process.kill)
        timeout.start()
        try:
            with log.open("a", encoding="utf-8", newline="\n") as handle:
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    if line.startswith("ATTN37_RESULT "):
                        lines.append(line)
                    else:
                        print(line, end="", flush=True)
            returncode = process.wait()
        finally:
            timeout.cancel()
    if returncode:
        raise RuntimeError("AttnSleep worker failed; inspect " + str(log))
    if len(lines) != 1:
        raise ValueError("AttnSleep worker returned no unique result")
    result = json.loads(lines[0][len("ATTN37_RESULT "):])
    json_text(result)
    return result


def train_or_load(root: Path, fold_id: int, cache_dir: Path, label_dir: Path,
                  run_dir: Path, seed: int = SEED) -> dict:
    from .attn37_verify import require_verification
    require_verification(root)
    _writable_path(run_dir)
    manifest = _manifest(root, fold_id, cache_dir, label_dir, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    atomic_json(manifest_path, manifest, immutable=True)
    result = _worker(root, "train", manifest_path, run_dir)
    if result.get("manifest_id") != manifest["manifest_id"]:
        raise ValueError("Worker result has a different fit ancestry")
    return result


def profile(root: Path, fold_id: int, cache_dir: Path, label_dir: Path,
            run_dir: Path, seed: int = SEED) -> dict:
    """One disposable development batch; no optimizer step or fit checkpoint."""
    _writable_path(run_dir)
    manifest = _manifest(root, fold_id, cache_dir, label_dir, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    atomic_json(manifest_path, manifest, immutable=True)
    result = _worker(root, "profile", manifest_path, run_dir)
    if (result.get("manifest_id") != manifest["manifest_id"] or
            result.get("artifact_type") != "attnsleep_disposable_development_profile" or
            result.get("cpu_threads") != 4):
        raise ValueError("Native disposable profile identity or resource limit differs")
    return result


def predict(root: Path, fold_id: int, cache_dir: Path, label_dir: Path,
            run_dir: Path, record_id: str, output: Path, seed: int = SEED) -> dict:
    _writable_path(output)
    manifest = _inference_manifest(root, fold_id, cache_dir, label_dir, run_dir, seed)
    manifest_path = run_dir / "manifest.json"
    if read_json(manifest_path) != manifest:
        raise ValueError("AttnSleep prediction manifest changed since fit")
    candidates = [r for r in manifest["validation"] if r["recording_id"] == record_id]
    if len(candidates) != 1 or record_id in {r["recording_id"] for r in manifest["train"]}:
        raise ValueError("Prediction requires an OOF development recording")
    with tempfile.TemporaryDirectory(dir=run_dir, prefix="predict-") as scratch:
        raw_path = Path(scratch) / "raw.npz"
        result = _worker(root, "predict", manifest_path, run_dir,
                         "--recording-id", record_id, "--output", str(raw_path))
        with np.load(raw_path, allow_pickle=False) as raw:
            hard = raw["hard_label"]
            prob = raw["probabilities"]
        entry = candidates[0]
        if (result.get("recording_id") != record_id or result.get("manifest_id") != manifest["manifest_id"] or
                result.get("checkpoint_sha256") != file_sha256(run_dir / "final.pt") or
                hard.shape != (entry["n_epochs"],) or prob.shape != (entry["n_epochs"], 5)):
            raise ValueError("Native prediction coverage or lineage mismatch")
        provenance = {"candidate": "native_attnsleep_1_4_cpu", "fold_id": fold_id,
                      "seed": seed, "training_participants": manifest["train_participants"],
                      "checkpoint_sha256": result["checkpoint_sha256"],
                      "fit_manifest_id": manifest["manifest_id"], "source_psg_sha256": entry["source_psg_sha256"],
                      "input_channel": CHANNEL, "input_unit": "uV", "sample_rate_hz": 100,
                      "selection_rule": "fixed_final_epoch_100"}
        return save_prediction(output, participant_id=entry["participant_id"], recording_id=record_id,
                               epoch_index=np.arange(entry["n_epochs"]),
                               onset_seconds=np.arange(entry["n_epochs"]) * 30.0,
                               hard_label=hard, probabilities=prob, model_id="native-attnsleep-v1",
                               protocol_hash=manifest["protocol_hash"],
                               registry_hash=manifest["registry_hash"], provenance=provenance)


def export_provenance(root: Path, fold_id: int, cache_dir: Path,
                      label_dir: Path, run_dir: Path, seed: int = SEED) -> dict:
    manifest = _inference_manifest(root, fold_id, cache_dir, label_dir, run_dir, seed)
    if read_json(run_dir / "manifest.json") != manifest:
        raise ValueError("AttnSleep fit ancestry changed")
    result = _worker(root, "inspect", run_dir / "manifest.json", run_dir)
    return {"manifest": manifest, "worker": result,
            "checkpoint_sha256": file_sha256(run_dir / "final.pt")}


def validate_output(root: Path, fold_id: int, cache_dir: Path, label_dir: Path,
                    run_dir: Path, output: Path, seed: int = SEED) -> dict:
    manifest = _inference_manifest(root, fold_id, cache_dir, label_dir, run_dir, seed)
    if read_json(run_dir / "manifest.json") != manifest:
        raise ValueError("AttnSleep fit ancestry changed")
    inspection = _worker(root, "inspect", run_dir / "manifest.json", run_dir)
    if inspection.get("manifest_id") != manifest["manifest_id"] or inspection.get("completed_epochs") != 100:
        raise ValueError("Native final checkpoint is incomplete")
    meta, arrays = load_prediction(output)
    entries = [r for r in manifest["validation"] if r["recording_id"] == arrays["recording_id"]]
    if len(entries) != 1:
        raise ValueError("Prediction is not an OOF validation recording")
    entry = entries[0]
    provenance = meta["provenance"]
    if (meta["model_id"] != "native-attnsleep-v1" or meta["protocol_hash"] != manifest["protocol_hash"] or
            meta["registry_hash"] != manifest["registry_hash"] or
            provenance.get("fit_manifest_id") != manifest["manifest_id"] or
            provenance.get("checkpoint_sha256") != file_sha256(run_dir / "final.pt") or
            provenance.get("source_psg_sha256") != entry["source_psg_sha256"] or
            arrays["participant_id"] != entry["participant_id"] or len(arrays["epoch_index"]) != entry["n_epochs"]):
        raise ValueError("AttnSleep output lineage or full timeline differs")
    return {"artifact_type": "attnsleep_prediction_validation", "recording_id": arrays["recording_id"],
            "epochs": len(arrays["epoch_index"]), "prediction_sha256": file_sha256(output)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "profile", "train_or_load", "predict", "export_provenance", "validate_output"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/attnsleep/eeg-cache"))
    parser.add_argument("--label-dir", type=Path, default=Path("runs/attnsleep/labels"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEED)
    parser.add_argument("--recording-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    cache, labels = (p if p.is_absolute() else root / p for p in (args.cache_dir, args.label_dir))
    run = args.run_dir or root / "runs" / "attnsleep" / ("fold-" + str(args.fold) + "-seed-" + str(args.seed))
    if not run.is_absolute():
        run = root / run
    if args.action == "prepare":
        if args.data_root is None:
            parser.error("prepare requires --data-root")
        with compute_lease(root, "attnsleep-prepare"):
            result = prepare(root, args.data_root.resolve(), cache, labels)
    else:
        if args.fold is None:
            parser.error("action requires --fold")
        if args.action == "profile":
            with compute_lease(root, "attnsleep-profile-fold-" + str(args.fold) + "-seed-" + str(args.seed)):
                result = profile(root, args.fold, cache, labels, run, args.seed)
        elif args.action == "train_or_load":
            with compute_lease(root, "attnsleep-fold-" + str(args.fold) + "-seed-" + str(args.seed)):
                result = train_or_load(root, args.fold, cache, labels, run, args.seed)
        elif args.action == "predict":
            if not args.recording_id or args.output is None:
                parser.error("predict requires --recording-id and --output")
            with compute_lease(root, "attnsleep-predict-" + str(args.fold)):
                result = predict(root, args.fold, cache, labels, run, args.recording_id, args.output, args.seed)
        elif args.action == "export_provenance":
            result = export_provenance(root, args.fold, cache, labels, run, args.seed)
        else:
            if args.output is None:
                parser.error("validate_output requires --output")
            result = validate_output(root, args.fold, cache, labels, run, args.output, args.seed)
    print(json_text(result))


if __name__ == "__main__":
    main()
