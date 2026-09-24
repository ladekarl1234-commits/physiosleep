"""Run all five native AttnSleep development folds under one resumable lease."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import numpy as np

from . import attnsleep_baseline as native
from .attn37_verify import require_verification
from .contracts import content_id, json_text
from .evaluation import evaluate_saved_records
from .predictions import load_prediction
from .protocol import development_records, load_development_truth
from .research import append_event, atomic_json, compute_lease, file_sha256


SOURCES = (
    "sleepedf/attnsleep_experiment.py", "sleepedf/attnsleep_baseline.py",
    "sleepedf/attn37_verify.py", "tools/verify_attn37_native.py",
    "tools/attn37_worker.py", "sleepedf/protocol.py", "sleepedf/contracts.py",
    "sleepedf/research.py", "sleepedf/predictions.py", "sleepedf/evaluation.py",
    "sleepedf/splits.py", "sleepedf/waveform.py", "sleepedf/dataset.py",
    "sleepedf/readers.py", "sleepedf/timing.py", "sleepedf/audit.py",
    "requirements/research.lock.txt", "requirements/attn37.in",
    "research/runtimes/attn37/freeze.txt", "research/runtimes/attn37/provenance.json",
)


def _snapshot(root: Path, output: Path, files: dict[str, str]) -> None:
    for name, expected in files.items():
        source, target = root / name, output / "source_snapshot" / name
        if file_sha256(source) != expected:
            raise ValueError("AttnSleep source changed before snapshot")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if file_sha256(target) != expected:
            raise ValueError("AttnSleep frozen source snapshot differs")
    atomic_json(output / "source_snapshot/manifest.json",
                {"capture_scope": "before fitting any fold", "files": files}, immutable=True)


def _same_fit(root: Path, cache: Path, labels: Path, frozen: dict) -> None:
    actual = native._manifest(root, frozen["fold_id"], cache, labels, frozen["seed"])
    if actual != frozen:
        raise ValueError("AttnSleep fold inputs changed after experiment freeze")


def _prediction(path: Path, record: dict, manifest: dict, checkpoint_sha256: str) -> None:
    metadata, arrays = load_prediction(path)
    provenance = metadata["provenance"]
    n = record["n_epochs"]
    if (metadata["model_id"] != "native-attnsleep-v1" or
            "probabilities" not in arrays or
            not np.array_equal(arrays["hard_label"], np.argmax(arrays["probabilities"], axis=1)) or
            metadata["protocol_hash"] != manifest["protocol_hash"] or
            metadata["registry_hash"] != manifest["registry_hash"] or
            arrays["recording_id"] != record["recording_id"] or
            arrays["participant_id"] != record["participant_id"] or
            arrays["participant_id"] not in manifest["validation_participants"] or
            arrays["participant_id"] in manifest["train_participants"] or
            not np.array_equal(arrays["epoch_index"], np.arange(n)) or
            not np.array_equal(arrays["onset_seconds"], np.arange(n) * 30.0) or
            provenance.get("fit_manifest_id") != manifest["manifest_id"] or
            provenance.get("checkpoint_sha256") != checkpoint_sha256 or
            provenance.get("source_psg_sha256") != record["psg_sha256"] or
            provenance.get("training_participants") != manifest["train_participants"] or
            provenance.get("fold_id") != manifest["fold_id"] or
            provenance.get("seed") != manifest["seed"] or
            provenance.get("input_channel") != native.CHANNEL or
            provenance.get("input_unit") != "uV" or
            provenance.get("sample_rate_hz") != 100 or
            provenance.get("selection_rule") != "fixed_final_epoch_100"):
        raise ValueError("AttnSleep saved prediction differs from frozen OOF ancestry or timeline")


def run(root: Path, cache: Path, labels: Path, seed: int) -> dict:
    with compute_lease(root, f"attnsleep-controller-seed-{seed}"):
        return _run_locked(root, cache, labels, seed)


def _run_locked(root: Path, cache: Path, labels: Path, seed: int) -> dict:
    if seed not in native.SEEDS:
        raise ValueError("Use a registered development seed")
    verification = require_verification(root)
    protocol, split, records = development_records(root)
    if [fold["fold_id"] for fold in split["folds"]] != list(range(5)):
        raise ValueError("The original five development folds are required")
    manifests = [native._manifest(root, fold["fold_id"], cache, labels, seed) for fold in split["folds"]]
    sources = {name: file_sha256(root / name) for name in SOURCES}
    sources.update({"vendor/attnsleep/" + name: digest
                    for name, digest in manifests[0]["source"]["files"].items()})
    config = {"experiment": "native_attnsleep_v1", "seed": seed,
              "protocol_hash": protocol["protocol_hash"], "split_id": split["split_id"],
              "epochs": 100, "selection": "fixed_final_epoch_100", "channel": native.CHANNEL,
              "fold_manifests": manifests, "sources": sources,
              "native_numerical_verification": verification,
              "truth_manifest_sha256": file_sha256(root / "research/development-truth-v1.json")}
    config_hash = content_id(config)
    run_id = f"attnsleep-s{seed}-{config_hash[-10:]}"
    output = root / "runs" / run_id
    native._writable_path(output)
    atomic_json(output / "config.json", config, immutable=True)
    _snapshot(root, output, sources)
    predictions, truths, fits = [], [], []
    for fold, manifest in zip(split["folds"], manifests):
        started = time.monotonic()
        fold_id = fold["fold_id"]
        fold_dir = output / f"fold-{fold_id}"
        if any(file_sha256(root / name) != expected for name, expected in sources.items()):
            raise ValueError("Frozen AttnSleep source or runtime changed before a fold")
        _same_fit(root, cache, labels, manifest)
        if require_verification(root) != config["native_numerical_verification"]:
            raise ValueError("Native AttnSleep numerical verification changed before a fold")
        trained = native.train_or_load(root, fold_id, cache, labels, fold_dir, seed)
        verified = native.export_provenance(root, fold_id, cache, labels, fold_dir, seed)
        inspection = verified["worker"]
        checkpoint_hash = file_sha256(fold_dir / "final.pt")
        if (verified["manifest"] != manifest or verified["checkpoint_sha256"] != checkpoint_hash or
                any(item.get("manifest_id") != manifest["manifest_id"] or
                    item.get("completed_epochs") != 100 or
                    item.get("checkpoint_sha256") != checkpoint_hash
                    for item in (trained, inspection))):
            raise ValueError("AttnSleep final checkpoint differs from the frozen complete fit")
        fit = {"fold_id": fold_id, "seed": seed, "manifest_id": manifest["manifest_id"],
               "checkpoint_sha256": checkpoint_hash, "completed_epochs": 100,
               "training_participants": manifest["train_participants"],
               "validation_participants": manifest["validation_participants"],
               "selection_rule": "fixed_final_epoch_100", "native_inspection": inspection}
        atomic_json(fold_dir / "fit.json", fit, immutable=True)
        fits.append(fit)
        for record in records:
            if record["participant_id"] not in fold["validation"]:
                continue
            path = fold_dir / "predictions" / (record["recording_id"] + ".npz")
            if not path.exists():
                native.predict(root, fold_id, cache, labels, fold_dir, record["recording_id"], path, seed)
            _prediction(path, record, manifest, checkpoint_hash)
            load_development_truth(record, split)
            predictions.append(path)
            truths.append(Path(record["truth_path"]))
        _same_fit(root, cache, labels, manifest)
        if file_sha256(fold_dir / "final.pt") != checkpoint_hash:
            raise ValueError("AttnSleep checkpoint changed during prediction")
        append_event(output / "events.jsonl", {"event": "fold_complete", "fold_id": fold_id,
                     "checkpoint_sha256": checkpoint_hash,
                     "elapsed_seconds_this_invocation": time.monotonic() - started})
        print(f"completed {run_id} fold {fold_id}", flush=True)
    if len(predictions) != len(records) or len(set(predictions)) != len(records):
        raise ValueError("AttnSleep OOF predictions do not cover all development nights exactly once")
    if any(file_sha256(root / name) != expected for name, expected in sources.items()):
        raise ValueError("Frozen AttnSleep source or runtime changed before result publication")
    _snapshot(root, output, sources)
    metrics = evaluate_saved_records(truths, predictions, protocol_hash=protocol["protocol_hash"],
                                     registry_hash=protocol["registry_hash"], model_id="native-attnsleep-v1")
    result = {"status": "DEVELOPMENT_OOF_COMPLETE", "confirmatory": False, "run_id": run_id,
              "config_hash": config_hash, "protocol_hash": protocol["protocol_hash"],
              "metrics": metrics, "folds": fits,
              "prediction_paths": [str(path.resolve()) for path in predictions],
              "truth_paths": [str(path.resolve()) for path in truths],
              "gate_A": "NOT_RUN", "gate_B": "NOT_RUN"}
    atomic_json(output / "result.json", result, immutable=True)
    append_event(root / "runs/experiments.jsonl", {"event": "development_run_complete",
                 "run_id": run_id, "report_sha256": file_sha256(output / "result.json"),
                 "macro_f1": metrics["macro_f1"], "confirmatory": False})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/attnsleep/eeg-cache"))
    parser.add_argument("--label-dir", type=Path, default=Path("runs/attnsleep/labels"))
    parser.add_argument("--seed", type=int, choices=native.SEEDS, default=17)
    args = parser.parse_args()
    root = args.root.resolve()
    cache, labels = (p if p.is_absolute() else root / p for p in (args.cache_dir, args.label_dir))
    print(json_text(run(root, cache.resolve(), labels.resolve(), args.seed)))


if __name__ == "__main__":
    main()
