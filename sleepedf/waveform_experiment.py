"""Resume the fixed five-fold compact waveform development experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

from . import waveform
from .contracts import content_id, json_text
from .evaluation import evaluate_saved_records
from .predictions import load_prediction
from .protocol import development_records, load_development_truth
from .research import append_event, atomic_json, compute_lease, file_sha256


def run(root: Path, data_root: Path, cache_dir: Path, seed: int) -> dict:
    with compute_lease(root, f"waveform-controller-seed-{seed}"):
        return _run_locked(root, data_root, cache_dir, seed)


def _run_locked(root: Path, data_root: Path, cache_dir: Path, seed: int) -> dict:
    if seed not in (17, 43, 101):
        raise ValueError("Use a registered development seed")
    protocol, split, records = development_records(root)
    implementation = waveform._implementation_fingerprint(root)
    controller_sources = {name: file_sha256(root / name) for name in
                          ("sleepedf/waveform_experiment.py", "sleepedf/evaluation.py")}
    config = {"experiment": "compact_epoch_cnn_v1", "seed": seed,
              "protocol_hash": protocol["protocol_hash"], "split_id": split["split_id"],
              "epochs": waveform.EPOCHS, "selection": "fixed_final_epoch_20",
              "channels": list(waveform.CHANNELS), "implementation": implementation,
              "controller_sources": controller_sources,
              "truth_manifest_sha256": file_sha256(root / "research/development-truth-v1.json"),
              "truth": waveform._truth_fingerprints(records),
              "cache": waveform._cache_fingerprints(records, cache_dir)}
    config_hash = content_id(config)
    run_id = f"waveform-s{seed}-{config_hash[-10:]}"
    out = root / "runs" / run_id
    waveform._writable_path(out, data_root)
    atomic_json(out / "config.json", config, immutable=True)
    sources = dict(implementation["source_sha256"], **controller_sources)
    sources[implementation["runtime_lock_path"]] = implementation["runtime_lock_sha256"]
    for relative, expected in sources.items():
        source, target = root / relative, out / "source_snapshot" / relative
        if file_sha256(source) != expected:
            raise ValueError("Source changed during the pre-fit snapshot")
        if target.exists():
            if file_sha256(target) != expected:
                raise ValueError("The frozen experiment source changed")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if file_sha256(target) != expected:
            raise ValueError("Copied source snapshot differs from the frozen configuration")
    atomic_json(out / "source_snapshot/manifest.json",
                {"capture_scope": "before fitting any fold", "files": sources}, immutable=True)
    predictions, truths, fits = [], [], []
    for fold in split["folds"]:
        started = time.monotonic()
        fold_id = fold["fold_id"]
        fold_dir = out / f"fold-{fold_id}"
        checkpoint = fold_dir / "model.pt"
        if (waveform._implementation_fingerprint(root) != config["implementation"] or
            waveform._cache_fingerprints(records, cache_dir) != config["cache"] or
            waveform._truth_fingerprints(records) != config["truth"] or
            any(file_sha256(root / name) != expected for name, expected in controller_sources.items())):
            raise ValueError("Frozen experiment inputs changed before a fold")
        report = waveform.fit_fold(root, fold_id, cache_dir, checkpoint, seed=seed)
        if report["complete"] is not True or report["completed_epochs"] != waveform.EPOCHS:
            raise ValueError("An incomplete fold cannot produce an OOF result")
        checkpoint_hash = file_sha256(checkpoint)
        state = waveform._torch().load(checkpoint, map_location="cpu", weights_only=True)
        identity = state["identity"]
        if (identity["implementation"] != config["implementation"] or
            identity["cache_sha256"] != config["cache"] or
            identity["development_truth_sha256"] != config["truth"] or
            identity["protocol_hash"] != config["protocol_hash"] or
            identity["split_id"] != config["split_id"] or
            identity["fold_id"] != fold_id or identity["seed"] != seed or
            identity["max_epochs"] != config["epochs"] or
            identity["channels"] != config["channels"] or
            identity["selection_rule"] != config["selection"] or
            identity["train_participants"] != sorted(fold["train"]) or
            identity["validation_participants"] != sorted(fold["validation"]) or
            state["completed_epochs"] != config["epochs"] or state.get("complete") is not True):
            raise ValueError("Checkpoint identity differs from the frozen experiment")
        del state
        fit = {"config_hash": config_hash, "checkpoint_sha256": checkpoint_hash,
               "fold_id": fold_id, "seed": seed,
               "training_participants": sorted(fold["train"]),
               "validation_participants": sorted(fold["validation"]),
               "selection_rule": report["selection_rule"],
               "completed_epochs": report["completed_epochs"]}
        atomic_json(fold_dir / "fit.json", fit, immutable=True)
        fits.append(fit)
        validation = [rec for rec in records if rec["participant_id"] in fold["validation"]]
        for rec in validation:
            path = fold_dir / "predictions" / (rec["recording_id"] + ".npz")
            if not path.exists():
                waveform.predict_fold(root, fold_id, checkpoint, rec["recording_id"], data_root, path)
            meta, prediction = load_prediction(path)
            provenance = meta["provenance"]
            if (meta["protocol_hash"] != protocol["protocol_hash"] or
                meta["registry_hash"] != protocol["registry_hash"] or
                meta["model_id"] != "compact-waveform-v1" or
                prediction["recording_id"] != rec["recording_id"] or
                prediction["participant_id"] != rec["participant_id"] or
                provenance["checkpoint_sha256"] != checkpoint_hash or
                provenance["fold_id"] != fold_id or provenance["seed"] != seed or
                provenance["training_participants"] != sorted(fold["train"]) or
                provenance["input_channels"] != list(waveform.CHANNELS) or
                provenance["source_psg_sha256"] != rec["psg_sha256"]):
                raise ValueError("Resumed prediction has different model or participant ancestry")
            load_development_truth(rec, split)
            predictions.append(path)
            truths.append(Path(rec["truth_path"]))
        append_event(out / "events.jsonl", {"event": "fold_complete", "fold_id": fold_id,
                     "checkpoint_sha256": checkpoint_hash,
                     "elapsed_seconds_this_invocation": time.monotonic() - started})
        print(f"completed {run_id} fold {fold_id}", flush=True)
    if len(predictions) != len(records) or len(set(predictions)) != len(records):
        raise ValueError("The five folds did not cover every development recording once")
    metrics = evaluate_saved_records(truths, predictions, protocol_hash=protocol["protocol_hash"],
                                     registry_hash=protocol["registry_hash"],
                                     model_id="compact-waveform-v1")
    result = {"status": "DEVELOPMENT_OOF_COMPLETE", "confirmatory": False,
              "run_id": run_id, "config_hash": config_hash,
              "protocol_hash": protocol["protocol_hash"], "metrics": metrics, "folds": fits,
              "prediction_paths": [str(path.resolve()) for path in predictions],
              "truth_paths": [str(path.resolve()) for path in truths],
              "gate_A": "NOT_RUN", "gate_B": "NOT_RUN"}
    atomic_json(out / "result.json", result, immutable=True)
    append_event(root / "runs/experiments.jsonl", {"event": "development_run_complete",
                 "run_id": run_id, "report_sha256": file_sha256(out / "result.json"),
                 "macro_f1": metrics["macro_f1"], "confirmatory": False})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/waveform/cache"))
    parser.add_argument("--seed", type=int, choices=(17, 43, 101), default=17)
    args = parser.parse_args()
    print(json_text(run(args.root.resolve(), args.data_root.resolve(), args.cache_dir.resolve(), args.seed)))


if __name__ == "__main__":
    main()
