"""Reproducible development-only ablations of a soft staging decoder."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import numpy as np

from .contracts import content_id, read_json
from .evaluation import evaluate_saved_records
from .predictions import load_prediction, save_prediction
from .protocol import development_records, load_development_truth
from .research import append_event, atomic_json, compute_lease, file_sha256
from .temporal import decode, fit_fold_priors

ABLATIONS = {"independent": (0.0, 0.0), "transition": (0.1, 0.0),
             "duration": (0.0, 0.1), "combined": (0.1, 0.1)}


def run(root: Path, source_report: Path, ablation: str, *, cutoff: int = 60) -> dict:
    """No audit participants or label-derived inference boundaries are accepted."""
    if ablation not in ABLATIONS or cutoff not in (30, 60, 120):
        raise ValueError("Unregistered development ablation or sensitivity cutoff")
    protocol, split, records = development_records(root)
    source = read_json(source_report)
    if (source.get("confirmatory") is not False or source.get("status") != "DEVELOPMENT_OOF_COMPLETE" or
            source.get("protocol_hash") != protocol["protocol_hash"]):
        raise ValueError("Decoder input must be a completed original-development OOF run")
    input_predictions = {}
    source_hashes = {}
    for name in source["prediction_paths"]:
        path = Path(name)
        meta, arrays = load_prediction(path)
        rid = arrays["recording_id"]
        if rid in input_predictions or meta["protocol_hash"] != protocol["protocol_hash"] or \
                meta["registry_hash"] != protocol["registry_hash"] or "probabilities" not in arrays:
            raise ValueError("Input predictions lack unique, bound OOF probabilities")
        input_predictions[rid] = (path, meta)
        source_hashes[rid] = {"payload_sha256": file_sha256(path),
                              "sidecar_sha256": file_sha256(path.with_suffix(".json"))}
    if set(input_predictions) != {r["recording_id"] for r in records}:
        raise ValueError("Decoder study requires every development recording and no audit records")
    weights = ABLATIONS[ablation]
    module_names = ("temporal.py", "temporal_experiment.py", "protocol.py", "splits.py", "research.py",
                    "contracts.py", "predictions.py", "evaluation.py")
    config = {"hypothesis": "H4_soft_temporal_priors", "ablation": ablation,
              "transition_weight": weights[0], "duration_weight": weights[1], "cutoff_epochs": cutoff,
              "protocol_hash": protocol["protocol_hash"], "split_id": split["split_id"],
              "source_run": source["run_id"], "source_report_sha256": file_sha256(source_report),
              "source_prediction_hashes": source_hashes,
              "implementation": {name: file_sha256(root / "sleepedf" / name) for name in module_names},
              "runtime_lock_sha256": file_sha256(root / "requirements" / "research.lock.txt"),
              "selection_scope": "development exploratory ablation; no confirmation or automatic adoption",
              "inference_blocks": "one continuous PSG recording; never reference-invalid-mask-derived"}
    config_hash = content_id(config)
    run_id = f"temporal-{ablation}-d{cutoff}-{config_hash[-10:]}"
    out = root / "runs" / run_id
    atomic_json(out / "config.json", config, immutable=True)
    snapshot = out / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in module_names:
        destination = snapshot / name
        if destination.exists() and file_sha256(destination) != config["implementation"][name]:
            raise ValueError("Frozen decoder implementation changed")
        if not destination.exists():
            shutil.copyfile(root / "sleepedf" / name, destination)
    predictions, truth_paths, parent_paths = [], [], []
    changed_predictions, emission_ties = 0, 0
    for fold in split["folds"]:
        fold_id = fold["fold_id"]
        train_ids = set(fold["train"])
        priors = fit_fold_priors(root, fold_id, cutoff=cutoff)
        atomic_json(out / f"priors-fold-{fold_id}.json", priors, immutable=True)
        for record in records:
            if record["participant_id"] not in fold["validation"]:
                continue
            rid = record["recording_id"]
            path, meta = input_predictions[rid]
            if set(meta["provenance"].get("fitted_participants", [])) != train_ids:
                raise ValueError("Parent model did not exclude exactly this held-out development fold")
            if (file_sha256(path) != source_hashes[rid]["payload_sha256"] or
                    file_sha256(path.with_suffix(".json")) != source_hashes[rid]["sidecar_sha256"]):
                raise ValueError("Parent predictions changed after experiment freeze")
            _, arrays = load_prediction(path)
            if arrays["participant_id"] != record["participant_id"] or len(arrays["epoch_index"]) != record["n_epochs"]:
                raise ValueError("Parent predictions changed identity or complete PSG grid")
            target = out / "predictions" / f"{rid}.npz"
            provenance = {"config_hash": config_hash, "prior_hash": priors["prior_hash"],
                          "fitted_participants": sorted(train_ids), "source_prediction": source_hashes[rid],
                          "channels": meta["provenance"]["channels"], "hypnogram_required": False,
                          "probability_meaning": "unchanged parent emissions; not decoder posterior",
                          "hard_label_rule": "exact structured objective", "duration_truncated": False}
            started = time.perf_counter()
            if target.exists():
                saved, _ = load_prediction(target)
                if (saved["provenance"] != provenance or saved["model_id"] != run_id or
                        saved["protocol_hash"] != protocol["protocol_hash"]):
                    raise ValueError("Saved decoder output differs from frozen experiment")
            else:
                result = decode(arrays["probabilities"], priors, transition_weight=weights[0], duration_weight=weights[1])
                save_prediction(target, participant_id=record["participant_id"], recording_id=rid,
                                epoch_index=arrays["epoch_index"], onset_seconds=arrays["onset_seconds"],
                                hard_label=result["hard_label"], probabilities=arrays["probabilities"],
                                model_id=run_id, protocol_hash=protocol["protocol_hash"],
                                registry_hash=protocol["registry_hash"], provenance=provenance)
                append_event(out / "events.jsonl", {"event": "record_decoded", "recording_id": rid,
                             "fold_id": fold_id, "elapsed_seconds": time.perf_counter()-started,
                             "payload_sha256": file_sha256(target), "objective": result["objective"],
                             "hard_label_changes": int(np.count_nonzero(result["hard_label"] != arrays["hard_label"]))})
            load_development_truth(record, split)
            _, saved_arrays = load_prediction(target)
            changed_predictions += int(np.count_nonzero(saved_arrays["hard_label"] != arrays["hard_label"]))
            p = arrays["probabilities"]
            emission_ties += int(np.count_nonzero((p == p.max(axis=1, keepdims=True)).sum(axis=1) > 1))
            predictions.append(target)
            truth_paths.append(Path(record["truth_path"]))
            parent_paths.append(path)
        print(f"{run_id} fold {fold_id} decoded", flush=True)
    metrics = evaluate_saved_records(truth_paths, predictions, protocol_hash=protocol["protocol_hash"],
                                      registry_hash=protocol["registry_hash"], model_id=run_id)
    parent_metrics = evaluate_saved_records(truth_paths, parent_paths, protocol_hash=protocol["protocol_hash"],
                                             registry_hash=protocol["registry_hash"])
    for key in ("macro_f1_exact", "confusion", "evaluated_epochs", "participant_count", "recording_count"):
        if parent_metrics[key] != source["metrics"][key]:
            raise ValueError("Parent metric report differs from recomputation on the same frozen development records")
    report = {"status": "DEVELOPMENT_OOF_COMPLETE", "confirmatory": False, "run_id": run_id,
              "config_hash": config_hash, "protocol_hash": protocol["protocol_hash"], "metrics": metrics,
              "parent_macro_f1": parent_metrics["macro_f1"], "parent_macro_f1_exact": parent_metrics["macro_f1_exact"],
              "development_delta": metrics["macro_f1"]-parent_metrics["macro_f1"],
              "changed_hard_predictions": changed_predictions, "emission_argmax_tie_epochs": emission_ties,
              "zero_weight_hard_tie_changes": changed_predictions if weights == (0., 0.) else None,
              "prediction_paths": [str(p.resolve()) for p in predictions],
              "truth_paths": [str(p.resolve()) for p in truth_paths], "gate_A": "NOT_RUN", "gate_B": "NOT_RUN"}
    atomic_json(out / "result.json", report)
    append_event(root / "runs" / "experiments.jsonl", {"event": "development_run_complete", "run_id": run_id,
                 "macro_f1": metrics["macro_f1"], "report_sha256": file_sha256(out / "result.json"), "confirmatory": False})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--ablation", choices=ABLATIONS, required=True)
    parser.add_argument("--cutoff", type=int, choices=(30, 60, 120), default=60)
    args = parser.parse_args()
    root = args.root.resolve()
    with compute_lease(root, f"temporal-{args.ablation}"):
        try:
            result = run(root, args.source_report.resolve(), args.ablation, cutoff=args.cutoff)
        except Exception as exc:
            append_event(root / "runs" / "experiments.jsonl", {"event": "development_run_failed",
                         "component": "temporal", "ablation": args.ablation, "error": str(exc)})
            raise
    print(result["status"], result["metrics"]["macro_f1"], flush=True)


if __name__ == "__main__":
    main()
