"""Evaluate fixed round prefixes of attested development-only YASA checkpoints."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time

import numpy as np

from .contracts import content_id, read_json
from .evaluation import evaluate_saved_records
from .predictions import load_prediction, save_prediction
from .protocol import development_records
from .research import atomic_json, compute_lease, file_sha256
from . import yasa_baseline as yasa
from tools.run_yasa_prospective import _limits


ROUNDS = (100, 200, 300, 400)
SOURCES = ("sleepedf/yasa_round_diagnostics.py", "sleepedf/yasa_baseline.py",
           "sleepedf/protocol.py", "sleepedf/predictions.py", "sleepedf/evaluation.py",
           "sleepedf/contracts.py", "sleepedf/splits.py", "sleepedf/research.py",
           "tools/run_yasa_prospective.py")


def _check(root: Path, started: float) -> None:
    _limits(root, started, [])
    if time.monotonic() - started > 900:
        raise TimeoutError("Round diagnostics reached the 900-second invocation bound")


def _output_path(path: Path, out: Path) -> None:
    if not path.is_relative_to(out) or path.resolve() != path:
        raise ValueError("Diagnostic output path is redirected")


def _truth_hashes(records: list[dict]) -> None:
    for rec in records:
        truth = Path(rec["truth_path"])
        expected = rec.get("frozen_truth")
        if (not isinstance(expected, dict) or
            file_sha256(truth) != expected.get("payload_sha256") or
            file_sha256(truth.with_suffix(".json")) != expected.get("sidecar_sha256")):
            raise ValueError("Development truth differs from its frozen payload or sidecar")


def _recipe(config: dict, chosen: dict) -> None:
    params = config.get("params", {})
    if (config.get("variant") not in yasa.CHANNELS or
        params.get("boosting_type") != "gbdt" or params.get("n_estimators") != 400 or params.get("n_jobs") != 4 or
        config.get("selection_evidence", {}).get("content", {}).get("chosen") != chosen or
        chosen.get("variant") != config.get("variant") or chosen.get("seed") != config.get("seed") or
        chosen.get("lr_multiplier") not in (0.3, 1, 3) or
        params.get("learning_rate") != 0.1 * chosen["lr_multiplier"]):
        raise ValueError("Round diagnostics require the attested native 400-round GBDT recipe")


def _completed_parent(result: dict, config: dict, protocol: dict, run_name: str) -> None:
    if (result.get("status") != "DEVELOPMENT_OOF_COMPLETE" or
        result.get("confirmatory") is not False or result.get("gate_A") != "NOT_RUN" or
        result.get("gate_B") != "NOT_RUN" or result.get("protocol_hash") != protocol["protocol_hash"] or
        result.get("run_id") != run_name or result.get("config_hash") != content_id(config) or
        result.get("selection_evidence_sha256") != config.get("selection_evidence", {}).get("sha256")):
        raise ValueError("Parent must be a bound nonconfirmatory development completion")


def run(root: Path, result_path: Path) -> dict:
    import joblib
    import psutil

    root = root.resolve()
    result_path = (root / result_path).resolve()
    if (result_path.parent.parent != root / "runs" or result_path.name != "result.json" or
        (root / "runs").resolve() != root / "runs"):
        raise ValueError("Parent and outputs must stay in the local runs directory")
    if any(os.environ.get(name) != "4" for name in
           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("Set native CPU thread environment limits to 4 before starting Python")
    started = time.monotonic()
    with compute_lease(root, "yasa-round-prefix-diagnostics"):
        _check(root, started)
        protocol, split, records = development_records(root)
        parent = result_path.parent
        result, config = read_json(result_path), read_json(parent / "config.json")
        _completed_parent(result, config, protocol, parent.name)
        chosen = config["selection_evidence"]["content"]["chosen"]
        _recipe(config, chosen)
        variant = config["variant"]
        manifest_path = root / "derived/yasa_features" / variant / "manifest.json"
        manifest = read_json(manifest_path)
        if (config.get("split_id") != split["split_id"] or
            config.get("feature_manifest_sha256") != file_sha256(manifest_path) or
            config.get("implementation_sha") != file_sha256(root / "sleepedf/yasa_baseline.py")):
            raise ValueError("Parent split, feature cache or YASA implementation changed")
        selection = yasa._selection_evidence(root, root / config["selection_evidence"]["path"],
                                             protocol, split, records, variant, config["seed"], chosen["lr_multiplier"])
        if selection != config["selection_evidence"]:
            raise ValueError("Parent selection content or bytes changed")
        yasa._verify_feature_cache(root, variant, records, manifest)
        expected = {r["recording_id"]: r for r in records}
        parents, parents_sha = {}, {}
        for name in result["prediction_paths"]:
            path = Path(name).resolve()
            if not path.is_relative_to(parent):
                raise ValueError("Parent prediction escaped its attested run")
            meta, arrays = load_prediction(path)
            rid = arrays["recording_id"]
            if rid in parents or rid not in expected or arrays["participant_id"] != expected[rid]["participant_id"]:
                raise ValueError("Parent predictions are duplicated or outside development")
            parents[rid] = path
            parents_sha[rid] = {"payload": file_sha256(path), "sidecar": file_sha256(path.with_suffix(".json"))}
        if set(parents) != set(expected):
            raise ValueError("Parent must cover every development recording")
        checkpoints = {}
        for fold in split["folds"]:
            dest = parent / f"fold-{fold['fold_id']}"
            fit = read_json(dest / "fit.json")
            roles = yasa._fold_roles(fold, config["selection_evidence"], split, manifest)
            yasa._check_fold_roles(fit, roles)
            if (fit.get("status") != "FIT_COMPLETE" or fit.get("config_hash") != content_id(config) or
                fit.get("protocol_hash") != protocol["protocol_hash"] or
                fit.get("fitted_participants") != fold["train"] or
                fit.get("excluded_validation_participants") != fold["validation"] or
                fit.get("checkpoint_sha") != file_sha256(dest / "model.joblib")):
                raise ValueError("Parent checkpoint or held-out exclusion changed")
            checkpoints[str(fold["fold_id"])] = {"model": fit["checkpoint_sha"], "fit": file_sha256(dest / "fit.json")}
        frozen = {"rounds": list(ROUNDS), "parent_result_sha256": file_sha256(result_path),
                  "parent_run": parent.name, "parent_config_hash": content_id(config),
                  "split_id": split["split_id"], "protocol_hash": protocol["protocol_hash"],
                  "parents": parents_sha, "checkpoints": checkpoints,
                  "sources": {p: file_sha256(root / p) for p in SOURCES},
                  "runtime_lock_sha256": file_sha256(root / "requirements/research.hashed.txt"),
                  "selection_participants": config["selection_evidence"]["content"]["selection_participants"],
                  "interpretation": "conditional development convergence sensitivity; no automatic recipe replacement"}
        run_id = "yasa-rounds-" + content_id(frozen)[-10:]
        out = root / "runs" / run_id
        if out.resolve() != out:
            raise ValueError("Diagnostic output is redirected")
        _output_path(out / "config.json", out)
        _output_path(out / "result.json", out)
        atomic_json(out / "config.json", frozen, immutable=True)
        for name, digest in frozen["sources"].items():
            dest = out / "source_snapshot" / name
            _output_path(dest, out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes((root / name).read_bytes())
            if file_sha256(dest) != digest:
                raise ValueError("Diagnostic source snapshot differs")
        outputs = {n: [] for n in ROUNDS}
        truths = []
        peak = 0
        for fold in split["folds"]:
            model_path = parent / f"fold-{fold['fold_id']}" / "model.joblib"
            if file_sha256(model_path) != checkpoints[str(fold["fold_id"])]["model"]:
                raise ValueError("Checkpoint changed before loading")
            model = joblib.load(model_path)
            if model.booster_.current_iteration() != 400:
                raise ValueError("Checkpoint does not contain 400 native rounds")
            for rec in records:
                if rec["participant_id"] not in fold["validation"]:
                    continue
                _check(root, started)
                rid = rec["recording_id"]
                path = parents[rid]
                if (file_sha256(path) != parents_sha[rid]["payload"] or
                    file_sha256(path.with_suffix(".json")) != parents_sha[rid]["sidecar"]):
                    raise ValueError("Parent prediction changed before comparison")
                meta, saved = load_prediction(path)
                if (set(meta["provenance"].get("fitted_participants", [])) != set(fold["train"]) or
                    meta["protocol_hash"] != protocol["protocol_hash"] or
                    meta["registry_hash"] != protocol["registry_hash"] or
                    meta["provenance"].get("checkpoint_sha") != checkpoints[str(fold["fold_id"])]["model"] or
                    not np.array_equal(saved["epoch_index"], np.arange(rec["n_epochs"])) or
                    not np.array_equal(saved["onset_seconds"], np.arange(rec["n_epochs"]) * 30)):
                    raise ValueError("Parent grid or fitted exclusion differs")
                frame = yasa._features(root, variant, rec, manifest)
                if len(frame) != rec["n_epochs"]:
                    raise ValueError("Feature grid differs")
                for n in ROUNDS:
                    labels = yasa.native_hard_labels(model.predict(frame[model.feature_name_], num_iteration=n))
                    if n == 400 and not np.array_equal(labels, saved["hard_label"]):
                        raise ValueError("400-round predictions differ from the parent")
                    target = out / f"round-{n}" / f"{rid}.npz"
                    _output_path(target, out)
                    _output_path(target.with_suffix(".json"), out)
                    provenance = {"config_hash": content_id(frozen), "parent_prediction": parents_sha[rid],
                                  "checkpoint_sha": checkpoints[str(fold["fold_id"])]["model"],
                                  "fitted_participants": fold["train"], "selection_participants": frozen["selection_participants"],
                                  "round_prefix": n, "channels": meta["provenance"]["channels"],
                                  "hypnogram_required": False}
                    if target.exists():
                        prior_meta, prior = load_prediction(target)
                        if (prior_meta["provenance"] != provenance or
                            prior["participant_id"] != rec["participant_id"] or prior["recording_id"] != rid or
                            any(not np.array_equal(prior[key], saved[key]) for key in ("epoch_index", "onset_seconds")) or
                            not np.array_equal(prior["hard_label"], labels)):
                            raise ValueError("Existing round-prefix output differs")
                    else:
                        save_prediction(target, participant_id=rec["participant_id"], recording_id=rid,
                        epoch_index=saved["epoch_index"], onset_seconds=saved["onset_seconds"],
                        hard_label=labels, model_id=f"{run_id}/round-{n}/fold-{fold['fold_id']}",
                        protocol_hash=protocol["protocol_hash"], registry_hash=protocol["registry_hash"],
                        provenance=provenance)
                    outputs[n].append(target)
                truths.append(Path(rec["truth_path"]))
                peak = max(peak, psutil.Process().memory_info().rss)
            print(f"round diagnostics fold {fold['fold_id']} complete", flush=True)
        _truth_hashes(records)
        metrics = {str(n): evaluate_saved_records(truths, outputs[n], protocol_hash=protocol["protocol_hash"],
                    registry_hash=protocol["registry_hash"]) for n in ROUNDS}
        _truth_hashes(records)
        if any(file_sha256(root / p) != sha for p, sha in frozen["sources"].items()):
            raise ValueError("Diagnostic sources changed during execution")
        report = {"status": "DEVELOPMENT_PREFIX_DIAGNOSTICS_COMPLETE", "confirmatory": False,
                  "run_id": run_id, "config_hash": content_id(frozen), "protocol_hash": protocol["protocol_hash"],
                  "metrics": metrics, "prediction_paths": {str(n): [str(p) for p in outputs[n]] for n in ROUNDS},
                  "truth_paths": [str(p) for p in truths], "elapsed_seconds": time.monotonic()-started,
                  "peak_observed_rss_bytes": peak, "gate_A": "NOT_RUN", "gate_B": "NOT_RUN"}
        _check(root, started)
        if (out / "result.json").exists():
            old = read_json(out / "result.json")
            changing = {"elapsed_seconds", "peak_observed_rss_bytes"}
            if (set(old) != set(report) or any(old[k] != report[k] for k in set(report) - changing) or
                any(type(old[k]) not in (int, float) or not math.isfinite(old[k]) or old[k] < 0 for k in changing) or
                old["elapsed_seconds"] > 900 or old["peak_observed_rss_bytes"] > 10 * 1024**3):
                raise ValueError("Completed round-prefix diagnostics differ on replay")
            return old
        _output_path(out / "result.json", out)
        atomic_json(out / "result.json", report, immutable=True)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.result)
    print(result["run_id"], flush=True)


if __name__ == "__main__":
    main()
