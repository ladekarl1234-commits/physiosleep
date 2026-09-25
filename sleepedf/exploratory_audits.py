"""Exploratory A/B evaluation of one frozen development-fitted YASA control.

The parent owns the compute lease, retirement record, subprocess supervision,
and resource limits. Each child predicts one PSG with no reference access.
Evaluation is a separate command and refuses to open any truth until every A/B
signal-only prediction has been verified on the full original epoch grid.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid

import numpy as np

from .contracts import content_id, read_json
from .dataset import _source_path
from .evaluation import _macro_f1_float, evaluate_saved_records
from .predictions import load_prediction, save_prediction
from .protocol import load_protocol
from .research import append_event, atomic_json, compute_lease, file_sha256
from .yasa_baseline import CHANNELS, predict


MODEL_REL = "runs/yasa-development-refit/yasa-d60-eeg_eog-s17-2e9f51a3fd/refit/model.joblib"
MODEL_SHA = "abdc64d3cd736f0a2e6c89fb5843eee06a3af26f64f0abb80120ade794d83860"
CONFIG_REL = "runs/yasa-development-refit/yasa-d60-eeg_eog-s17-2e9f51a3fd/config.json"
FIT_REL = "runs/yasa-development-refit/yasa-d60-eeg_eog-s17-2e9f51a3fd/refit/fit.json"
AUTH_REL = "research/exploratory-audits-authorization-v1.json"
SELECTION_REL = "research/exploratory-audits-v1.json"
SELECTION_KEYS = {"artifact_type", "schema_version", "scope", "project_root", "data_root",
                  "run_dir", "model_path", "model_sha256", "model_id", "variant",
                  "protocol_hash", "split_id", "registry_hash", "source_sha256",
                  "authorization_path", "authorization_sha256", "config_path",
                  "config_sha256", "config_hash", "fit_path", "fit_sha256",
                  "bootstrap", "selection_id"}
SCOPE = "EXPLORATORY_A_AND_B_NO_GATE_PASS"
BOOTSTRAP = {"draws": 10000, "rng": "PCG64",
             "seeds": {"A": 2026092301, "B": 2026092302},
             "quantiles": [0.025, 0.975], "strata": ["SC", "ST"],
             "unit": "participant_all_nights",
             "coverage": "individual_95_percent_not_simultaneous"}
REQUIRED_SOURCES = {"sleepedf/exploratory_audits.py", "sleepedf/exploratory_access.py",
                    "sleepedf/audit_access.py", "sleepedf/yasa_baseline.py",
                    "sleepedf/predictions.py", "sleepedf/evaluation.py",
                    "sleepedf/protocol.py", "sleepedf/dataset.py",
                    "sleepedf/research.py", "sleepedf/contracts.py"}
ENVIRONMENT = {name: "4" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                       "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                                       "NUMBA_NUM_THREADS")}
ENVIRONMENT["CUDA_VISIBLE_DEVICES"] = "-1"
MAX_RSS = 10 * 1024**3
MIN_RAM = 4 * 1024**3
MIN_DISK = 20 * 1024**3
RUN_SECONDS = 2 * 3600
RECORD_SECONDS = 180


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _require_lease(root: Path, selection_path: Path, *, owner: bool,
                   recording_id: str | None = None,
                   recompute_existing: bool = False) -> None:
    """Require the live controller, or an exact one-record Python launcher hop."""
    import psutil

    lease = read_json(root / "runs/compute.lock")
    expected_pid = os.getpid() if owner else os.getppid()
    if not owner and lease.get("pid") != expected_pid:
        launcher = psutil.Process(expected_pid)
        command = [str(root / ".venvs/research/Scripts/python.exe"), "-B", "-m",
                   "sleepedf.exploratory_audits", "predict-one", "--selection",
                   str(selection_path), "--recording", recording_id]
        if recompute_existing:
            command.append("--recompute-existing")
        _require(type(recording_id) is str and
                 Path(launcher.exe()).resolve() == Path(command[0]).resolve() and
                 launcher.cmdline() == command and launcher.parent() is not None,
                 "Exploratory prediction requires the exact parent launcher")
        expected_pid = launcher.parent().pid
    try:
        live = psutil.Process(expected_pid)
        birth = live.create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        raise RuntimeError("Exploratory A/B compute owner disappeared") from None
    _require(lease.get("pid") == expected_pid and lease.get("process_start") == birth and
             lease.get("host") == socket.gethostname() and
             lease.get("run_id") == "exploratory-audits-v1",
             "Exploratory A/B command lacks its exact live compute lease")


def _resources(root: Path, started: float, process: object | None = None) -> int:
    import psutil

    rss = psutil.Process().memory_info().rss
    if process is not None:
        try:
            members = [process, *process.children(recursive=True)]
        except psutil.NoSuchProcess:
            members = []
        for child in members:
            try:
                rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
    if (rss > MAX_RSS or psutil.virtual_memory().available < MIN_RAM or
            shutil.disk_usage(root).free < MIN_DISK or
            time.monotonic() - started > RUN_SECONDS):
        raise RuntimeError("Exploratory A/B exceeded its memory, disk or two-hour bound")
    return rss


def _selection(root: Path, path: Path) -> dict:
    root, path = Path(root).resolve(), Path(path)
    _require(path == root / SELECTION_REL and path.resolve() == path,
             "Exploratory A/B selection path differs from immutable freeze")
    item = read_json(path)
    _require(type(item) is dict and set(item) == SELECTION_KEYS and
             item["artifact_type"] == "physiosleep_exploratory_audits_selection" and
             item["schema_version"] == "1.0" and item["scope"] == SCOPE and
             item["selection_id"] == content_id({k: v for k, v in item.items()
                                                  if k != "selection_id"}) and
             item["project_root"] == str(root) and item["variant"] == "eeg_eog" and
             item["model_path"] == str(root / MODEL_REL) and
             item["model_sha256"] == MODEL_SHA and
             item["model_id"] == "sha256:" + MODEL_SHA and
             item["bootstrap"] == BOOTSTRAP and
             item["authorization_path"] == str(root / AUTH_REL) and
             item["config_path"] == str(root / CONFIG_REL) and
             item["fit_path"] == str(root / FIT_REL) and
             item["run_dir"] == str(root / "runs/exploratory-audits-v1") and
             item["authorization_sha256"] == file_sha256(root / AUTH_REL) and
             item["config_sha256"] == file_sha256(root / CONFIG_REL) and
             item["fit_sha256"] == file_sha256(root / FIT_REL),
             "Exploratory A/B model or selection identity differs")
    config, fit = read_json(root / CONFIG_REL), read_json(root / FIT_REL)
    authorization = read_json(root / AUTH_REL)
    _require(authorization.get("authorization_id") == content_id({
                 key: value for key, value in authorization.items() if key != "authorization_id"}) and
             authorization.get("artifact_type") == "owner_exploratory_audit_authorization" and
             authorization.get("schema_version") == "1.0" and
             authorization.get("phases") == ["A", "B"] and
             authorization.get("protocol_hash") == item["protocol_hash"] and
             authorization.get("split_id") == item["split_id"] and
             authorization.get("formal_gate_criteria_unchanged") is True and
             authorization.get("fresh_data_required_for_confirmation") is True and
             authorization.get("owner_reply") == "Run both now as exploratory evaluations" and
             authorization.get("scope") == (
                 "Score the existing D60 seed17 EEG+EOG model; no audit-informed tuning "
                 "or refitting; no all-baseline victory or Gate A/B pass") and
             authorization.get("deadline_seconds") == RUN_SECONDS and
             authorization.get("cpu_threads") == 4 and
             authorization.get("host_memory_limit_gib") == 10 and
             authorization.get("minimum_free_memory_gib") == 4 and
             authorization.get("minimum_free_disk_gib") == 20,
             "Exploratory A/B authorization scope or identity differs")
    _require(item["config_hash"] == fit["config_hash"] and
             item["config_hash"] == content_id(config) and
             fit["checkpoint_sha"] == MODEL_SHA and
             fit["protocol_hash"] == item["protocol_hash"] and
             len(fit["fitted_participants"]) == 60 and
             config["variant"] == "eeg_eog" and config["seed"] == 17,
             "Exploratory A/B control configuration or fit ancestry differs")
    data_root, run_dir = Path(item["data_root"]), Path(item["run_dir"])
    _require(data_root.is_absolute() and data_root.resolve() == data_root and
             data_root.is_dir() and not root.is_relative_to(data_root) and
             run_dir.is_absolute() and run_dir.resolve() == run_dir and
             run_dir.is_relative_to(root / "runs") and
             not run_dir.is_relative_to(data_root) and
             type(item["source_sha256"]) is dict and
             REQUIRED_SOURCES <= set(item["source_sha256"]) and
             item["source_sha256"].get("sleepedf/exploratory_audits.py") ==
             file_sha256(Path(__file__)), "Exploratory A/B paths or runner source differ")
    for relative, expected in item["source_sha256"].items():
        _require(type(relative) is str and re.fullmatch(r"(?:sleepedf|tools)/[a-z0-9_/]+\.py", relative)
                 and type(expected) is str and expected == file_sha256(root / relative),
                 "Exploratory A/B inference source differs")
    _require(file_sha256(root / MODEL_REL) == MODEL_SHA,
             "Exploratory A/B control checkpoint bytes changed")
    return item


def _records(root: Path, selection: dict) -> tuple[dict, dict[str, list[dict]]]:
    """Read only frozen protocol/split/readiness metadata, never truth files."""
    protocol, split, readiness = load_protocol(root)
    _require(selection["protocol_hash"] == protocol["protocol_hash"] and
             selection["split_id"] == split["split_id"] and
             selection["registry_hash"] == protocol["registry_hash"],
             "Exploratory A/B protocol or registry differs")
    fit = read_json(root / FIT_REL)
    development = split["participants"]["development"]
    _require(len(fit["fitted_participants"]) == len(development) == 60 and
             len(set(fit["fitted_participants"])) == 60 and
             set(fit["fitted_participants"]) == set(development),
             "Exploratory A/B control checkpoint did not fit exact D60 participants")
    by_phase = {}
    seen = set()
    for phase, key in (("A", "audit_a"), ("B", "audit_b")):
        people = set(split["participants"][key])
        rows = sorted((r for r in readiness["records"] if r["participant_id"] in people),
                      key=lambda r: r["recording_id"])
        frozen = protocol["audit_recordings"][phase]
        _require(rows and {r["recording_id"] for r in rows} == set(frozen) and
                 len({r["recording_id"] for r in rows}) == len(rows) and
                 {r["participant_id"] for r in rows} == people,
                 "Exploratory A/B phase lacks a complete frozen participant/night roster")
        for row in rows:
            rid = row["recording_id"]
            expected = frozen[rid]
            _require(rid not in seen and expected == {
                "participant_id": row["participant_id"], "n_epochs": row["n_epochs"],
                "psg_sha256": row["psg_sha256"],
                "hypnogram_sha256": row["hypnogram_sha256"],
                "truth_payload_sha256": expected["truth_payload_sha256"],
                "truth_sidecar_sha256": expected["truth_sidecar_sha256"]},
                "Exploratory A/B frozen recording metadata differs")
            seen.add(rid)
        by_phase[phase] = rows
    _require(len(seen) == 78, "Exploratory A/B requires all 78 original audit nights")
    return protocol, by_phase


def _prediction_path(selection: dict, recording_id: str) -> Path:
    return Path(selection["run_dir"]) / "predictions" / recording_id / "prediction.npz"


def _require_retired(root: Path, selection_path: Path, protocol_hash: str) -> None:
    from .audit_access import read_ledger

    rows = read_ledger(root / "runs/audit-access.jsonl", protocol_hash)
    expected_sha = file_sha256(selection_path)
    for phase in ("A", "B"):
        same = [row for row in rows if row["phase"] == phase]
        _require(len(same) == 1 and same[0]["event"] == "EXPLORATORY_RETIRED" and
                 same[0]["evidence"]["selection"] == {
                     "path": str(selection_path), "sha256": expected_sha},
                 "Exploratory A/B labels must be retired by the owner before inference")


def _provenance(selection: dict, row: dict) -> dict:
    return {"selection_id": selection["selection_id"],
            "model_sha256": MODEL_SHA, "variant": "eeg_eog",
            "channels": list(CHANNELS["eeg_eog"]),
            "psg_sha256": row["psg_sha256"],
            "inference_route": "pinned native YASA PSG-only predict",
            "hypnogram_required": False,
            "source_sha256": selection["source_sha256"]}


def _verify_prediction(path: Path, selection: dict, row: dict, protocol: dict) -> dict:
    meta, data = load_prediction(path)
    _require(meta["model_id"] == selection["model_id"] and
             meta["protocol_hash"] == protocol["protocol_hash"] and
             meta["registry_hash"] == protocol["registry_hash"] and
             meta["provenance"] == _provenance(selection, row) and
             data["recording_id"] == row["recording_id"] and
             data["participant_id"] == row["participant_id"] and
             len(data["epoch_index"]) == row["n_epochs"] and
             "probabilities" in data,
             "Exploratory A/B saved prediction does not match frozen PSG/model/grid")
    progress = read_json(path.with_name("progress.json"))
    _require(progress == {"artifact_type": "exploratory_audit_prediction_progress",
                          "schema_version": "1.0", "recording_id": row["recording_id"],
                          "selection_id": selection["selection_id"],
                          "model_sha256": MODEL_SHA,
                          "prediction_sha256": file_sha256(path),
                          "sidecar_sha256": file_sha256(path.with_suffix(".json"))},
             "Exploratory A/B prediction progress differs from committed bytes")
    return progress


def predict_one(root: Path, selection_path: Path, recording_id: str, *,
                recompute_existing: bool = False) -> dict:
    root = Path(root).resolve()
    _require_lease(root, selection_path, owner=False, recording_id=recording_id,
                   recompute_existing=recompute_existing)
    selection = _selection(root, selection_path)
    protocol, groups = _records(root, selection)
    _require_retired(root, selection_path, protocol["protocol_hash"])
    matches = [r for rows in groups.values() for r in rows if r["recording_id"] == recording_id]
    _require(len(matches) == 1, "Exploratory A/B recording is outside frozen audit A/B")
    row = matches[0]
    output = _prediction_path(selection, recording_id)
    if output.parent.exists():
        progress = _verify_prediction(output, selection, row, protocol)
        if recompute_existing:
            psg = _source_path(Path(selection["data_root"]), row["psg"])
            _require(file_sha256(psg) == row["psg_sha256"], "Exploratory A/B PSG bytes differ")
            hard, probabilities = predict(root, psg, CHANNELS["eeg_eog"],
                                          n_epochs=row["n_epochs"],
                                          model_path=Path(selection["model_path"]))
            _, saved = load_prediction(output)
            _require(np.array_equal(hard, saved["hard_label"]) and
                     np.array_equal(probabilities, saved["probabilities"]),
                     "Exploratory A/B resumed prediction differs from fresh native inference")
        return progress
    psg = _source_path(Path(selection["data_root"]), row["psg"])
    _require(file_sha256(psg) == row["psg_sha256"], "Exploratory A/B PSG bytes differ")
    hard, probabilities = predict(root, psg, CHANNELS["eeg_eog"],
                                  n_epochs=row["n_epochs"], model_path=Path(selection["model_path"]))
    stage = output.parent.parent / (".partial-" + recording_id + "-" + uuid.uuid4().hex)
    _require(not stage.exists() and not output.parent.exists(),
             "Exploratory A/B prediction already has an unverified commit")
    stage.mkdir(parents=True)
    staged = stage / "prediction.npz"
    save_prediction(staged, participant_id=row["participant_id"], recording_id=recording_id,
                    epoch_index=np.arange(row["n_epochs"], dtype=np.int64),
                    onset_seconds=np.arange(row["n_epochs"], dtype=np.float64) * 30,
                    hard_label=hard, probabilities=probabilities,
                    model_id=selection["model_id"], protocol_hash=protocol["protocol_hash"],
                    registry_hash=protocol["registry_hash"],
                    provenance=_provenance(selection, row))
    progress = {"artifact_type": "exploratory_audit_prediction_progress",
                "schema_version": "1.0", "recording_id": recording_id,
                "selection_id": selection["selection_id"], "model_sha256": MODEL_SHA,
                "prediction_sha256": file_sha256(staged),
                "sidecar_sha256": file_sha256(staged.with_suffix(".json"))}
    atomic_json(stage / "progress.json", progress, immutable=True)
    # Commit all three files as one same-parent directory operation.
    os.replace(stage, output.parent)
    return _verify_prediction(output, selection, row, protocol)


def _complete_predictions(selection: dict, protocol: dict,
                          groups: dict[str, list[dict]], *,
                          require_receipts: bool = True) -> dict[str, list[Path]]:
    result = {}
    for phase, rows in groups.items():
        paths = []
        for row in rows:
            path = _prediction_path(selection, row["recording_id"])
            _verify_prediction(path, selection, row, protocol)
            if require_receipts:
                _verify_receipt(selection, row, path)
            paths.append(path)
        result[phase] = paths
    return result


def _receipt_path(selection: dict, recording_id: str) -> Path:
    return Path(selection["run_dir"]) / "receipts" / (recording_id + ".json")


def _verify_receipt(selection: dict, row: dict, prediction_path: Path) -> dict:
    import math
    path = _receipt_path(selection, row["recording_id"])
    receipt = read_json(path)
    _require(type(receipt) is dict and type(receipt.get("log_path")) is str,
             "Exploratory A/B execution receipt lacks its log")
    log = Path(receipt["log_path"])
    _require(type(receipt) is dict and set(receipt) == {
                 "artifact_type", "schema_version", "recording_id", "selection_id",
                 "command", "environment", "exit_code", "elapsed_seconds",
                 "peak_combined_rss_bytes", "log_path", "log_sha256", "prediction_sha256"} and
             receipt["artifact_type"] == "exploratory_audit_prediction_execution" and
             receipt["schema_version"] == "1.0" and
             receipt["recording_id"] == row["recording_id"] and
             receipt["selection_id"] == selection["selection_id"] and
             receipt["environment"] == ENVIRONMENT and receipt["exit_code"] == 0 and
             type(receipt["elapsed_seconds"]) in (int, float) and
             math.isfinite(receipt["elapsed_seconds"]) and
             0 < receipt["elapsed_seconds"] <= RECORD_SECONDS and
             type(receipt["peak_combined_rss_bytes"]) is int and
             0 < receipt["peak_combined_rss_bytes"] <= MAX_RSS and
             log.is_absolute() and log.resolve() == log and
             log.is_relative_to(Path(selection["run_dir"]) / "logs") and
             file_sha256(log) == receipt["log_sha256"] and
             file_sha256(prediction_path) == receipt["prediction_sha256"],
             "Exploratory A/B prediction lacks a successful bound child receipt")
    command = [str(Path(selection["project_root"]) / ".venvs/research/Scripts/python.exe"),
               "-B", "-m", "sleepedf.exploratory_audits", "predict-one", "--selection",
               str(Path(selection["project_root"]) / SELECTION_REL), "--recording",
               row["recording_id"]]
    _require(receipt["command"] == command or
             receipt["command"] == command + ["--recompute-existing"],
             "Exploratory A/B prediction invocation differs")
    markers = [line[len("EXPLORATORY_PREDICTION "):]
               for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
               if line.startswith("EXPLORATORY_PREDICTION ")]
    _require(len(markers) == 1 and json.loads(markers[0]) ==
             read_json(prediction_path.with_name("progress.json")),
             "Exploratory A/B child completion marker differs from committed prediction")
    return receipt


def _mark_truth_opening(root: Path, phase: str, selection_sha: str) -> None:
    ledger = root / "runs/exposure.jsonl"
    previous = ([json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
                if ledger.exists() else [])
    matching = [row for row in previous if row.get("event") == "exploratory_audit_labels_opening"
                and row.get("phase") == phase]
    expected = {"event": "exploratory_audit_labels_opening", "phase": phase,
                "selection_sha256": selection_sha, "scope": SCOPE,
                "gate_status": "NOT_A_GATE_PASS"}
    _require(len(matching) <= 1 and
             (not matching or {k: matching[0].get(k) for k in expected} == expected),
             "Exploratory A/B exposure chronology differs")
    if not matching:
        append_event(ledger, expected)


def _descriptive_interval(result: dict, phase: str, protocol: dict, selection: dict) -> dict:
    pids = sorted(result["per_participant_confusion"])
    matrices = np.asarray([result["per_participant_confusion"][pid] for pid in pids], dtype=np.int64)
    cohorts = [pid[:2] for pid in pids]
    strata = [[i for i, cohort in enumerate(cohorts) if cohort == name] for name in ("SC", "ST")]
    strata = [group for group in strata if group]
    recipe = protocol["bootstrap"]
    _require(selection["bootstrap"] == BOOTSTRAP and
             recipe["draws"] == 10000 and recipe["rng"] == "PCG64" and
             recipe["strata"] == ["SC", "ST"] and recipe["cluster"] == "participant_all_nights",
             "Exploratory A/B bootstrap recipe differs")
    seed = recipe["seeds"][phase]
    rng = np.random.Generator(np.random.PCG64(seed))
    pooled = np.empty((10000, 5, 5), dtype=np.int64)
    for draw in range(10000):
        chosen = np.concatenate([rng.choice(group, size=len(group), replace=True)
                                 for group in strata])
        pooled[draw] = matrices[chosen].sum(axis=0)
    scores = _macro_f1_float(pooled)
    low, high = np.quantile(scores, [.025, .975], method="linear")
    return {"method": "single_model_participant_cluster_stratified_SC_ST_percentile",
            "scope": "exploratory_descriptive_not_paired_margin_or_gate",
            "rng": "PCG64", "seed": seed, "draws": 10000,
            "confidence_level": 0.95, "quantile_method": "linear",
            "lower": float(low), "upper": float(high),
            "participant_count": len(pids),
            "cohort_counts": {name: cohorts.count(name) for name in ("SC", "ST")}}


def evaluate(root: Path, selection_path: Path, *, lease_owner: bool = False) -> dict:
    """Consume both phase references only after all 78 predictions are fixed."""
    root = Path(root).resolve()
    _require_lease(root, selection_path, owner=lease_owner)
    selection = _selection(root, selection_path)
    protocol, groups = _records(root, selection)
    _require_retired(root, selection_path, protocol["protocol_hash"])
    prediction_paths = _complete_predictions(selection, protocol, groups)
    selection_sha = file_sha256(selection_path)
    # Both exposure events precede the first truth-sidecar or payload access.
    for phase in ("A", "B"):
        _mark_truth_opening(root, phase, selection_sha)
    results = {}
    for phase in ("A", "B"):
        truth_paths = []
        for row in groups[phase]:
            path = Path(row["truth_path"])
            frozen = protocol["audit_recordings"][phase][row["recording_id"]]
            _require(path.is_absolute() and path.resolve() == path and
                     path == root / "derived/evaluator_truth" / (row["recording_id"] + ".npz") and
                     file_sha256(path) == frozen["truth_payload_sha256"] and
                     file_sha256(path.with_suffix(".json")) == frozen["truth_sidecar_sha256"],
                     "Exploratory A/B truth bytes differ from frozen protocol")
            truth_paths.append(path)
        scored = evaluate_saved_records(truth_paths, prediction_paths[phase],
                                        protocol_hash=protocol["protocol_hash"],
                                        registry_hash=protocol["registry_hash"],
                                        model_id=selection["model_id"])
        scored["descriptive_macro_f1_interval"] = _descriptive_interval(scored, phase, protocol, selection)
        result = {"artifact_type": "physiosleep_exploratory_audit_result",
                  "schema_version": "1.0", "status": "EXPLORATORY_EVALUATED",
                  "gate_status": "NOT_RUN", "phase": phase,
                  "selection_sha256": selection_sha,
                  "model_sha256": MODEL_SHA,
                  "protocol_hash": protocol["protocol_hash"],
                  "scope": SCOPE,
                  "prediction_sha256": {r["recording_id"]: file_sha256(p)
                                        for r, p in zip(groups[phase], prediction_paths[phase])},
                  "truth_payload_sha256": {r["recording_id"]:
                                           protocol["audit_recordings"][phase][r["recording_id"]]["truth_payload_sha256"]
                                           for r in groups[phase]},
                  "metrics": scored}
        atomic_json(Path(selection["run_dir"]) / "results" / f"{phase}.json", result, immutable=True)
        results[phase] = result
    return results


def run(root: Path, selection_path: Path) -> dict:
    """Serialize one-child-at-a-time predictions, then consume both references."""
    import psutil
    from .exploratory_access import append_exploratory_retirement

    root = Path(root).resolve()
    selection = _selection(root, selection_path)
    protocol, groups = _records(root, selection)
    started = time.monotonic()
    with compute_lease(root, "exploratory-audits-v1"):
        _resources(root, started)
        append_exploratory_retirement(root, selection_path)
        for row in [*groups["A"], *groups["B"]]:
            rid = row["recording_id"]
            output = _prediction_path(selection, rid)
            committed = output.parent.exists()
            if committed:
                _verify_prediction(output, selection, row, protocol)
                if _receipt_path(selection, rid).exists():
                    _verify_receipt(selection, row, output)
                    continue
            _resources(root, started)
            command = [str(root / ".venvs/research/Scripts/python.exe"), "-B", "-m",
                       "sleepedf.exploratory_audits", "predict-one", "--selection",
                       str(selection_path), "--recording", rid]
            if committed:
                command.append("--recompute-existing")
            environment = os.environ.copy()
            environment.update(ENVIRONMENT)
            log = Path(selection["run_dir"]) / "logs" / rid / (uuid.uuid4().hex + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            child_started = time.monotonic()
            peak = 0
            with log.open("x", encoding="utf-8") as stream:
                child = subprocess.Popen(command, cwd=root, env=environment,
                                         stdout=stream, stderr=subprocess.STDOUT)
                process = psutil.Process(child.pid)
                try:
                    while child.poll() is None:
                        peak = max(peak, _resources(root, started, process))
                        if time.monotonic() - child_started > RECORD_SECONDS:
                            raise RuntimeError("Exploratory A/B one-record inference exceeded 180 seconds")
                        time.sleep(.25)
                    peak = max(peak, _resources(root, started))
                except BaseException:
                    try:
                        descendants = process.children(recursive=True)
                    except psutil.NoSuchProcess:
                        descendants = []
                    for descendant in descendants:
                        try:
                            descendant.kill()
                        except psutil.NoSuchProcess:
                            pass
                    if child.poll() is None:
                        child.kill()
                    child.wait()
                    raise
            _require(child.returncode == 0,
                     "Exploratory A/B one-record prediction failed; preserve its log and partial staging")
            progress = _verify_prediction(output, selection, row, protocol)
            receipt = {"artifact_type": "exploratory_audit_prediction_execution",
                       "schema_version": "1.0", "recording_id": rid,
                       "selection_id": selection["selection_id"], "command": command,
                       "environment": ENVIRONMENT, "exit_code": child.returncode,
                       "elapsed_seconds": time.monotonic() - child_started,
                       "peak_combined_rss_bytes": peak,
                       "log_path": str(log), "log_sha256": file_sha256(log),
                       "prediction_sha256": progress["prediction_sha256"]}
            atomic_json(_receipt_path(selection, rid), receipt, immutable=True)
            _verify_receipt(selection, row, output)
        _complete_predictions(selection, protocol, groups)
        _resources(root, started)
        return evaluate(root, selection_path, lease_owner=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("predict-one", "evaluate", "run"))
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--recording")
    parser.add_argument("--recompute-existing", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "predict-one":
        _require(type(args.recording) is str and args.recording,
                 "A frozen A/B recording ID is required for one prediction")
        result = predict_one(root, args.selection, args.recording,
                             recompute_existing=args.recompute_existing)
        print("EXPLORATORY_PREDICTION " + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    elif args.action == "evaluate":
        _require(args.recording is None and not args.recompute_existing,
                 "Evaluation consumes all A/B predictions together")
        result = evaluate(root, args.selection)
        print("EXPLORATORY_EVALUATION " + json.dumps({phase: result[phase]["metrics"]["macro_f1"]
                                                          for phase in ("A", "B")},
                                                       sort_keys=True, allow_nan=False), flush=True)
    else:
        _require(args.recording is None and not args.recompute_existing,
                 "Full exploratory run selects all frozen A/B nights")
        result = run(root, args.selection)
        print("EXPLORATORY_RUN_COMPLETE " + json.dumps({phase: result[phase]["metrics"]["macro_f1"]
                                                          for phase in ("A", "B")},
                                                       sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
