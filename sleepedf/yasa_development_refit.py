"""Bounded experimental D60 YASA refit from a frozen completed CV configuration.

This produces a checkpoint for the existing signal-only predict command. It
reports no in-sample performance and grants no audit or confirmatory status.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np

from .contracts import content_id, read_json
from .development_ensemble import _frozen_truth, _member
from .protocol import development_records, load_development_truth
from .research import atomic_json, compute_lease, file_sha256
from . import yasa_baseline as native


SECONDS = 600
LEASE = "yasa-development-d60-refit"
IDENTITY = "YASA_D60_REFIT_IDENTITY "
RESULT = "YASA_D60_REFIT_RESULT "
SOURCE_FILES = ("sleepedf/yasa_development_refit.py", "tests/test_yasa_development_refit.py",
                "sleepedf/development_ensemble.py", "sleepedf/yasa_baseline.py",
                "sleepedf/protocol.py", "sleepedf/predictions.py", "sleepedf/evaluation.py",
                "sleepedf/contracts.py", "sleepedf/research.py", "sleepedf/splits.py")


def _limits(root: Path, deadline: float, descendants_rss: int = 0) -> None:
    import psutil
    if (time.monotonic() >= deadline or
        psutil.Process().memory_info().rss + descendants_rss > 10 * 1024**3 or
        psutil.virtual_memory().available < 4 * 1024**3 or
        shutil.disk_usage(root).free < 20 * 1024**3):
        raise RuntimeError("YASA D60 refit exceeded 600s, 10GiB RSS, 4GiB free RAM, or 20GiB disk")


def _hash(path: Path, root: Path, deadline: float) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            _limits(root, deadline)
    return digest.hexdigest()


def _snapshot(root: Path, deadline: float) -> dict:
    return {"sources": {name: _hash(root / name, root, deadline) for name in SOURCE_FILES},
            "runtime_lock_sha256": _hash(root / "requirements/research.hashed.txt", root, deadline),
            "python_sha256": _hash(root / ".venvs/research/Scripts/python.exe", root, deadline),
            "scope": "experimental_D60_400_round_refit_only", "fit_authorized": True,
            "gate_A": "NOT_RUN", "gate_B": "NOT_RUN"}


def _base(root: Path) -> Path:
    root = root.resolve()
    base = root / "runs/yasa-development-refit"
    if any(path.resolve() != path for path in (root / "runs", root / "runs/compute.lock",
                                              base, base / "attempts")):
        raise ValueError("YASA D60 output redirects outside canonical runs")
    return base


def _canonical_leaf(path: Path) -> Path:
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError("D60 refit output leaf redirects outside its canonical path: " + str(path))
    return path


def _selected(root: Path, cv_result: Path, deadline: float) -> tuple[dict, dict, dict, list[dict], dict]:
    protocol, split, records = development_records(root)
    if len(split["participants"]["development"]) != 60 or len(records) != 119:
        raise ValueError("D60/119 frozen development membership differs")
    cv_result = cv_result.resolve()
    member = _member(root, cv_result, protocol, split, records, deadline)
    config = read_json(Path(member["config_path"]))
    lr_multiplier = config["selection_evidence"]["content"]["chosen"]["lr_multiplier"]
    expected_params = dict(boosting_type="gbdt", n_estimators=400, max_depth=5, num_leaves=90,
        colsample_bytree=.5, importance_type="gain", learning_rate=.1 * lr_multiplier,
        class_weight={"N1": 2.2, "N2": 1, "N3": 1.2, "R": 1.4, "W": 1},
        n_jobs=4, random_state=config["seed"], deterministic=True, force_col_wise=True,
        verbosity=-1)
    if (config.get("runtime_lock_sha256") != _hash(root / "requirements/research.hashed.txt", root, deadline) or
        Path(config.get("runtime", {}).get("executable", "")).resolve() !=
            (root / ".venvs/research/Scripts/python.exe").resolve() or
        config.get("source_sha") != native.SOURCE_SHA or
        config.get("recipe_sha") != native.RECIPE_SHA or
        config.get("implementation_sha") != _hash(root / "sleepedf/yasa_baseline.py", root, deadline) or
        config.get("truth_manifest_sha256") != _hash(
            root / "research/development-truth-v1.json", root, deadline) or
        config.get("params") != expected_params):
        raise ValueError("Selected completed CV runtime, source, or native 400-round recipe differs")
    return protocol, split, config, records, member


def _refit_config(config: dict, member: dict, snapshot: dict) -> dict:
    return dict(config, development_refit={"schema_version": "1.0",
        "cv_run_id": member["run_id"], "cv_result_sha256": member["result_sha256"],
        "cv_config_sha256": member["config_sha256"],
        "source_snapshot": snapshot, "training_membership": "all_60_frozen_development_participants",
        "selection_rule": "reuse_frozen_completed_CV_configuration_no_refit_score",
        "confirmatory": False, "gate_A": "NOT_RUN", "gate_B": "NOT_RUN"})


def _roles(split: dict, selection: dict, feature_manifest: dict) -> dict:
    return native._fold_roles({"train": sorted(split["participants"]["development"])},
                              selection, split, feature_manifest)


def _fit_record(protocol: dict, split: dict, config_id: str, roles: dict,
                cv_config: dict, member: dict, training_epochs: int,
                checkpoint_sha: str) -> dict:
    return {"status": "FIT_COMPLETE", "protocol_hash": protocol["protocol_hash"],
            "config_hash": config_id, "roles": roles, "roles_id": content_id(roles),
            "fitted_participants": sorted(split["participants"]["development"]),
            "excluded_validation_participants": [], "training_epochs": training_epochs,
            "trees": 2000, "checkpoint_sha": checkpoint_sha,
            "feature_manifest_sha256": cv_config["feature_manifest_sha256"],
            "truth_manifest_sha256": cv_config["truth_manifest_sha256"],
            "cv_result_sha256": member["result_sha256"],
            "experimental_no_in_sample_score": True}


def _recheck_selected(root: Path, protocol: dict, split: dict, records: list[dict],
                      cv_config: dict, member: dict, deadline: float) -> None:
    selection = cv_config["selection_evidence"]
    if (native._selection_evidence(root, root / selection["path"], protocol, split, records,
            cv_config["variant"], cv_config["seed"],
            selection["content"]["chosen"]["lr_multiplier"]) != selection or
        _hash(Path(member["result_path"]), root, deadline) != member["result_sha256"] or
        _hash(Path(member["config_path"]), root, deadline) != member["config_sha256"]):
        raise ValueError("Selected completed CV result/config/selection changed during D60 refit")


def _fit(root: Path, cv_result: Path, out: Path, snapshot: dict, deadline: float) -> dict:
    import importlib.metadata
    import joblib
    import lightgbm as lgb
    import pandas as pd
    protocol, split, cv_config, records, member = _selected(root, cv_result, deadline)
    installed = dict(sorted((dist.metadata["Name"].lower(), dist.version)
                            for dist in importlib.metadata.distributions()))
    if (cv_config.get("runtime", {}).get("python") != sys.version or
        cv_config["runtime"].get("packages") != installed or
        Path(sys.executable).resolve() != Path(cv_config["runtime"]["executable"]).resolve()):
        raise ValueError("D60 refit installed Python/packages differ from completed CV")
    variant = cv_config["variant"]
    feature_manifest_path = root / "derived/yasa_features" / variant / "manifest.json"
    feature_manifest = read_json(feature_manifest_path)
    if (feature_manifest.get("config") != native._feature_config(protocol, variant) or
        feature_manifest.get("config_hash") != cv_config.get("feature_config_hash") or
        feature_manifest.get("recordings") != 119 or
        feature_manifest.get("participants") != 60 or
        _hash(feature_manifest_path, root, deadline) != cv_config.get("feature_manifest_sha256")):
        raise ValueError("D60 feature manifest differs from completed CV configuration")
    native._verify_feature_cache(root, variant, records, feature_manifest)
    native._verify_source(root)
    _frozen_truth(records, root, deadline)
    selection = cv_config["selection_evidence"]
    if native._selection_evidence(root, root / selection["path"], protocol, split, records,
            variant, cv_config["seed"], selection["content"]["chosen"]["lr_multiplier"]) != selection:
        raise ValueError("D60 refit selection bytes or contents changed")
    roles = _roles(split, selection, feature_manifest)
    refit_config = _refit_config(cv_config, member, snapshot)
    config_id = content_id(refit_config)
    config_path = _canonical_leaf(out / "config.json")
    atomic_json(config_path, refit_config, immutable=True)
    source_dir = _canonical_leaf(out / "source_snapshot")
    source_dir.mkdir(exist_ok=True)
    for name, digest in snapshot["sources"].items():
        target = _canonical_leaf(source_dir / Path(name).name)
        if target.exists():
            if _hash(target, root, deadline) != digest:
                raise ValueError("Refit source snapshot changed")
        else:
            shutil.copyfile(root / name, target)
            if _hash(target, root, deadline) != digest:
                raise ValueError("Refit copied source changed")
    atomic_json(_canonical_leaf(source_dir / "manifest.json"), snapshot, immutable=True)
    folder = _canonical_leaf(out / "refit")
    folder.mkdir(exist_ok=True)
    model_path, fit_path = (_canonical_leaf(folder / "model.joblib"),
                            _canonical_leaf(folder / "fit.json"))
    if fit_path.exists():
        fit = read_json(fit_path)
        training_epochs = sum(int(load_development_truth(rec, split)["valid_mask"].sum())
                              for rec in records)
        expected = _fit_record(protocol, split, config_id, roles, cv_config, member,
                               training_epochs, _hash(model_path, root, deadline))
        if fit != expected:
            raise ValueError("Completed D60 refit no longer matches its checkpoint/roles")
        if (_hash(root / "research/development-truth-v1.json", root, deadline) !=
            cv_config["truth_manifest_sha256"]):
            raise ValueError("Frozen D truth manifest changed during resume")
        _recheck_selected(root, protocol, split, records, cv_config, member, deadline)
        return {"status": "REFIT_COMPLETE", "resumed": True, "run_id": out.name,
                "config_hash": config_id, "fit_sha256": _hash(fit_path, root, deadline),
                "checkpoint_sha256": fit["checkpoint_sha"], "model_path": str(model_path),
                "confirmatory": False, "gate_A": "NOT_RUN", "gate_B": "NOT_RUN"}
    if model_path.exists():
        model_path.rename(_canonical_leaf(folder / (
            "unattested-" + _hash(model_path, root, deadline) + ".joblib")))
    xs, ys = [], []
    for rec in records:
        _limits(root, deadline)
        truth = load_development_truth(rec, split)
        frame = native._features(root, variant, rec, feature_manifest)
        if len(frame) != len(truth["epoch_index"]):
            raise ValueError("D60 feature/truth full grid differs")
        xs.append(frame.loc[truth["valid_mask"]])
        ys.append(native.LABEL_NAMES[truth["reference_label"][truth["valid_mask"]]])
    X = pd.concat(xs, ignore_index=True)
    y = np.concatenate(ys)
    if set(y) != set(native.LABEL_NAMES):
        raise ValueError("D60 labels lack a class")
    _limits(root, deadline)
    clf = lgb.LGBMClassifier(**cv_config["params"])
    clf.fit(X, y)
    if clf.booster_.num_trees() != 2000:
        raise ValueError("D60 refit did not produce the native 400-round five-class tree count")
    _limits(root, deadline)
    partial = _canonical_leaf(folder / "model.partial.joblib")
    if partial.exists():
        partial.rename(_canonical_leaf(folder / (
            "unattested-" + _hash(partial, root, deadline) + ".partial.joblib")))
    joblib.dump(clf, partial, compress=3)
    _limits(root, deadline)
    os.replace(partial, model_path)
    checkpoint_sha = _hash(model_path, root, deadline)
    if (_hash(root / "research/development-truth-v1.json", root, deadline) !=
        cv_config["truth_manifest_sha256"]):
        raise ValueError("Frozen D truth manifest changed during refit")
    _frozen_truth(records, root, deadline)
    native._verify_feature_cache(root, variant, records, feature_manifest)
    _recheck_selected(root, protocol, split, records, cv_config, member, deadline)
    fit = _fit_record(protocol, split, config_id, roles, cv_config, member,
                      len(y), checkpoint_sha)
    atomic_json(fit_path, fit, immutable=True)
    return {"status": "REFIT_COMPLETE", "resumed": False, "run_id": out.name,
            "config_hash": config_id, "fit_sha256": _hash(fit_path, root, deadline),
            "checkpoint_sha256": checkpoint_sha, "model_path": str(model_path),
            "confirmatory": False, "gate_A": "NOT_RUN", "gate_B": "NOT_RUN"}


def worker_main(request_path: Path) -> None:
    import psutil
    root = Path(__file__).resolve().parents[1]
    request_path = request_path.resolve()
    request = read_json(request_path)
    lease = read_json(root / "runs/compute.lock")
    current = psutil.Process()
    parent = current.parent()
    owner = psutil.Process(request["owner_pid"])
    command = request["command"]
    remaining = request.get("remaining_seconds")
    out = Path(request["out"])
    if (request_path.name != "request.json" or
        request_path.parent.parent != _base(root) / "attempts" or
        request.get("schema_version") != "1.0" or
        request.get("scope") != "experimental_D60_400_round_refit_only" or
        request.get("root") != str(root) or
        type(remaining) not in (int, float) or not math.isfinite(remaining) or
        not 0 < remaining <= SECONDS or
        out.resolve() != out or out.parent != _base(root) or
        (out / "refit").resolve() != out / "refit" or
        (out / "source_snapshot").resolve() != out / "source_snapshot" or
        lease.get("pid") != request.get("owner_pid") or
        lease.get("process_start") != request.get("owner_birth") or
        lease.get("run_id") != LEASE or lease.get("host") != socket.gethostname() or
        owner.create_time() != request["owner_birth"] or parent is None or
        (parent.pid != owner.pid and
         (parent.cmdline() != command or parent.parent() is None or
          parent.parent().pid != owner.pid or parent.parent().create_time() != owner.create_time())) or
        Path(sys.prefix).resolve() != (root / ".venvs/research").resolve() or
        any(os.environ.get(key) != "4" for key in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"))):
        raise RuntimeError("D60 refit worker has no exact runtime, command ancestry, or compute lease")
    deadline = time.monotonic() + remaining
    if request["snapshot"] != _snapshot(root, deadline):
        raise ValueError("D60 refit source/runtime changed before worker execution")
    identity = {"pid": current.pid, "birth": current.create_time(), "ppid": parent.pid,
                "owner_pid": owner.pid, "owner_birth": owner.create_time(),
                "request_sha256": file_sha256(request_path)}
    print(IDENTITY + json.dumps(identity, sort_keys=True), flush=True)
    result = _fit(root, Path(request["cv_result"]), Path(request["out"]),
                  request["snapshot"], deadline)
    print(RESULT + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def _supervise(root: Path, child: subprocess.Popen, log_path: Path, request: dict,
               deadline: float, evidence: dict) -> dict:
    import psutil
    owner = psutil.Process()
    observed = {}
    identity = result = None
    error = None
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        try:
            while True:
                for line in stream.readlines():
                    if line.startswith(IDENTITY):
                        if identity is not None:
                            raise RuntimeError("Duplicate D60 worker identity")
                        identity = json.loads(line[len(IDENTITY):])
                        worker = psutil.Process(identity["pid"])
                        if (worker.create_time() != identity.get("birth") or
                            worker.ppid() != identity.get("ppid") or
                            worker.ppid() not in (owner.pid, child.pid) or
                            identity.get("owner_pid") != owner.pid or
                            identity.get("owner_birth") != owner.create_time() or
                            identity.get("request_sha256") != request["request_sha256"]):
                            raise RuntimeError("D60 worker PID/birth/lease marker differs")
                        if worker.ppid() == child.pid:
                            launcher = psutil.Process(child.pid)
                            launcher_parent = launcher.parent()
                            if (launcher.cmdline() != request["command"] or
                                launcher_parent is None or launcher_parent.pid != owner.pid or
                                launcher_parent.create_time() != owner.create_time()):
                                raise RuntimeError("D60 Windows launcher ancestry differs")
                        observed[worker.pid] = worker.create_time()
                    elif line.startswith(RESULT):
                        if identity is None or result is not None:
                            raise RuntimeError("D60 result lacks one authenticated worker")
                        result = json.loads(line[len(RESULT):])
                if log_path.stat().st_size > 16 * 1024**2:
                    raise RuntimeError("D60 refit worker log exceeded 16MiB")
                try:
                    launcher = psutil.Process(child.pid)
                    observed[launcher.pid] = launcher.create_time()
                    for process in launcher.children(recursive=True):
                        observed[process.pid] = process.create_time()
                except psutil.NoSuchProcess:
                    pass
                if identity is not None:
                    try:
                        worker = psutil.Process(identity["pid"])
                        if worker.create_time() == identity["birth"]:
                            for process in worker.children(recursive=True):
                                observed[process.pid] = process.create_time()
                    except psutil.NoSuchProcess:
                        pass
                live = []
                for pid, birth in observed.items():
                    try:
                        process = psutil.Process(pid)
                        if process.create_time() == birth and process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                            live.append(process)
                    except psutil.NoSuchProcess:
                        pass
                try:
                    rss = sum(process.memory_info().rss for process in live)
                except psutil.NoSuchProcess:
                    continue
                evidence.update(identity=identity, observed_pid_birth=sorted(observed.items()),
                                peak_combined_rss_bytes=max(evidence.get("peak_combined_rss_bytes", 0),
                                                            owner.memory_info().rss + rss), result=result)
                _limits(root, deadline, rss)
                if child.poll() is not None and not live:
                    break
                time.sleep(.25)
        except BaseException as exc:
            error = exc
    pending = []
    inaccessible = []
    for pid, birth in observed.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == birth and process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                process.terminate()
                pending.append(process)
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            inaccessible.append((pid, birth))
    try:
        _, remaining = psutil.wait_procs(pending, timeout=3)
    except psutil.AccessDenied:
        inaccessible.extend((process.pid, observed.get(process.pid)) for process in pending)
        remaining = pending
    for process in remaining:
        try:
            if process.create_time() == observed[process.pid]:
                process.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            inaccessible.append((process.pid, observed[process.pid]))
    try:
        _, remaining = psutil.wait_procs(remaining, timeout=3)
    except psutil.AccessDenied:
        inaccessible.extend((process.pid, observed.get(process.pid)) for process in remaining)
    if child.poll() is None:
        try:
            child.kill()
        except PermissionError:
            inaccessible.append((child.pid, observed.get(child.pid)))
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        inaccessible.append((child.pid, observed.get(child.pid)))
    evidence["cleanup_survivors"] = [(p.pid, observed.get(p.pid)) for p in remaining] + inaccessible
    if evidence["cleanup_survivors"]:
        raise RuntimeError("D60 refit cleanup left live or inaccessible PID/birth: " +
                           repr(evidence["cleanup_survivors"])) from error
    if error is not None:
        raise error
    if identity is None or result is None or child.returncode != 0:
        raise RuntimeError("D60 refit worker failed before a complete authenticated result")
    return result


def run(root: Path, cv_result: Path) -> dict:
    root = root.resolve()
    cv_result = (root / cv_result).resolve()
    base = _base(root)
    started = time.monotonic()
    deadline = started + SECONDS
    with compute_lease(root, LEASE) as lease:
        attempt = base / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
                                       "-" + uuid.uuid4().hex[:8])
        attempt.mkdir(parents=True, exist_ok=False)
        evidence = {"identity": None, "observed_pid_birth": [], "cleanup_survivors": [],
                    "peak_combined_rss_bytes": 0, "result": None}
        request_sha = log_sha = None
        failure = None
        status = "FAILED"
        try:
            snapshot = _snapshot(root, deadline)
            _, _, config, _, member = _selected(root, cv_result, deadline)
            refit_config = _refit_config(config, member, snapshot)
            out = base / ("yasa-d60-" + config["variant"] + "-s" + str(config["seed"]) +
                          "-" + content_id(refit_config)[-10:])
            if (out.resolve() != out or (out / "refit").resolve() != out / "refit" or
                (out / "source_snapshot").resolve() != out / "source_snapshot"):
                raise ValueError("D60 refit output junction differs")
            out.mkdir(exist_ok=True)
            request_path = _canonical_leaf(attempt / "request.json")
            command = [str(root / ".venvs/research/Scripts/python.exe"), "-B", "-m",
                       "sleepedf.yasa_development_refit", "--child", str(request_path)]
            request = {"schema_version": "1.0", "scope": "experimental_D60_400_round_refit_only",
                       "root": str(root), "cv_result": str(Path(cv_result).resolve()),
                       "out": str(out), "owner_pid": lease["pid"],
                       "owner_birth": lease["process_start"], "command": command,
                       "snapshot": snapshot, "remaining_seconds": deadline - time.monotonic()}
            atomic_json(request_path, request, immutable=True)
            request_sha = _hash(request_path, root, deadline)
            request["request_sha256"] = request_sha
            env = os.environ.copy()
            env.update({key: "4" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                               "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
            log_path = _canonical_leaf(attempt / "worker.log")
            with log_path.open("xb") as log:
                child = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                         stderr=subprocess.STDOUT)
                result = _supervise(root, child, log_path, request, deadline, evidence)
            log_sha = _hash(log_path, root, deadline)
            if (result.get("status") != "REFIT_COMPLETE" or result.get("run_id") != out.name or
                result.get("config_hash") != content_id(refit_config) or
                result.get("confirmatory") is not False or result.get("gate_A") != "NOT_RUN" or
                result.get("gate_B") != "NOT_RUN" or
                result.get("model_path") != str(out / "refit/model.joblib") or
                result.get("checkpoint_sha256") != _hash(out / "refit/model.joblib", root, deadline) or
                result.get("fit_sha256") != _hash(out / "refit/fit.json", root, deadline)):
                raise ValueError("D60 refit result differs from immutable source/config/checkpoint")
            _, _, _, _, after_member = _selected(root, cv_result, deadline)
            if after_member != member or _snapshot(root, deadline) != snapshot:
                raise ValueError("D60 refit source/runtime changed during the worker attempt")
            _limits(root, deadline)
            status = "EXPERIMENTAL_D60_REFIT_COMPLETE"
        except BaseException as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
            if 'log_path' in locals() and log_path.exists() and log_path.stat().st_size <= 16 * 1024**2:
                log_sha = file_sha256(log_path)
        record = {"schema_version": "1.0", "artifact_type": "yasa_development_d60_refit_attempt",
                  "status": status, "failure": failure, "request_sha256": request_sha,
                  "log_sha256": log_sha, "worker_evidence": evidence,
                  "elapsed_seconds": time.monotonic() - started,
                  "confirmatory": False, "gate_A": "NOT_RUN", "gate_B": "NOT_RUN"}
        atomic_json(_canonical_leaf(attempt / "attempt.json"), record, immutable=True)
        if status != "EXPERIMENTAL_D60_REFIT_COMPLETE":
            raise RuntimeError("D60 refit failed; retained attempt: " + str(attempt))
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cv-result", type=Path)
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    if args.child is not None:
        if args.cv_result is not None:
            parser.error("child mode accepts only its immutable request")
        worker_main(args.child)
    elif args.cv_result is None:
        parser.error("refit requires --cv-result")
    else:
        record = run(args.root, args.cv_result)
        print(record["status"], flush=True)


if __name__ == "__main__":
    main()
