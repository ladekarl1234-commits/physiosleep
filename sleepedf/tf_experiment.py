"""Bounded original U-Time inner-selection and fresh outer-refit controller.

Each ``epoch`` invocation runs one complete training epoch under the shared
compute lease. No TensorFlow model is imported by this module.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace
import uuid

import numpy as np

from .contracts import content_id, json_text, read_json
from .protocol import development_records, load_development_truth
from .research import atomic_json, compute_lease, file_sha256
from .tf_native import (_build_inner_manifest, _preprocess_identity,
                        load_signal_cache, locked_packages)

SEEDS = (17, 43, 101)
MAX_EPOCHS = 2000
PATIENCE = 80
MEMORY_GIB = 10
EPOCH_SECONDS = 8 * 3600
EXECUTION_MODE = "native_compiled_train_on_batch_v1"
CHECKPOINT_SCHEMA = "model_adam_generator_loss_mean_train_counter_v1"
FIT_DEPENDENCIES = ("sleepedf/tf_experiment.py", "sleepedf/tf_native.py",
                    "sleepedf/tf_prepare.py",
                    "sleepedf/contracts.py", "sleepedf/protocol.py",
                    "sleepedf/evaluation.py", "sleepedf/research.py")
SAMPLER_BASE = Path("runs/utime-sampler-parity")
SAMPLER_SEEDS = (17, 43, 101)
SAMPLER_BATCHES = 8


def _fit_implementation(root: Path) -> dict[str, str]:
    return {name: file_sha256(root / name) for name in FIT_DEPENDENCIES}


def _fit_output_path(root: Path, output: Path, data_root: Path) -> Path:
    root, output, data_root = root.resolve(), output.resolve(), data_root.resolve()
    runs = (root / "runs").resolve()
    prep = (root / "runs/utime/full-record-v2").resolve()
    if (runs != root / "runs" or not runs.is_relative_to(root) or
            not output.is_relative_to(runs) or output.is_relative_to(data_root) or
            output.is_relative_to(prep) or runs.is_relative_to(data_root)):
        raise ValueError("U-Time fit output must stay in project runs outside data/preparation")
    return output


def _fit_artifact_paths(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    targets = (run_dir / "manifest.json", run_dir / "state.json", run_dir / "states",
               run_dir / "checkpoints", run_dir / "selected", run_dir / "invocations")
    if any(path.resolve() != path for path in targets):
        raise ValueError("U-Time fit artifact path redirects outside its run")


def _utc_stamp(value: object) -> datetime:
    if type(value) is not str:
        raise ValueError("U-Time invocation timestamp is missing")
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None or stamp.utcoffset() != timezone.utc.utcoffset(stamp):
        raise ValueError("U-Time invocation timestamp is not UTC")
    return stamp


def _sampler_sources(root: Path) -> dict[str, str]:
    paths = ("vendor/utime/utime/sequences/balanced_random_batch_sequence.py",
             "vendor/utime/utime/sequences/batch_sequence.py",
             ".venvs/utime/Lib/site-packages/psg_utils/dataset/queue/base_queue.py",
             ".venvs/utime/Lib/site-packages/psg_utils/errors/__init__.py")
    source_manifest = read_json(root / "research/sources/utime.json")
    for name in paths[:2]:
        relative = name.removeprefix("vendor/utime/")
        if file_sha256(root / name) != source_manifest["files"][relative]["sha256"]:
            raise ValueError("Pinned native sampler source changed")
    return {name: file_sha256(root / name) for name in paths}


def _sampler_identity(root: Path) -> dict:
    lock = root / "requirements/utime.lock.txt"
    installation = root / "research/runtimes/utime/install.json"
    if (locked_packages(lock).get("numpy") != "1.23.5" or
            file_sha256(lock) != read_json(installation)["freeze_sha256"]):
        raise ValueError("Sampler native NumPy runtime changed")
    return {"schema_version": "1.0", "artifact_type": "utime_native_sampler_inputs",
            "decision_sha256": file_sha256(root / "research/utime-fit-readiness-decision-v1.json"),
            "worker_sha256": file_sha256(root / "tools/tf_train_worker.py"),
            "checker_sha256": file_sha256(root / "sleepedf/tf_experiment.py"),
            "native_sources_sha256": _sampler_sources(root),
            "native_manifest_sha256": file_sha256(root / "research/sources/utime.json"),
            "runtime_lock_sha256": file_sha256(lock),
            "runtime_install_sha256": file_sha256(installation),
            "native_executable_sha256": file_sha256(root / ".venvs/utime/Scripts/python.exe"),
            "numpy_version": "1.23.5", "seeds": list(SAMPLER_SEEDS),
            "batches_per_seed": SAMPLER_BATCHES, "batch_size": 12,
            "fixture_lengths": [35, 36, 73], "context": 35, "target_samples": 3840}


def _sampler_output_base(root: Path) -> Path:
    root = root.resolve()
    base = root / SAMPLER_BASE
    targets = (root / "runs", root / "runs/compute.lock", base,
               base / "attempts", base / "current.json")
    if any(path.resolve() != path for path in targets):
        raise ValueError("Sampler qualification output redirects outside canonical project runs")
    return base


def _native_sampler_function(path: Path, name: str, *, owner: str | None = None,
                             namespace: dict) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body if owner is None else next(node.body for node in tree.body
                                                   if isinstance(node, ast.ClassDef) and node.name == owner)
    matches = [node for node in nodes if isinstance(node, (ast.FunctionDef, ast.ClassDef))
               and node.name == name]
    if len(matches) != 1:
        raise ValueError("Pinned native sampler function is missing or ambiguous")
    code = compile(ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[])),
                   str(path), "exec")
    exec(code, namespace)
    return namespace[name]


def _sampler_proof(root: Path) -> dict:
    """Compare raw native and adapter draws before model or augmentation imports."""
    import psutil
    if np.__version__ != "1.23.5":
        raise ValueError("Native sampler proof requires pinned NumPy 1.23.5")
    started = time.monotonic()
    module_spec = importlib.util.spec_from_file_location("tf_train_sampler_proof",
                                                         root / "tools/tf_train_worker.py")
    worker = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(worker)
    sources = list(_sampler_sources(root))
    namespace = {"np": np, "logger": SimpleNamespace(warning=lambda *args: None)}
    namespace["MarginError"] = _native_sampler_function(root / sources[3], "MarginError",
                                                          namespace=namespace)
    namespace["_check_margin"] = _native_sampler_function(root / sources[1], "_check_margin",
                                                            namespace=namespace)
    get_period = _native_sampler_function(root / sources[1], "get_period",
                                           owner="BatchSequence", namespace=namespace)
    get_random_study = _native_sampler_function(root / sources[2], "get_random_study",
                                                 owner="BaseQueue", namespace=namespace)
    get_balanced = _native_sampler_function(root / sources[0],
                                             "get_class_balanced_random_period",
                                             owner="BalancedRandomBatchSequence", namespace=namespace)

    class Study:
        def __init__(self, number, length, classes):
            self.x = np.broadcast_to((np.arange(length, dtype=np.float32) + 1000 * number)
                                     [:, None, None], (length, 3840, 1)).copy()
            self.y = np.asarray([classes[index % len(classes)] for index in range(length)],
                                dtype=np.int8)
            self.y[length // 2] = -1
            self.n_periods = length

        def get_class_indices(self, stage):
            return np.flatnonzero(self.y == stage)

        def get_periods_by_idx(self, start_idx, n_periods):
            return (self.x[start_idx:start_idx + n_periods],
                    self.y[start_idx:start_idx + n_periods, None])

    studies = [Study(0, 35, [0, 1]), Study(1, 36, [2, 3]),
               Study(2, 73, [0, 1, 2, 3, 4])]
    queue = SimpleNamespace(dataset=SimpleNamespace(pairs=studies))

    @contextmanager
    def random_study():
        yield get_random_study(queue)

    queue.get_random_study = random_study
    reference = SimpleNamespace(n_classes=5, sample_prob=[0.2] * 5, margin=17,
                                dataset_queue=queue)
    reference.get_period = MethodType(get_period, reference)
    records = [{"recording_id": str(index), "n_epochs": len(study.y),
                "signal_path": str(root / f"sampler-fixture-{index}-x.npy"),
                "label_path": str(root / f"sampler-fixture-{index}-y.npy"),
                "continuous_blocks": [[0, len(study.y)]]}
               for index, study in enumerate(studies)]
    original_load = np.load

    def fixture_load(path, *args, **kwargs):
        name = Path(path).name
        match = re.fullmatch(r"sampler-fixture-([0-2])-([xy])\.npy", name)
        if match is None:
            raise ValueError("Sampler proof tried to read a non-fixture path")
        study = studies[int(match[1])]
        return study.x if match[2] == "x" else study.y

    np.load = fixture_load
    try:
        adapter = worker.NativeBalancedSampler(records)
    finally:
        np.load = original_load
    details = []

    def state_digest():
        state = np.random.get_state()
        return content_id({"name": state[0], "keys": state[1].tolist(),
                           "position": state[2], "has_gauss": state[3],
                           "cached_gauss": state[4]})

    old_negative = []
    for seed in SAMPLER_SEEDS:
        np.random.seed(seed)
        expected = []
        for _ in range(SAMPLER_BATCHES):
            x_rows, y_rows = [], []
            for _ in range(12):
                x, y = get_balanced(reference)
                x_rows.append(x)
                y_rows.append(y.astype(np.int32))
            expected.append((np.stack(x_rows), np.stack(y_rows), state_digest()))
        negative_reference = expected[0]
        np.random.seed(seed)
        for batch_number, (x_ref, y_ref, state_ref) in enumerate(expected, 1):
            x, y, mask = adapter.batch(12)
            if (x.shape != (12, 35, 3840, 1) or x.dtype != np.float32 or
                    y.shape != (12, 35, 1) or y.dtype != np.int32 or
                    mask.shape != y.shape or mask.dtype != np.bool_ or
                    x_ref.shape != x.shape or x_ref.dtype != x.dtype or
                    y_ref.shape != y.shape or y_ref.dtype != y.dtype or
                    x.tobytes() != x_ref.tobytes() or y.tobytes() != y_ref.tobytes() or
                    mask.tobytes() != (y_ref >= 0).tobytes() or
                    state_digest() != state_ref):
                raise ValueError("Native raw sampler or complete MT19937 state differs")
            details.append({"seed": seed, "batch": batch_number,
                            "x_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                            "y_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
                            "mt19937_id": state_ref})
            if (time.monotonic() - started > 120 or
                    psutil.Process().memory_info().rss > 1024**3):
                raise RuntimeError("Sampler proof exceeded two-minute or 1 GiB bound")
        np.random.seed(seed)
        x_ref, y_ref, native_state = negative_reference
        original_choice = np.random.choice

        def old_class_draw(values, *args, **kwargs):
            if (np.array_equal(values, np.arange(5)) and
                    kwargs.get("p") == [0.2] * 5):
                return original_choice(values, size=1)
            return original_choice(values, *args, **kwargs)

        np.random.choice = old_class_draw
        try:
            x_old, y_old, _ = adapter.batch(12)
            old_state = state_digest()
        finally:
            np.random.choice = original_choice
        diverged = (x_old.tobytes() != x_ref.tobytes() or
                    y_old.tobytes() != y_ref.tobytes() or old_state != native_state)
        if not diverged:
            raise ValueError("Sampler negative control did not distinguish old kernel")
        old_negative.append({"seed": seed, "diverged": diverged})
    if "tensorflow" in sys.modules:
        raise ValueError("Sampler qualification imported TensorFlow")
    np.random.seed(17)
    first_native = int(np.random.choice(np.arange(5), size=1, p=[0.2] * 5)[0])
    np.random.seed(17)
    first_old = int(np.random.choice(np.arange(5)))
    return {"checked_batches": len(details), "batch_evidence": details,
            "old_no_probability_kernel_diverged": True,
            "old_negative_by_seed": old_negative,
            "old_first_class": first_old, "native_first_class": first_native,
            "tensorflow_imported": "tensorflow" in sys.modules}


def qualify_sampler(root: Path) -> dict:
    """Run only the bounded synthetic native sampler comparison under a lease."""
    root = Path(root).resolve()
    if Path(sys.executable).resolve() != (root / ".venvs/utime/Scripts/python.exe").resolve():
        raise ValueError("Sampler qualification requires the isolated native runtime")
    if (any(os.environ.get(name) != "4" for name in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS")) or
            shutil.disk_usage(root).free < 20 * 1024**3):
        raise ValueError("Sampler qualification requires four threads and 20 GiB free disk")
    identity = _sampler_identity(root)
    base = _sampler_output_base(root)
    from datetime import datetime, timezone
    with compute_lease(root, "utime-sampler-parity"):
        attempt = base / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
                                        "-" + uuid.uuid4().hex[:8])
        attempt.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        try:
            proof = _sampler_proof(root)
            if proof["checked_batches"] != 24 or proof["tensorflow_imported"]:
                raise ValueError("Sampler parity proof is incomplete")
            result = {"schema_version": "1.0", "artifact_type": "utime_sampler_parity",
                      "status": "PASS", "inputs": identity, "proof": proof,
                      "elapsed_seconds": time.monotonic() - started}
            if not 0 < result["elapsed_seconds"] <= 120:
                raise RuntimeError("Sampler parity exceeded two-minute bound")
            atomic_json(attempt / "result.json", result, immutable=True)
            atomic_json(base / "current.json", {"status": "PASS",
                        "result_path": str((attempt / "result.json").resolve()),
                        "result_sha256": file_sha256(attempt / "result.json")})
        except BaseException as exc:
            atomic_json(attempt / "failure.json", {"status": "FAILED", "error": str(exc),
                        "inputs": identity}, immutable=True)
            atomic_json(base / "current.json", {"status": "FAILED",
                        "failure_path": str((attempt / "failure.json").resolve())})
            raise
    return require_sampler_qualification(root)


def require_sampler_qualification(root: Path) -> dict:
    root = Path(root).resolve()
    base = _sampler_output_base(root)
    current = read_json(base / "current.json")
    if (type(current) is not dict or set(current) !=
            {"status", "result_path", "result_sha256"} or current["status"] != "PASS"):
        raise ValueError("Current native sampler parity is absent")
    path = Path(current["result_path"])
    latest = max((item for item in (base / "attempts").iterdir() if item.is_dir()), default=None)
    if (not path.is_absolute() or path != path.resolve() or
            path.parent != latest or path.name != "result.json" or
            file_sha256(path) != current["result_sha256"]):
        raise ValueError("Sampler parity result path or bytes changed")
    result = read_json(path)
    proof = result.get("proof")
    if (result.get("schema_version") != "1.0" or
            result.get("artifact_type") != "utime_sampler_parity" or
            result.get("status") != "PASS" or
            result.get("inputs") != _sampler_identity(root) or
            type(result.get("elapsed_seconds")) not in (int, float) or
            not math.isfinite(result["elapsed_seconds"]) or
            not 0 < result["elapsed_seconds"] <= 120 or
            type(proof) is not dict or proof.get("checked_batches") != 24 or
            proof.get("old_no_probability_kernel_diverged") is not True or
            proof.get("old_negative_by_seed") !=
            [{"seed": seed, "diverged": True} for seed in SAMPLER_SEEDS] or
            proof.get("tensorflow_imported") is not False or
            type(proof.get("batch_evidence")) is not list or
            len(proof["batch_evidence"]) != 24):
        raise ValueError("Native sampler parity evidence differs")
    for seed in SAMPLER_SEEDS:
        if [row.get("batch") for row in proof["batch_evidence"]
            if row.get("seed") == seed] != list(range(1, SAMPLER_BATCHES + 1)):
            raise ValueError("Native sampler parity seed/batch grid differs")
    return {"artifact_type": "utime_sampler_parity_verification",
            "result_path": str(path), "result_sha256": file_sha256(path),
            "inputs_id": content_id(result["inputs"]), "checked_batches": 24}


def select_inner_epoch(history: list[dict]) -> tuple[int | None, float, int, bool]:
    """Registered float64 pooled F1 rounded four decimals; earliest tie."""
    best_epoch, best_score, wait = None, -1.0, 0
    for index, row in enumerate(history, 1):
        if wait >= PATIENCE:
            raise ValueError("U-Time inner history continues after first patience stop")
        score = row.get("val_macro_f1_rounded4")
        if (row.get("epoch") != index or type(score) not in (int, float) or
                not math.isfinite(score) or not 0 <= score <= 1 or
                float(np.round(score, 4)) != float(score)):
            raise ValueError("U-Time inner history has non-sequential or non-finite metric")
        if score > best_score:
            best_epoch, best_score, wait = index, float(score), 0
        else:
            wait += 1
    if len(history) > MAX_EPOCHS:
        raise ValueError("U-Time history exceeds native maximum")
    return best_epoch, best_score, wait, wait >= PATIENCE or len(history) == MAX_EPOCHS


def validate_state_chain(run_dir: Path, pointer: dict, manifest: dict,
                         *, pending: bool = False) -> None:
    """Verify every immutable epoch state and the mutable latest pointer."""
    epoch = pointer.get("completed_epochs")
    if type(epoch) is not int or not 1 <= epoch <= MAX_EPOCHS:
        raise ValueError("U-Time state chain has invalid epoch")
    states = run_dir / "states"
    expected_names = {f"epoch-{number:04d}.json" for number in range(1, epoch + 1)}
    expected_names.update(f"epoch-{number:04d}.commit.json" for number in
                          range(1, epoch if pending else epoch + 1))
    if (not states.is_dir() or {path.name for path in states.iterdir()} != expected_names or
            pointer.get("state_snapshot_sha256") is None):
        raise ValueError("U-Time immutable state chain is incomplete")
    prior_sha = None
    for number in range(1, epoch + 1):
        path = states / f"epoch-{number:04d}.json"
        snapshot = read_json(path)
        if (snapshot.get("run_id") != manifest["run_id"] or
                snapshot.get("completed_epochs") != number or
                snapshot.get("prior_state_sha256") != prior_sha or
                snapshot.get("history_prefix_id") != content_id(snapshot.get("history")) or
                type(snapshot.get("history")) is not list or
                len(snapshot["history"]) != number or
                snapshot["history"] != pointer["history"][:number] or
                snapshot.get("rng") is None):
            raise ValueError("U-Time immutable epoch ancestry differs")
        prior_sha = file_sha256(path)
        if not pending or number < epoch:
            commit = read_json(states / f"epoch-{number:04d}.commit.json")
            execution_path = Path(commit.get("execution_path", ""))
            if (commit.get("run_id") != manifest["run_id"] or
                    commit.get("epoch") != number or
                    commit.get("snapshot_sha256") != prior_sha or
                    not execution_path.is_absolute() or
                    execution_path != execution_path.resolve() or
                    execution_path.name != "execution.json" or
                    execution_path.parent.parent != run_dir.resolve() / "invocations" or
                    file_sha256(execution_path) != commit.get("execution_sha256")):
                raise ValueError("U-Time epoch commit or execution ancestry differs")
            execution = read_json(execution_path)
            report = execution.get("report")
            invocation_path = Path(execution.get("invocation_path", ""))
            if (not invocation_path.is_absolute() or
                    invocation_path != invocation_path.resolve() or
                    invocation_path != execution_path.parent / "invocation.json" or
                    file_sha256(invocation_path) != execution.get("invocation_sha256")):
                raise ValueError("U-Time committed invocation digest differs")
            invocation = read_json(invocation_path)
            project_root = Path(manifest["project_root"])
            command = [str(project_root / ".venvs/utime/Scripts/python.exe"), "-I",
                       str(project_root / "tools/tf_train_worker.py"),
                       "epoch", "--manifest", str(run_dir.resolve() / "manifest.json")]
            required_environment = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                                    "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4",
                                    "TF_NUM_INTRAOP_THREADS": "4", "TF_NUM_INTEROP_THREADS": "4",
                                    "CUDA_VISIBLE_DEVICES": "-1", "TF_DETERMINISTIC_OPS": "1",
                                    "WANDB_MODE": "disabled"}
            prior_pointer_sha = None
            if number > 1:
                earlier = read_json(states / f"epoch-{number - 1:04d}.json")
                earlier["state_snapshot_sha256"] = snapshot["prior_state_sha256"]
                prior_pointer_sha = hashlib.sha256(json_text(earlier).encode("utf-8")).hexdigest()
            lease = invocation.get("lease_owner")
            if (execution.get("status") != "SUCCESS" or execution.get("exit_code") != 0 or
                    type(report) is not dict or report.get("run_id") != manifest["run_id"] or
                    report.get("epoch") != number or report.get("state_sha256") != prior_sha or
                    invocation.get("run_id") != manifest["run_id"] or
                    invocation.get("manifest_sha256") != file_sha256(run_dir / "manifest.json") or
                    invocation.get("command") != command or
                    invocation.get("thread_environment") != required_environment or
                    invocation.get("prior_state_sha256") != prior_pointer_sha or
                    invocation.get("prior_state_snapshot_sha256") !=
                    snapshot.get("prior_state_sha256") or
                    invocation.get("worker_sha256") != manifest.get("worker_sha256") or
                    invocation.get("fit_implementation_sha256") !=
                    manifest.get("fit_implementation_sha256") or
                    type(lease) is not dict or
                    type(lease.get("pid")) is not int or lease["pid"] < 1 or
                    type(lease.get("process_start")) not in (int, float) or
                    not math.isfinite(lease["process_start"]) or
                    lease.get("run_id") != "utime-" + manifest["run_id"][7:23] or
                    type(lease.get("host")) is not str or not lease["host"] or
                    execution.get("log_sha256") != file_sha256(execution_path.parent / "epoch.log") or
                    type(execution.get("peak_observed_host_bytes")) is not int or
                    not 0 < execution["peak_observed_host_bytes"] <= MEMORY_GIB * 1024**3 or
                    type(execution.get("elapsed_seconds")) not in (int, float) or
                    not math.isfinite(execution["elapsed_seconds"]) or
                    not 0 < execution["elapsed_seconds"] <= EPOCH_SECONDS or
                    _utc_stamp(execution.get("ended_utc")) <
                    _utc_stamp(invocation.get("started_utc"))):
                raise ValueError("U-Time committed epoch lacks successful execution")
        if number == epoch and (pointer.get("state_snapshot_sha256") != prior_sha or
                                {key: value for key, value in pointer.items()
                                 if key != "state_snapshot_sha256"} != snapshot):
            raise ValueError("U-Time latest state pointer differs from epoch snapshot")


def validate_fit_history(history: list[dict], manifest: dict, validation_total: int) -> None:
    """Recompute stored epoch evidence before either resume or selection."""
    from .evaluation import macro_f1_fraction
    if type(history) is not list or not history or len(history) > MAX_EPOCHS:
        raise ValueError("U-Time fit history is missing or exceeds the native maximum")
    for epoch, row in enumerate(history, 1):
        if (type(row) is not dict or type(row.get("epoch")) is not int or row["epoch"] != epoch or
                row.get("steps") != manifest["steps_per_epoch"] or row.get("batch_size") != 12 or
                type(row.get("valid_targets")) is not int or
                not 1 <= row["valid_targets"] <= manifest["steps_per_epoch"] * 12 * 35 or
                type(row.get("train_loss")) not in (int, float) or
                not math.isfinite(row["train_loss"]) or row["train_loss"] < 0 or
                type(row.get("wall_seconds")) not in (int, float) or
                not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] <= 0):
            raise ValueError("U-Time fit history has invalid sequential training evidence")
        if manifest["phase"] == "inner":
            matrix = np.asarray(row.get("val_confusion"))
            if (matrix.shape != (5, 5) or matrix.dtype.kind not in "iu" or np.any(matrix < 0) or
                    type(row.get("val_invalid_epochs")) is not int or row["val_invalid_epochs"] < 0 or
                    int(matrix.sum()) + row["val_invalid_epochs"] != validation_total):
                raise ValueError("U-Time inner history lacks full-grid confusion evidence")
            exact = float(macro_f1_fraction(matrix))
            if (row.get("val_macro_f1") != exact or
                    row.get("val_macro_f1_rounded4") != float(np.round(exact, 4))):
                raise ValueError("U-Time inner history differs from recomputed pooled F1")
        elif manifest["phase"] != "outer" or any(row.get(key, "missing") is not None for key in
                ("val_macro_f1", "val_macro_f1_rounded4", "val_confusion", "val_invalid_epochs")):
            raise ValueError("U-Time outer history cannot contain validation selection")


def validate_checkpoint_files(state: dict, run_dir: Path) -> None:
    """Bind restored TensorFlow prefix to exactly the verified run-local shards."""
    epoch = state.get("completed_epochs")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("U-Time checkpoint epoch is invalid")
    expected = run_dir.resolve() / "checkpoints" / f"ckpt-{epoch}"
    prefix = Path(state.get("checkpoint_prefix", ""))
    if prefix != expected or prefix.resolve() != expected:
        raise ValueError("U-Time checkpoint prefix differs from the canonical run-local epoch")
    paths = sorted(prefix.parent.glob(prefix.name + ".*"))
    shards = []
    for path in paths:
        if path.resolve().parent != expected.parent or not path.is_file():
            raise ValueError("U-Time checkpoint shard escapes its run")
        if path.name == prefix.name + ".index":
            continue
        match = re.fullmatch(re.escape(prefix.name) + r"\.data-(\d{5})-of-(\d{5})", path.name)
        if match is None:
            raise ValueError("U-Time checkpoint has an unexpected shard")
        shards.append(tuple(map(int, match.groups())))
    if (not prefix.with_suffix(".index").is_file() or not shards or
            len({total for _, total in shards}) != 1 or shards[0][1] != len(shards) or
            sorted(index for index, _ in shards) != list(range(len(shards))) or
            state.get("checkpoint_files") != {str(path): file_sha256(path) for path in paths}):
        raise ValueError("U-Time checkpoint index/data inventory or content differs")


def _snapshot(root: Path, frozen_path: Path) -> tuple[dict, dict, list[dict]]:
    protocol, split, records = development_records(root)
    frozen = read_json(frozen_path)
    if type(frozen) is not dict or frozen != _build_inner_manifest(
            root, frozen["partition"]["fold_id"], protocol, split, records):
        raise ValueError("U-Time fit requires the exact frozen original development manifest")
    lock = root / "requirements" / "utime.lock.txt"
    installation = read_json(root / "research" / "runtimes" / "utime" / "install.json")
    if (file_sha256(lock) != installation["freeze_sha256"] or
            frozen["preprocess_identity"] != _preprocess_identity() or
            locked_packages(lock).get("numpy") != "1.23.5"):
        raise ValueError("U-Time fit runtime/preprocessing identity changed")
    return frozen, split, records


def _items(root: Path, frozen: dict, split: dict, records: list[dict],
           participants: list[str], cache_dir: Path, label_dir: Path) -> list[dict]:
    allowed = set(participants)
    selected = [record for record in records if record["participant_id"] in allowed]
    if {record["participant_id"] for record in selected} != allowed:
        raise ValueError("U-Time fit records do not cover every selected participant")
    items = []
    for record in selected:
        meta = load_signal_cache(record, cache_dir, metadata_only=True)
        label_path = label_dir / (record["recording_id"] + ".npy")
        labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
        truth = load_development_truth(record, split)
        expected = np.where(truth["valid_mask"], truth["reference_label"], -1)
        if (labels.shape != (record["n_epochs"],) or labels.dtype != np.int8 or
                not np.array_equal(labels, expected)):
            raise ValueError("U-Time fit label cache differs from frozen original D truth")
        signal_path = cache_dir / (record["recording_id"] + ".npy")
        items.append({"recording_id": record["recording_id"],
                      "participant_id": record["participant_id"],
                      "n_epochs": record["n_epochs"],
                      "source_psg_sha256": record["psg_sha256"],
                      "continuous_blocks": meta["continuous_blocks"],
                      "signal_path": str(signal_path.resolve()),
                      "signal_sha256": meta["payload_sha256"],
                      "label_path": str(label_path.resolve()),
                      "label_sha256": file_sha256(label_path)})
    return sorted(items, key=lambda row: row["recording_id"])


def _completed_inner_state(path: Path, frozen: dict, seed: int) -> tuple[dict, int]:
    from .evaluation import macro_f1_fraction
    from .tf_prepare import verify_fit_attestation
    if path.name != "state.json":
        raise ValueError("Outer selection requires a canonical inner state pointer")
    inner = read_json(path)
    run = read_json(path.parent / "manifest.json")
    root = Path(__file__).resolve().parents[1]
    preparation = run.get("preparation")
    if (type(preparation) is not dict or
            _fit_output_path(root, path.parent, Path(preparation["data_root"])) !=
            path.parent.resolve() or
            preparation != verify_fit_attestation(root, Path(preparation["data_root"]))):
        raise ValueError("Outer selection lacks current canonical D preparation")
    _fit_artifact_paths(path.parent)
    validate_state_chain(path.parent, inner, run)
    partition = frozen["partition"]
    history = inner.get("history")
    if type(history) is not list or not history:
        raise ValueError("Outer refit has no genuine inner epoch history")
    selected, score, wait, stopped = select_inner_epoch(history)
    if (not stopped or inner.get("status") != "COMPLETE" or
            inner.get("artifact_type") != "utime_native_fit_state" or
            inner.get("phase") != "inner" or inner.get("seed") != seed or
            inner.get("frozen_manifest_id") != frozen["manifest_id"] or
            inner.get("selected_inner_epoch") != selected or
            inner.get("best_epoch") != selected or
            inner.get("best_score_rounded4") != score or
            inner.get("patience_wait") != wait or
            inner.get("completed_epochs") != len(history) or
            run.get("run_id") != inner.get("run_id") or
            run.get("run_id") != content_id({k: v for k, v in run.items() if k != "run_id"}) or
            run.get("run_dir") != str(path.parent.resolve()) or
            run.get("project_root") != str(root) or
            run.get("frozen_manifest_id") != frozen["manifest_id"] or
            read_json(Path(run["frozen_manifest_path"])) != frozen or
            file_sha256(Path(run["frozen_manifest_path"])) != run.get("frozen_manifest_sha256") or
            run.get("train_participants") != partition["inner_train"] or
            run.get("validation_participants") != partition["inner_validation"] or
            run.get("seed") != seed or run.get("phase") != "inner" or
            run.get("steps_per_epoch") != frozen["recipe"]["inner_steps_per_epoch"] or
            run.get("fit_implementation_sha256") != _fit_implementation(Path(__file__).resolve().parents[1]) or
            run.get("sampler_qualification") != require_sampler_qualification(
                Path(__file__).resolve().parents[1]) or
            run.get("worker_sha256") != file_sha256(Path(__file__).resolve().parents[1] /
                                                     "tools/tf_train_worker.py") or
            run.get("selection_arithmetic") != "exact_fraction_macro_f1_to_float64_then_numpy_round4" or
            run.get("execution_mode") != EXECUTION_MODE or
            run.get("checkpoint_schema") != CHECKPOINT_SCHEMA or
            inner.get("execution_mode") != EXECUTION_MODE or
            inner.get("checkpoint_schema") != CHECKPOINT_SCHEMA or
            type(inner.get("train_counter")) is not int or
            inner["train_counter"] != inner.get("optimizer_iterations") or
            type(inner.get("loss_metric_total")) not in (int, float) or
            type(inner.get("loss_metric_count")) not in (int, float) or
            not np.isfinite(inner["loss_metric_total"]) or
            not np.isfinite(inner["loss_metric_count"]) or
            inner["loss_metric_total"] < 0 or
            inner["loss_metric_count"] <= 0 or
            run.get("recipe_id") != content_id(frozen["recipe"]) or
            run.get("protocol_hash") != frozen["protocol_hash"] or
            run.get("split_id") != frozen["split_id"] or
            run.get("source_manifest_sha256") != frozen["source_manifest_sha256"] or
            run.get("native_hparams_sha256") != frozen["native_hparams_sha256"] or
            run.get("batch_size") != 12 or run.get("threads") != 4 or
            run.get("memory_limit_gib") != MEMORY_GIB or
            inner.get("optimizer_iterations") != len(history) * run["steps_per_epoch"]):
        raise ValueError("Outer refit has no completed, selected inner run")
    validation_total = sum(value["n_epochs"] for value in frozen["recordings"].values()
                           if value["participant_id"] in partition["inner_validation"])
    validate_fit_history(history, run, validation_total)
    validate_checkpoint_files(inner, path.parent)
    best_path = Path(inner["best_weights_path"])
    if (best_path.parent != path.parent / "selected" or
            not re.fullmatch(rf"inner-epoch-{selected:04d}-[0-9a-f]{{32}}\.weights\.h5",
                             best_path.name) or
            not best_path.is_file() or file_sha256(best_path) != inner.get("best_weights_sha256")):
        raise ValueError("U-Time selected model or resumable optimizer state differs")
    return inner, selected


def _validate_outer_selection(manifest: dict, frozen: dict) -> None:
    if manifest["phase"] != "outer":
        return
    path = Path(manifest["inner_state_path"])
    if file_sha256(path) != manifest["inner_state_sha256"]:
        raise ValueError("Fresh outer refit selection evidence changed")
    _, selected = _completed_inner_state(path, frozen, manifest["seed"])
    if selected != manifest["selected_inner_epoch"]:
        raise ValueError("Fresh outer refit epoch differs from completed inner selection")


def create_run(root: Path, frozen_path: Path, cache_dir: Path, label_dir: Path,
               output_dir: Path, *, data_root: Path, phase: str, seed: int,
               inner_state_path: Path | None = None) -> dict:
    from .tf_verify import require_verification
    from .tf_prepare import verify_fit_attestation
    if phase not in ("inner", "outer") or seed not in SEEDS:
        raise ValueError("U-Time phase or registered seed differs")
    output_dir = _fit_output_path(root, output_dir, data_root)
    _fit_artifact_paths(output_dir)
    native_verification = require_verification(root)
    sampler_qualification = require_sampler_qualification(root)
    preparation = verify_fit_attestation(root, data_root)
    if (cache_dir.resolve() != Path(preparation["cache_dir"]) or
            label_dir.resolve() != Path(preparation["label_dir"])):
        raise ValueError("U-Time fit cache and label paths differ from D preparation")
    frozen, split, records = _snapshot(root, frozen_path)
    partition = frozen["partition"]
    selected_epoch = None
    inner_state_sha256 = None
    if phase == "outer":
        if inner_state_path is None:
            raise ValueError("Fresh outer refit requires completed inner selection")
        _, selected_epoch = _completed_inner_state(inner_state_path, frozen, seed)
        inner_state_sha256 = file_sha256(inner_state_path)
    train = partition["inner_train"] if phase == "inner" else partition["outer_train"]
    validation = partition["inner_validation"] if phase == "inner" else []
    items = _items(root, frozen, split, records, train + validation, cache_dir, label_dir)
    worker_path = root / "tools" / "tf_train_worker.py"
    result = {"schema_version": "1.0", "artifact_type": "utime_native_fit_run",
              "run_dir": str(output_dir.resolve()),
              "project_root": str(root.resolve()),
              "phase": phase, "seed": seed, "fold_id": partition["fold_id"],
              "frozen_manifest_path": str(frozen_path.resolve()),
              "frozen_manifest_sha256": file_sha256(frozen_path),
              "frozen_manifest_id": frozen["manifest_id"],
              "protocol_hash": frozen["protocol_hash"], "split_id": frozen["split_id"],
              "runtime_lock_sha256": frozen["preprocess_identity"]["runtime_lock_sha256"],
              "adapter_sha256": frozen["preprocess_identity"]["implementation_sha256"]["sleepedf/tf_native.py"],
              "worker_sha256": file_sha256(worker_path),
              "fit_implementation_sha256": _fit_implementation(root),
              "native_verification": native_verification,
              "sampler_qualification": sampler_qualification,
              "preparation": preparation,
              "source_manifest_sha256": frozen["source_manifest_sha256"],
              "native_hparams_sha256": frozen["native_hparams_sha256"],
              "recipe_id": content_id(frozen["recipe"]),
              "selection_arithmetic": "exact_fraction_macro_f1_to_float64_then_numpy_round4",
              "execution_mode": EXECUTION_MODE, "checkpoint_schema": CHECKPOINT_SCHEMA,
              "train_participants": train, "validation_participants": validation,
              "records": items, "selected_inner_epoch": selected_epoch,
              "inner_state_path": str(inner_state_path.resolve()) if inner_state_path else None,
              "inner_state_sha256": inner_state_sha256,
              "steps_per_epoch": frozen["recipe"][phase + "_steps_per_epoch"] if phase == "inner"
                                 else frozen["recipe"]["outer_steps_per_epoch"],
              "batch_size": 12, "memory_limit_gib": MEMORY_GIB,
              "threads": 4, "checkpoint_every_epochs": 1,
              "epoch_limit_seconds": EPOCH_SECONDS,
              "min_free_ram_bytes": 4 * 1024**3,
              "min_free_disk_bytes": 20 * 1024**3}
    result["run_id"] = content_id(result)
    atomic_json(output_dir / "manifest.json", result, immutable=True)
    return result


def _supervise_epoch(child: subprocess.Popen, log_path: Path, manifest: dict, root: Path,
                     manifest_path: Path,
                     *, timeout_seconds: float = EPOCH_SECONDS,
                     deadline: float | None = None) -> int:
    """Bound the exact leased worker tree through setup, training and validation."""
    import psutil
    started = last_progress = time.monotonic()
    deadline = started + timeout_seconds if deadline is None else deadline
    owner = psutil.Process()
    owner_birth = owner.create_time()
    observed: dict[int, float] = {}
    worker: tuple[int, float] | None = None
    peak = 0
    stage_order = {"setup": 0, "train": 1, "validation": 2, "sealed": 3}
    last_stage, last_index = -1, -1
    train_count = validation_count = 0
    error = None
    with log_path.open("r", encoding="utf-8", errors="replace") as reader:
        try:
            while True:
                for line in reader.readlines():
                    if line.startswith("UTIME_WORKER_IDENTITY "):
                        if worker is not None:
                            raise RuntimeError("Duplicate U-Time worker identity")
                        identity = json.loads(line[len("UTIME_WORKER_IDENTITY "):])
                        pid, birth = identity.get("pid"), identity.get("birth")
                        process = psutil.Process(pid)
                        if (type(pid) is not int or type(birth) not in (int, float) or
                                process.create_time() != birth or
                                identity.get("ppid") not in (child.pid, owner.pid) or
                                process.ppid() != identity["ppid"] or
                                identity.get("parent_pid") != owner.pid or
                                identity.get("parent_birth") != owner_birth or
                                identity.get("run_id") != manifest["run_id"] or
                                identity.get("manifest_sha256") !=
                                file_sha256(manifest_path)):
                            raise RuntimeError("U-Time worker PID, birth or lease ancestry differs")
                        if process.ppid() == child.pid:
                            launcher = psutil.Process(child.pid)
                            observed[child.pid] = launcher.create_time()
                        worker = pid, birth
                        observed[pid] = birth
                        last_progress = time.monotonic()
                    elif line.startswith("UTIME_PROGRESS "):
                        if worker is None:
                            raise RuntimeError("U-Time progress preceded worker authorization")
                        progress = json.loads(line[len("UTIME_PROGRESS "):])
                        stage, index = stage_order.get(progress.get("phase")), progress.get("index")
                        if (stage is None or type(index) is not int or index < 0 or
                                stage < last_stage or (stage == last_stage and index <= last_index) or
                                (stage == 1 and index > manifest["steps_per_epoch"]) or
                                (stage == 3 and index < 1)):
                            raise RuntimeError("U-Time worker progress is non-monotone")
                        last_stage, last_index = stage, index
                        if stage == 1:
                            train_count = index
                        elif stage == 2:
                            validation_count = index
                        last_progress = time.monotonic()
                try:
                    launcher = psutil.Process(child.pid)
                    if launcher.is_running():
                        observed[launcher.pid] = launcher.create_time()
                        for item in launcher.children(recursive=True):
                            observed[item.pid] = item.create_time()
                except psutil.NoSuchProcess:
                    pass
                except psutil.AccessDenied as exc:
                    raise RuntimeError("U-Time launcher tree became inaccessible") from exc
                if worker is not None:
                    try:
                        process = psutil.Process(worker[0])
                        if process.create_time() == worker[1]:
                            for item in process.children(recursive=True):
                                observed[item.pid] = item.create_time()
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.AccessDenied as exc:
                        raise RuntimeError("U-Time worker tree became inaccessible: " +
                                           repr(worker)) from exc
                live = []
                for pid, birth in observed.items():
                    try:
                        process = psutil.Process(pid)
                        if process.create_time() == birth and process.is_running():
                            live.append(process)
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.AccessDenied as exc:
                        raise RuntimeError("U-Time observed process became inaccessible: " +
                                           repr((pid, birth))) from exc
                rss = owner.memory_info().rss
                for process in live:
                    try:
                        rss += process.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.AccessDenied as exc:
                        raise RuntimeError("U-Time observed RSS became inaccessible: " +
                                           repr((process.pid, observed.get(process.pid)))) from exc
                peak = max(peak, rss)
                now = time.monotonic()
                if (peak > MEMORY_GIB * 1024**3 or
                        psutil.virtual_memory().available < 4 * 1024**3 or
                        shutil.disk_usage(root).free < 20 * 1024**3 or
                        now >= deadline or now - last_progress > 1800):
                    raise RuntimeError("U-Time epoch exceeded memory, disk, time or progress bound")
                if child.poll() is not None and not live:
                    if worker is None:
                        raise RuntimeError("U-Time worker exited without PID/birth handshake")
                    if (child.returncode == 0 and
                            (last_stage != 3 or train_count != manifest["steps_per_epoch"] or
                             (manifest["phase"] == "inner" and validation_count < 1) or
                             (manifest["phase"] == "outer" and validation_count != 0))):
                        raise RuntimeError("U-Time successful epoch lacks complete progress evidence")
                    break
                time.sleep(.25)
        except BaseException as exc:
            error = exc
    survivors = []
    unresolved = []
    for pid, birth in observed.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == birth and process.is_running():
                process.terminate()
                survivors.append(process)
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            unresolved.append((pid, birth))
    try:
        _, remaining = psutil.wait_procs(survivors, timeout=3)
    except psutil.AccessDenied:
        unresolved.extend((item.pid, observed.get(item.pid)) for item in survivors)
        remaining = survivors
    for process in remaining:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            unresolved.append((process.pid, observed.get(process.pid)))
    try:
        _, remaining = psutil.wait_procs(remaining, timeout=3)
    except psutil.AccessDenied:
        unresolved.extend((item.pid, observed.get(item.pid)) for item in remaining)
    if child.poll() is None:
        try:
            child.kill()
        except (PermissionError, psutil.AccessDenied):
            unresolved.append((child.pid, observed.get(child.pid)))
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        unresolved.append((child.pid, observed.get(child.pid)))
    if remaining or unresolved:
        raise RuntimeError("U-Time epoch cleanup left live or inaccessible process identities: " +
                           repr([(item.pid, observed.get(item.pid)) for item in remaining] +
                                unresolved)) from error
    if error is not None:
        raise error
    return peak


def _epoch_bound(root: Path, deadline: float) -> None:
    """Check host resources and the invocation-wide deadline outside the child."""
    import psutil
    if (time.monotonic() >= deadline or
            psutil.Process().memory_info().rss > MEMORY_GIB * 1024**3 or
            psutil.virtual_memory().available < 4 * 1024**3 or
            shutil.disk_usage(root).free < 20 * 1024**3):
        raise RuntimeError("U-Time epoch exceeded memory, disk or whole-invocation time bound")


def run_one_epoch(root: Path, manifest_path: Path) -> dict:
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    deadline = started + EPOCH_SECONDS
    from .tf_verify import require_verification
    from .tf_prepare import verify_fit_attestation
    manifest = read_json(manifest_path)
    if type(manifest) is not dict or manifest.get("artifact_type") != "utime_native_fit_run":
        raise ValueError("U-Time fit manifest is missing")
    if (manifest_path.name != "manifest.json" or _fit_output_path(
            root, manifest_path.parent, Path(manifest["preparation"]["data_root"])) !=
            manifest_path.parent.resolve() or
            manifest.get("run_dir") != str(manifest_path.parent.resolve()) or
            manifest.get("project_root") != str(root.resolve())):
        raise ValueError("U-Time fit manifest path is not canonical")
    _fit_artifact_paths(manifest_path.parent)
    if manifest.get("native_verification") != require_verification(root):
        raise ValueError("U-Time fit lacks current numerical and production batch readiness")
    if manifest.get("sampler_qualification") != require_sampler_qualification(root):
        raise ValueError("U-Time fit lacks current native sampler parity")
    if (manifest.get("preparation") != verify_fit_attestation(
            root, Path(manifest["preparation"]["data_root"])) or
            manifest.get("epoch_limit_seconds") != EPOCH_SECONDS or
            manifest.get("min_free_ram_bytes") != 4 * 1024**3 or
            manifest.get("min_free_disk_bytes") != 20 * 1024**3):
        raise ValueError("U-Time fit preparation or whole-epoch bounds changed")
    frozen, _, _ = _snapshot(root, Path(manifest["frozen_manifest_path"]))
    if (manifest["frozen_manifest_id"] != frozen["manifest_id"] or
            manifest["worker_sha256"] != file_sha256(root / "tools" / "tf_train_worker.py") or
            manifest["fit_implementation_sha256"] != _fit_implementation(root) or
            manifest["selection_arithmetic"] != "exact_fraction_macro_f1_to_float64_then_numpy_round4" or
            manifest["run_id"] != content_id({k: v for k, v in manifest.items() if k != "run_id"})):
        raise ValueError("U-Time run identity changed")
    _validate_outer_selection(manifest, frozen)
    state_path = manifest_path.parent / "state.json"
    if state_path.exists():
        validate_state_chain(manifest_path.parent, read_json(state_path), manifest)
    elif (manifest_path.parent / "states").exists() and any(
            (manifest_path.parent / "states").iterdir()):
        raise ValueError("U-Time uncommitted epoch snapshot blocks a fresh fit")
    environment = os.environ.copy()
    environment.update(TF_NUM_INTRAOP_THREADS="4", TF_NUM_INTEROP_THREADS="4",
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4",
                       CUDA_VISIBLE_DEVICES="-1",
                       TF_DETERMINISTIC_OPS="1", WANDB_MODE="disabled")
    command = [str(root / ".venvs" / "utime" / "Scripts" / "python.exe"), "-I",
               str(root / "tools" / "tf_train_worker.py"), "epoch",
               "--manifest", str(manifest_path.resolve())]
    _epoch_bound(root, deadline)
    with compute_lease(root, "utime-" + manifest["run_id"][7:23]) as lease:
        if manifest["native_verification"] != require_verification(root):
            raise ValueError("U-Time native readiness changed before the leased epoch")
        if manifest["sampler_qualification"] != require_sampler_qualification(root):
            raise ValueError("U-Time sampler qualification changed before the leased epoch")
        if manifest["preparation"] != verify_fit_attestation(
                root, Path(manifest["preparation"]["data_root"])):
            raise ValueError("U-Time D preparation changed before the leased epoch")
        _epoch_bound(root, deadline)
        attempt = manifest_path.parent / "invocations" / uuid.uuid4().hex
        attempt.mkdir(parents=True, exist_ok=False)
        log = attempt / "epoch.log"
        execution_path = attempt / "execution.json"
        invocation_path = attempt / "invocation.json"
        prior_state = read_json(state_path) if state_path.exists() else None
        required_env = {name: environment[name] for name in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
                         "TF_NUM_INTEROP_THREADS", "CUDA_VISIBLE_DEVICES",
                         "TF_DETERMINISTIC_OPS", "WANDB_MODE")}
        atomic_json(invocation_path, {"schema_version": "1.0",
                    "artifact_type": "utime_native_epoch_invocation",
                    "started_utc": started_utc,
                    "command": command, "thread_environment": required_env,
                    "run_id": manifest["run_id"],
                    "manifest_sha256": file_sha256(manifest_path),
                    "worker_sha256": manifest["worker_sha256"],
                    "fit_implementation_sha256": manifest["fit_implementation_sha256"],
                    "native_verification": manifest["native_verification"],
                    "sampler_qualification": manifest["sampler_qualification"],
                    "preparation": manifest["preparation"],
                    "prior_state_sha256": file_sha256(state_path) if prior_state else None,
                    "prior_state_snapshot_sha256": (prior_state["state_snapshot_sha256"]
                                                     if prior_state else None),
                    "lease_owner": lease}, immutable=True)
        try:
            _epoch_bound(root, deadline)
            with log.open("xb") as handle:
                child = subprocess.Popen(command, cwd=root, env=environment, stdout=handle, stderr=subprocess.STDOUT)
                peak = _supervise_epoch(child, log, manifest, root, manifest_path,
                                        deadline=deadline)
            if child.returncode:
                raise RuntimeError("Isolated native U-Time epoch failed; inspect " + str(log))
            lines = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                     if line.startswith("UTIME_TRAIN_RESULT ")]
            if len(lines) != 1:
                raise ValueError("U-Time epoch worker returned no unique result")
            report = json.loads(lines[0][len("UTIME_TRAIN_RESULT "):])
            json.dumps(report, allow_nan=False)
            if (type(report) is not dict or report.get("run_id") != manifest["run_id"] or
                    report.get("phase") != manifest["phase"] or
                    type(report.get("epoch")) is not int or
                    not 1 <= report["epoch"] <= MAX_EPOCHS):
                raise ValueError("U-Time epoch worker result identity differs")
            snapshot_path = manifest_path.parent / "states" / f"epoch-{report['epoch']:04d}.json"
            snapshot_sha = file_sha256(snapshot_path)
            snapshot = read_json(snapshot_path)
            pointer = dict(snapshot, state_snapshot_sha256=snapshot_sha)
            if (report.get("state_sha256") != snapshot_sha or
                    report.get("status") != pointer.get("status") or
                    report.get("train_loss") != pointer["history"][-1]["train_loss"] or
                    report.get("val_macro_f1_rounded4") !=
                    pointer["history"][-1]["val_macro_f1_rounded4"]):
                raise ValueError("U-Time worker report differs from sealed epoch state")
            validate_state_chain(manifest_path.parent, pointer, manifest, pending=True)
            validate_checkpoint_files(pointer, manifest_path.parent)
            validation_total = sum(item["n_epochs"] for item in manifest["records"]
                                   if item["participant_id"] in manifest["validation_participants"])
            validate_fit_history(pointer["history"], manifest, validation_total)
            if manifest["phase"] == "inner":
                selected, score, wait, stopped = select_inner_epoch(pointer["history"])
                if (pointer["status"] != ("COMPLETE" if stopped else "RUNNING") or
                        pointer.get("selected_inner_epoch") != selected or
                        pointer.get("best_score_rounded4") != score or
                        pointer.get("patience_wait") != wait):
                    raise ValueError("U-Time sealed inner stop or selection differs")
            elif pointer["status"] != ("COMPLETE" if report["epoch"] ==
                                            manifest["selected_inner_epoch"] else "RUNNING"):
                raise ValueError("U-Time sealed outer target differs")
            if (manifest["native_verification"] != require_verification(root) or
                    manifest["sampler_qualification"] != require_sampler_qualification(root) or
                    manifest["preparation"] != verify_fit_attestation(
                        root, Path(manifest["preparation"]["data_root"])) or
                    manifest["worker_sha256"] != file_sha256(root / "tools/tf_train_worker.py") or
                    manifest["fit_implementation_sha256"] != _fit_implementation(root)):
                raise ValueError("U-Time qualification or source changed during the epoch")
            _epoch_bound(root, deadline)
            atomic_json(execution_path, {"status": "SUCCESS", "exit_code": 0,
                        "run_id": manifest["run_id"], "report": report,
                        "invocation_path": str(invocation_path.resolve()),
                        "invocation_sha256": file_sha256(invocation_path),
                        "ended_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_observed_host_bytes": peak,
                        "log_sha256": file_sha256(log)}, immutable=True)
            commit = {"run_id": manifest["run_id"], "epoch": report["epoch"],
                      "snapshot_sha256": snapshot_sha,
                      "execution_path": str(execution_path.resolve()),
                      "execution_sha256": file_sha256(execution_path)}
            atomic_json(manifest_path.parent / "states" /
                        f"epoch-{report['epoch']:04d}.commit.json", commit, immutable=True)
            validate_state_chain(manifest_path.parent, pointer, manifest)
            _epoch_bound(root, deadline)
            atomic_json(manifest_path.parent / "state.json", pointer)
            return report
        except BaseException as exc:
            failure = {"status": "FAILED", "error": str(exc),
                       "invocation_path": str(invocation_path.resolve()),
                       "invocation_sha256": file_sha256(invocation_path),
                       "ended_utc": datetime.now(timezone.utc).isoformat(),
                       "elapsed_seconds": time.monotonic() - started,
                       "log_sha256": file_sha256(log) if log.exists() else None}
            target = attempt / ("commit_failure.json" if execution_path.exists() else "execution.json")
            atomic_json(target, failure, immutable=True)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create_inner", "create_outer", "epoch", "qualify_sampler"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--label-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--inner-state", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "qualify_sampler":
        print(json.dumps(qualify_sampler(root), sort_keys=True, allow_nan=False))
    elif args.action == "epoch":
        if args.manifest is None:
            parser.error("epoch requires --manifest")
        print(json.dumps(run_one_epoch(root, args.manifest), sort_keys=True, allow_nan=False))
    else:
        if any(value is None for value in (args.frozen, args.cache_dir, args.label_dir,
                                           args.data_root,
                                           args.output_dir, args.seed)):
            parser.error("create requires data/frozen/cache/label/output paths and seed")
        result = create_run(root, args.frozen, args.cache_dir, args.label_dir,
                            args.output_dir, phase=args.action.split("_")[1],
                            data_root=args.data_root, seed=args.seed,
                            inner_state_path=args.inner_state)
        print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
