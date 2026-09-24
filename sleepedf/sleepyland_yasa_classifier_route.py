"""Explicit clean-classifier compatibility for the pinned Sleepyland YASA branch.

This executes selected unchanged vendor functions in the research interpreter. It
is neither the unmodified native container nor a released-model reproduction.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
from pathlib import Path
import re
import sys
import time
import types

import numpy as np

from . import sleepyland_yasa_bridge as bridge
from . import sleepyland_yasa_experiment as experiment
from .contracts import read_json
from .protocol import development_records
from .research import file_sha256


PINNED = {
    "vendor/usleepyland/utime/bin/predict_one.py":
        "29ac04fc453a589f6115694bcdd53ee4994eed5eaff0d38d76d003547ec795d9",
    "vendor/usleepyland/psg_utils/io/channels/channels.py":
        "f1d4f40f6d079d7583a82193bac07f82ecde13249e8e975f408285bd7ac5f060",
    "vendor/usleepyland/psg_utils/io/channels/channel_types.py":
        "025c99891eec40cad75b37b7c2d87e5da86f8220145f7ffd15133a11fddfc1a1",
    "vendor/yasa/src/yasa/staging.py":
        "dd68b5a7a9756674084a0671afa35dc6cc60e0931c1e98c26e45507497df44b1",
}
CONTRACT = ("research/sleepyland-yasa-probability-contract-v2.json",
            "ee229865171de93889905b2caee42988abe0261b124793b14a8684dbc52070b3")
FUNCTIONS = ("get_save_path", "get_updated_majority_voted", "run_pred_on", "predict_study")
ORDER = tuple(bridge.GROUPS)
LABELS = ("W", "N1", "N2", "N3", "R")


def _pinned(root: Path) -> dict[str, str]:
    actual = {name: file_sha256(root / name) for name in PINNED}
    if actual != PINNED or file_sha256(root / CONTRACT[0]) != CONTRACT[1]:
        raise ValueError("Pinned Sleepyland classifier branch or channel semantics changed")
    return actual


def _channel_functions(root: Path) -> dict:
    """Execute pinned channel definitions with only their package import injected."""
    import mne
    first = root / "vendor/usleepyland/psg_utils/io/channels/channels.py"
    second = root / "vendor/usleepyland/psg_utils/io/channels/channel_types.py"
    namespace = {"__name__": "pinned_sleepyland_channels"}
    exec(compile(first.read_text(encoding="utf-8"), str(first), "exec"), namespace)
    parsed = ast.parse(second.read_text(encoding="utf-8"), filename=str(second))
    injected = [node for node in parsed.body if isinstance(node, ast.ImportFrom)
                and node.module == "psg_utils.io.channels"]
    if (len(injected) != 1 or [alias.name for alias in injected[0].names] !=
            ["ChannelMontageTuple", "ChannelMontage"]):
        raise ValueError("Pinned channel-type import differs from scoped injection")
    parsed.body.remove(injected[0])
    namespace.update(logging=logging, re=re, make_standard_montage=mne.channels.make_standard_montage)
    exec(compile(parsed, str(second), "exec"), namespace)
    for group in ORDER:
        names = list(bridge.GROUPS[group])
        if (namespace["infer_channel_types"](names) != ["EEG", "EOG"] or
                not namespace["is_eeg_central"](names[0])):
            raise ValueError("Pinned EEG/EOG type or central predicate excludes a registered group")
    return namespace


def _vendor_functions(root: Path, channels: dict, mne_module) -> dict:
    source = root / "vendor/usleepyland/utime/bin/predict_one.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                and node.name in FUNCTIONS]
    if len(selected) != len(FUNCTIONS) or {node.name for node in selected} != set(FUNCTIONS):
        raise ValueError("Pinned vendor prediction functions changed")
    namespace = {"os": os, "np": np, "mne": mne_module,
                 "logger": logging.getLogger(__name__),
                 "infer_channel_types": channels["infer_channel_types"],
                 "is_eeg_central": channels["is_eeg_central"]}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def _five(values: object, n_epochs: int, *, dtype: object | None = None) -> np.ndarray:
    array = np.asarray(values)
    if (array.shape != (n_epochs, 5) or array.dtype.kind != "f" or
            (dtype is not None and array.dtype != dtype) or
            not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1) or
            not np.allclose(array.sum(axis=1), 1., rtol=0, atol=1e-6)):
        raise ValueError("Packaged YASA probabilities lack finite fixed-five-class original grid")
    return array


def _native_mean(cached: dict[str, np.ndarray], n_epochs: int) -> tuple[dict, np.ndarray]:
    if tuple(cached) != ORDER:
        raise ValueError("Both exact named groups in pinned order are required")
    groups = {group: _five(cached[group], n_epochs).astype(np.float32)
              for group in ORDER}
    majority = groups[ORDER[0]].copy()
    majority += groups[ORDER[1]]
    majority = majority / 2
    return groups, _five(majority, n_epochs, dtype=np.float32)


def _wide_labels_model(clf):
    """Give pinned YASA room for WAKE/REM without changing the fitted classifier."""
    classes = np.asarray(clf.classes_)
    if (classes.shape != (5,) or not all(isinstance(label, str) for label in classes)
            or len(set(classes)) != 5 or set(classes) != set(LABELS)):
        raise ValueError("Clean YASA classifier has unknown, colliding or missing classes")
    return types.SimpleNamespace(classes_=classes.astype(object, copy=True),
                                 feature_name_=clf.feature_name_,
                                 predict=clf.predict, predict_proba=clf.predict_proba)


def _safe_work(root: Path, data_root: Path) -> Path:
    expected_base = root / "derived/sleepyland_yasa"
    expected_work = expected_base / "tmp"
    work = expected_work.resolve()
    if (expected_base.resolve() != expected_base or work != expected_work or
            work.is_relative_to(data_root)):
        raise ValueError("Native reader work directory escapes local derived artifacts")
    return work


def _execute_vendor(root: Path, signal: np.ndarray, recording_id: str, model_path: Path,
                    cached: dict[str, np.ndarray], *, mne_module=None, yasa_module=None) -> dict:
    """Compare pinned branch arrays without writing its `save_file` outputs."""
    _pinned(root)
    if (model_path != model_path.resolve() or not model_path.is_absolute() or
            model_path.name != "model.joblib" or signal.dtype != np.float32 or
            signal.ndim != 2 or signal.shape[1] != len(bridge.CHANNELS) or
            signal.shape[0] % (30 * 128) or not np.isfinite(signal).all()):
        raise ValueError("Explicit classifier path or 128-Hz volts signal differs")
    n_epochs = signal.shape[0] // (30 * 128)
    if n_epochs < 1 or not re.fullmatch(r"(?:SC|ST)[0-9]{4}", recording_id):
        raise ValueError("Invalid original recording grid or identity")
    expected_groups, expected_majority = _native_mean(cached, n_epochs)
    channels = _channel_functions(root)
    if mne_module is None:
        import mne as mne_module
    namespace = _vendor_functions(root, channels, mne_module)
    added_path = None
    previous_yasa = sys.modules.get("yasa")
    if yasa_module is None:
        pinned_yasa = (root / "vendor/yasa/src").resolve()
        if str(pinned_yasa) not in sys.path:
            sys.path.insert(0, str(pinned_yasa))
            added_path = str(pinned_yasa)
        try:
            import yasa as yasa_module
            if not Path(yasa_module.__file__).resolve().is_relative_to(pinned_yasa):
                raise ValueError("Compatibility branch resolved unpinned YASA")
        except BaseException:
            if added_path is not None:
                sys.path.remove(added_path)
            if previous_yasa is None:
                sys.modules.pop("yasa", None)
            raise
    saved = {}
    def capture(path, arr, argmax):
        key = Path(path).parent.name
        if (argmax or key in saved or key not in
                ({"+".join(bridge.GROUPS[group]) for group in ORDER} | {"majority"})):
            raise ValueError("Packaged route saved an unexpected group or hard-label artifact")
        saved[key] = np.asarray(arr).copy()
    namespace["save_file"] = capture
    facade_names = ("psg_utils", "psg_utils.io", "psg_utils.io.channels", "yasa")
    previous = {name: sys.modules.get(name) for name in facade_names}
    previous["yasa"] = previous_yasa
    facade = {name: types.ModuleType(name) for name in facade_names}
    facade["psg_utils"].__path__ = []
    facade["psg_utils.io"].__path__ = []
    facade["psg_utils.io.channels"].infer_channel_types = channels["infer_channel_types"]
    facade["yasa"] = yasa_module
    original = yasa_module.SleepStaging.predict_proba
    original_load = yasa_module.SleepStaging._load_model
    routed = []
    def explicit_model(native, path_to_model):
        if path_to_model != str(model_path):
            raise ValueError("Native YASA classifier must load the explicit fold model")
        return _wide_labels_model(original_load(native, path_to_model))
    def explicit_only(native, *args, **kwargs):
        if args or kwargs or hasattr(native, "_proba"):
            raise ValueError("Native YASA call must use a fresh classifier and no auto arguments")
        frame = original(native, path_to_model=str(model_path))
        names = list(frame.columns)
        aliases = ["W" if name == "WAKE" else "R" if name == "REM" else name for name in names]
        if (len(names) != 5 or set(aliases) != set(LABELS) or len(set(aliases)) != 5):
            raise ValueError("YASA classifier class aliases collide or omit a fixed class")
        routed.append(str(model_path))
        copy = frame.copy()
        copy.columns = aliases
        return copy
    try:
        sys.modules.update(facade)
        yasa_module.SleepStaging._load_model = explicit_model
        yasa_module.SleepStaging.predict_proba = explicit_only
        study = types.SimpleNamespace(psg=signal, sample_rate=128,
                                      psg_file_path=recording_id + ".edf")
        groups = [types.SimpleNamespace(channel_names=list(bridge.GROUPS[group]),
                                        channel_indices=[bridge.CHANNELS.index(name)
                                                         for name in bridge.GROUPS[group]])
                  for group in ORDER]
        args = types.SimpleNamespace(out_dir="in-memory", overwrite=True,
                                     no_argmax=True, majority=True)
        namespace["predict_study"](study, None, groups, "yasa", args, None)
    finally:
        yasa_module.SleepStaging.predict_proba = original
        yasa_module.SleepStaging._load_model = original_load
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        if added_path is not None:
            sys.path.remove(added_path)
    if (len(routed) != 2 or set(saved) !=
            ({"+".join(bridge.GROUPS[group]) for group in ORDER} | {"majority"})):
        raise ValueError("Pinned vendor branch did not route and retain both groups")
    native = {}
    for group in ORDER:
        value = _five(saved["+".join(bridge.GROUPS[group])], n_epochs, dtype=np.float32)
        if (value.tobytes() != expected_groups[group].tobytes() or
                not np.array_equal(value.argmax(axis=1),
                                   expected_groups[group].argmax(axis=1))):
            raise ValueError("Pinned group differs from explicit clean-classifier cache")
        native[group] = value
    majority = _five(saved["majority"], n_epochs, dtype=np.float32)
    if majority.tobytes() != expected_majority.tobytes() or not np.array_equal(
            majority.argmax(axis=1), expected_majority.argmax(axis=1)):
        raise ValueError("Pinned float32 accumulation or hard labels differ")
    return {"groups": native, "majority": majority,
            "cached_groups": expected_groups, "cached_majority": expected_majority,
            "hard_label": majority.argmax(axis=1).astype(np.int8),
            "evidence": {"recording_id": recording_id, "n_epochs": n_epochs,
                         "model_path": str(model_path), "checkpoint_sha256": file_sha256(model_path),
                         "vendor_sha256": dict(PINNED), "classifier_calls": len(routed),
                         "probability_contract_sha256": CONTRACT[1],
                         "label_storage_compatibility": "copied_object_dtype_classes_facade",
                         "group_order": list(ORDER),
                         "aggregation": "float32_cast_each_then_copy_add_inplace_divide_2"}}


def compare_record(root: Path, data_root: Path, record: dict, model_path: Path,
                   config: dict, fold: dict, manifest: dict) -> dict:
    """Later caller-owned fit lease: compare one held D record; no training."""
    import joblib
    root, data_root, model_path = root.resolve(), data_root.resolve(), model_path.resolve()
    route_sha = file_sha256(Path(__file__))
    lease = bridge._owned_lease_run_id(root)
    protocol, split, records = development_records(root)
    if (not lease.startswith("sleepyland-yasa-fit-") or
        fold not in split["folds"] or record not in records or
        model_path.parent.parent.parent != (root / "runs").resolve() or
        record["participant_id"] not in fold["validation"] or
        record["participant_id"] in fold["train"] or
        len(fold["train"]) != 48 or len(fold["validation"]) != 12 or
        config.get("protocol_hash") != protocol["protocol_hash"] or
        config.get("split_id") != split["split_id"] or
        experiment._sources(root) != config.get("sources")):
        raise ValueError("Packaged compatibility needs exact held D fold and current source")
    fit = experiment.validate_fit(model_path, config, fold)
    checkpoint = fit["checkpoint_sha256"]
    _pinned(root)
    manifest_path = root / "derived/sleepyland_yasa/manifest.json"
    feature_config = bridge.feature_config(root)
    if (manifest_path.resolve() != manifest_path or
        read_json(manifest_path) != manifest or
        file_sha256(manifest_path) != config.get("feature_manifest_sha256") or
        manifest.get("manifest_id") != config.get("feature_manifest_id") or
        manifest.get("config") != config.get("feature_config") or
        feature_config != config.get("feature_config")):
        raise ValueError("Classifier cache does not match the fitted feature manifest")
    psg = bridge._safe_psg(data_root, record)
    started = time.monotonic()
    model = joblib.load(model_path)
    cached = experiment._cached_probabilities(root, record, manifest, model)
    work = _safe_work(root, data_root)
    work.mkdir(parents=True, exist_ok=True)
    transfer = bridge._run_reader(root, bridge._reader_request(
        root, data_root, record, feature_config, psg, lease), work, started)
    try:
        signal = bridge._checked_signal(transfer["path"], record["n_epochs"])
        result = _execute_vendor(root, signal, record["recording_id"], model_path, cached)
    finally:
        for key in ("path", "request_path", "log_path"):
            transfer[key].unlink(missing_ok=True)
    if (file_sha256(model_path) != checkpoint or
        file_sha256(manifest_path) != config["feature_manifest_sha256"] or
        read_json(manifest_path) != manifest or
        file_sha256(psg) != record["psg_sha256"] or
        file_sha256(Path(__file__)) != route_sha or
        experiment.validate_fit(model_path, config, fold) != fit or
        experiment._sources(root) != config["sources"] or
        bridge.feature_config(root) != feature_config or
        bridge._owned_lease_run_id(root) != lease):
        raise ValueError("Compatibility source, fit, runtime or lease changed during prediction")
    result["evidence"].update(fold_id=fold["fold_id"],
                              fitted_participants=sorted(fold["train"]),
                              excluded_participants=sorted(fold["validation"]),
                              source_psg_sha256=record["psg_sha256"],
                              classifier_route_sha256=route_sha,
                              feature_config_id=feature_config["config_id"])
    return result
