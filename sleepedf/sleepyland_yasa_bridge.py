"""D-only, signal-only bridge from packaged Sleepyland to pinned YASA features.

The native EDF reader runs in isolated Python 3.9; feature extraction runs in
the isolated research Python 3.11. No classifier or Hypnogram is imported here.
"""
from __future__ import annotations

import importlib.metadata
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import sysconfig
import time
import uuid

import numpy as np

from .contracts import content_id, read_json
from .protocol import development_records
from .research import atomic_json, compute_lease, file_sha256

GROUPS = {"fpz_eog": ("EEG Fpz-Cz", "EOG horizontal"),
          "pz_eog": ("EEG Pz-Oz", "EOG horizontal")}
CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal")
RATE_HZ = 128
EPOCH_SECONDS = 30
MAX_RECORD_SECONDS = 900
MAX_TOTAL_SECONDS = 8 * 3600
MAX_BYTES = 10 * 1024**3
LEASE_NAME = "sleepyland-yasa-bridge"
PREFIX = "SLEEPYLAND_SIGNAL_RESULT "


def _allowed_lease_run_id(value: object) -> bool:
    return value == LEASE_NAME or (type(value) is str and
                                   re.fullmatch(r"sleepyland-yasa-fit-[0-9a-f]{12,64}", value) is not None)


def _limit_cpu_threads() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "4"
    if "numba" in sys.modules:
        import numba
        if numba.get_num_threads() > 4:
            numba.set_num_threads(4)


def _owned_lease_run_id(root: Path) -> str:
    import psutil
    lease = read_json(root / "runs/compute.lock")
    if (lease.get("pid") != os.getpid() or
            lease.get("process_start") != psutil.Process().create_time() or
            lease.get("host") != socket.gethostname() or
            not _allowed_lease_run_id(lease.get("run_id"))):
        raise RuntimeError("Sleepyland/YASA bridge requires its current process's live matching lease")
    return lease["run_id"]


def _packages(lock: Path) -> dict[str, str]:
    expected = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, sep, version = line.partition("==")
        if not sep or not version:
            raise ValueError("Sleepyland/YASA runtime lock has an unpinned package")
        name = re.sub(r"[-_.]+", "-", name).lower()
        if name in expected:
            raise ValueError("Sleepyland/YASA runtime lock has duplicate package")
        expected[name] = version
    return expected


def _verify_research_runtime(root: Path) -> None:
    if (sys.version_info[:3] != (3, 11, 9) or
            Path(sys.executable).resolve() != (root / ".venvs/research/Scripts/python.exe").resolve()):
        raise ValueError("Pinned YASA extraction requires isolated research Python 3.11.9")
    site = Path(sysconfig.get_paths()["purelib"]).resolve()
    observed = {}
    for package in importlib.metadata.distributions():
        if Path(package.locate_file("")).resolve() != site:
            continue
        name = re.sub(r"[-_.]+", "-", package.metadata["Name"]).lower()
        if name in observed:
            raise ValueError("Research runtime has duplicate distribution")
        observed[name] = package.version
    if observed != _packages(root / "requirements/research.lock.txt"):
        raise ValueError("Research installed package set differs from exact lock")


def _verify_source(root: Path, package: str) -> str:
    manifest_path = root / "research" / "sources" / (package + ".json")
    manifest = read_json(manifest_path)
    if manifest.get("weights_downloaded") is not False:
        raise ValueError("Bridge requires source-only vendor acquisition")
    for relative, evidence in manifest["files"].items():
        if file_sha256(root / "vendor" / package / relative) != evidence["sha256"]:
            raise ValueError("Pinned bridge vendor source changed: " + package)
    return file_sha256(manifest_path)


def feature_config(root: Path) -> dict:
    """Return a current runtime/source/recipe identity before any real input read."""
    root = root.resolve()
    _verify_research_runtime(root)
    native_lock = root / "requirements/usleepyland.lock.txt"
    install = read_json(root / "research/runtimes/usleepyland/install.json")
    if install.get("freeze_sha256") != file_sha256(native_lock):
        raise ValueError("uSLEEPYLAND installed lock provenance differs")
    config_files = ("vendor/sleepyland/usleepyland/model/yasa/hyperparameters/hparams.yaml",
                    "vendor/sleepyland/usleepyland/model/yasa/hyperparameters/dataset_configurations/abc.yaml")
    config = {"schema_version": "1.0", "artifact_type": "sleepyland_yasa_feature_config",
              "groups": {name: list(channels) for name, channels in GROUPS.items()},
              "signal_recipe": {"source_hz": 100, "model_hz": RATE_HZ, "epoch_seconds": EPOCH_SECONDS,
                                "trim": "native trim_psg_trailing", "clip": "native global IQR x20",
                                "resample": "native polyphase", "signal_dtype": "float32",
                                "signal_unit": "V", "scaler": None,
                                "yasa_feature_hz": 100, "metadata_inputs": []},
              "source_manifest_sha256": {package: _verify_source(root, package)
                                         for package in ("sleepyland", "usleepyland", "yasa")},
              "native_config_sha256": {name: file_sha256(root / name) for name in config_files},
              "runtime_lock_sha256": {"research": file_sha256(root / "requirements/research.lock.txt"),
                                      "usleepyland": file_sha256(native_lock)},
              "runtime_install_sha256": file_sha256(root / "research/runtimes/usleepyland/install.json"),
              "python_executable_sha256": {
                  "research": file_sha256(root / ".venvs/research/Scripts/python.exe"),
                  "usleepyland": file_sha256(root / ".venvs/usleepyland/Scripts/python.exe")},
              "implementation_sha256": {
                  "sleepedf/sleepyland_yasa_bridge.py": file_sha256(Path(__file__)),
                  "tools/sleepyland_yasa_reader.py": file_sha256(root / "tools/sleepyland_yasa_reader.py")}}
    config["config_id"] = content_id(config)
    return config


def _development(root: Path):
    protocol, split, records = development_records(root)
    allowed = set(split["participants"]["development"])
    if (not records or {record["participant_id"] for record in records} != allowed or
            len({record["recording_id"] for record in records}) != len(records)):
        raise ValueError("Sleepyland/YASA bridge requires unique original development recordings")
    return protocol, split, sorted(records, key=lambda record: record["recording_id"])


def _checked_record(record: dict) -> dict:
    channels = record["channels"]
    if (type(record.get("n_epochs")) is not int or record["n_epochs"] <= 0 or
            type(channels) is not list or
            any(sum(item.get("label") == name for item in channels) != 1 for name in CHANNELS)):
        raise ValueError("Sleepyland/YASA bridge requires three unique named original channels")
    for name in CHANNELS:
        item = next(item for item in channels if item["label"] == name)
        if item.get("unit") != "uV" or item.get("sample_rate_hz") != 100:
            raise ValueError("Sleepyland/YASA bridge requires verified 100-Hz uV EEG/EOG")
    return {"recording_id": record["recording_id"], "participant_id": record["participant_id"],
            "source_psg_sha256": record["psg_sha256"], "n_epochs": record["n_epochs"]}


def _safe_psg(data_root: Path, record: dict) -> Path:
    path = (data_root / record["psg"]).resolve()
    if not path.is_relative_to(data_root.resolve()) or file_sha256(path) != record["psg_sha256"]:
        raise ValueError("Sleepyland/YASA PSG bytes differ from frozen D readiness")
    return path


def _resource_check(started: float, child_pid: int | None = None) -> None:
    import psutil
    rss = psutil.Process().memory_info().rss
    if child_pid is not None:
        try:
            rss += psutil.Process(child_pid).memory_info().rss
        except psutil.NoSuchProcess:
            pass
    if (rss > MAX_BYTES or psutil.virtual_memory().available < 4 * 1024**3 or
            time.monotonic() - started > MAX_TOTAL_SECONDS):
        raise RuntimeError("Sleepyland/YASA bridge exceeded combined memory or time bound")


def _supervise_reader(child, started: float, begin: float) -> None:
    """Track the venv launcher and actual interpreter, then reap both."""
    import psutil
    owned = {}
    try:
        while True:
            try:
                launcher = psutil.Process(child.pid)
                discovered = [launcher] + launcher.children(recursive=True)
                for process in discovered:
                    owned[process.pid] = process.create_time()
            except psutil.NoSuchProcess:
                pass
            live = []
            for pid, birth in owned.items():
                try:
                    process = psutil.Process(pid)
                    if process.create_time() == birth and process.is_running():
                        live.append(process)
                except psutil.NoSuchProcess:
                    pass
            combined = psutil.Process().memory_info().rss + sum(
                process.memory_info().rss for process in live)
            if (combined > MAX_BYTES or psutil.virtual_memory().available < 4 * 1024**3 or
                    time.monotonic() - started > MAX_TOTAL_SECONDS or
                    time.monotonic() - begin > MAX_RECORD_SECONDS):
                raise RuntimeError("Sleepyland native reader exceeded combined memory or time bound")
            if child.poll() is not None and not live:
                break
            time.sleep(0.25)
        child.wait()
    except BaseException:
        survivors = []
        for pid, birth in owned.items():
            try:
                process = psutil.Process(pid)
                if process.create_time() == birth and process.is_running():
                    survivors.append(process)
            except psutil.NoSuchProcess:
                pass
        for process in reversed(survivors):
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, remaining = psutil.wait_procs(survivors, timeout=3)
        for process in remaining:
            process.kill()
        psutil.wait_procs(remaining, timeout=3)
        child.wait()
        raise


def _run_reader(root: Path, request: dict, work: Path, started: float) -> dict:
    stem = request["recording_id"] + "-" + uuid.uuid4().hex
    request_path, output_path, log_path = (work / (stem + suffix)
                                         for suffix in (".request.json", ".signal.npz", ".reader.log"))
    if any(path.exists() for path in (request_path, output_path, log_path)):
        raise ValueError("Sleepyland bridge temporary artifact already exists")
    request = dict(request, output_path=str(output_path.resolve()))
    atomic_json(request_path, request, immutable=True)
    command = [str(root / ".venvs/usleepyland/Scripts/python.exe"), "-I",
               str(root / "tools/sleepyland_yasa_reader.py"), "--request", str(request_path.resolve())]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       NUMEXPR_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="-1")
    try:
        begin = time.monotonic()
        with log_path.open("xb") as log:
            child = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT)
            _supervise_reader(child, started, begin)
        output = log_path.read_text(encoding="utf-8", errors="replace")
        if child.returncode != 0:
            raise RuntimeError("Native Sleepyland reader failed; local log: " + str(log_path))
        lines = [line[len(PREFIX):] for line in output.splitlines() if line.startswith(PREFIX)]
        if len(lines) != 1:
            raise ValueError("Native Sleepyland reader returned no unique result")
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("Duplicate native reader result key")
                value[key] = item
            return value
        result = json.loads(lines[0], object_pairs_hook=unique,
                            parse_constant=lambda _: (_ for _ in ()).throw(
                                ValueError("Non-finite native reader result")))
        json.dumps(result, allow_nan=False)
        if (type(result) is not dict or set(result) != {
                "schema_version", "artifact_type", "recording_id", "n_epochs",
                "signal_sha256", "channel_names", "runtime", "unit",
                "sample_rate_hz", "shape", "seconds"} or
                result.get("schema_version") != "1.0" or
                result.get("artifact_type") != "sleepyland_yasa_signal_result" or
                result.get("recording_id") != request["recording_id"] or
                result.get("n_epochs") != request["n_epochs"] or
                result.get("channel_names") != list(CHANNELS) or
                result.get("unit") != "V" or result.get("sample_rate_hz") != RATE_HZ or
                result.get("shape") != [request["n_epochs"] * EPOCH_SECONDS * RATE_HZ, 3] or
                result.get("runtime") != {key: request[key] for key in (
                    "runtime_lock_sha256", "reader_sha256", "usleepyland_source_manifest_sha256",
                    "sleepyland_source_manifest_sha256")} or
                result.get("signal_sha256") != file_sha256(output_path) or
                type(result.get("seconds")) not in (int, float) or
                not math.isfinite(result["seconds"]) or not 0 < result["seconds"] <= MAX_RECORD_SECONDS):
            raise ValueError("Native Sleepyland signal result differs from request/artifact")
        return {"path": output_path, "result": result, "request_path": request_path, "log_path": log_path}
    except BaseException:
        request_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise


def _checked_signal(path: Path, n_epochs: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"signal", "epoch_index", "channel_names"}:
            raise ValueError("Sleepyland signal transfer has unexpected fields")
        signal = archive["signal"]
        if (signal.shape != (n_epochs * EPOCH_SECONDS * RATE_HZ, 3) or
                signal.dtype != np.float32 or not np.isfinite(signal).all() or
                not np.array_equal(archive["epoch_index"], np.arange(n_epochs, dtype=np.int32)) or
                archive["channel_names"].tolist() != list(CHANNELS)):
            raise ValueError("Sleepyland signal transfer changed volts, channels or original grid")
    return signal


def _native_group(root: Path, signal: np.ndarray, group: str, n_epochs: int):
    if group not in GROUPS:
        raise ValueError("Unknown Sleepyland/YASA channel group")
    _verify_source(root, "yasa")
    import mne
    source = root / "vendor/yasa/src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import yasa
    if not Path(yasa.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError("Unexpected YASA implementation resolved")
    names = GROUPS[group]
    indices = [CHANNELS.index(name) for name in names]
    volts = signal[:, indices].T.astype(np.float64)
    info = mne.create_info(list(names), RATE_HZ, ["eeg", "eog"])
    raw = mne.io.RawArray(volts, info, verbose="ERROR")
    if not np.array_equal(raw.get_data(), volts):
        raise ValueError("MNE RawArray changed bridged physical volts")
    if not np.allclose(raw.get_data(units={"eeg": "uV", "eog": "uV"}),
                       volts * 1e6, rtol=0, atol=1e-9):
        raise ValueError("MNE channel-unit conversion differs from one volts-to-uV step")
    native = yasa.SleepStaging(raw, eeg_name=names[0], eog_name=names[1])
    if native.sf != 100:
        raise ValueError("Pinned YASA did not establish its 100-Hz feature grid")
    frame = native.get_features()
    if (len(frame) != n_epochs or frame.columns.has_duplicates or
            not all(np.issubdtype(dtype, np.number) for dtype in frame.dtypes)):
        raise ValueError("Pinned YASA features lack original complete epochs or named numeric columns")
    values = frame.to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1 or np.isinf(values).any():
        raise ValueError("Pinned YASA features are empty or infinite")
    return native, frame


def _feature_frame(root: Path, signal: np.ndarray, group: str, n_epochs: int):
    return _native_group(root, signal, group, n_epochs)[1]


def _reader_request(root: Path, data_root: Path, record: dict,
                    config: dict, psg: Path, lease_run_id: str) -> dict:
    if not _allowed_lease_run_id(lease_run_id):
        raise ValueError("Sleepyland/YASA reader request has an unregistered lease run ID")
    return {"schema_version": "1.0", "artifact_type": "sleepyland_yasa_signal_request",
            "lease_run_id": lease_run_id,
            "recording_id": record["recording_id"], "n_epochs": record["n_epochs"],
            "source_psg_sha256": record["psg_sha256"], "source_channels": record["channels"],
            "data_root": str(data_root), "psg_path": str(psg),
            "channel_names": list(CHANNELS),
            "reader_sha256": config["implementation_sha256"]["tools/sleepyland_yasa_reader.py"],
            "runtime_lock_sha256": config["runtime_lock_sha256"]["usleepyland"],
            "usleepyland_source_manifest_sha256": config["source_manifest_sha256"]["usleepyland"],
            "sleepyland_source_manifest_sha256": config["source_manifest_sha256"]["sleepyland"]}


def _feature_artifact(path: Path, frame, identity: dict) -> dict:
    if path.exists() or path.with_suffix(".json").exists():
        raise ValueError("Sleepyland/YASA feature artifact already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + "-" + uuid.uuid4().hex + ".tmp.npz")
    columns = list(frame.columns)
    dtypes = [str(dtype) for dtype in frame.dtypes]
    values = frame.to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("Sleepyland/YASA feature artifact contains infinity")
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, features=values,
                     epoch_index=np.arange(len(frame), dtype=np.int32))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    meta = {"schema_version": "1.0", "artifact_type": "sleepyland_yasa_feature_cache",
            **identity, "n_rows": len(frame), "n_columns": len(columns),
            "columns": columns, "column_dtypes": dtypes,
            "native_nan_values": int(np.isnan(values).sum()),
            "nan_policy": "native LightGBM missing-value support",
            "payload_sha256": file_sha256(path)}
    atomic_json(path.with_suffix(".json"), meta, immutable=True)
    return {"path": str(path.resolve()), "sidecar_path": str(path.with_suffix(".json").resolve()),
            "payload_sha256": meta["payload_sha256"],
            "sidecar_sha256": file_sha256(path.with_suffix(".json"))}


def _quarantine_orphan(root: Path, path: Path) -> dict | None:
    """Preserve a lone cache file after interrupted two-file publication."""
    root = root.resolve()
    cache = (root / "derived/sleepyland_yasa/features").resolve()
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if (path.parent.resolve() != cache or path.suffix != ".npz" or
            not re.fullmatch(r"[A-Z]{2}[0-9]{4}-(fpz_eog|pz_eog)\.npz", path.name)):
        raise ValueError("Sleepyland/YASA orphan target escapes the feature cache")
    if path.exists() == sidecar.exists():
        return None
    orphan = path if path.exists() else sidecar
    resolved = orphan.resolve()
    if resolved.parent != cache:
        raise ValueError("Sleepyland/YASA orphan is a symlink or escaped cache path")
    digest = file_sha256(orphan)
    destination = (root / "derived/sleepyland_yasa/orphans" /
                   (path.stem + "-" + uuid.uuid4().hex)).resolve()
    if (not destination.is_relative_to(root / "derived/sleepyland_yasa/orphans") or
            destination.exists()):
        raise ValueError("Sleepyland/YASA orphan quarantine path differs")
    destination.mkdir(parents=True, exist_ok=False)
    moved = destination / orphan.name
    os.replace(orphan, moved)
    if file_sha256(moved) != digest:
        raise ValueError("Sleepyland/YASA orphan changed during quarantine")
    evidence = {"schema_version": "1.0", "artifact_type": "sleepyland_yasa_orphan_quarantine",
                "original_path": str(resolved), "preserved_path": str(moved.resolve()),
                "sha256": digest, "reason": "interrupted_payload_sidecar_publication"}
    atomic_json(destination / "manifest.json", evidence, immutable=True)
    return evidence


def _artifact_header(root: Path, record: dict, group: str, config: dict,
                     artifact: dict) -> dict:
    cache = (root / "derived/sleepyland_yasa/features").resolve()
    path = Path(artifact["path"])
    sidecar = Path(artifact["sidecar_path"])
    if (not path.is_absolute() or path != path.resolve() or path.parent != cache or
            path.name != record["recording_id"] + "-" + group + ".npz" or
            sidecar != path.with_suffix(".json") or
            file_sha256(path) != artifact["payload_sha256"] or
            file_sha256(sidecar) != artifact["sidecar_sha256"]):
        raise ValueError("Sleepyland/YASA feature artifact path or bytes changed")
    meta = read_json(sidecar)
    expected = _checked_record(record)
    for key, value in dict(expected, group=group, channels=list(GROUPS[group]),
                           config_id=config["config_id"], signal_unit="V",
                           signal_rate_hz=RATE_HZ, feature_rate_hz=100).items():
        if meta.get(key) != value:
            raise ValueError("Sleepyland/YASA feature sidecar differs from D source")
    if (meta.get("artifact_type") != "sleepyland_yasa_feature_cache" or
            meta.get("n_rows") != record["n_epochs"] or
            type(meta.get("columns")) is not list or not meta["columns"] or
            len(meta["columns"]) != meta.get("n_columns") or
            len(set(meta["columns"])) != len(meta["columns"]) or
            len(meta.get("column_dtypes", [])) != len(meta["columns"]) or
            type(meta.get("native_nan_values")) is not int or
            not 0 <= meta["native_nan_values"] <= record["n_epochs"] * meta["n_columns"] or
            meta.get("nan_policy") != "native LightGBM missing-value support" or
            meta.get("payload_sha256") != artifact["payload_sha256"]):
        raise ValueError("Sleepyland/YASA feature columns or coverage changed")
    try:
        if any(not np.issubdtype(np.dtype(dtype), np.number)
               for dtype in meta["column_dtypes"]):
            raise ValueError("Sleepyland/YASA feature cache has nonnumeric column dtype")
    except (TypeError, ValueError) as exc:
        raise ValueError("Sleepyland/YASA feature cache has invalid column dtype") from exc
    return meta


def _feature_values(path: Path, meta: dict, n_epochs: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"features", "epoch_index"}:
            raise ValueError("Sleepyland/YASA feature payload has unexpected fields")
        values = archive["features"]
        if (values.shape != (n_epochs, meta["n_columns"]) or
                values.dtype != np.float64 or np.isinf(values).any() or
                int(np.isnan(values).sum()) != meta["native_nan_values"] or
                not np.array_equal(archive["epoch_index"],
                                   np.arange(n_epochs, dtype=np.int32))):
            raise ValueError("Sleepyland/YASA feature payload changed original grid")
    return values


def _manifest_header(manifest: dict, protocol: dict, split: dict,
                     records: list[dict], config: dict) -> None:
    expected = {record["recording_id"]: _checked_record(record) for record in records}
    if (manifest.get("schema_version") != "1.0" or
            manifest.get("artifact_type") != "sleepyland_yasa_development_features" or
            manifest.get("manifest_id") != content_id({key: value for key, value in manifest.items()
                                                       if key != "manifest_id"}) or
            manifest.get("protocol_hash") != protocol["protocol_hash"] or
            manifest.get("split_id") != split["split_id"] or
            manifest.get("config") != config or
            set(manifest.get("records", {})) != set(expected)):
        raise ValueError("Sleepyland/YASA feature manifest differs from original development protocol")
    for record_id, identity in expected.items():
        row = manifest["records"][record_id]
        if (any(row.get(key) != value for key, value in identity.items()) or
                set(row.get("groups", {})) != set(GROUPS)):
            raise ValueError("Sleepyland/YASA feature manifest has missing or altered channel groups")


def verify_manifest(root: Path, manifest: dict, protocol: dict,
                    records: list[dict]) -> None:
    """Revalidate D membership, full two-group coverage, and all cached bytes."""
    root = root.resolve()
    current_protocol, split, canonical = _development(root)
    if protocol != current_protocol or records != canonical:
        raise ValueError("Sleepyland/YASA manifest inputs are not current canonical development")
    config = feature_config(root)
    _manifest_header(manifest, protocol, split, records, config)
    for record in records:
        for group in GROUPS:
            artifact = manifest["records"][record["recording_id"]]["groups"][group]
            meta = _artifact_header(root, record, group, config, artifact)
            _feature_values(Path(artifact["path"]), meta, record["n_epochs"])


def load_features(root: Path, record: dict, manifest: dict, group: str):
    """Return one untouched native feature frame for an explicit named group."""
    if group not in GROUPS:
        raise ValueError("Unknown Sleepyland/YASA channel group")
    root = root.resolve()
    protocol, split, records = _development(root)
    matching = [item for item in records if item["recording_id"] == record.get("recording_id")]
    if len(matching) != 1 or record != matching[0]:
        raise ValueError("Sleepyland/YASA load requires an original D record")
    config = feature_config(root)
    _manifest_header(manifest, protocol, split, records, config)
    artifact = manifest["records"][record["recording_id"]]["groups"][group]
    meta = _artifact_header(root, record, group, config, artifact)
    values = _feature_values(Path(artifact["path"]), meta, record["n_epochs"])
    import pandas as pd
    frame = pd.DataFrame(values, columns=meta["columns"])
    for name, dtype in zip(meta["columns"], meta["column_dtypes"]):
        frame[name] = frame[name].astype(dtype)
    if not np.array_equal(frame.to_numpy(dtype=np.float64), values, equal_nan=True):
        raise ValueError("Sleepyland/YASA feature dtype restoration changed values")
    return frame


def prepare_features(root: Path, data_root: Path) -> dict:
    """Prepare both named groups for all original D records under one lease."""
    root, data_root = root.resolve(), data_root.resolve()
    protocol, split, records = _development(root)
    config = feature_config(root)
    base = (root / "derived/sleepyland_yasa").resolve()
    cache, work = base / "features", base / "tmp"
    cache.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = base / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        verify_manifest(root, manifest, protocol, records)
        return manifest
    from threadpoolctl import threadpool_limits
    started = time.monotonic()
    rows = {}
    with compute_lease(root, LEASE_NAME), threadpool_limits(limits=4):
        _limit_cpu_threads()
        for record in records:
            record_started = time.monotonic()
            _resource_check(started)
            identity = _checked_record(record)
            group_artifacts = {}
            for group in GROUPS:
                path = cache / (record["recording_id"] + "-" + group + ".npz")
                sidecar = path.with_suffix(".json")
                _quarantine_orphan(root, path)
                if path.exists() and sidecar.exists():
                    artifact = {"path": str(path.resolve()), "sidecar_path": str(sidecar.resolve()),
                                "payload_sha256": file_sha256(path),
                                "sidecar_sha256": file_sha256(sidecar)}
                    meta = _artifact_header(root, record, group, config, artifact)
                    _feature_values(path, meta, record["n_epochs"])
                    group_artifacts[group] = artifact
            if len(group_artifacts) != len(GROUPS):
                psg = _safe_psg(data_root, record)
                request = _reader_request(root, data_root, record, config, psg,
                                          _owned_lease_run_id(root))
                transfer = _run_reader(root, request, work, started)
                try:
                    signal = _checked_signal(transfer["path"], record["n_epochs"])
                    for group in GROUPS:
                        if group in group_artifacts:
                            continue
                        _resource_check(started)
                        frame = _feature_frame(root, signal, group, record["n_epochs"])
                        artifact_identity = dict(identity, group=group, channels=list(GROUPS[group]),
                                                 config_id=config["config_id"], signal_unit="V",
                                                 signal_rate_hz=RATE_HZ, feature_rate_hz=100,
                                                 signal_transfer_sha256=transfer["result"]["signal_sha256"])
                        group_artifacts[group] = _feature_artifact(
                            cache / (record["recording_id"] + "-" + group + ".npz"),
                            frame, artifact_identity)
                        if time.monotonic() - record_started > MAX_RECORD_SECONDS:
                            raise RuntimeError("Sleepyland/YASA feature extraction exceeded per-record deadline")
                finally:
                    transfer["path"].unlink(missing_ok=True)
                    transfer["request_path"].unlink(missing_ok=True)
                    transfer["log_path"].unlink(missing_ok=True)
            rows[record["recording_id"]] = dict(identity, groups=group_artifacts)
    manifest = {"schema_version": "1.0", "artifact_type": "sleepyland_yasa_development_features",
                "protocol_hash": protocol["protocol_hash"], "split_id": split["split_id"],
                "config": config, "records": rows}
    manifest["manifest_id"] = content_id(manifest)
    atomic_json(manifest_path, manifest, immutable=True)
    verify_manifest(root, manifest, protocol, records)
    return manifest


def prepare_record(root: Path, data_root: Path, record: dict,
                   work_dir: Path | None = None) -> dict:
    """Acquire the bridge lease and return native objects for one D-only parity.

    This performs no classifier loading. A caller invoking ``native.predict``
    later must acquire its own compute lease and validate its clean model path.
    """
    root = root.resolve()
    with compute_lease(root, LEASE_NAME):
        return prepare_record_locked(root, data_root, record, work_dir)


def prepare_record_locked(root: Path, data_root: Path, record: dict,
                          work_dir: Path | None = None) -> dict:
    """Use the caller's live bridge or fit lease for native objects and frames."""
    root, data_root = root.resolve(), data_root.resolve()
    lease_run_id = _owned_lease_run_id(root)
    _, _, records = _development(root)
    matching = [item for item in records if item["recording_id"] == record.get("recording_id")]
    if len(matching) != 1 or matching[0] != record:
        raise ValueError("Sleepyland/YASA parity requires an exact original D recording")
    _checked_record(record)
    config = feature_config(root)
    base = (root / "derived/sleepyland_yasa").resolve()
    work = (base / "tmp" if work_dir is None else work_dir.resolve())
    if not work.is_relative_to(base) or work == base:
        raise ValueError("Sleepyland/YASA parity work directory escapes local derived artifacts")
    work.mkdir(parents=True, exist_ok=True)
    from threadpoolctl import threadpool_limits
    started = time.monotonic()
    with threadpool_limits(limits=4):
        _limit_cpu_threads()
        record_started = time.monotonic()
        psg = _safe_psg(data_root, record)
        transfer = _run_reader(root, _reader_request(root, data_root, record, config, psg,
                                                    lease_run_id),
                               work, started)
        try:
            signal = _checked_signal(transfer["path"], record["n_epochs"])
            groups = {}
            feature_evidence = {}
            for group in GROUPS:
                _resource_check(started)
                native, frame = _native_group(root, signal, group, record["n_epochs"])
                groups[group] = {"native": native, "features": frame}
                values = np.ascontiguousarray(frame.to_numpy(dtype=np.float64))
                feature_evidence[group] = {
                    "columns": list(frame.columns),
                    "column_dtypes": [str(dtype) for dtype in frame.dtypes],
                    "values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                    "native_nan_values": int(np.isnan(values).sum())}
                if time.monotonic() - record_started > MAX_RECORD_SECONDS:
                    raise RuntimeError("Sleepyland/YASA parity exceeded per-record deadline")
            _resource_check(started)
            evidence = {"config_id": config["config_id"], **_checked_record(record),
                        "signal_transfer_sha256": transfer["result"]["signal_sha256"],
                        "unit": "V", "sample_rate_hz": RATE_HZ,
                        "feature_rate_hz": 100, "groups": feature_evidence}
            return {"groups": groups, "evidence": evidence}
        finally:
            transfer["path"].unlink(missing_ok=True)
            transfer["request_path"].unlink(missing_ok=True)
            transfer["log_path"].unlink(missing_ok=True)
