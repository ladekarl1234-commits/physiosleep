"""Bounded synthetic readiness for the separate native AttnSleep GPU route."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

from .contracts import content_id, json_text, read_json
from .research import atomic_json, compute_lease, file_sha256

BASE = "runs/attn37-cuda-verify"
RUN_ID = "attn37-cuda-verify"
PREFIX = "ATTN37_CUDA_PROBE "
STAGES = ("smoke", "continuous", "resume", "compare", "profile")
MAX_SECONDS = 900
MAX_STAGE_SECONDS = 180
MAX_BYTES = 10 * 1024**3
WHEEL_SHA = "374a7f1265b6a24a3ef07b8719fd850213ab911b189ade787305c482538598b7"


def snapshot(root: Path) -> dict:
    evidence = root / "research/runtimes/attn37-cuda"
    provenance = read_json(evidence / "provenance.json")
    verification = provenance.get("verification")
    if (not verification or not provenance.get("install") or
            provenance["install"]["wheel_sha256"] != WHEEL_SHA or
            verification["freeze_sha256"] != file_sha256(evidence / "freeze.txt") or
            verification["installed_torch_sha256"] != file_sha256(evidence / "installed-torch.json")):
        raise ValueError("Separate CUDA runtime lacks current wheel-bound installation evidence")
    inventory = read_json(evidence / "installed-torch.json")
    if inventory["wheel_sha256"] != WHEEL_SHA:
        raise ValueError("Installed CUDA payload belongs to another wheel")
    site = (root / ".venvs/attn37-cuda/Lib/site-packages").resolve()
    for files in (provenance["copied_files"], inventory["files"]):
        for name, item in files.items():
            path = (site / name).resolve()
            if (not path.is_relative_to(site) or not path.is_file() or
                    path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]):
                raise ValueError("Installed CUDA runtime bytes changed: " + name)
    sources = ["sleepedf/attn37_cuda_verify.py", "tools/verify_attn37_cuda.py",
               "tools/attn37_worker.py", "tools/verify_attn37_native.py",
               "tools/bootstrap_attn37_cuda.py", "tools/bootstrap_attn37.py",
               "research/runtimes/attn37-cuda/provenance.json",
               "research/runtimes/attn37-cuda/freeze.txt",
               "research/runtimes/attn37-cuda/installed-torch.json",
               "research/sources/attnsleep.json", ".venvs/attn37-cuda/python.exe",
               ".venvs/attn37-cuda/python37._pth"]
    source_manifest = read_json(root / "research/sources/attnsleep.json")
    for name, item in source_manifest["files"].items():
        relative = "vendor/attnsleep/" + name
        if file_sha256(root / relative) != item["sha256"]:
            raise ValueError("Pinned native AttnSleep source changed")
        sources.append(relative)
    versions = {name.lower().replace("_", "-"): version for name, version in provenance["source"]["packages"].items()}
    versions["torch"] = "1.4.0"
    hashes = {name: file_sha256(root / name) for name in sources}
    if (hashes[".venvs/attn37-cuda/python.exe"] != provenance["extraction"]["python_exe_sha256"] or
            hashes[".venvs/attn37-cuda/python37._pth"] != provenance["extraction"]["configured_pth_sha256"] or
            hashes["tools/bootstrap_attn37_cuda.py"] != provenance["bootstrap_sha256"] or
            hashes["tools/bootstrap_attn37.py"] != provenance["helper_sha256"]):
        raise ValueError("CUDA extraction/bootstrap identity changed")
    return {"source_hashes": hashes, "versions": versions, "wheel_sha256": WHEEL_SHA,
            "stages": list(STAGES), "batch_size": 128, "precision": "float32",
            "threads": 4, "cudnn_deterministic": True, "cudnn_benchmark": False,
            "max_seconds": MAX_SECONDS, "max_stage_seconds": MAX_STAGE_SECONDS,
            "max_host_bytes": MAX_BYTES}


def supervise(child: subprocess.Popen, total_start: float) -> int:
    import psutil
    begin, observed, peak = time.monotonic(), {}, 0
    try:
        while True:
            try:
                process = psutil.Process(child.pid)
                for item in [process] + process.children(recursive=True):
                    observed[item.pid] = item.create_time()
            except psutil.NoSuchProcess:
                pass
            live = []
            for pid, birth in observed.items():
                try:
                    item = psutil.Process(pid)
                    if item.create_time() == birth and item.is_running():
                        live.append(item)
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, psutil.Process().memory_info().rss + sum(p.memory_info().rss for p in live))
            if (peak > MAX_BYTES or psutil.virtual_memory().available < 4 * 1024**3 or
                    time.monotonic() - total_start > MAX_SECONDS or time.monotonic() - begin > MAX_STAGE_SECONDS):
                raise RuntimeError("CUDA probe exceeded its shared host memory or time bounds")
            if child.poll() is not None and not live:
                return peak
            time.sleep(.25)
    finally:
        survivors = []
        for pid, birth in observed.items():
            try:
                item = psutil.Process(pid)
                if item.create_time() == birth and item.is_running():
                    survivors.append(item)
                    item.terminate()
            except psutil.NoSuchProcess:
                pass
        _, remaining = psutil.wait_procs(survivors, timeout=3)
        for item in remaining:
            try:
                item.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(remaining, timeout=3)
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)


def parse(path: Path) -> dict:
    import json
    lines = [line[len(PREFIX):] for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise ValueError("No unique CUDA probe stage result")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate CUDA result key")
            result[key] = value
        return result
    return json.loads(lines[0], object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite GPU result")))


def validate_results(results: dict, probe_id: str) -> None:
    if type(results) is not dict or set(results) != set(STAGES):
        raise ValueError("Every registered CUDA readiness stage is required")
    def number(value, low=0., high=float("inf")):
        return type(value) in (float, int) and math.isfinite(value) and low <= value <= high
    def integer(value, low, high):
        return type(value) is int and low <= value <= high
    common = {"mode", "probe_id", "hardware", "device_after", "wall_seconds"}
    extras = {"smoke": {"kernel_exact"}, "resume": {"traces"}, "compare": {"fresh_process_resume_exact"},
              "continuous": {"loss_error", "gradient_error", "invalid_gradient", "traces",
                             "cpu_gpu_evaluation_max_abs_error", "cpu_gpu_evaluation_scaled_error",
                             "cpu_gpu_evaluation_reference_max_abs", "cpu_gpu_evaluation_atol", "cpu_gpu_evaluation_rtol"},
              "profile": {"batch_size", "warmup_seconds", "update_seconds", "optimizer_steps",
                          "peak_allocated_bytes", "peak_reserved_bytes"}}
    keys = ("uuid", "name", "driver", "total_mib", "capability", "cuda", "cudnn", "torch",
            "cudnn_deterministic", "cudnn_benchmark", "precision")
    identity = None
    for mode, value in results.items():
        if (type(value) is not dict or set(value) != common | extras[mode] or
                value.get("mode") != mode or value.get("probe_id") != probe_id or
                not number(value["wall_seconds"], 0., MAX_STAGE_SECONDS)):
            raise ValueError("CUDA stage identity, hardware or bounded execution differs")
        for where in ("hardware", "device_after"):
            hardware = value[where]
            if (type(hardware) is not dict or set(hardware) != set(keys) | {"free_mib"} or
                    any(type(hardware[key]) is not str or not hardware[key] for key in ("uuid", "name", "driver")) or
                    not integer(hardware["total_mib"], 1, 1024 * 1024) or
                    not integer(hardware["free_mib"], 512, hardware["total_mib"]) or
                    type(hardware["capability"]) is not list or len(hardware["capability"]) != 2 or
                    any(not integer(item, 0, 100) for item in hardware["capability"]) or
                    not integer(hardware["cudnn"], 1, 1000000)):
                raise ValueError("GPU hardware or device memory evidence is invalid")
            this = {key: hardware[key] for key in keys}
            if identity is None:
                identity = this
            elif this != identity:
                raise ValueError("GPU hardware/backend changed between probe stages")
    if (identity["torch"] != "1.4.0" or identity["cuda"] != "10.1" or identity["precision"] != "float32" or
            identity["cudnn_deterministic"] is not True or identity["cudnn_benchmark"] is not False or
            results["smoke"].get("kernel_exact") is not True or
            results["compare"].get("fresh_process_resume_exact") is not True):
        raise ValueError("CUDA readiness runtime, kernel or exact resume check failed")
    numeric, profile = results["continuous"], results["profile"]
    if (not number(numeric["loss_error"], 0., 1e-7) or not number(numeric["gradient_error"], 0., 1e-7) or
            not number(numeric["invalid_gradient"], 0., 0.) or
            not number(numeric["cpu_gpu_evaluation_max_abs_error"]) or
            not number(numeric["cpu_gpu_evaluation_scaled_error"], 0., 1.) or
            not number(numeric["cpu_gpu_evaluation_reference_max_abs"]) or
            numeric["cpu_gpu_evaluation_atol"] != 1e-5 or numeric["cpu_gpu_evaluation_rtol"] != 1e-4 or
            numeric["cpu_gpu_evaluation_max_abs_error"] > 1e-5 + 1e-4 * numeric["cpu_gpu_evaluation_reference_max_abs"] or
            type(numeric["traces"]) is not list or len(numeric["traces"]) != 2 or
            numeric["traces"] != results["resume"]["traces"] or
            not integer(profile["batch_size"], 128, 128) or not integer(profile["optimizer_steps"], 3, 3) or
            type(profile["update_seconds"]) is not list or len(profile["update_seconds"]) != 2 or
            any(not number(t, 1e-12, profile["wall_seconds"]) for t in [profile["warmup_seconds"]] + profile["update_seconds"]) or
            sum(profile["update_seconds"]) + profile["warmup_seconds"] > profile["wall_seconds"] or
            not integer(profile["peak_reserved_bytes"], 1, identity["total_mib"] * 1024**2) or
            not integer(profile["peak_allocated_bytes"], 1, profile["peak_reserved_bytes"])):
        raise ValueError("CUDA numerical or full native batch qualification is incomplete")
    for index, trace in enumerate(numeric["traces"]):
        if (type(trace) is not dict or set(trace) != {"lr", "loss", "order"} or
                trace["lr"] != [.001, .0001][index] or not number(trace["loss"]) or
                type(trace["order"]) is not list or len(trace["order"]) != 3 or
                any(type(value) is not int for value in trace["order"]) or sorted(trace["order"]) != [0, 1, 2]):
            raise ValueError("GPU continuation loss, learning rate or next batch order is invalid")


def run(root: Path) -> dict:
    root = root.resolve()
    with compute_lease(root, RUN_ID):
        return _run_locked(root)


def _run_locked(root: Path) -> dict:
    import psutil
    start = time.monotonic()
    before = snapshot(root)
    probe_id = content_id(before)
    base = root / BASE
    attempt = base / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    current = base / "current.json"
    atomic_json(current, {"status": "PENDING", "attempt": str(attempt), "probe_id": probe_id})
    owner = psutil.Process()
    request = {"schema_version": "1.0", "artifact_type": "attn37_cuda_synthetic_probe_request",
               "attempt_directory": str(attempt), "probe_id": probe_id, "lease_pid": owner.pid,
               "lease_birth": owner.create_time(), **before}
    request_path = attempt / "request.json"
    atomic_json(request_path, request, immutable=True)
    results, logs = {}, {}
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1")
    try:
        for mode in STAGES:
            log = attempt / (mode + ".log")
            command = [str(root / ".venvs/attn37-cuda/python.exe"), "-I", "-B",
                       str(root / "tools/verify_attn37_cuda.py"), mode, "--request", str(request_path)]
            with log.open("xb") as handle:
                child = subprocess.Popen(command, cwd=root, env=environment, stdout=handle, stderr=subprocess.STDOUT)
                peak = supervise(child, start)
            if child.returncode != 0:
                raise RuntimeError("CUDA stage failed: " + str(log))
            results[mode] = parse(log)
            logs[mode] = {"path": str(log), "sha256": file_sha256(log), "peak_observed_host_bytes": peak}
        validate_results(results, probe_id)
        if snapshot(root) != before or time.monotonic() - start > MAX_SECONDS:
            raise ValueError("CUDA probe source/runtime changed or total deadline expired")
        record = {"status": "SUCCESS", "probe_id": probe_id, "snapshot": before, "results": results,
                  "logs": logs, "elapsed_seconds": time.monotonic() - start,
                  "scientific_training_performed": False, "audits_opened": False}
        atomic_json(attempt / "result.json", record, immutable=True)
        receipt = {"status": "SUCCESS", "probe_id": probe_id, "result_path": str(attempt / "result.json"),
                   "result_sha256": file_sha256(attempt / "result.json")}
        atomic_json(base / "receipts" / (attempt.name + ".json"), receipt, immutable=True)
        atomic_json(current, receipt)
        return receipt
    except BaseException as exc:
        record = {"status": "FAILED", "probe_id": probe_id, "snapshot": before, "results": results,
                  "logs": logs, "error_type": type(exc).__name__, "error": str(exc),
                  "elapsed_seconds": time.monotonic() - start}
        atomic_json(attempt / "result.json", record, immutable=True)
        atomic_json(current, {"status": "FAILED", "probe_id": probe_id, "attempt": str(attempt)})
        raise


def require_verification(root: Path) -> dict:
    root = root.resolve()
    current = read_json(root / BASE / "current.json")
    result_path = Path(current.get("result_path", "")).resolve()
    attempts = (root / BASE / "attempts").resolve()
    if (current.get("status") != "SUCCESS" or result_path.name != "result.json" or
            result_path.parent.parent != attempts or
            result_path.parent != max(path for path in attempts.iterdir() if path.is_dir()) or
            file_sha256(result_path) != current["result_sha256"]):
        raise ValueError("CUDA readiness has no current successful immutable attempt")
    result = read_json(result_path)
    now = snapshot(root)
    probe_id = content_id(now)
    if (result["status"] != "SUCCESS" or result["snapshot"] != now or result["probe_id"] != probe_id or
            current["probe_id"] != probe_id or not 0 <= result["elapsed_seconds"] <= MAX_SECONDS):
        raise ValueError("CUDA readiness is stale or incomplete")
    validate_results(result["results"], probe_id)
    if (type(result["logs"]) is not dict or set(result["logs"]) != set(STAGES) or
            sum(item["wall_seconds"] for item in result["results"].values()) > result["elapsed_seconds"]):
        raise ValueError("CUDA readiness requires every raw stage log and consistent elapsed time")
    for mode, item in result["logs"].items():
        path = result_path.parent / (mode + ".log")
        if (type(item) is not dict or set(item) != {"path", "sha256", "peak_observed_host_bytes"} or
                type(item["peak_observed_host_bytes"]) is not int or
                not 0 < item["peak_observed_host_bytes"] <= MAX_BYTES or
                item["path"] != str(path) or file_sha256(path) != item["sha256"] or parse(path) != result["results"][mode]):
            raise ValueError("CUDA native execution log changed")
    return {"probe_id": probe_id, "result_sha256": current["result_sha256"],
            "qualified_training_batch_size": 128, "resume_probe_batch_size": 3,
            "hardware": result["results"]["smoke"]["hardware"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    print(json_text(run(parser.parse_args().root)))
