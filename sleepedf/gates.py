"""Recompute confirmatory gates from bound run artifacts, never status text alone."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import math
from pathlib import Path

from .contracts import CLASS_ORDER, content_id, json_text, read_json
from .evaluation import evaluate_saved_records, paired_cluster_intervals
from .predictions import load_prediction, load_truth
from .research import artifact_id
from .splits import validate_split_v2

REQUIRED_SLOTS = (
    "yasa_native", "msa_cnn_official", "attnsleep_official", "xsleepnet_official",
    "tinysleepnet_official", "utime_official", "usleep_official", "sleepyland_yasa",
    "sleepyland_usleep", "sleepyland_deepresnet", "sleepyland_sleeptransformer",
)
_FIT_ROLES = ("fitted_participants", "selection_participants", "pretraining_participants",
              "scaler_participants", "calibration_participants", "teacher_participants",
              "pseudolabel_participants")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_file(path: Path, expected_sha256: str) -> None:
    actual = _sha256(path)
    if expected_sha256 not in (actual, "sha256:" + actual):
        raise ValueError(f"artifact SHA256 mismatch: {path}")


def _path(base: Path, supplied: str) -> Path:
    if not isinstance(supplied, str) or not supplied:
        raise ValueError("artifact path must be a nonempty string")
    candidate = Path(supplied)
    return candidate if candidate.is_absolute() else base / candidate


def _load_run(path: Path, expected_sha256: str) -> dict:
    _bound_file(path, expected_sha256)
    run = read_json(path)
    if type(run) is not dict or run.get("schema_version") != "1.0":
        raise ValueError("invalid run record")
    return run


def _list_of_ids(run: dict, name: str) -> set[str]:
    value = run.get(name)
    if type(value) is not list or any(not isinstance(v, str) or not v for v in value) or len(value) != len(set(value)):
        raise ValueError(f"run record requires explicit unique {name}")
    return set(value)


def _positive(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _evidence(run: dict, path_key: str, hash_key: str, kind: str) -> dict:
    path = _path(Path(run["_record_path"]).parent, run[path_key])
    _bound_file(path, run[hash_key])
    try:
        value = read_json(path)
    except ValueError as exc:
        raise ValueError(f"{kind} evidence requires a structured JSON artifact") from exc
    if type(value) is not dict or value.get("schema_version") != "1.0" or value.get("artifact_type") != kind:
        raise ValueError(f"{kind} evidence requires a structured JSON artifact")
    for key in ("system_id", "model_id", "config_hash", "source_sha", "checkpoint_sha", "implementation_sha"):
        if value.get(key) != run.get(key):
            raise ValueError(f"{kind} {key} differs from bound run")
    return value


def _execution(run: dict, evidence: dict) -> None:
    invocation = evidence.get("execution")
    required = {"command", "entrypoint_path", "entrypoint_sha256", "log_path", "log_sha256",
                "runtime", "runtime_version", "exit_code", "wall_seconds", "cpu_threads",
                "gpu_jobs", "peak_host_bytes", "peak_gpu_bytes"}
    if type(invocation) is not dict or set(invocation) != required:
        raise ValueError("native execution invocation is incomplete")
    command = invocation["command"]
    if type(command) is not list or len(command) < 2 or any(type(arg) is not str or not arg for arg in command):
        raise ValueError("native execution command is missing")
    entrypoint = _path(Path(run["_record_path"]).parent, invocation["entrypoint_path"])
    if invocation["entrypoint_path"] not in command and str(entrypoint) not in command:
        raise ValueError("native entrypoint absent from recorded command")
    for key in ("runtime", "runtime_version"):
        if type(invocation[key]) is not str or not invocation[key]:
            raise ValueError("native execution runtime/version is missing")
    if type(invocation["exit_code"]) is not int or invocation["exit_code"] != 0 or \
            not _positive(invocation["wall_seconds"]) or \
            type(invocation["cpu_threads"]) is not int or not 1 <= invocation["cpu_threads"] <= 4 or \
            type(invocation["gpu_jobs"]) is not int or not 0 <= invocation["gpu_jobs"] <= 1 or \
            type(invocation["peak_host_bytes"]) is not int or not 0 < invocation["peak_host_bytes"] <= 10*1024**3 or \
            type(invocation["peak_gpu_bytes"]) is not int or not 0 <= invocation["peak_gpu_bytes"] <= 4*1024**3:
        raise ValueError("native execution failed or exceeded resource bounds")
    _bound_file(entrypoint, invocation["entrypoint_sha256"])
    _bound_file(_path(Path(run["_record_path"]).parent, invocation["log_path"]), invocation["log_sha256"])


def _checkpoint(path: Path, expected_sha256: str) -> None:
    _bound_file(path, expected_sha256)
    if path.suffix.lower() in {".md", ".txt", ".json", ".yaml", ".yml"} or path.stat().st_size < 128:
        raise ValueError("checkpoint is not a native model artifact")
    with path.open("rb") as stream:
        head = stream.read(4096)
    if all(byte in (9, 10, 13) or 32 <= byte <= 126 for byte in head):
        raise ValueError("checkpoint is a text document, not a native model artifact")


def _adequacy(run: dict, selected: dict) -> None:
    evidence = _evidence(run, "adequacy_evidence_path", "adequacy_evidence_sha256", "native_training_adequacy")
    _execution(run, evidence)
    loaded = evidence.get("checkpoint_load")
    if (type(loaded) is not dict or
            set(loaded) != {"command", "exit_code", "loaded_checkpoint_sha256", "log_path", "log_sha256"} or
            type(loaded["command"]) is not list or len(loaded["command"]) < 2 or
            any(type(arg) is not str or not arg for arg in loaded["command"]) or
            type(loaded["exit_code"]) is not int or loaded["exit_code"] != 0 or
            loaded["loaded_checkpoint_sha256"] != run["checkpoint_sha"] or
            not any(arg in (selected["checkpoint_path"], str(_path(Path(run["_record_path"]).parent,
                                                                selected["checkpoint_path"])))
                    for arg in loaded["command"])):
        raise ValueError("native checkpoint load verification is incomplete")
    _bound_file(_path(Path(run["_record_path"]).parent, loaded["log_path"]), loaded["log_sha256"])
    training = evidence.get("training")
    recipe = selected.get("native_recipe")
    if type(training) is not dict or type(recipe) is not dict or training.get("recipe") != recipe:
        raise ValueError("native training recipe differs from frozen selection")
    if (set(recipe) != {"mode", "unit", "minimum_steps", "variants"} or
            recipe["mode"] not in ("local_fit", "released_weights") or
            recipe["unit"] not in ("epoch", "iteration", "release") or
            type(recipe["minimum_steps"]) is not int or recipe["minimum_steps"] < 0 or
            type(recipe["variants"]) is not list or not recipe["variants"] or
            any(type(v) is not str or not v for v in recipe["variants"]) or
            len(set(recipe["variants"])) != len(recipe["variants"])):
        raise ValueError("frozen native recipe is incomplete")
    if (run["system_id"] == "yasa_native" and recipe["mode"] == "local_fit" and
            (recipe["unit"] != "iteration" or recipe["minimum_steps"] < 400)) or \
            (run["system_id"] == "msa_cnn_official" and recipe["mode"] == "local_fit" and
             (recipe["unit"] != "epoch" or recipe["minimum_steps"] < 100 or
              set(recipe["variants"]) != {"small_mono", "large_mono", "small_multi", "large_multi"})):
        raise ValueError("frozen native recipe is below approved adequacy")
    if (training.get("mode") != recipe["mode"] or
            training.get("selected_variant") not in recipe["variants"] or
            training.get("selected_variant") != selected.get("selected_variant") or
            training.get("selected_step") != selected.get("selected_step")):
        raise ValueError("selected native variant or fit mode differs")
    if recipe["mode"] == "local_fit":
        fit_ids = training.get("fit_participants")
        if recipe["minimum_steps"] < 1 or training.get("released_lineage") is not None or \
                type(fit_ids) is not list or any(type(pid) is not str or not pid for pid in fit_ids) or \
                len(fit_ids) != len(set(fit_ids)) or \
                set(fit_ids) != set(run["fitted_participants"]) or \
                not run["fitted_participants"]:
            raise ValueError("local native fit ancestry is incomplete")
        histories = training.get("histories")
        if type(histories) is not dict or set(histories) != set(recipe["variants"]):
            raise ValueError("native recipe variant histories are incomplete")
        for variant, rows in histories.items():
            if type(rows) is not list or not rows:
                raise ValueError("native fit history is empty")
            steps = []
            for row in rows:
                if type(row) is not dict or set(row) != {"step", "train_examples", "validation_examples",
                                                       "train_loss", "validation_macro_f1"} or \
                        type(row["step"]) is not int or row["step"] < 1 or \
                        type(row["train_examples"]) is not int or row["train_examples"] < 1 or \
                        type(row["validation_examples"]) is not int or row["validation_examples"] < 1 or \
                        not _positive(row["train_loss"]) or \
                        type(row["validation_macro_f1"]) not in (int, float) or \
                        not 0 <= row["validation_macro_f1"] <= 1 or \
                        not math.isfinite(row["validation_macro_f1"]):
                    raise ValueError("native fit history contains invalid numerical evidence")
                steps.append(row["step"])
            if steps != sorted(set(steps)) or steps[-1] < recipe["minimum_steps"]:
                raise ValueError("native fit history did not complete frozen recipe")
        selected_step = training.get("selected_step")
        if type(selected_step) is not int or selected_step not in [
                row["step"] for row in histories[training["selected_variant"]]]:
            raise ValueError("released checkpoint selection has no native fit history")
    else:
        lineage = training.get("released_lineage")
        if (recipe["unit"] != "release" or recipe["minimum_steps"] != 0 or
                training.get("histories") != {} or training.get("fit_participants") != [] or
                run["fitted_participants"] or training.get("selected_step") is not None or
                type(lineage) is not dict or
                set(lineage) != {"origin", "release_version", "training_datasets", "checkpoint_sha",
                                "checkpoint_license", "training_participant_manifest_sha256"} or
                any(type(lineage[k]) is not str or not lineage[k] for k in
                    ("origin", "release_version", "checkpoint_sha", "checkpoint_license",
                     "training_participant_manifest_sha256")) or
                type(lineage["training_datasets"]) is not list or not lineage["training_datasets"] or
                lineage["checkpoint_sha"] != run["checkpoint_sha"]):
            raise ValueError("released checkpoint lineage is incomplete")


def _overlap(run: dict, all_audit_participants: set[str]) -> None:
    evidence = _evidence(run, "overlap_evidence_path", "overlap_evidence_sha256", "participant_overlap_review")
    ids = evidence.get("source_training_participants")
    if (type(ids) is not list or any(type(pid) is not str or not pid for pid in ids) or
            len(ids) != len(set(ids)) or
            set(ids) != set(run["fitted_participants"]) | set(run["pretraining_participants"]) or
            set(ids) & all_audit_participants or evidence.get("matched_audit_participants") != [] or
            evidence.get("method") != "source_manifest_participant_id_match"):
        raise ValueError("participant overlap review lacks verified disjoint ancestry")
    for prefix in ("source_manifest", "review_log"):
        key, hash_key = prefix + "_path", prefix + "_sha256"
        if type(evidence.get(key)) is not str or not evidence[key] or \
                type(evidence.get(hash_key)) is not str or not evidence[hash_key]:
            raise ValueError("participant overlap review lacks bound source or review artifact")
        _bound_file(_path(Path(run["_record_path"]).parent, evidence[key]), evidence[hash_key])
    manifest = read_json(_path(Path(run["_record_path"]).parent, evidence["source_manifest_path"]))
    if (type(manifest) is not dict or manifest.get("schema_version") != "1.0" or
            manifest.get("artifact_type") != "training_participant_manifest" or
            manifest.get("participant_ids") != ids):
        raise ValueError("source training participant manifest differs from overlap review")


def _inference(run: dict, expected_records: dict) -> None:
    evidence = _evidence(run, "inference_evidence_path", "inference_evidence_sha256", "signal_only_inference")
    rows = evidence.get("records")
    if type(rows) is not list or len(rows) != len(expected_records):
        raise ValueError("native inference must evidence every audit recording")
    paths = set(run["prediction_paths"])
    seen = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"recording_id", "participant_id", "n_epochs", "source_psg_sha256",
                                                   "prediction_path", "prediction_sha256", "command", "entrypoint_path",
                                                   "entrypoint_sha256", "log_path", "log_sha256", "exit_code",
                                                   "wall_seconds", "input_kind"}:
            raise ValueError("native inference record evidence is incomplete")
        rid = row["recording_id"]
        if type(rid) is not str or type(row["prediction_path"]) is not str:
            raise ValueError("native inference record identity is invalid")
        expected = expected_records.get(rid)
        if rid in seen or type(expected) is not dict or row["participant_id"] != expected["participant_id"] or \
                row["n_epochs"] != expected["n_epochs"] or row["source_psg_sha256"] != expected["psg_sha256"] or \
                row["prediction_path"] not in paths or \
                row["prediction_sha256"] != run["prediction_sha256"][row["prediction_path"]] or \
                row["input_kind"] != "psg_only" or type(row["exit_code"]) is not int or row["exit_code"] != 0 or \
                not _positive(row["wall_seconds"]):
            raise ValueError("native inference record differs from frozen audit or run")
        command = row["command"]
        entrypoint = _path(Path(run["_record_path"]).parent, row["entrypoint_path"])
        if type(command) is not list or len(command) < 2 or \
                any(type(arg) is not str or not arg for arg in command) or \
                (row["entrypoint_path"] not in command and str(entrypoint) not in command):
            raise ValueError("native inference command is missing")
        _bound_file(entrypoint, row["entrypoint_sha256"])
        _bound_file(_path(Path(run["_record_path"]).parent, row["log_path"]), row["log_sha256"])
        seen.add(rid)
    if seen != set(expected_records):
        raise ValueError("native inference evidence omitted audit recording")


def _validate_run(run: dict, *, system_id: str, selected: dict,
                  all_audit_participants: set[str], development_participants: set[str],
                  expected_records: dict, synthetic_test: bool) -> None:
    if run.get("system_id") != system_id or run.get("model_id") != selected.get("model_id"):
        raise ValueError("run system/model differs from frozen selection")
    if run.get("config_hash") != selected.get("config_hash") or not run.get("config_hash"):
        raise ValueError("run config differs from frozen selection")
    for key in ("source_sha", "checkpoint_sha", "implementation_sha"):
        if run.get(key) != selected.get(key) or not run.get(key):
            raise ValueError(f"run {key} differs from frozen selection")
    checkpoint_path = selected.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError("frozen selection lacks checkpoint artifact path")
    _checkpoint(_path(Path(run["_record_path"]).parent, checkpoint_path), run["checkpoint_sha"])
    if run.get("class_order") != list(CLASS_ORDER):
        raise ValueError("wrong run class order")
    if run.get("status") not in ("EVALUATED_CLEAN", "EXECUTED_EXACT_DUPLICATE"):
        raise ValueError(f"required run is not cleanly executed: {system_id}")
    if run.get("synthetic") is not synthetic_test or run.get("baseline_adequate") is not True:
        raise ValueError("synthetic or inadequate run cannot enter confirmatory gate")
    if run.get("selection_frozen_before_audit") is not True or run.get("rights_reviewed") is not True:
        raise ValueError("run selection or rights evidence is incomplete")
    for name in ("source_sha", "checkpoint_sha", "implementation_sha", "code_license",
                 "checkpoint_license", "adequacy_evidence_path", "adequacy_evidence_sha256",
                 "overlap_evidence_path", "overlap_evidence_sha256",
                 "inference_evidence_path", "inference_evidence_sha256"):
        if not isinstance(run.get(name), str) or not run[name]:
            raise ValueError(f"run record lacks {name}")
    if run.get("external_overlap_verified") is not True:
        raise ValueError("external/pretraining overlap is not verified")
    for name in _FIT_ROLES:
        ids = _list_of_ids(run, name)
        if ids & all_audit_participants:
            raise ValueError(f"audit participant appears in {name}")
        if name != "pretraining_participants" and not ids <= development_participants:
            raise ValueError(f"{name} includes non-development participant")
    _adequacy(run, selected)
    _overlap(run, all_audit_participants)
    _inference(run, expected_records)
    if run["status"] == "EXECUTED_EXACT_DUPLICATE":
        for name in ("duplicate_of", "parity_evidence_path", "parity_evidence_sha256"):
            if not isinstance(run.get(name), str) or not run[name]:
                raise ValueError("exact duplicate lacks executed parity evidence")
        parity = _evidence(run, "parity_evidence_path", "parity_evidence_sha256", "executed_duplicate_parity")
        if parity.get("original_system_id") != run["duplicate_of"] or \
                parity.get("compared_recording_ids") != sorted(expected_records):
            raise ValueError("exact duplicate parity does not cover frozen recordings")
        _execution(run, parity)


def _prediction_paths(run: dict, *, protocol_hash: str, registry_hash: str) -> list[Path]:
    paths = run.get("prediction_paths")
    hashes = run.get("prediction_sha256")
    if type(paths) is not list or not paths or type(hashes) is not dict or set(hashes) != set(paths):
        raise ValueError("run requires every prediction file and its hash")
    result = []
    for supplied in paths:
        path = _path(Path(run["_record_path"]).parent, supplied)
        _bound_file(path, hashes[supplied])
        metadata, _ = load_prediction(path)
        if metadata["protocol_hash"] != protocol_hash or metadata["registry_hash"] != registry_hash:
            raise ValueError("prediction protocol/registry hash differs")
        if metadata["model_id"] != run["model_id"]:
            raise ValueError("prediction model differs from run")
        provenance = metadata["provenance"]
        if any(provenance.get(key) != run[key] for key in
               ("config_hash", "source_sha", "checkpoint_sha", "implementation_sha")):
            raise ValueError("prediction provenance differs from frozen run")
        result.append(path)
    return result


def _same_predictions(left: list[Path], right: list[Path]) -> bool:
    def indexed(paths: list[Path]) -> dict:
        records = {}
        for path in paths:
            _, data = load_prediction(path)
            records[data["recording_id"]] = data
        return records
    first, second = indexed(left), indexed(right)
    if first.keys() != second.keys():
        return False
    import numpy as np
    return all(
        ("probabilities" in first[rid]) == ("probabilities" in second[rid])
        and all(np.array_equal(first[rid][key], second[rid][key])
                for key in ("epoch_index", "onset_seconds", "hard_label"))
        and ("probabilities" not in first[rid]
             or np.array_equal(first[rid]["probabilities"], second[rid]["probabilities"]))
        for rid in first)


def passes_absolute_margin(candidate: Fraction, comparators: list[Fraction]) -> bool:
    """No rounding and no relative or average-baseline substitution."""
    if not comparators:
        raise ValueError("at least one comparator required")
    return candidate - max(comparators) >= Fraction(1, 50)


def compute_gate(*, phase: str, protocol_hash: str, registry_hash: str,
                 freeze_manifest: dict, run_evidence: dict[str, dict],
                 truth_paths: list[Path], participant_cohort: dict[str, str],
                 development_participants: list[str], candidate_id: str,
                 draws: int = 10_000, synthetic_test: bool = False) -> dict:
    """Return a gate report from immutable inputs; reject missing or inconsistent evidence."""
    if phase not in ("A", "B") or not protocol_hash or not registry_hash:
        raise ValueError("phase and frozen hashes are required")
    if candidate_id in REQUIRED_SLOTS:
        raise ValueError("candidate must be distinct from every mandatory comparator slot")
    if not synthetic_test and draws != 10_000:
        raise ValueError("confirmatory paired bootstrap requires 10,000 frozen draws")
    if type(freeze_manifest) is not dict or freeze_manifest.get("protocol_hash") != protocol_hash or \
            freeze_manifest.get("registry_hash") != registry_hash or freeze_manifest.get("phase") != phase or \
            freeze_manifest.get("candidate_id") != candidate_id or freeze_manifest.get("selection_frozen") is not True:
        raise ValueError("freeze manifest does not bind this gate")
    for key in ("split_id", "readiness_id", "split_sha256", "readiness_sha256",
                "registry_sha256", "protocol_path", "protocol_sha256", "readiness_path"):
        if not isinstance(freeze_manifest.get(key), str) or not freeze_manifest[key]:
            raise ValueError(f"freeze manifest lacks {key}")
    _bound_file(Path(freeze_manifest["protocol_path"]), freeze_manifest["protocol_sha256"])
    protocol = read_json(Path(freeze_manifest["protocol_path"]))
    if (type(protocol) is not dict or protocol.get("protocol_hash") != artifact_id(protocol, "protocol_hash") or
            protocol["protocol_hash"] != protocol_hash or protocol.get("split_id") != freeze_manifest["split_id"] or
            protocol.get("readiness_id") != freeze_manifest["readiness_id"] or
            protocol.get("registry_hash") != registry_hash or
            protocol.get("split_sha256") != freeze_manifest["split_sha256"] or
            protocol.get("readiness_sha256") != freeze_manifest["readiness_sha256"] or
            protocol.get("registry_sha256") != freeze_manifest["registry_sha256"] or
            protocol.get("class_order") != list(CLASS_ORDER) or protocol.get("epoch_seconds") != 30):
        raise ValueError("frozen protocol identity or data bindings differ")
    selected = freeze_manifest.get("selected_models")
    if type(selected) is not dict or not set(REQUIRED_SLOTS) <= set(selected) or candidate_id not in selected:
        raise ValueError("freeze manifest omits a mandatory slot or candidate")
    if type(run_evidence) is not dict or set(run_evidence) != set(selected):
        raise ValueError("run evidence must cover exactly the frozen systems")
    if type(development_participants) is not list or len(development_participants) != len(set(development_participants)):
        raise ValueError("development participant membership is invalid")
    frozen_development = freeze_manifest.get("development_participants")
    frozen_a = freeze_manifest.get("audit_a_participants")
    frozen_b = freeze_manifest.get("audit_b_participants")
    if (type(frozen_development) is not list or type(frozen_a) is not list
            or type(frozen_b) is not list or set(frozen_development) != set(development_participants)
            or any(len(group) != len(set(group)) for group in (frozen_development, frozen_a, frozen_b))
            or set(frozen_development) & (set(frozen_a) | set(frozen_b))
            or set(frozen_a) & set(frozen_b)):
        raise ValueError("frozen development/A/B participant membership is invalid")
    if set(freeze_manifest["audit_participants"]) != set(frozen_a if phase == "A" else frozen_b):
        raise ValueError("phase audit membership differs from frozen partitions")
    if type(participant_cohort) is not dict or set(participant_cohort) != set(freeze_manifest.get("audit_participants", [])):
        raise ValueError("participant cohort mapping differs from frozen audit")
    truth_hashes = freeze_manifest.get("truth_artifact_sha256")
    if type(truth_hashes) is not dict or set(truth_hashes) != {str(Path(p)) for p in truth_paths}:
        raise ValueError("frozen truth artifact hashes are missing or differ")
    expected_records = freeze_manifest.get("audit_recordings")
    if type(expected_records) is not dict or not expected_records:
        raise ValueError("freeze manifest lacks authoritative audit recording inventory")
    if protocol.get("audit_recordings", {}).get(phase) != expected_records:
        raise ValueError("audit recording inventory differs from original frozen protocol")
    _bound_file(Path(freeze_manifest["readiness_path"]), freeze_manifest["readiness_sha256"])
    readiness = read_json(Path(freeze_manifest["readiness_path"]))
    if readiness.get("manifest_id") != freeze_manifest["readiness_id"]:
        raise ValueError("frozen readiness identity differs")
    if not synthetic_test:
        for name in ("split", "readiness", "registry"):
            path_key, hash_key = name + "_path", name + "_sha256"
            if not isinstance(freeze_manifest.get(path_key), str) or not isinstance(freeze_manifest.get(hash_key), str):
                raise ValueError(f"frozen {name} artifact path/hash is missing")
            _bound_file(Path(freeze_manifest[path_key]), freeze_manifest[hash_key])
        split = read_json(Path(freeze_manifest["split_path"]))
        registry = read_json(Path(freeze_manifest["registry_path"]))
        validate_split_v2(split, readiness)
        if (split["split_id"] != freeze_manifest["split_id"]
                or readiness["manifest_id"] != freeze_manifest["readiness_id"]
                or registry.get("registry_hash") != artifact_id(registry, "registry_hash")
                or registry["registry_hash"] != registry_hash):
            raise ValueError("frozen split, readiness, or registry identity differs")
        for part, key in (("development", "development_participants"),
                          ("audit_a", "audit_a_participants"),
                          ("audit_b", "audit_b_participants")):
            if set(split["participants"][part]) != set(freeze_manifest[key]):
                raise ValueError("frozen participant partitions differ from validated split")
        source_records = {rec["recording_id"]: rec for rec in readiness["records"]
                          if rec["participant_id"] in set(freeze_manifest["audit_participants"])}
        if set(source_records) != set(expected_records):
            raise ValueError("audit inventory is an incomplete subset of validated readiness")
        for rid, rec in source_records.items():
            frozen = expected_records[rid]
            if any(frozen[key] != rec[source] for key, source in
                   (("participant_id", "participant_id"), ("n_epochs", "n_epochs"),
                    ("psg_sha256", "psg_sha256"),
                    ("hypnogram_sha256", "hypnogram_sha256"))):
                raise ValueError("frozen audit recording differs from validated readiness")
        registry_slots = {slot["id"] for repo in registry["repositories"]
                          for slot in repo["execution_slots"] if slot["required"]}
        additional = {item["id"] for item in registry.get("additional_comparisons", [])}
        if registry_slots != set(REQUIRED_SLOTS) or not registry_slots | additional <= set(selected):
            raise ValueError("frozen selection omits required registry comparison")
        from .audit_access import validate_audit_opening
        validate_audit_opening(freeze_manifest)
    source_records = {rec["recording_id"]: rec for rec in readiness["records"]
                      if rec["recording_id"] in expected_records}
    if set(source_records) != set(expected_records):
        raise ValueError("authoritative readiness omits frozen audit recording")
    authoritative_cohorts = {}
    for rid, rec in source_records.items():
        expected = expected_records[rid]
        if rec.get("participant_id") != expected["participant_id"] or \
                rec.get("cohort") not in ("SC", "ST") or \
                rec["participant_id"] != f"{rec['cohort']}:{rec['participant_id'].split(':')[-1]}":
            raise ValueError("authoritative cohort or recording identity differs")
        pid = rec["participant_id"]
        if pid in authoritative_cohorts and authoritative_cohorts[pid] != rec["cohort"]:
            raise ValueError("participant cohort differs between nights")
        authoritative_cohorts[pid] = rec["cohort"]
    if participant_cohort != authoritative_cohorts:
        raise ValueError("participant cohort differs from authoritative readiness")
    for path in truth_paths:
        _bound_file(Path(path), truth_hashes[str(Path(path))])
    actual_records = {}
    required_record = {"participant_id", "n_epochs", "psg_sha256", "hypnogram_sha256",
                       "truth_payload_sha256", "truth_sidecar_sha256"}
    for path in truth_paths:
        path = Path(path)
        sidecar, record = load_truth(path)
        rid = record["recording_id"]
        if rid in actual_records:
            raise ValueError("duplicate audit recording")
        expected = expected_records.get(rid)
        if type(expected) is not dict or set(expected) != required_record:
            raise ValueError("audit recording differs from authoritative inventory")
        for key, actual in (("participant_id", record["participant_id"]),
                            ("n_epochs", len(record["epoch_index"])),
                            ("psg_sha256", sidecar["source_psg_sha256"]),
                            ("hypnogram_sha256", sidecar["source_hypnogram_sha256"])):
            if expected[key] != actual:
                raise ValueError(f"audit recording {rid} {key} differs from frozen inventory")
        _bound_file(path, expected["truth_payload_sha256"])
        _bound_file(path.with_suffix(".json"), expected["truth_sidecar_sha256"])
        actual_records[rid] = record["participant_id"]
    if set(actual_records) != set(expected_records):
        raise ValueError("missing or extra audit recording against frozen inventory")
    audit = set(participant_cohort)
    all_audit = set(frozen_a) | set(frozen_b)
    if set(actual_records.values()) != audit:
        raise ValueError("audit participant coverage differs from frozen recordings")
    development = set(development_participants)
    if audit & development:
        raise ValueError("development and audit participants overlap")
    runs = {}
    scores = {}
    prediction_paths = {}
    for system_id in sorted(selected):
        item = run_evidence[system_id]
        if type(item) is not dict or set(item) != {"path", "sha256"}:
            raise ValueError("run evidence must bind a JSON path and SHA256")
        path = Path(item["path"])
        run = _load_run(path, item["sha256"])
        run["_record_path"] = str(path)
        _validate_run(run, system_id=system_id, selected=selected[system_id],
                      all_audit_participants=all_audit, development_participants=development,
                      expected_records=expected_records, synthetic_test=synthetic_test)
        paths = _prediction_paths(run, protocol_hash=protocol_hash, registry_hash=registry_hash)
        prediction_paths[system_id] = paths
        scores[system_id] = evaluate_saved_records(truth_paths, paths,
                                                   protocol_hash=protocol_hash,
                                                   registry_hash=registry_hash,
                                                   model_id=run["model_id"])
        if set(scores[system_id]["per_participant_confusion"]) != audit:
            raise ValueError("scored participants differ from frozen audit")
        runs[system_id] = run
    distinct = {}
    for system_id, run in runs.items():
        if run["status"] != "EXECUTED_EXACT_DUPLICATE":
            distinct[system_id] = scores[system_id]
            continue
        parent = run["duplicate_of"]
        if parent == system_id or parent not in runs or parent == candidate_id:
            raise ValueError("exact duplicate points to missing/self/candidate run")
        original = runs[parent]
        if original["status"] != "EVALUATED_CLEAN" or run["model_id"] != original["model_id"]:
            raise ValueError("exact duplicate must name its clean original model")
        for key in ("implementation_sha", "checkpoint_sha", "config_hash"):
            if run[key] != original[key]:
                raise ValueError("exact duplicate artifact identity differs")
        if not _same_predictions(prediction_paths[system_id], prediction_paths[parent]):
            raise ValueError("duplicate predictions differ despite parity claim")
    if candidate_id not in distinct:
        raise ValueError("candidate cannot be an exact duplicate")
    comparators = {key: value for key, value in distinct.items() if key != candidate_id}
    if not comparators:
        raise ValueError("no distinct comparator")
    candidate_f1 = Fraction(scores[candidate_id]["macro_f1_exact"])
    deltas = {name: candidate_f1 - Fraction(item["macro_f1_exact"])
              for name, item in comparators.items()}
    best_name = max(comparators, key=lambda name: Fraction(comparators[name]["macro_f1_exact"]))
    best_delta = candidate_f1 - Fraction(comparators[best_name]["macro_f1_exact"])
    uncertainty = paired_cluster_intervals(scores[candidate_id], comparators,
                                           participant_cohort, phase=phase, draws=draws)
    lower = {name: item["lower"] for name, item in uncertainty["intervals"].items()}
    flags = {"all_executed": True, "point_margin_pass": passes_absolute_margin(
        candidate_f1, [Fraction(item["macro_f1_exact"]) for item in comparators.values()]),
             "positive_superiority_supported": all(value > 0 for value in lower.values()),
             "minimum_margin_supported": all(value >= 0.02 for value in lower.values())}
    integrity = {"frozen_registry_and_protocol": True, "all_records_hash_bound": True,
                 "same_full_epoch_set": True, "all_audit_participants_disjoint": True,
                 "baseline_adequacy_evidenced": True, "selection_before_audit": True,
                 "truth_separate_from_predictions": True}
    passed = all((flags["all_executed"], flags["point_margin_pass"],
                  flags["positive_superiority_supported"], all(integrity.values())))
    if synthetic_test:
        status = "NOT_CONFIRMATORY"
    else:
        status = "PASSED" if passed else "FAILED"
    report = {"schema_version": "1.0", "phase": phase, "status": status,
              "synthetic_test": bool(synthetic_test), "protocol_hash": protocol_hash,
              "registry_hash": registry_hash, "freeze_manifest_hash": content_id(freeze_manifest),
              "truth_artifact_sha256": truth_hashes,
              "candidate_id": candidate_id, "required_slots": list(REQUIRED_SLOTS),
              "distinct_comparators": sorted(comparators), "best_comparator": best_name,
              "candidate_macro_f1": float(candidate_f1),
              "best_comparator_macro_f1": comparators[best_name]["macro_f1"],
              "point_margin": float(best_delta),
              "point_margin_exact": f"{best_delta.numerator}/{best_delta.denominator}",
              "per_comparison_delta_exact": {name: f"{delta.numerator}/{delta.denominator}"
                                             for name, delta in sorted(deltas.items())},
              "scores": {name: {key: val for key, val in item.items()
                                if key != "per_participant_confusion"} for name, item in sorted(scores.items())},
              "uncertainty": uncertainty, "flags": flags, "integrity": integrity,
              "unmet_conditions": [name for name, value in {**flags, **integrity}.items()
                                   if name != "minimum_margin_supported" and not value],
              "run_record_sha256": {name: item["sha256"] for name, item in sorted(run_evidence.items())}}
    json_text(report)
    return report


def _verify_attestation(attestation_path: Path, report_path: Path, report: dict) -> None:
    """Check a separate verifier's recorded invocation and hash-bound result."""
    attestation_path = Path(attestation_path)
    attestation = read_json(attestation_path)
    required = {"schema_version", "artifact_type", "report_sha256", "verifier_id",
                "invocation_id", "entrypoint_path", "entrypoint_sha256", "argv",
                "exit_code", "result_path", "result_sha256"}
    if type(attestation) is not dict or set(attestation) != required or \
            attestation["schema_version"] != "1.0" or \
            attestation["artifact_type"] != "independent_gate_verification":
        raise ValueError("verifier attestation schema differs")
    if not isinstance(attestation["verifier_id"], str) or not attestation["verifier_id"] or \
            not isinstance(attestation["invocation_id"], str) or not attestation["invocation_id"]:
        raise ValueError("verifier identity or invocation identity is missing")
    if type(attestation["argv"]) is not list or not attestation["argv"] or \
            any(not isinstance(arg, str) or not arg for arg in attestation["argv"]):
        raise ValueError("verifier invocation argv is missing")
    if type(attestation["exit_code"]) is not int or attestation["exit_code"] != 0:
        raise ValueError("verifier invocation did not succeed")
    _bound_file(report_path, attestation["report_sha256"])
    entrypoint = _path(attestation_path.parent, attestation["entrypoint_path"]).resolve()
    result_path = _path(attestation_path.parent, attestation["result_path"]).resolve()
    if entrypoint == Path(__file__).resolve() or entrypoint in (report_path.resolve(), result_path):
        raise ValueError("verifier entrypoint must be a separate executable artifact")
    if str(entrypoint) not in attestation["argv"] and attestation["entrypoint_path"] not in attestation["argv"]:
        raise ValueError("verifier entrypoint missing from recorded invocation")
    _bound_file(entrypoint, attestation["entrypoint_sha256"])
    _bound_file(result_path, attestation["result_sha256"])
    result = read_json(result_path)
    expected = {"schema_version": "1.0", "status": "VERIFIED",
                "invocation_id": attestation["invocation_id"],
                "report_sha256": attestation["report_sha256"],
                "protocol_hash": report["protocol_hash"],
                "registry_hash": report["registry_hash"],
                "freeze_manifest_hash": report["freeze_manifest_hash"],
                "truth_artifact_sha256": report["truth_artifact_sha256"],
                "run_record_sha256": report["run_record_sha256"]}
    if result != expected:
        raise ValueError("verifier result does not bind the recomputed report and artifacts")


def require_gate(report_path: Path, *, attestation_path: Path, **recompute_arguments) -> dict:
    """Launch guard: recompute the report and check independent verifier evidence."""
    report_path = Path(report_path)
    stored = read_json(report_path)
    computed = compute_gate(**recompute_arguments)
    if stored != computed or computed["status"] != "PASSED":
        raise ValueError("confirmatory gate has not passed from current evidence")
    _verify_attestation(attestation_path, report_path, computed)
    return computed
