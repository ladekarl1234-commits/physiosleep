import hashlib
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from sleepedf.contracts import content_id, json_text, read_json
from sleepedf.gates import (REQUIRED_SLOTS, _checkpoint, _verify_attestation, compute_gate,
                            passes_absolute_margin, require_gate)
from sleepedf.predictions import save_prediction
from test_predictions import write_truth


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root, *, duplicate=False):
    truth_paths = [write_truth(root / "truth_sc.npz", tuple(range(5)) * 2, "SC:01", "night_sc"),
                   write_truth(root / "truth_st.npz", tuple(range(5)) * 2, "ST:01", "night_st")]
    audit_recordings = {("night_sc" if "sc" in path.stem else "night_st"): {
        "participant_id": "SC:01" if "sc" in path.stem else "ST:01", "n_epochs": 10,
        "psg_sha256": "a" * 64, "hypnogram_sha256": "b" * 64,
        "truth_payload_sha256": digest(path),
        "truth_sidecar_sha256": digest(path.with_suffix(".json"))} for path in truth_paths}
    readiness_path = root / "readiness.json"
    readiness_path.write_text(json_text({"manifest_id": "fixture-readiness", "records": [
        {"recording_id": rid, "participant_id": row["participant_id"],
         "cohort": row["participant_id"].split(":")[0]} for rid, row in audit_recordings.items()]}), encoding="utf-8")
    protocol = {"schema_version": "1.0", "split_id": "fixture-split", "readiness_id": "fixture-readiness",
                "registry_hash": "r", "split_sha256": "c" * 64,
                "readiness_sha256": digest(readiness_path), "registry_sha256": "e" * 64,
                "class_order": ["W", "N1", "N2", "N3", "REM"], "epoch_seconds": 30,
                "audit_recordings": {"A": audit_recordings, "B": {}}}
    protocol["protocol_hash"] = content_id(protocol)
    protocol_path = root / "protocol.json"
    protocol_path.write_text(json_text(protocol), encoding="utf-8")
    protocol_hash = protocol["protocol_hash"]
    common = root / "synthetic-checkpoint.pt"
    common.write_bytes(b"PK\x03\x04" + b"\x00" * 256)
    checkpoint_hash = digest(common)
    program = root / "native.py"
    program.write_text("# Fixture entrypoint; never executed as a model.\n", encoding="utf-8")
    log = root / "execution.log"
    log.write_text("Synthetic invocation fixture only.\n", encoding="utf-8")
    manifest = root / "source-participants.json"
    manifest.write_text(json_text({"schema_version": "1.0", "artifact_type": "training_participant_manifest",
                                   "participant_ids": ["D:01"]}), encoding="utf-8")
    runs, selected = {}, {}
    system_ids = list(REQUIRED_SLOTS) + ["candidate"]
    for system_id in system_ids:
        same_as_yasa = duplicate and system_id == "sleepyland_yasa"
        model_id = "yasa_native" if same_as_yasa else system_id
        config_hash = "config_yasa" if duplicate and system_id in ("yasa_native", "sleepyland_yasa") else "config_" + system_id
        prediction_paths = []
        prediction_hashes = {}
        for truth_path in truth_paths:
            rid = "night_sc" if "sc" in truth_path.stem else "night_st"
            pid = "SC:01" if rid == "night_sc" else "ST:01"
            hard = np.tile(np.arange(5, dtype=np.int8), 2) if system_id == "candidate" else np.zeros(10, dtype=np.int8)
            if same_as_yasa:
                hard = np.zeros(10, dtype=np.int8)
            prediction_path = root / f"{system_id}_{rid}.npz"
            save_prediction(prediction_path, participant_id=pid, recording_id=rid,
                            epoch_index=np.arange(10, dtype=np.int64),
                            onset_seconds=np.arange(10, dtype=float) * 30,
                            hard_label=hard, model_id=model_id, protocol_hash=protocol_hash, registry_hash="r",
                            provenance={"channel_set": ["EEG"], "config_hash": config_hash,
                                        "source_sha": "synthetic_source", "checkpoint_sha": checkpoint_hash,
                                        "implementation_sha": "synthetic_implementation"})
            prediction_paths.append(str(prediction_path))
            prediction_hashes[str(prediction_path)] = digest(prediction_path)
        identity = {"system_id": system_id, "model_id": model_id, "config_hash": config_hash,
                    "source_sha": "synthetic_source", "checkpoint_sha": checkpoint_hash,
                    "implementation_sha": "synthetic_implementation"}
        execution = {"command": ["python", str(program), system_id],
                     "entrypoint_path": str(program), "entrypoint_sha256": digest(program),
                     "log_path": str(log), "log_sha256": digest(log), "runtime": "python",
                     "runtime_version": "fixture", "exit_code": 0, "wall_seconds": 1.0,
                     "cpu_threads": 1, "gpu_jobs": 0, "peak_host_bytes": 1024,
                     "peak_gpu_bytes": 0}
        variants = (["small_mono", "large_mono", "small_multi", "large_multi"]
                    if system_id == "msa_cnn_official" else ["main"])
        steps = 400 if system_id == "yasa_native" else 100 if system_id == "msa_cnn_official" else 1
        unit = "iteration" if system_id == "yasa_native" else "epoch"
        recipe = {"mode": "local_fit", "unit": unit, "minimum_steps": steps, "variants": variants}
        row = lambda step: {"step": step, "train_examples": 100, "validation_examples": 20,
                            "train_loss": 0.5, "validation_macro_f1": 0.3}
        histories = {variant: [row(1)] if steps == 1 else [row(1), row(steps)] for variant in variants}
        adequacy = {"schema_version": "1.0", "artifact_type": "native_training_adequacy", **identity,
                    "execution": execution, "checkpoint_load": {"command": ["python", str(program), "load", str(common)],
                    "exit_code": 0, "loaded_checkpoint_sha256": checkpoint_hash,
                    "log_path": str(log), "log_sha256": digest(log)},
                    "training": {"mode": "local_fit", "recipe": recipe, "histories": histories,
                                 "selected_variant": variants[0], "selected_step": steps,
                                 "fit_participants": ["D:01"], "released_lineage": None}}
        overlap = {"schema_version": "1.0", "artifact_type": "participant_overlap_review", **identity,
                   "source_training_participants": ["D:01"], "matched_audit_participants": [],
                   "method": "source_manifest_participant_id_match",
                   "source_manifest_path": str(manifest), "source_manifest_sha256": digest(manifest),
                   "review_log_path": str(log), "review_log_sha256": digest(log)}
        inference = {"schema_version": "1.0", "artifact_type": "signal_only_inference", **identity,
                     "records": [{"recording_id": "night_sc" if "night_sc" in path else "night_st",
                                  "participant_id": "SC:01" if "night_sc" in path else "ST:01",
                                  "n_epochs": 10, "source_psg_sha256": "a" * 64,
                                  "prediction_path": path, "prediction_sha256": prediction_hashes[path],
                                  "command": ["python", str(program), "predict", path],
                                  "entrypoint_path": str(program), "entrypoint_sha256": digest(program),
                                  "log_path": str(log), "log_sha256": digest(log),
                                  "exit_code": 0, "wall_seconds": 1.0, "input_kind": "psg_only"}
                                 for path in prediction_paths]}
        evidence_files = {}
        for kind, payload in (("adequacy", adequacy), ("overlap", overlap), ("inference", inference)):
            evidence_path = root / f"{system_id}-{kind}.json"
            evidence_path.write_text(json_text(payload), encoding="utf-8")
            evidence_files[kind] = evidence_path
        run = {"schema_version": "1.0", "system_id": system_id, "model_id": model_id,
               "config_hash": config_hash, "class_order": ["W", "N1", "N2", "N3", "REM"],
               "status": "EXECUTED_EXACT_DUPLICATE" if same_as_yasa else "EVALUATED_CLEAN",
               "synthetic": True, "baseline_adequate": True, "selection_frozen_before_audit": True,
               "rights_reviewed": True, "external_overlap_verified": True,
               "source_sha": "synthetic_source", "checkpoint_sha": checkpoint_hash,
               "implementation_sha": "synthetic_implementation", "code_license": "fixture",
               "checkpoint_license": "fixture", "adequacy_evidence_path": str(evidence_files["adequacy"]),
               "adequacy_evidence_sha256": digest(evidence_files["adequacy"]),
               "overlap_evidence_path": str(evidence_files["overlap"]),
               "overlap_evidence_sha256": digest(evidence_files["overlap"]),
               "inference_evidence_path": str(evidence_files["inference"]),
               "inference_evidence_sha256": digest(evidence_files["inference"]), "fitted_participants": ["D:01"],
               "selection_participants": ["D:01"], "pretraining_participants": [],
               "scaler_participants": ["D:01"], "calibration_participants": [], "teacher_participants": [],
               "pseudolabel_participants": [], "prediction_paths": prediction_paths,
               "prediction_sha256": prediction_hashes}
        if same_as_yasa:
            parity = {"schema_version": "1.0", "artifact_type": "executed_duplicate_parity", **identity,
                      "original_system_id": "yasa_native", "compared_recording_ids": ["night_sc", "night_st"],
                      "execution": execution}
            parity_path = root / f"{system_id}-parity.json"
            parity_path.write_text(json_text(parity), encoding="utf-8")
            run.update(duplicate_of="yasa_native", parity_evidence_path=str(parity_path),
                       parity_evidence_sha256=digest(parity_path), config_hash="config_yasa")
        path = root / f"run_{system_id}.json"
        path.write_text(json_text(run), encoding="utf-8")
        runs[system_id] = {"path": str(path), "sha256": digest(path)}
        selected[system_id] = {"model_id": model_id, "config_hash": config_hash,
                               "source_sha": "synthetic_source", "checkpoint_sha": checkpoint_hash,
                               "implementation_sha": "synthetic_implementation",
                               "checkpoint_path": str(common), "native_recipe": recipe,
                               "selected_variant": variants[0], "selected_step": steps}
    freeze = {"phase": "A", "protocol_hash": protocol_hash, "registry_hash": "r",
              "development_participants": ["D:01"],
              "audit_a_participants": ["SC:01", "ST:01"], "audit_b_participants": ["SC:02", "ST:02"],
              "split_id": "fixture-split", "readiness_id": "fixture-readiness",
              "split_sha256": "c" * 64, "readiness_sha256": digest(readiness_path),
              "registry_sha256": "e" * 64, "readiness_path": str(readiness_path),
              "protocol_path": str(protocol_path), "protocol_sha256": digest(protocol_path),
              "audit_recordings": audit_recordings,
              "candidate_id": "candidate", "selection_frozen": True,
              "selected_models": selected, "audit_participants": ["SC:01", "ST:01"],
              "truth_artifact_sha256": {str(path): digest(path) for path in truth_paths}}
    args = dict(phase="A", protocol_hash=protocol_hash, registry_hash="r", freeze_manifest=freeze,
                run_evidence=runs, truth_paths=truth_paths,
                participant_cohort={"SC:01": "SC", "ST:01": "ST"},
                development_participants=["D:01"], candidate_id="candidate", draws=40,
                synthetic_test=True)
    return args


class GateTests(unittest.TestCase):
    def test_structured_native_evidence_rejects_markdown_flags_and_missing_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            markdown = root / "pretend.pt"
            markdown.write_text("# A checkpoint claim\n" * 20, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "native model artifact|text document"):
                _checkpoint(markdown, digest(markdown))
        for replacement, message in (("# adequate\n", "structured JSON"),
                                     (json_text({"baseline_adequate": True}), "structured JSON")):
            with tempfile.TemporaryDirectory() as tmp:
                args = fixture(Path(tmp))
                item = args["run_evidence"]["yasa_native"]
                run_path = Path(item["path"])
                run = read_json(run_path)
                evidence = Path(run["adequacy_evidence_path"])
                evidence.write_text(replacement, encoding="utf-8")
                run["adequacy_evidence_sha256"] = digest(evidence)
                run_path.write_text(json_text(run), encoding="utf-8")
                item["sha256"] = digest(run_path)
                with self.assertRaisesRegex(ValueError, message):
                    compute_gate(**args)
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            item = args["run_evidence"]["yasa_native"]
            path = Path(item["path"])
            run = read_json(path)
            run["fitted_participants"] = []
            path.write_text(json_text(run), encoding="utf-8")
            item["sha256"] = digest(path)
            with self.assertRaisesRegex(ValueError, "fit ancestry"):
                compute_gate(**args)
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            item = args["run_evidence"]["yasa_native"]
            run_path = Path(item["path"])
            run = read_json(run_path)
            evidence_path = Path(run["overlap_evidence_path"])
            evidence = read_json(evidence_path)
            evidence["source_training_participants"] = [{}]
            evidence_path.write_text(json_text(evidence), encoding="utf-8")
            run["overlap_evidence_sha256"] = digest(evidence_path)
            run_path.write_text(json_text(run), encoding="utf-8")
            item["sha256"] = digest(run_path)
            with self.assertRaisesRegex(ValueError, "overlap review"):
                compute_gate(**args)

    def test_authoritative_cohort_protocol_and_inference_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            args["participant_cohort"]["ST:01"] = "SC"
            with self.assertRaisesRegex(ValueError, "authoritative readiness"):
                compute_gate(**args)
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            protocol_path = Path(args["freeze_manifest"]["protocol_path"])
            protocol = read_json(protocol_path)
            protocol["audit_recordings"]["A"].pop("night_st")
            protocol["protocol_hash"] = content_id({k: v for k, v in protocol.items() if k != "protocol_hash"})
            protocol_path.write_text(json_text(protocol), encoding="utf-8")
            args["freeze_manifest"]["protocol_sha256"] = digest(protocol_path)
            with self.assertRaisesRegex(ValueError, "frozen protocol"):
                compute_gate(**args)
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            item = args["run_evidence"]["yasa_native"]
            run_path = Path(item["path"])
            run = read_json(run_path)
            evidence_path = Path(run["inference_evidence_path"])
            evidence = read_json(evidence_path)
            evidence["records"].pop()
            evidence_path.write_text(json_text(evidence), encoding="utf-8")
            run["inference_evidence_sha256"] = digest(evidence_path)
            run_path.write_text(json_text(run), encoding="utf-8")
            item["sha256"] = digest(run_path)
            with self.assertRaisesRegex(ValueError, "every audit recording"):
                compute_gate(**args)

    def test_exact_absolute_margin_uses_strongest_not_average_or_relative(self):
        self.assertFalse(passes_absolute_margin(Fraction(70, 100),
                          [Fraction(681, 1000), Fraction(1, 10)]))
        self.assertTrue(passes_absolute_margin(Fraction(70, 100),
                         [Fraction(68, 100), Fraction(1, 10)]))
        self.assertFalse(passes_absolute_margin(Fraction(70, 100), [Fraction(689, 1000)]))

    def test_synthetic_report_never_unlocks_and_status_tamper_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = fixture(root)
            report = compute_gate(**args)
            self.assertEqual(report["status"], "NOT_CONFIRMATORY")
            self.assertTrue(report["flags"]["all_executed"])
            self.assertTrue(report["flags"]["point_margin_pass"])
            self.assertTrue(report["flags"]["positive_superiority_supported"])
            report["status"] = "PASSED"
            path = root / "gate_A.json"
            path.write_text(json_text(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                require_gate(path, attestation_path=root / "missing_attestation.json", **args)

    def test_attestation_binds_separate_invocation_and_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = compute_gate(**fixture(root))
            report_path = root / "gate_A.json"
            report_path.write_text(json_text(report), encoding="utf-8")
            entrypoint = root / "independent_verifier.py"
            entrypoint.write_text("# Synthetic verifier identity fixture.\n", encoding="utf-8")
            result_path = root / "verification_result.json"
            result = {"schema_version": "1.0", "status": "VERIFIED", "invocation_id": "fixture-1",
                      "report_sha256": digest(report_path),
                      "protocol_hash": report["protocol_hash"],
                      "registry_hash": report["registry_hash"],
                      "freeze_manifest_hash": report["freeze_manifest_hash"],
                      "truth_artifact_sha256": report["truth_artifact_sha256"],
                      "run_record_sha256": report["run_record_sha256"]}
            result_path.write_text(json_text(result), encoding="utf-8")
            attestation_path = root / "attestation.json"
            attestation = {"schema_version": "1.0", "artifact_type": "independent_gate_verification",
                           "report_sha256": digest(report_path), "verifier_id": "test-verifier",
                           "invocation_id": "fixture-1", "entrypoint_path": str(entrypoint),
                           "entrypoint_sha256": digest(entrypoint), "argv": ["python", str(entrypoint)],
                           "exit_code": 0, "result_path": str(result_path),
                           "result_sha256": digest(result_path)}
            attestation_path.write_text(json_text(attestation), encoding="utf-8")
            _verify_attestation(attestation_path, report_path, report)
            result["status"] = "FAILED"
            result_path.write_text(json_text(result), encoding="utf-8")
            attestation["result_sha256"] = digest(result_path)
            attestation_path.write_text(json_text(attestation), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verifier result"):
                _verify_attestation(attestation_path, report_path, report)

    def test_missing_slot_or_leakage_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            del args["run_evidence"]["tinysleepnet_official"]
            with self.assertRaisesRegex(ValueError, "exactly"):
                compute_gate(**args)
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            evidence = args["run_evidence"]["yasa_native"]
            path = Path(evidence["path"])
            run = read_json(path)
            run["teacher_participants"] = ["SC:01"]
            path.write_text(json_text(run), encoding="utf-8")
            evidence["sha256"] = digest(path)
            with self.assertRaisesRegex(ValueError, "audit participant"):
                compute_gate(**args)
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            evidence = args["run_evidence"]["yasa_native"]
            path = Path(evidence["path"])
            run = read_json(path)
            run["pretraining_participants"] = ["SC:02"]  # B is sealed even during A.
            path.write_text(json_text(run), encoding="utf-8")
            evidence["sha256"] = digest(path)
            with self.assertRaisesRegex(ValueError, "audit participant"):
                compute_gate(**args)

    def test_frozen_record_inventory_candidate_role_and_draw_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            args["freeze_manifest"]["audit_recordings"]["missing_second_night"] = {
                **args["freeze_manifest"]["audit_recordings"]["night_sc"]}
            with self.assertRaisesRegex(ValueError, "original frozen protocol"):
                compute_gate(**args)
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            args["candidate_id"] = "yasa_native"
            with self.assertRaisesRegex(ValueError, "distinct"):
                compute_gate(**args)
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            args["synthetic_test"] = False
            args["draws"] = 1
            with self.assertRaisesRegex(ValueError, "10,000"):
                compute_gate(**args)

    def test_exact_duplicate_requires_matching_artifacts_and_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp), duplicate=True)
            self.assertEqual(compute_gate(**args)["status"], "NOT_CONFIRMATORY")
            evidence = args["run_evidence"]["sleepyland_yasa"]
            path = Path(evidence["path"])
            run = read_json(path)
            run["implementation_sha"] = "other"
            path.write_text(json_text(run), encoding="utf-8")
            evidence["sha256"] = digest(path)
            with self.assertRaisesRegex(ValueError, "differs from frozen selection"):
                compute_gate(**args)


if __name__ == "__main__":
    unittest.main()
