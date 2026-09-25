"""Synthetic guard tests; no audit references, EDFs, or models are opened."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from sleepedf import exploratory_audits as audits
from sleepedf.research import file_sha256


class ExploratoryFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.data = self.root / "original"
        self.data.mkdir()
        self.psg = self.data / "SC4001-PSG.edf"
        self.psg.write_bytes(b"offline synthetic PSG placeholder")
        self.selection_path = self.root / audits.SELECTION_REL
        self.selection = {
            "run_dir": str(self.root / "runs/exploratory-audits-v1"),
            "data_root": str(self.data), "model_path": str(self.root / audits.MODEL_REL),
            "project_root": str(self.root), "model_id": "sha256:" + audits.MODEL_SHA,
            "selection_id": "sha256:" + "a" * 64,
            "source_sha256": {"sleepedf/exploratory_audits.py": "b" * 64},
            "bootstrap": audits.BOOTSTRAP,
        }
        self.protocol = {"protocol_hash": "sha256:" + "c" * 64,
                         "registry_hash": "sha256:" + "d" * 64,
                         "bootstrap": {"draws": 10000, "rng": "PCG64",
                                       "strata": ["SC", "ST"],
                                       "cluster": "participant_all_nights",
                                       "seeds": {"A": 2026092301, "B": 2026092302}}}
        self.row = {"recording_id": "SC4001", "participant_id": "SC:00",
                    "n_epochs": 2, "psg": self.psg.name,
                    "psg_sha256": file_sha256(self.psg)}
        self.hard = np.array([0, 4], dtype=np.int8)
        self.prob = np.array([[.8, .1, .05, .03, .02],
                              [.02, .03, .05, .1, .8]], dtype=np.float64)


class TestPrediction(ExploratoryFixture):
    def test_signal_only_atomic_commit_resume_and_native_recheck(self):
        groups = {"A": [self.row], "B": []}
        with mock.patch.object(audits, "_selection", return_value=self.selection), \
             mock.patch.object(audits, "_records", return_value=(self.protocol, groups)), \
             mock.patch.object(audits, "_require_lease"), \
             mock.patch.object(audits, "_require_retired"), \
             mock.patch.object(audits, "predict", return_value=(self.hard, self.prob)) as native:
            first = audits.predict_one(self.root, self.selection_path, self.row["recording_id"])
            path = audits._prediction_path(self.selection, self.row["recording_id"])
            self.assertTrue(path.is_file())
            self.assertEqual(first["prediction_sha256"], file_sha256(path))
            self.assertEqual(native.call_count, 1)
            self.assertEqual(audits.predict_one(self.root, self.selection_path,
                                                self.row["recording_id"]), first)
            self.assertEqual(native.call_count, 1)
            audits.predict_one(self.root, self.selection_path, self.row["recording_id"],
                               recompute_existing=True)
            self.assertEqual(native.call_count, 2)
            native.return_value = (np.array([4, 4], dtype=np.int8), self.prob)
            with self.assertRaises(ValueError):
                audits.predict_one(self.root, self.selection_path, self.row["recording_id"],
                                   recompute_existing=True)
            self.assertEqual(file_sha256(path), first["prediction_sha256"])
            self.assertFalse(list(path.parent.parent.glob(".partial-*")))

    def test_incomplete_predictions_stop_before_any_reference_exposure(self):
        groups = {"A": [self.row], "B": []}
        with mock.patch.object(audits, "_selection", return_value=self.selection), \
             mock.patch.object(audits, "_records", return_value=(self.protocol, groups)), \
             mock.patch.object(audits, "_require_lease"), \
             mock.patch.object(audits, "_require_retired"), \
             mock.patch.object(audits, "append_event") as exposure, \
             mock.patch.object(audits, "evaluate_saved_records") as score:
            with self.assertRaises(FileNotFoundError):
                audits.evaluate(self.root, self.selection_path)
            exposure.assert_not_called()
            score.assert_not_called()

    def test_prediction_without_successful_child_receipt_cannot_expose_reference(self):
        groups = {"A": [self.row], "B": []}
        with mock.patch.object(audits, "_selection", return_value=self.selection), \
             mock.patch.object(audits, "_records", return_value=(self.protocol, groups)), \
             mock.patch.object(audits, "_require_lease"), \
             mock.patch.object(audits, "_require_retired"), \
             mock.patch.object(audits, "predict", return_value=(self.hard, self.prob)), \
             mock.patch.object(audits, "append_event") as exposure, \
             mock.patch.object(audits, "evaluate_saved_records") as score:
            audits.predict_one(self.root, self.selection_path, self.row["recording_id"])
            with self.assertRaises(FileNotFoundError):
                audits.evaluate(self.root, self.selection_path)
            exposure.assert_not_called()
            score.assert_not_called()

    def test_descriptive_interval_is_single_model_participant_cluster(self):
        cm_sc = [[0] * 5 for _ in range(5)]
        cm_st = [[0] * 5 for _ in range(5)]
        cm_sc[0][0], cm_st[4][4] = 2, 3
        result = {"per_participant_confusion": {"SC:00": cm_sc, "ST:00": cm_st}}
        interval = audits._descriptive_interval(result, "A", self.protocol, self.selection)
        self.assertEqual(interval["cohort_counts"], {"SC": 1, "ST": 1})
        self.assertEqual(interval["draws"], 10000)
        self.assertEqual(interval["lower"], interval["upper"])
        self.assertEqual(interval["scope"], "exploratory_descriptive_not_paired_margin_or_gate")
        changed = dict(self.selection, bootstrap=dict(audits.BOOTSTRAP, draws=9999))
        with self.assertRaises(ValueError):
            audits._descriptive_interval(result, "A", self.protocol, changed)


class TestRoster(ExploratoryFixture):
    def test_audit_roster_requires_all_78_and_exact_d60_training_membership(self):
        development = [f"SC:{i:02d}" for i in range(60)]
        rows = []
        groups = {"A": [], "B": []}
        for i in range(78):
            phase = "A" if i < 39 else "B"
            row = {"recording_id": f"R{i:04d}", "participant_id": f"ST:{i:02d}",
                   "n_epochs": 2, "psg_sha256": "a" * 64,
                   "hypnogram_sha256": "b" * 64}
            rows.append(row)
            groups[phase].append(row)
        protocol = {"protocol_hash": self.protocol["protocol_hash"],
                    "registry_hash": self.protocol["registry_hash"],
                    "audit_recordings": {phase: {r["recording_id"]: {
                        "participant_id": r["participant_id"], "n_epochs": r["n_epochs"],
                        "psg_sha256": r["psg_sha256"],
                        "hypnogram_sha256": r["hypnogram_sha256"],
                        "truth_payload_sha256": "c" * 64,
                        "truth_sidecar_sha256": "d" * 64}
                        for r in members} for phase, members in groups.items()}}
        split = {"split_id": "sha256:" + "e" * 64,
                 "participants": {"development": development,
                                  "audit_a": [r["participant_id"] for r in groups["A"]],
                                  "audit_b": [r["participant_id"] for r in groups["B"]]}}
        selection = {"protocol_hash": protocol["protocol_hash"],
                     "registry_hash": protocol["registry_hash"],
                     "split_id": split["split_id"]}
        original_read = audits.read_json
        fitted = development.copy()
        fit_path = self.root / audits.FIT_REL
        fit_path.parent.mkdir(parents=True)
        fit_path.write_text("{}")
        def read(path):
            if Path(path) == fit_path:
                return {"fitted_participants": fitted}
            return original_read(path)
        with mock.patch.object(audits, "load_protocol", return_value=(protocol, split,
                                                                        {"records": rows})), \
             mock.patch.object(audits, "read_json", side_effect=read):
            observed, by_phase = audits._records(self.root, selection)
            self.assertEqual(sum(map(len, by_phase.values())), 78)
            self.assertIs(observed, protocol)
            fitted[0] = "SC:99"
            with self.assertRaises(ValueError):
                audits._records(self.root, selection)
        self.assertFalse((self.root / "runs/exposure.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
