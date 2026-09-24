"""Synthetic private-truth and signal-only interface checks."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sleepedf.dataset import build_readiness, iter_signal_epochs
from sleepedf.readers import PHYSIOLOGICAL_CHANNELS
from test_audit import write_edf


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DatasetTests(unittest.TestCase):
    def test_private_truth_and_signal_only_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root, private = base / "source", base / "private"
            root.mkdir()
            psg_name, hyp_name = "SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf"
            write_edf(root / psg_name, duration=60,
                      channels=[(name, "uV", 1 if name == "EMG submental" else 100)
                                for name in PHYSIOLOGICAL_CHANNELS])
            write_edf(root / hyp_name, annotation=True)
            names = [psg_name, hyp_name]
            for index in range(396):
                name = f"aux-{index:03}.txt"
                (root / name).write_text("fixture", encoding="ascii")
                names.append(name)
            (root / "SHA256SUMS.txt").write_text(
                "".join(f"{digest(root / name)} {name}\n" for name in names), encoding="utf-8")
            inventory = {"report_kind": "audit", "manifest_id": "fixture",
                         "recordings": [{"recording_id": "SC4001", "participant_id": "SC:00",
                                         "cohort": "SC", "night": 1,
                                         "identity_status": "resolved",
                                         "psg": psg_name, "hypnogram": hyp_name,
                                         "evidence": {"pair_unique": True,
                                                      "metadata_verified": True}}]}
            from sleepedf.contracts import inventory_id
            inventory["manifest_id"] = inventory_id(inventory)
            inventory_path = base / "inventory.json"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            result = build_readiness(inventory_path, root, private,
                                     digest(root / "SHA256SUMS.txt"),
                                     recording_ids={"SC4001"})
            self.assertEqual(result["errors"], [])
            record = result["records"][0]
            self.assertEqual(record["n_epochs"], 2)
            self.assertTrue(all(record["checks"].values()))
            with np.load(record["truth_path"], allow_pickle=False) as truth:
                self.assertEqual(truth["epoch_index"].tolist(), [0, 1])
                self.assertEqual(truth["reference_label"].tolist(), [-1, -1])
                self.assertEqual(truth["valid_mask"].tolist(), [False, False])
            sidecar = json.loads((private / "SC4001.json").read_text())
            self.assertEqual(sidecar["payload_sha256"], digest(Path(record["truth_path"])))
            public_record = dict(record)
            public_record.pop("truth_path")
            self.assertNotIn("reference_label", str(public_record))
            epochs = list(iter_signal_epochs(record, root))
            self.assertEqual([e["epoch_index"] for e in epochs], [0, 1])
            self.assertEqual(len(epochs[0]["channels"]["EEG Fpz-Cz"]["samples_uv"]), 3000)
            first_hash = digest(Path(record["truth_path"]))
            repeated = build_readiness(inventory_path, root, private,
                                       digest(root / "SHA256SUMS.txt"),
                                       recording_ids={"SC4001"})
            self.assertEqual(repeated["errors"], [])
            self.assertEqual(digest(Path(record["truth_path"])), first_hash)
            (private / "SC4001.json").write_text("{}", encoding="utf-8")
            blocked = build_readiness(inventory_path, root, private,
                                      digest(root / "SHA256SUMS.txt"),
                                      recording_ids={"SC4001"})
            self.assertEqual(len(blocked["errors"]), 1)
            self.assertEqual(digest(Path(record["truth_path"])), first_hash)

    def test_swapped_hypnogram_identity_is_rejected_even_when_hash_matches(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "source"
            root.mkdir()
            psg_name, hyp_name = "SC4001E0-PSG.edf", "SC4002EC-Hypnogram.edf"
            (root / psg_name).write_bytes(b"a")
            (root / hyp_name).write_bytes(b"b")
            inventory = {"report_kind": "audit", "manifest_id": "fixture",
                         "recordings": [{"recording_id": "SC4001", "participant_id": "SC:00",
                                         "cohort": "SC", "night": 1,
                                         "identity_status": "resolved", "psg": psg_name,
                                         "hypnogram": hyp_name,
                                         "evidence": {"pair_unique": True,
                                                      "metadata_verified": True}}]}
            from sleepedf.contracts import inventory_id
            inventory["manifest_id"] = inventory_id(inventory)
            inventory_path = base / "inventory.json"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            with patch("sleepedf.dataset._checksum_entries", return_value={
                psg_name: digest(root / psg_name), hyp_name: digest(root / hyp_name)}):
                report = build_readiness(inventory_path, root, base / "private",
                                         "0" * 64, recording_ids={"SC4001"})
            self.assertEqual(len(report["errors"]), 1)
            self.assertFalse(report["records"][0]["checks"]["content_verified"])


if __name__ == "__main__":
    unittest.main()
