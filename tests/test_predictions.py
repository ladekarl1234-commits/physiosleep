import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sleepedf.contracts import json_text
from sleepedf.predictions import load_prediction, load_truth, save_prediction


def write_truth(path, labels=(0, 1, -1, 4), participant="SC:01", recording="night1"):
    labels = np.asarray(labels, dtype=np.int8)
    np.savez_compressed(path, participant_id=np.asarray(participant), recording_id=np.asarray(recording),
                        epoch_index=np.arange(len(labels), dtype=np.int64),
                        onset_seconds=np.arange(len(labels), dtype=np.float64) * 30,
                        reference_label=labels, valid_mask=labels >= 0)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json_text({"schema_version": "sleepedf-truth-v1",
        "payload_sha256": digest, "source_psg_sha256": "a" * 64,
        "source_hypnogram_sha256": "b" * 64, "participant_id": participant,
        "recording_id": recording, "timing_tolerance_us": 0, "epoch_seconds": 30,
        "reader_evidence": {"fixture": True}, "valid_epochs": int((labels >= 0).sum()),
        "n_epochs": len(labels)}),
        encoding="utf-8")
    return path


def write_prediction(path, labels=(0, 1, 2, 4), **overrides):
    kwargs = dict(participant_id="SC:01", recording_id="night1",
                  epoch_index=np.arange(len(labels), dtype=np.int64),
                  onset_seconds=np.arange(len(labels), dtype=np.float64) * 30,
                  hard_label=np.asarray(labels, dtype=np.int8), model_id="candidate",
                  protocol_hash="sha256:protocol", registry_hash="sha256:registry", provenance={})
    kwargs.update(overrides)
    save_prediction(path, **kwargs)
    return path


class PredictionArtifactTests(unittest.TestCase):
    def test_signal_only_roundtrip_keeps_invalid_time_and_private_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_path = write_prediction(root / "pred.npz")
            truth_path = write_truth(root / "truth.npz")
            meta, pred = load_prediction(pred_path)
            _, truth = load_truth(truth_path)
            self.assertEqual(meta["class_order"], ["W", "N1", "N2", "N3", "REM"])
            self.assertEqual(len(pred["epoch_index"]), 4)
            self.assertNotIn("reference_label", pred)
            self.assertEqual(truth["reference_label"][2], -1)

    def test_rejects_nonfinite_probability_metadata_and_wrong_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                write_prediction(root / "bad.npz", probabilities=np.full((4, 5), np.nan))
            with self.assertRaises(ValueError):
                write_prediction(root / "bad.npz", provenance={"threshold": float("nan")})
            with self.assertRaises(ValueError):
                write_prediction(root / "bad.npz", provenance={"reference_label": [0]})
            pred_path = write_prediction(root / "good.npz")
            sidecar = pred_path.with_suffix(".json")
            sidecar.write_text(sidecar.read_text(encoding="utf-8").replace('"REM"', '"R"'),
                               encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "class order"):
                load_prediction(pred_path)

    def test_rejects_hash_mismatch_and_missing_truth_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_path = write_prediction(root / "pred.npz")
            with pred_path.open("ab") as out:
                out.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_prediction(pred_path)
            truth_path = write_truth(root / "truth.npz")
            np.savez_compressed(truth_path, participant_id=np.asarray("SC:01"),
                                recording_id=np.asarray("night1"), epoch_index=np.arange(4),
                                onset_seconds=np.arange(4) * 30., reference_label=np.array([0, 1, 2, 4]))
            digest = hashlib.sha256(truth_path.read_bytes()).hexdigest()
            truth_path.with_suffix(".json").write_text(json_text({"payload_sha256": digest}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_truth(truth_path)


if __name__ == "__main__":
    unittest.main()
