import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from sleepedf.evaluation import (evaluate_saved_records, macro_f1_fraction,
                                 paired_cluster_intervals)
from sleepedf.predictions import save_prediction
from test_predictions import write_truth


def prediction(path, participant, recording, hard, epochs=None):
    hard = np.asarray(hard, dtype=np.int8)
    if epochs is None:
        epochs = np.arange(len(hard), dtype=np.int64)
    save_prediction(path, participant_id=participant, recording_id=recording,
                    epoch_index=epochs, onset_seconds=epochs.astype(float) * 30,
                    hard_label=hard, model_id="candidate", protocol_hash="p",
                    registry_hash="r", provenance={})
    return path


class EvaluationTests(unittest.TestCase):
    def test_fixed_five_pooled_macro_f1_exact(self):
        cm = np.array([[2, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], dtype=int)
        # Explicit independent class-wise calculation: 1, 2/3, 0, 1, 1.
        self.assertEqual(macro_f1_fraction(cm), Fraction(11, 15))
        self.assertEqual(macro_f1_fraction(np.zeros((5, 5), dtype=int)), 0)
        with self.assertRaises(ValueError):
            macro_f1_fraction(np.zeros((4, 4), dtype=int))

    def test_no_evaluable_participant_and_undefined_kappa(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth = write_truth(root / "all_invalid.npz", (-1, -1), "SC:01", "night1")
            pred = prediction(root / "pred.npz", "SC:01", "night1", (0, 0))
            with self.assertRaisesRegex(ValueError, "positive reference-valid support"):
                evaluate_saved_records([truth], [pred])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth = write_truth(root / "wake.npz", (0, 0), "SC:01", "night1")
            pred = prediction(root / "pred.npz", "SC:01", "night1", (0, 0))
            score = evaluate_saved_records([truth], [pred])
            self.assertIsNone(score["kappa"])
            self.assertEqual(score["kappa_unavailable_reason"], "zero_expected_disagreement")

    def test_full_timeline_alignment_and_secondary_participant_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            t1 = write_truth(root / "t1.npz", (0, 0, -1), "SC:01", "n1")
            t2 = write_truth(root / "t2.npz", (1, 1), "ST:02", "n2")
            p1 = prediction(root / "p1.npz", "SC:01", "n1", (0, 0, 3))
            p2 = prediction(root / "p2.npz", "ST:02", "n2", (0, 0))
            result = evaluate_saved_records([t1, t2], [p2, p1], protocol_hash="p",
                                            registry_hash="r", model_id="candidate")
            self.assertEqual(result["evaluated_epochs"], 4)
            self.assertEqual(result["invalid_epochs"], 1)
            self.assertEqual(result["support"], [2, 2, 0, 0, 0])
            self.assertEqual(result["predicted_support"], [4, 0, 0, 0, 0])
            self.assertEqual(result["per_class_precision"], [0.5, 0, 0, 0, 0])
            self.assertEqual(result["per_class_recall"], [1, 0, 0, 0, 0])
            self.assertEqual(result["complete_psg_epochs"], 5)
            self.assertEqual(result["reference_valid_fraction"], 0.8)
            self.assertEqual(result["prediction_coverage_fraction"], 1.0)
            self.assertEqual(result["macro_f1_exact"], "2/15")  # W F1 2/3, five-class mean
            self.assertNotEqual(result["macro_f1"], result["participant_macro_f1_mean"])
            dropped = prediction(root / "dropped.npz", "SC:01", "n1", (0, 0),
                                 epochs=np.array([0, 1], dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "epoch_index"):
                evaluate_saved_records([t1], [dropped])

    def test_paired_bootstrap_draws_people_and_is_deterministic(self):
        c = {"per_participant_confusion": {
            "SC:01": np.diag([2, 1, 1, 1, 1]).tolist(),
            "SC:02": np.diag([1, 2, 1, 1, 1]).tolist(),
            "ST:01": np.diag([1, 1, 2, 1, 1]).tolist()}}
        b = {"per_participant_confusion": {
            "SC:01": [[2, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
            "SC:02": [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
            "ST:01": [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 2, 0, 0],
                      [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]}}
        cohorts = {"SC:01": "SC", "SC:02": "SC", "ST:01": "ST"}
        one = paired_cluster_intervals(c, {"baseline": b}, cohorts, phase="A", draws=100)
        two = paired_cluster_intervals(c, {"baseline": b}, cohorts, phase="A", draws=100)
        self.assertEqual(one, two)
        self.assertEqual(one["cohort_counts"], {"SC": 2, "ST": 1})
        self.assertEqual(one["seed"], 2026092301)
        self.assertGreater(one["intervals"]["baseline"]["lower"], 0)
        with self.assertRaises(ValueError):
            paired_cluster_intervals(c, {"baseline": b}, {"SC:01": "SC"}, phase="A", draws=10)


if __name__ == "__main__":
    unittest.main()
