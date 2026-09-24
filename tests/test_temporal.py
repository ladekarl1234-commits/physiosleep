import itertools
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sleepedf.temporal import decode, fit_fold_priors, fit_priors, objective


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.priors = fit_priors([np.asarray([0, 1, 1, 2, 2, 2, 2, 3, 4, 4, 0])], cutoff=2)

    def test_exact_cost_matches_exhaustive_sequences_and_zero_emissions(self):
        rng = np.random.default_rng(1701)
        for zeros in (False, True):
            for repetition in range(3):
                p = rng.dirichlet(np.ones(5), size=4 if not zeros else 8)
                if zeros:
                    for row in p:
                        row[rng.choice(5, 3, replace=False)] = 0
                    p /= p.sum(axis=1, keepdims=True)
                choices = [np.flatnonzero(row > 0) for row in p]
                for lam, mu in ((0, 0), (0.4, 0), (0, 1.3), (0.4, 1.3)):
                    actual = decode(p, self.priors, transition_weight=lam, duration_weight=mu)
                    minimum = min(objective(p, np.asarray(y), self.priors,
                                            transition_weight=lam, duration_weight=mu)
                                  for y in itertools.product(*choices))
                    self.assertAlmostEqual(actual["objective"], minimum, places=10)
                    self.assertAlmostEqual(objective(p, actual["hard_label"], self.priors,
                                                     transition_weight=lam, duration_weight=mu), minimum, places=10)

    def test_long_tail_has_no_duration_limit(self):
        p = np.zeros((1000, 5))
        p[:, 2] = 1
        result = decode(p, self.priors, transition_weight=0.3, duration_weight=2)
        np.testing.assert_array_equal(result["hard_label"], np.full(1000, 2))
        self.assertTrue(np.isfinite(result["objective"]))
        self.assertFalse(result["duration_truncated"])
        self.assertAlmostEqual(result["objective"], objective(p, np.full(1000, 2), self.priors,
                                                            transition_weight=0.3, duration_weight=2), places=9)

    def test_subnormal_positive_hazard_does_not_underflow_tail_cost(self):
        priors = fit_priors([np.asarray([0, 1, 2])], cutoff=1)
        priors["tail_hazard"] = [float(np.nextafter(0., 1.))] * 5
        p = np.zeros((5, 5))
        p[:, 2] = 1
        for mu in (0., .4):
            result = decode(p, priors, transition_weight=.4, duration_weight=mu)
            np.testing.assert_array_equal(result["hard_label"], np.full(5, 2))
            self.assertTrue(np.isfinite(result["objective"]))
            self.assertAlmostEqual(result["objective"], objective(p, np.full(5, 2), priors,
                                                                 transition_weight=.4, duration_weight=mu), places=10)

    def test_zero_weights_and_signal_boundaries(self):
        p = np.array([[.5, .5, 0, 0, 0], [.1, .1, .7, .1, 0], [0, 0, 0, 0, 1]])
        np.testing.assert_array_equal(decode(p, self.priors)["hard_label"], p.argmax(axis=1))
        starts = np.array([True, False, True])
        joined = decode(p, self.priors, transition_weight=.3, duration_weight=.4, block_start=starts)
        first = decode(p[:2], self.priors, transition_weight=.3, duration_weight=.4)
        second = decode(p[2:], self.priors, transition_weight=.3, duration_weight=.4)
        np.testing.assert_array_equal(joined["hard_label"], np.r_[first["hard_label"], second["hard_label"]])
        self.assertAlmostEqual(joined["objective"], first["objective"] + second["objective"])
        self.assertEqual(joined["block_count"], 2)
        self.assertEqual(decode(np.empty((0, 5)), self.priors)["block_count"], 0)

    def test_censoring_and_invalid_time_do_not_form_transitions_or_runs(self):
        result = fit_priors([np.asarray([0, 0, 1, 1, 2, -1, 3, 4, 4, 0])], cutoff=2)
        counts = np.asarray(result["transition_counts"])
        self.assertEqual(counts[2, 3], 0)
        durations = np.asarray(result["duration_counts"])
        self.assertEqual(durations.sum(), 2)
        self.assertEqual(durations[1, 1], 1)
        self.assertEqual(durations[4, 1], 1)
        self.assertTrue(np.all(np.asarray(result["transition_probability"]) > 0))
        self.assertTrue(np.all(np.asarray(result["tail_mass"]) > 0))

    def test_invalid_inputs_and_audit_membership_fail_before_truth_read(self):
        for labels in (np.array([], dtype=int), np.array([-1, -1])):
            with self.assertRaisesRegex(ValueError, "observed valid"):
                fit_priors([labels])
        with self.assertRaises(ValueError):
            decode(np.zeros((1, 5)), self.priors)
        with self.assertRaises(ValueError):
            decode(np.full((1, 5), .2), self.priors, transition_weight=-1)
        with self.assertRaises(ValueError):
            decode(np.full((1, 5), .2), self.priors, block_start=np.array([1]))
        split = {"folds": [{"train": [f"D:{i}" for i in range(47)] + ["A:01"]}]}
        records = [{"participant_id": f"D:{i}"} for i in range(60)]
        with patch("sleepedf.temporal.development_records", return_value=({}, split, records)), \
                patch("sleepedf.temporal.load_development_truth") as reader:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                fit_fold_priors(Path("."), 0)
            reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
