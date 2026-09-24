import unittest
import numpy as np
from sleepedf.sleep_summary import summarize, experimental_score


def summary(stages, **kwargs):
    return summarize(np.asarray(stages, dtype=np.int8), window_start_seconds=0,
                     window_end_seconds=kwargs.pop('end', len(stages) * 30),
                     window_provenance='synthetic declared interval', **kwargs)


class SleepSummaryTests(unittest.TestCase):
    def test_sleep_period_excludes_initial_and_terminal_wake(self):
        result = summary([0, 2, 0, 0, 3, 0])
        self.assertEqual(result['tst_minutes'], 1)
        self.assertEqual(result['spt_minutes'], 2)
        self.assertEqual(result['waso_minutes'], 1)
        self.assertEqual(result['awakening_count'], 1)
        self.assertEqual(result['continuity_fraction'], .5)
        self.assertIsNone(result['se_percent'])
        self.assertIsNone(result['sol_minutes'])

    def test_all_wake_and_one_sleep_epoch(self):
        wake = summary([0, 0])
        self.assertEqual(wake['tst_minutes'], 0)
        self.assertIsNone(wake['continuity_fraction'])
        self.assertEqual(wake['unavailable_reasons']['spt_minutes'], 'no_sleep')
        sleep = summary([0, 1, 0])
        self.assertEqual(sleep['spt_minutes'], .5)
        self.assertEqual(sleep['waso_minutes'], 0)
        self.assertEqual(sleep['continuity_fraction'], 1)

    def test_gaps_unknowns_and_partial_tail_never_disappear(self):
        gap = summary([2, 2], epoch_index=[0, 2], end=90)
        self.assertEqual(gap['coverage']['missing_seconds'], 30)
        self.assertIsNone(gap['tst_minutes'])
        self.assertEqual(gap['observed_tst_minutes'], 1)
        missing = summary([2, -1, 2])
        self.assertEqual(missing['coverage'], gap['coverage'])
        tail = summary([2, 2], end=80)
        self.assertEqual(tail['coverage']['fraction'], .75)
        self.assertIsNone(tail['spt_minutes'])

    def test_independent_anchors_and_bed_interval(self):
        result = summary([0, 2, 0, 3, 0, 0], attempt_to_sleep_seconds=0,
                         attempt_provenance='independent diary', time_in_bed=(0, 120),
                         time_in_bed_provenance='independent lights events')
        self.assertEqual(result['sol_minutes'], .5)
        self.assertEqual(result['time_in_bed_minutes'], 2)
        self.assertEqual(result['se_percent'], 50)
        self.assertIsNone(summary([2, 2], time_in_bed=(0, 90),
                                  time_in_bed_provenance='diary')['se_percent'])

    def test_no_sleep_after_attempt_has_no_invented_latency(self):
        result = summary([2, 0, 0], attempt_to_sleep_seconds=60, attempt_provenance='diary')
        self.assertIsNone(result['sol_minutes'])
        self.assertEqual(result['unavailable_reasons']['sol_minutes'], 'no_sleep_after_attempt_anchor')

    def test_subepoch_anchor_and_partial_epoch_window_do_not_imply_exact_measurement(self):
        anchor = summary([2, 0, 2], attempt_to_sleep_seconds=15, attempt_provenance='independent diary')
        self.assertIsNone(anchor['sol_minutes'])
        self.assertEqual(anchor['unavailable_reasons']['sol_minutes'], 'attempt_anchor_not_epoch_aligned')
        partial = summarize([2], window_start_seconds=15, window_end_seconds=30,
                            window_provenance='synthetic partial epoch')
        self.assertEqual(partial['coverage']['missing_seconds'], 15)
        self.assertIsNone(partial['tst_minutes'])
        bed = summary([2, 0, 2], time_in_bed=(15, 90), time_in_bed_provenance='diary')
        self.assertIsNone(bed['se_percent'])
        self.assertEqual(bed['unavailable_reasons']['se_percent'], 'time_in_bed_boundaries_not_epoch_aligned')

    def test_invalid_values_and_unproven_anchors_rejected(self):
        for stages in ([5], [-2]):
            with self.assertRaises(ValueError):
                summary(stages)
        for kwargs in ({'epoch_index': [1, 1]}, {'valid_mask': [1, 1]},
                       {'attempt_to_sleep_seconds': 0}, {'time_in_bed': (0, 60)}, {'end': float('nan')}):
            with self.assertRaises(ValueError):
                summary([0, 2], **kwargs)


class ExperimentalScoreTests(unittest.TestCase):
    def test_changed_parameters_have_distinct_identity_and_boolean_cannot_certify_night(self):
        raw = summary([2] * 960, independent_night_boundaries=True)
        primary = experimental_score(raw)
        changed = experimental_score(raw, target_minutes=450)
        self.assertNotEqual(primary['score_definition'], changed['score_definition'])
        self.assertNotEqual(primary['score_config_id'], changed['score_config_id'])
        self.assertIsNone(primary['whole_night_score'])

    def test_rounded_coverage_cannot_hide_a_small_unobserved_tail(self):
        result = experimental_score(summary([2] * 960, end=28800 + 1e-8))
        self.assertIsNone(result['score'])
        self.assertIn('insufficient_coverage', result['unavailable_reasons'])
    def test_exact_component_arithmetic(self):
        result = experimental_score(summary([0] * 60 + [2] * 420 + [0] * 120 + [3] * 420 + [0] * 60))
        self.assertAlmostEqual(result['score'], 93.5414346693, places=9)
        self.assertEqual(result['duration_component'], 100)
        self.assertEqual(result['continuity_component'], 87.5)
        self.assertIsNone(result['whole_night_score'])

    def test_short_observation_missing_time_and_all_wake_are_unavailable(self):
        for stages, kwargs, reason in (([2] * 720, {}, 'short_observation'),
                                      ([0] * 960, {}, 'no_sleep'),
                                      ([2] * 960, {'end': 28820}, 'insufficient_coverage'),
                                      ([2] * 959 + [-1], {}, 'insufficient_coverage')):
            result = experimental_score(summary(stages, **kwargs))
            self.assertIsNone(result['score'])
            self.assertIn(reason, result['unavailable_reasons'])
        wake = experimental_score(summary([0] * 960))
        self.assertEqual(wake['duration_component'], 0)
        self.assertIsNone(wake['continuity_component'])

    def test_geometric_avoids_arithmetic_compensation_floor(self):
        score = experimental_score(summary([0] * 479 + [2] + [0] * 480))
        self.assertAlmostEqual(score['score'], 3.4503277967, places=9)
        self.assertAlmostEqual(score['arithmetic_sensitivity_score'], 50.0595238095, places=9)

    def test_stage_substitution_and_external_wake_invariance(self):
        one = experimental_score(summary([0] * 60 + [1] * 840 + [0] * 60))
        two = experimental_score(summary([0] * 120 + [4] * 840 + [0] * 120))
        self.assertEqual(one['score'], two['score'])

    def test_boundary_error_can_reduce_score_when_sleep_minutes_increase(self):
        labels = [0] * 120 + [2] * 720 + [0] * 120
        before = experimental_score(summary(labels))
        labels[0] = 2
        after = experimental_score(summary(labels))
        self.assertAlmostEqual(before['score'], 92.5820099773, places=9)
        self.assertAlmostEqual(after['score'], 85.8333333333, places=9)
        self.assertLess(after['score'], before['score'])

    def test_internal_wake_replacement_increases_score_at_fixed_boundaries(self):
        labels = [2] * 420 + [0] * 120 + [2] * 420
        first = experimental_score(summary(labels))
        labels[450] = 2
        second = experimental_score(summary(labels))
        self.assertGreater(second['score'], first['score'])

    def test_wake_bout_count_is_not_a_hidden_score_component(self):
        first = summary([2] * 420 + [0] * 120 + [2] * 420)
        second = summary([2] * 210 + [0] * 60 + [2] * 420 + [0] * 60 + [2] * 210)
        self.assertEqual(experimental_score(first)['score'], experimental_score(second)['score'])
        self.assertEqual((first['awakening_count'], second['awakening_count']), (1, 2))


if __name__ == '__main__':
    unittest.main()
