import unittest

from sleepedf.timing import EPOCH_US, project_annotations, to_microseconds


W = "Sleep stage W"
N2 = "Sleep stage 2"


class TimingProjectionTests(unittest.TestCase):
    def test_long_interval_and_adjacent_same_class(self):
        result = project_annotations("SC0001", 90, [(0, 45, N2), (45, 45, N2)])
        self.assertEqual([e["label"] for e in result["epochs"]], [2, 2, 2])
        self.assertEqual([e["index"] for e in result["epochs"]], [0, 1, 2])

    def test_offset_and_original_grid(self):
        result = project_annotations("ST0001", 60, [(-10, 30, W), (20, 30, N2)], offset_seconds=10)
        self.assertEqual([e["label"] for e in result["epochs"]], [0, 2])
        self.assertEqual(result["epochs"][1]["onset_us"], EPOCH_US)

    def test_gap_overlap_and_conflict_remain_in_place(self):
        gap = project_annotations("r", 60, [(0, 29, W), (30, 30, N2)])
        self.assertEqual(gap["epochs"][0]["invalid_reasons"], ["annotation_gap"])
        self.assertEqual(gap["epochs"][1]["label"], 2)
        overlap = project_annotations("r", 30, [(0, 30, W), (10, 10, W)])
        self.assertIn("annotation_overlap", overlap["epochs"][0]["invalid_reasons"])
        conflict = project_annotations("r", 30, [(0, 15, W), (15, 15, N2)])
        self.assertIn("label_conflict", conflict["epochs"][0]["invalid_reasons"])

    def test_invalid_label_stage_merge_and_partial_tail(self):
        result = project_annotations("r", 110, [(0, 30, "Sleep stage 3"),
                                                (30, 30, "Sleep stage 4"),
                                                (60, 30, "Movement time"),
                                                (90, 20, "Sleep stage ?")])
        self.assertEqual([e["label"] for e in result["epochs"]], [3, 3, None])
        self.assertIn("invalid_label_movement", result["epochs"][2]["invalid_reasons"])
        self.assertEqual(result["incomplete_trailing_epoch"]["duration_us"], 20_000_000)

    def test_segment_boundary_off_record_and_rounding(self):
        result = project_annotations("r", 60, [(-30, 90, W), (60, 30, W)],
                                     segments=[(0, 15, "a"), (15, 60, "b")])
        self.assertIn("segment_gap_or_boundary", result["epochs"][0]["invalid_reasons"])
        self.assertEqual(result["epochs"][1]["segment_id"], "b")
        self.assertEqual(len(result["off_record_intervals"]), 2)
        self.assertEqual(to_microseconds(30.0000004), 30_000_000)

    def test_reject_bad_inputs(self):
        for value in (float("nan"), float("inf"), True, "oops"):
            with self.assertRaises(ValueError):
                to_microseconds(value)
        with self.assertRaises(ValueError):
            project_annotations("r", 30, [(0, 0, W)])
        with self.assertRaises(ValueError):
            project_annotations("r", 30, [], segments=[(0, 20, "a"), (10, 30, "b")])


if __name__ == "__main__":
    unittest.main()
