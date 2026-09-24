import copy
from pathlib import Path
import tempfile
import unittest

from sleepedf.contracts import content_id
from sleepedf.research import artifact_id, atomic_json
from sleepedf.splits import CHECKS, make_split, validate_split_v2
from sleepedf.protocol import load_development_truth


def fixtures():
    records, people = [], {}
    for cohort, count, prefix in (("SC", 78, "SC4"), ("ST", 22, "ST7")):
        for i in range(count):
            pid, rid = f"{cohort}:{i:02d}", f"{prefix}{i:02d}1"
            people[pid] = {"participant_id": pid, "cohort": cohort, "age": 18 + i, "sex": "F"}
            records.append(dict(recording_id=rid, participant_id=pid, cohort=cohort, night=1,
                psg=rid + "E0-PSG.edf", psg_sha256="a"*64, hypnogram_sha256="b"*64,
                duration_seconds=60, n_epochs=2, channels=["EEG Fpz-Cz"], reader_evidence={"reader": "fixture"},
                checks={k: True for k in CHECKS}, errors=[]))
    for first in list(records)[:97]:
        second = copy.deepcopy(first)
        second["recording_id"] = first["recording_id"][:-1] + "2"
        second["psg"] = second["recording_id"] + "E0-PSG.edf"
        second["night"] = 2
        records.append(second)
    manifest = dict(schema_version="2.0", report_kind="data-readiness", errors=[], records=records,
                    complete_dataset_scope=True, expected_recording_ids=sorted(r["recording_id"] for r in records))
    manifest["manifest_id"] = content_id(manifest)
    metadata = dict(participants=people, source_sha256={})
    metadata["metadata_id"] = content_id(metadata)
    return manifest, metadata


class SplitV2Tests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.metadata = fixtures()
        self.split = make_split(self.manifest, self.metadata)

    def test_fixed_quota_exposure_folds_and_order_independence(self):
        counts = validate_split_v2(self.split, self.manifest)["participant_counts"]
        self.assertEqual(counts, dict(development=60, audit_a=20, audit_b=20))
        self.assertTrue({"SC:36", "ST:01"} <= set(self.split["participants"]["development"]))
        for i, fold in enumerate(self.split["folds"]):
            self.assertEqual(sum(p.startswith("SC:") for p in fold["validation"]), 10 if i == 0 else 9)
        other = copy.deepcopy(self.metadata)
        other["participants"] = dict(reversed(list(other["participants"].items())))
        self.assertEqual(make_split(self.manifest, other), self.split)

    def test_night_not_participant_split_rejected(self):
        broken = copy.deepcopy(self.split)
        broken["participants"]["audit_a"][0] = broken["participants"]["development"][0]
        broken["split_id"] = artifact_id(broken, "split_id")
        with self.assertRaises(ValueError):
            validate_split_v2(broken, self.manifest)

    def test_unresolved_record_or_changed_metadata_rejected(self):
        self.manifest["records"][0]["checks"]["timing_verified"] = False
        self.manifest["manifest_id"] = artifact_id(self.manifest, "manifest_id")
        with self.assertRaises(ValueError):
            make_split(self.manifest, self.metadata)
        self.manifest, _ = fixtures()
        self.metadata["participants"]["SC:00"]["age"] = 120
        with self.assertRaises(ValueError):
            make_split(self.manifest, self.metadata)

    def test_audit_truth_rejected_before_file_open(self):
        record = {"participant_id": self.split["participants"]["audit_a"][0], "truth_path": "not-a-real-file"}
        with self.assertRaisesRegex(ValueError, "audit truth"):
            load_development_truth(record, self.split)

    def test_missing_night_and_shortened_grid_rejected(self):
        for defect in ("missing_night", "short_grid"):
            broken = copy.deepcopy(self.manifest)
            if defect == "missing_night":
                broken["records"].pop()
            else:
                broken["records"][0]["n_epochs"] = 1
            broken["manifest_id"] = artifact_id(broken, "manifest_id")
            with self.assertRaises(ValueError):
                make_split(broken, self.metadata)

    def test_immutable_file_and_raw_path_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "split.json"
            atomic_json(p, self.split, immutable=True)
            atomic_json(p, self.split, immutable=True)
            with self.assertRaises(ValueError):
                atomic_json(p, {}, immutable=True)
            with self.assertRaises(ValueError):
                atomic_json(Path(folder) / "sleep-edf-database-expanded-1.0.0" / "bad.json", {})


if __name__ == "__main__":
    unittest.main()
