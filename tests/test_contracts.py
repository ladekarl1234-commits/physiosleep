import copy
import math
import tempfile
import unittest
from pathlib import Path

from sleepedf.contracts import (code_identity, content_id, inventory_id, json_text,
                                map_annotation, read_json, validate_split)


def valid_fixture():
    records = []
    files = []
    for number in range(3):
        rid = f"SC4{number:02}1"
        psg = f"{rid}E0-PSG.edf"
        hyp = f"{rid}EC-Hypnogram.edf"
        psg_header = {"kind": "sampled_signals", "duration_seconds": 30.0,
                      "data_record_seconds": 30.0, "data_records": 1,
                      "channels": [{"label": "EEG Fpz-Cz", "samples_per_data_record": 3000,
                                    "sample_rate_hz": 100.0, "unit": "uV",
                                    "physical_min": 1.0, "physical_max": -1.0,
                                    "digital_min": -32768, "digital_max": 32767}]}
        hyp_header = {"kind": "annotation_only", "channels": [],
                      "duration_seconds": None, "data_record_seconds": 0.0,
                      "data_records": 1}
        records.append({"recording_id": rid, "participant_id": f"SC:{number:02}",
                        "cohort": "SC", "night": 1, "identity_status": "resolved",
                        "psg": psg, "hypnogram": hyp, "psg_header": psg_header,
                        "hypnogram_header": hyp_header,
                        "start_alignment": {"equal": True, "offset_seconds": 0.0},
                        "metadata": {"lights_off_available": True, "condition": None},
                        "evidence": {"pair_unique": True, "metadata_verified": True,
                                     "headers_verified": True}})
        files.extend([{"path": psg, "size_bytes": 1000}, {"path": hyp, "size_bytes": 200}])
    report = {"report_kind": "audit", "schema_version": "1.0", "tool_version": "0.1.0",
              "scope": {"data_root": "/synthetic", "headers_requested": True,
                        "headers_completed": 6, "reader_version": "fixture"},
              "recordings": records, "files": files,
              "errors": [], "warnings": []}
    report["manifest_id"] = inventory_id(report)
    split = {"schema_version": "1.0", "protocol_id": "sleepedf-bootstrap-v1",
             "inventory_id": report["manifest_id"], "seed": 7, "method": "synthetic",
             "participants": {"train": ["SC:00"], "validation": ["SC:01"], "test": ["SC:02"]}}
    return split, report


class ContractTests(unittest.TestCase):
    def test_label_mapping_is_strict(self):
        self.assertEqual(map_annotation("Sleep stage 4")["label"], 3)
        self.assertEqual(map_annotation("Movement time")["reason"], "movement")
        self.assertFalse(map_annotation("sleep stage W")["valid"])
        with self.assertRaises(ValueError):
            map_annotation(None)

    def test_strict_json_rejects_nonfinite_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            for text in ('{"x":NaN}', '{"x":1e999}', '{"x":1,"x":2}'):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_json(path)
        with self.assertRaises(ValueError):
            json_text({"x": math.nan})

    def test_source_identity_does_not_claim_raw_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sleepedf").mkdir()
            (root / "sleepedf" / "a.py").write_text("x = 1\n")
            (root / "raw.edf").write_bytes(b"private")
            identity = code_identity(root)
            self.assertFalse(identity["includes_raw_data"])
            self.assertEqual(set(identity["files"]), {"sleepedf/a.py"})

    def test_complete_audit_and_split_pass(self):
        split, report = valid_fixture()
        self.assertTrue(validate_split(split, report)["valid"])
        report["recordings"][0]["start_alignment"] = {"equal": False, "offset_seconds": 30.0}
        report["manifest_id"] = inventory_id(report)
        split["inventory_id"] = report["manifest_id"]
        self.assertTrue(validate_split(split, report)["valid"])

    def test_incomplete_or_untrusted_inventory_rejected(self):
        mutations = [
            lambda r: r["scope"].update(headers_requested=False),
            lambda r: r["scope"].update(headers_completed=0),
            lambda r: r["scope"].update(headers_completed=7),
            lambda r: r["scope"].update(reader_version=None),
            lambda r: r["recordings"][0]["evidence"].update(metadata_verified=False),
            lambda r: r["recordings"][0].update(metadata=None),
            lambda r: r["recordings"][0].update(metadata={}),
            lambda r: r["recordings"][0]["metadata"].update(lights_off_available="yes"),
            lambda r: r["recordings"][0]["metadata"].pop("condition"),
            lambda r: r["recordings"][0].update(psg_header=None),
            lambda r: r["recordings"][0].update(psg_header={}),
            lambda r: r["recordings"][0].update(hypnogram_header={}),
            lambda r: r["recordings"][0]["psg_header"].update(duration_seconds=0),
            lambda r: r["recordings"][0]["psg_header"].update(data_records=2),
            lambda r: r["recordings"][0]["psg_header"]["channels"][0].update(sample_rate_hz=99),
            lambda r: r["recordings"][0]["psg_header"]["channels"][0].update(physical_max=1),
            lambda r: r["recordings"][0]["hypnogram_header"].update(kind="sampled_signals"),
            lambda r: r["recordings"][0].update(start_alignment={}),
            lambda r: r["recordings"][0]["start_alignment"].update(equal=False),
            lambda r: r["recordings"][0].update(recording_id="unknown"),
            lambda r: r["recordings"][1].update(recording_id=r["recordings"][0]["recording_id"]),
            lambda r: r["recordings"][1].update(psg=r["recordings"][0]["psg"]),
            lambda r: r["recordings"][0].update(participant_id="SC:99"),
            lambda r: r.update(files=[]),
            lambda r: r.update(report_kind="files-only"),
            lambda r: r["errors"].append({"code": "unresolved"}),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                split, report = valid_fixture()
                mutate(report)
                report["manifest_id"] = inventory_id(report)
                split["inventory_id"] = report["manifest_id"]
                with self.assertRaises(ValueError):
                    validate_split(split, report)

    def test_swapped_unique_psg_paths_cannot_change_participant_attribution(self):
        split, report = valid_fixture()
        records = report["recordings"]
        records[0]["psg"], records[1]["psg"] = records[1]["psg"], records[0]["psg"]
        report["manifest_id"] = inventory_id(report)
        split["inventory_id"] = report["manifest_id"]
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_split(split, report)

    def test_tamper_or_added_record_invalidates_digest(self):
        split, report = valid_fixture()
        report["files"][0]["size_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_split(split, report)
        split, report = valid_fixture()
        report["recordings"].append(copy.deepcopy(report["recordings"][0]))
        with self.assertRaises(ValueError):
            validate_split(split, report)

    def test_unknown_duplicate_overlap_and_unassigned_rejected(self):
        for mutate in [
            lambda g: g["test"].append("SC:99"),
            lambda g: g["train"].append("SC:00"),
            lambda g: g["test"].append("SC:00"),
            lambda g: g["validation"].clear(),
        ]:
            split, report = valid_fixture()
            mutate(split["participants"])
            with self.assertRaises(ValueError):
                validate_split(split, report)


if __name__ == "__main__":
    unittest.main()
