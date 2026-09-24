"""Offline disposable fixtures for file identity and established-reader audits."""

import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sleepedf.audit import audit_dataset


def write_edf(path, *, annotation=False, duration=30, channels=None, start="20.00.00",
              physical_min=-1, physical_max=1):
    """Minimal synthetic EDF bytes; this helper is a fixture writer, not a reader."""
    channels = channels or [("EEG Fpz-Cz", "uV", 100), ("EMG submental", "uV", 1)]
    if annotation:
        channels = [("EDF Annotations", "", None)]
        duration = 0
    count = len(channels)

    def field(value, width):
        return str(value).encode("ascii").ljust(width, b" ")

    fixed = b"".join(field(value, width) for value, width in [
        ("0", 8), ("X X X X" if annotation else "synthetic", 80),
        ("Startdate 01-JAN-2000 X X X" if annotation else "fixture", 80),
        ("01.01.00", 8), (start, 8), (256 * (count + 1), 8),
        ("EDF+C" if annotation else "", 44), (1, 8), (duration, 8), (count, 4),
    ])
    samples = [32 if annotation else int(rate * duration) for _, _, rate in channels]
    columns = [
        ([label for label, _, _ in channels], 16), ([""] * count, 80),
        ([unit for _, unit, _ in channels], 8), ([physical_min] * count, 8),
        ([physical_max] * count, 8),
        ([-32768] * count, 8), ([32767] * count, 8), ([""] * count, 80),
        (samples, 8), ([""] * count, 32),
    ]
    signal_headers = b"".join(field(value, width) for values, width in columns for value in values)
    payload = b"+0\x14\x14\x00".ljust(64, b"\x00") if annotation else bytes(2 * sum(samples))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fixed + signal_headers + payload)


class AuditFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def touch(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def codes(self, report):
        return {issue["code"] for issue in report["errors"]}


class AuditIdentityTests(AuditFixture, unittest.TestCase):

    def test_pairs_different_suffixes_and_keeps_both_nights_one_participant(self):
        for name in ["SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf",
                     "SC4002E0-PSG.edf", "SC4002EW-Hypnogram.edf"]:
            self.touch(name)
        report = audit_dataset(self.root, headers=False)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["summary"]["recordings"], 2)
        self.assertEqual(report["summary"]["participants"], 0)
        self.assertEqual({item["participant_id"] for item in report["recordings"]}, {"SC:00"})
        self.assertEqual({item["night"] for item in report["recordings"]}, {1, 2})
        self.assertTrue(all(item["identity_status"] == "unresolved" for item in report["recordings"]))
        self.assertTrue(all(item["evidence"] == {"pair_unique": True,
            "metadata_verified": False, "headers_verified": False} for item in report["recordings"]))

    def test_f_and_g_equipment_and_st_j_are_supported(self):
        for stem in ["SC4261F", "SC4281G", "ST7011J"]:
            self.touch(stem + "0-PSG.edf")
            self.touch(stem + "P-Hypnogram.edf")
        report = audit_dataset(self.root, headers=False)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["summary"]["participants"], 0)
        self.assertEqual({item["participant_id"] for item in report["recordings"]},
                         {"SC:26", "SC:28", "ST:01"})

    def test_missing_pair_is_an_error(self):
        self.touch("SC4001E0-PSG.edf")
        report = audit_dataset(self.root, headers=False)
        self.assertIn("missing_pair", self.codes(report))
        self.assertIsNone(report["recordings"][0]["hypnogram"])

    def test_duplicate_candidates_are_not_arbitrarily_selected(self):
        self.touch("one/SC4001E0-PSG.edf")
        self.touch("two/SC4001E0-PSG.edf")
        self.touch("SC4001EC-Hypnogram.edf")
        report = audit_dataset(self.root, headers=False)
        self.assertIn("duplicate_pair", self.codes(report))
        recording = report["recordings"][0]
        self.assertIsNone(recording["psg"])
        self.assertEqual(recording["identity_status"], "unresolved")
        self.assertEqual(len(recording["psg_candidates"]), 2)

    def test_equipment_mismatch_is_unresolved(self):
        self.touch("SC4001E0-PSG.edf")
        self.touch("SC4001FC-Hypnogram.edf")
        report = audit_dataset(self.root, headers=False)
        self.assertIn("inconsistent_pair_identity", self.codes(report))
        self.assertEqual(report["recordings"][0]["identity_status"], "unresolved")

    def test_renamed_and_invalid_scorer_names_do_not_gain_identity(self):
        for name in ["renamed-PSG.edf", "SC4001EC-PSG.edf", "SC4001E0-Hypnogram.edf"]:
            self.touch(name)
        report = audit_dataset(self.root, headers=False)
        self.assertEqual(len(report["recordings"]), 3)
        self.assertEqual(report["summary"]["participants"], 0)
        self.assertTrue(all(item["identity_status"] == "unresolved" for item in report["recordings"]))

    def test_missing_dependency_has_actionable_error(self):
        self.touch("SC4001E0-PSG.edf")
        with patch("sleepedf.audit.importlib.import_module", side_effect=ImportError):
            report = audit_dataset(self.root)
        errors = [item for item in report["errors"] if item["code"] == "missing_dependency"]
        self.assertEqual(len(errors), 1)
        self.assertIn("uv sync --frozen --extra audit --group dev", errors[0]["message"])

    def test_manifest_presence_is_not_content_verification(self):
        self.touch("SC4001E0-PSG.edf")
        self.touch("SC4001EC-Hypnogram.edf")
        (self.root / "RECORDS").write_text("SC4001E0-PSG.edf\n")
        (self.root / "SHA256SUMS.txt").write_text("0" * 64 + " SC4001E0-PSG.edf\n")
        report = audit_dataset(self.root, headers=False)
        self.assertTrue(report["source_manifests"]["SHA256SUMS.txt"]["presence_verified"])
        self.assertFalse(report["source_manifests"]["SHA256SUMS.txt"]["content_hash_verified"])
        first = json.dumps(report, allow_nan=False, sort_keys=True)
        second = json.dumps(audit_dataset(self.root, headers=False), allow_nan=False, sort_keys=True)
        self.assertEqual(first, second)

    def test_missing_manifest_file_and_unsafe_manifest_are_errors(self):
        (self.root / "RECORDS").write_text("missing.edf\n")
        (self.root / "SHA256SUMS.txt").write_text("0" * 64 + " ../outside.edf\n")
        report = audit_dataset(self.root, headers=False)
        self.assertTrue({"manifest_file_missing", "invalid_source_manifest"} <= self.codes(report))

    def test_absent_root_is_actionable(self):
        report = audit_dataset(self.root / "absent", headers=False)
        self.assertEqual(self.codes(report), {"dataset_not_found"})

    def test_metadata_row_is_required_for_resolved_identity(self):
        self.touch("SC4001E0-PSG.edf")
        self.touch("SC4001EC-Hypnogram.edf")
        self.touch("SC-subjects.xls")
        sheet = SimpleNamespace(nrows=2, row_values=lambda index:
                                ["subject", "night"] if index == 0 else [0, 1, "", "", "20:00"])
        book = SimpleNamespace(sheet_by_index=lambda index: sheet)
        xlrd = SimpleNamespace(open_workbook=lambda path: book, XLRDError=Exception)
        with patch("sleepedf.audit._dependency", return_value=xlrd):
            report = audit_dataset(self.root, headers=False)
        record = report["recordings"][0]
        self.assertEqual(record["identity_status"], "resolved")
        self.assertEqual(record["evidence"], {"pair_unique": True,
            "metadata_verified": True, "headers_verified": False})
        self.assertEqual(report["summary"]["participants"], 1)

    def test_sorted_files_include_sizes_and_change_with_file(self):
        first = self.touch("b/SC4001E0-PSG.edf")
        self.touch("a/SC4001EC-Hypnogram.edf")
        report = audit_dataset(self.root, headers=False)
        self.assertEqual(report["files"], [
            {"path": "a/SC4001EC-Hypnogram.edf", "size_bytes": 0},
            {"path": "b/SC4001E0-PSG.edf", "size_bytes": 0},
        ])
        first.write_bytes(b"x")
        changed = audit_dataset(self.root, headers=False)
        self.assertEqual(changed["files"][1]["size_bytes"], 1)

    def test_checksum_match_mismatch_and_exact_coverage(self):
        psg = self.touch("SC4001E0-PSG.edf")
        hyp = self.touch("SC4001EC-Hypnogram.edf")
        psg.write_bytes(b"psg")
        hyp.write_bytes(b"hyp")
        sums = self.root / "SHA256SUMS.txt"
        sums.write_text("\n".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()} {path.name}"
                                   for path in (psg, hyp)) + "\n")
        good = audit_dataset(self.root, headers=False, verify_checksums=True)
        manifest = good["source_manifests"]["SHA256SUMS.txt"]
        self.assertTrue(manifest["content_hash_verified"])
        self.assertTrue(good["scope"]["content_hash_verified"])
        self.assertEqual(manifest["unlisted_local_files"], [])
        self.assertEqual(manifest["hashes_checked"], 2)
        self.assertEqual(manifest["hash_bytes_read"], 6)
        self.assertEqual([item["status"] for item in manifest["checksums"]],
                         ["match_local_manifest", "match_local_manifest"])
        hyp.write_bytes(b"changed")
        bad = audit_dataset(self.root, headers=False, verify_checksums=True)
        self.assertIn("checksum_mismatch", self.codes(bad))
        self.assertFalse(bad["scope"]["content_hash_verified"])
        self.assertEqual(bad["source_manifests"]["SHA256SUMS.txt"]["checksums"][1]["status"], "mismatch")

    def test_checksum_budget_preflight_and_partial_manifest(self):
        psg = self.touch("SC4001E0-PSG.edf")
        psg.write_bytes(b"test")
        self.touch("SC4001EC-Hypnogram.edf")
        (self.root / "SHA256SUMS.txt").write_text(
            f"{hashlib.sha256(b'test').hexdigest()} {psg.name}\n")
        bounded = audit_dataset(self.root, headers=False, verify_checksums=True, max_hash_bytes=3)
        self.assertEqual(bounded["scope"]["hash_bytes_read"], 0)
        self.assertEqual(bounded["source_manifests"]["SHA256SUMS.txt"]["checksums"][0]["status"],
                         "omitted_budget")
        self.assertFalse(bounded["scope"]["content_hash_verified"])
        partial = audit_dataset(self.root, headers=False, verify_checksums=True)
        self.assertFalse(partial["source_manifests"]["SHA256SUMS.txt"]["content_hash_verified"])
        self.assertEqual(partial["source_manifests"]["SHA256SUMS.txt"]["hashes_matched"], 1)
        self.assertFalse(partial["scope"]["content_hash_verified"])
        self.assertEqual(partial["source_manifests"]["SHA256SUMS.txt"]["unlisted_local_files"],
                         ["SC4001EC-Hypnogram.edf"])
        with self.assertRaises(ValueError):
            audit_dataset(self.root, headers=False, max_hash_bytes=-1)

    def test_duplicate_manifest_paths_are_rejected(self):
        self.touch("SC4001E0-PSG.edf")
        line = "0" * 64 + " SC4001E0-PSG.edf\n"
        (self.root / "SHA256SUMS.txt").write_text(line * 2)
        duplicate = audit_dataset(self.root, headers=False, verify_checksums=True)
        self.assertIn("duplicate_manifest_entry", self.codes(duplicate))
        self.assertFalse(duplicate["scope"]["content_hash_verified"])

    def test_symlink_escape_manifest_path_is_rejected(self):
        link = self.touch("escape.edf")
        outside = self.root.parent / f"{self.root.name}-outside.edf"
        (self.root / "SHA256SUMS.txt").write_text("0" * 64 + " escape.edf\n")
        original_resolve = Path.resolve

        def escaping_resolve(path, *args, **kwargs):
            return outside if path == link else original_resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", escaping_resolve):
            unsafe = audit_dataset(self.root, headers=False, verify_checksums=True)
        self.assertIn("invalid_source_manifest", self.codes(unsafe))
        self.assertIn("unsafe_dataset_path", self.codes(unsafe))

    def test_checksum_read_error_blocks_verification(self):
        psg = self.touch("SC4001E0-PSG.edf")
        psg.write_bytes(b"test")
        (self.root / "SHA256SUMS.txt").write_text(
            f"{hashlib.sha256(b'test').hexdigest()} {psg.name}\n")
        original_open = Path.open

        def failing_open(path, *args, **kwargs):
            if path == psg and args and args[0] == "rb":
                raise OSError("simulated denied read")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", failing_open):
            report = audit_dataset(self.root, headers=False, verify_checksums=True)
        self.assertIn("checksum_read_error", self.codes(report))
        self.assertFalse(report["scope"]["content_hash_verified"])
        self.assertEqual(report["source_manifests"]["SHA256SUMS.txt"]["checksums"][0]["status"],
                         "read_error")


@unittest.skipUnless(importlib.util.find_spec("pyedflib"), "audit extra required for EDF reader fixtures")
class AuditReaderTests(AuditFixture, unittest.TestCase):
    def pair(self, **options):
        psg = self.root / "SC4001E0-PSG.edf"
        write_edf(psg, **options)
        write_edf(self.root / "SC4001EC-Hypnogram.edf", annotation=True)
        return psg

    def test_mixed_native_rates_and_zero_duration_hypnogram(self):
        self.pair()
        report = audit_dataset(self.root)
        self.assertEqual(report["errors"], [])
        recording = report["recordings"][0]
        self.assertEqual([channel["sample_rate_hz"] for channel in recording["psg_header"]["channels"]], [100.0, 1.0])
        self.assertEqual([channel["samples_per_data_record"] for channel in
                          recording["psg_header"]["channels"]], [3000, 30])
        self.assertEqual(recording["hypnogram_header"]["kind"], "annotation_only")
        self.assertEqual(recording["hypnogram_header"]["data_record_seconds"], 0)
        self.assertIsNone(recording["hypnogram_header"]["duration_seconds"])
        self.assertEqual(recording["start_alignment"], {"equal": True, "offset_seconds": 0.0})
        self.assertFalse(report["scope"]["annotations_read"])
        self.assertEqual(report["scope"]["waveform_reads"], 0)
        self.assertEqual(report["scope"]["headers_completed"], 2)
        self.assertTrue(recording["evidence"]["headers_verified"])
        self.assertFalse(recording["evidence"]["metadata_verified"])
        json.dumps(report, allow_nan=False)

    def test_sixty_second_data_record_preserves_native_rate(self):
        self.pair(duration=60)
        report = audit_dataset(self.root)
        self.assertEqual(report["errors"], [])
        header = report["recordings"][0]["psg_header"]
        self.assertEqual(header["data_record_seconds"], 60)
        self.assertEqual(header["channels"][0]["sample_rate_hz"], 100)
        self.assertEqual(header["channels"][0]["samples_per_data_record"], 6000)

    def test_negative_physical_gain_is_valid(self):
        self.pair(physical_min=1, physical_max=-1)
        report = audit_dataset(self.root)
        self.assertEqual(report["errors"], [])
        self.assertTrue(report["recordings"][0]["evidence"]["headers_verified"])

    def test_truncated_payload_and_invalid_header_fail(self):
        psg = self.pair()
        psg.write_bytes(psg.read_bytes()[:-2])
        report = audit_dataset(self.root)
        self.assertIn("invalid_edf_header", self.codes(report))
        psg.write_bytes(b"not an EDF header")
        report = audit_dataset(self.root)
        self.assertIn("invalid_edf_header", self.codes(report))

    def test_reader_incompatible_recordingfield_error_is_sanitized(self):
        self.pair()
        class RejectingReader:
            def __init__(self, *args, **kwargs):
                raise OSError("EDF+ Recordingfield mismatch PATIENT_SECRET")
        module = SimpleNamespace(EdfReader=RejectingReader, DO_NOT_READ_ANNOTATIONS=0)
        with patch("sleepedf.audit._dependency", return_value=module):
            report = audit_dataset(self.root)
        self.assertIn("reader_incompatible_header", self.codes(report))
        self.assertNotIn("PATIENT_SECRET", json.dumps(report))
        self.assertEqual(report["scope"]["headers_completed"], 0)
        self.assertFalse(report["recordings"][0]["evidence"]["headers_verified"])

    def test_pair_offset_is_reported_without_raw_dates(self):
        self.pair(start="20.00.30")
        report = audit_dataset(self.root)
        recording = report["recordings"][0]
        self.assertEqual(recording["start_alignment"], {"equal": False, "offset_seconds": -30.0})
        serialized = json.dumps(report)
        self.assertNotIn("2000-01-01", serialized)
        self.assertIn("pair_start_offset", serialized)


if __name__ == "__main__":
    unittest.main()
