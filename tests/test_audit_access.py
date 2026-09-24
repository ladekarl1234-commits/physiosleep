from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sleepedf.audit_access import (append_audit_event, canonical_ledger, read_ledger,
                                   validate_audit_opening, _require_original_a)
from sleepedf.contracts import content_id
from sleepedf.research import atomic_json, file_sha256


def fixture(directory):
    root = Path(directory).resolve()
    protocol = {"split_path": str(root / "research" / "split-v2.json")}
    protocol["protocol_hash"] = content_id(protocol)
    atomic_json(root / "research" / "protocol-v1.json", protocol)
    return root / "runs" / "audit-access.jsonl", protocol["protocol_hash"]


class AuditAccessTests(unittest.TestCase):
    def test_no_open_without_registered_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path, protocol = fixture(directory)
            with self.assertRaises(ValueError):
                append_audit_event(path, protocol, "A", "AUDIT_OPENED", {})

    def test_no_reselection_or_boolean_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            path, protocol = fixture(directory)
            append_audit_event(path, protocol, "A", "SELECTION_FROZEN", {"selection_sha256": "one"})
            with self.assertRaises(ValueError):
                append_audit_event(path, protocol, "A", "SELECTION_FROZEN", {"selection_sha256": "two"})
            ready = Path(directory) / "fake.json"
            atomic_json(ready, {"status": "PASSED"})
            with self.assertRaisesRegex(ValueError, "Boolean status"):
                append_audit_event(path, protocol, "A", "AUDIT_OPENED", {
                    "selection_sha256": "one", "readiness_receipt": {"path": str(ready), "sha256": file_sha256(ready)}})

    def test_ledger_tampering_and_unopened_gate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, protocol = fixture(directory)
            append_audit_event(path, protocol, "A", "SELECTION_FROZEN", {"selection_sha256": "one"})
            with self.assertRaises(ValueError):
                validate_audit_opening({"protocol_hash": protocol, "phase": "A"})
            path.write_text(path.read_text().replace('"one"', '"two"'))
            with self.assertRaises(ValueError):
                read_ledger(path, protocol)

    def test_relocated_or_renamed_ledger_cannot_reset_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            path, protocol = fixture(directory)
            with self.assertRaisesRegex(ValueError, "canonical"):
                canonical_ledger(path.with_name("retry.jsonl"), protocol)
            relocated = Path(directory) / "copied"
            original = Path(directory) / "research" / "protocol-v1.json"
            import json
            atomic_json(relocated / "research" / "protocol-v1.json", json.loads(original.read_text()))
            with self.assertRaisesRegex(ValueError, "original frozen protocol"):
                canonical_ledger(relocated / "runs" / "audit-access.jsonl", protocol)

    def test_b_does_not_trust_pass_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path, protocol = fixture(directory)
            with self.assertRaisesRegex(ValueError, "Gate A"):
                append_audit_event(path, protocol, "B", "SELECTION_FROZEN", {})
            rows = [{"phase": "A", "event": "VERIFICATION_RECORDED", "evidence": {"outcome": "VERIFIED_PASS"}}]
            with self.assertRaisesRegex(ValueError, "bound report"):
                _require_original_a(rows, protocol, path)

    def test_b_recomputes_a_and_rejects_current_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path, protocol = fixture(directory)
            refs = {}
            for name, value in (("report", {"status": "PASSED"}), ("attestation", {}),
                                ("recompute_arguments", {"phase": "A", "protocol_hash": protocol,
                                 "freeze_manifest": {"access_ledger_path": str(path)}})):
                artifact = Path(directory) / (name + ".json")
                atomic_json(artifact, value)
                refs[name] = {"path": str(artifact), "sha256": file_sha256(artifact)}
            rows = [{"phase": "A", "event": "VERIFICATION_RECORDED",
                     "evidence": dict(refs, outcome="VERIFIED_PASS")}]
            with patch("sleepedf.gates.require_gate", side_effect=ValueError("actual margin fails")) as check:
                with self.assertRaisesRegex(ValueError, "actual margin fails"):
                    _require_original_a(rows, protocol, path)
                self.assertEqual(check.call_count, 1)


if __name__ == "__main__":
    unittest.main()
