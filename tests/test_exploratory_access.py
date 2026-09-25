import tempfile
from pathlib import Path
import unittest

from sleepedf.audit_access import append_audit_event, read_ledger
from sleepedf.contracts import content_id
from sleepedf.research import atomic_json, file_sha256
from test_audit_access import fixture


class ExploratoryAccessTests(unittest.TestCase):
    def test_retirement_is_not_confirmation_and_cannot_be_reselected(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger, protocol = fixture(temp)
            auth_path, selection_path = Path(temp) / 'auth.json', Path(temp) / 'selection.json'
            auth = {'artifact_type': 'owner_exploratory_audit_authorization',
                    'owner_reply': 'Run both now as exploratory evaluations',
                    'protocol_hash': protocol, 'fresh_data_required_for_confirmation': True}
            selection = {'artifact_type': 'physiosleep_exploratory_audits_selection',
                         'scope': 'EXPLORATORY_A_AND_B_NO_GATE_PASS', 'protocol_hash': protocol}
            selection['selection_id'] = content_id(selection)
            atomic_json(auth_path, auth)
            atomic_json(selection_path, selection)
            evidence = {'authorization': {'path': str(auth_path), 'sha256': file_sha256(auth_path)},
                        'selection': {'path': str(selection_path), 'sha256': file_sha256(selection_path)}}
            for phase in ('A', 'B'):
                append_audit_event(ledger, protocol, phase, 'EXPLORATORY_RETIRED', evidence)
                for event in ('SELECTION_FROZEN', 'AUDIT_OPENED', 'EVALUATION_RECORDED',
                              'VERIFICATION_RECORDED', 'EXPLORATORY_RETIRED'):
                    with self.assertRaises(ValueError):
                        append_audit_event(ledger, protocol, phase, event, evidence)
            self.assertEqual([r['event'] for r in read_ledger(ledger, protocol)],
                             ['EXPLORATORY_RETIRED', 'EXPLORATORY_RETIRED'])

    def test_status_file_cannot_retire_or_pass_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger, protocol = fixture(temp)
            path = Path(temp) / 'fake.json'
            atomic_json(path, {'status': 'PASSED'})
            ref = {'path': str(path), 'sha256': file_sha256(path)}
            with self.assertRaises(ValueError):
                append_audit_event(ledger, protocol, 'A', 'EXPLORATORY_RETIRED',
                                   {'authorization': ref, 'selection': ref})
            self.assertEqual(read_ledger(ledger, protocol), [])


if __name__ == '__main__':
    unittest.main()
