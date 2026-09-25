"""Replay exploratory audit point metrics from public aggregate counts."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def verify_phase(row):
    c = row['confusion']
    assert len(c) == 5 and all(len(r) == 5 for r in c)
    assert all(type(v) is int and v >= 0 for r in c for v in r)
    support = [sum(r) for r in c]
    predicted = [sum(c[j][i] for j in range(5)) for i in range(5)]
    n = sum(support)
    correct = sum(c[i][i] for i in range(5))
    f1 = [Fraction(2*c[i][i], support[i]+predicted[i])
          if support[i]+predicted[i] else Fraction(0) for i in range(5)]
    macro = sum(f1, Fraction(0))/5
    expected = sum(a*b for a, b in zip(support, predicted))
    values = {'macro_f1': float(macro), 'accuracy': float(Fraction(correct, n)),
              'kappa': float(Fraction(correct*n-expected, n*n-expected))}
    for key, value in values.items():
        assert abs(value-row[key]) < 1e-12, (key, value, row[key])
    assert all(abs(float(v)-w) < 1e-12 for v, w in zip(f1, row['per_class_f1']))
    assert support == row['reference_support'] and predicted == row['predicted_support']
    assert n == row['evaluated_epochs']
    assert n+row['invalid_epochs'] == row['complete_psg_epochs']
    assert row['prediction_coverage_fraction'] == 1
    assert row['recording_count'] == 39 and row['participant_count'] == 20
    assert row['cohort_participants'] == {'SC': 16, 'ST': 4}
    return {'status': 'PASS', **values, 'macro_f1_exact': str(macro), 'valid_epochs': n}


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root/'reports/exploratory-audits-v1'
    data = json.loads((folder/'aggregate.json').read_text(encoding='utf-8'))
    assert data['status'] == 'EXPLORATORY_EVALUATED' and data['gate_status'] == 'NOT_RUN'
    assert data['total_recordings'] == 78 and data['total_participants'] == 40
    result = {phase: verify_phase(data['phases'][phase]) for phase in ('A', 'B')}
    for name, expected in data['figure_sha256'].items():
        path = (folder/name).resolve()
        assert path.is_relative_to(folder.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    print(json.dumps({'scope': 'public pooled-confusion replay; participant-level bootstrap '
                      'requires excluded private cluster counts', 'phases': result}, indent=2))


if __name__ == '__main__':
    main()
