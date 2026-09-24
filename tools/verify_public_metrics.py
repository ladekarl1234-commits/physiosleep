"""Recompute headline metrics from public aggregate confusion counts."""
from pathlib import Path
from fractions import Fraction
import json


def main():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / 'evidence/development-metrics.json').read_text())
    c = data['confusion']
    assert len(c) == 5 and all(len(row) == 5 for row in c)
    assert all(type(x) is int and x >= 0 for row in c for x in row)
    support = [sum(row) for row in c]
    predicted = [sum(c[j][i] for j in range(5)) for i in range(5)]
    total = sum(support)
    correct = sum(c[i][i] for i in range(5))
    f1 = sum((Fraction(2*c[i][i], support[i]+predicted[i])
              if support[i]+predicted[i] else Fraction(0) for i in range(5)), Fraction(0))/5
    expected = sum(a*b for a,b in zip(support,predicted))
    kappa = Fraction(correct*total-expected,total*total-expected)
    values={'macro_f1':float(f1),'accuracy':float(Fraction(correct,total)), 'kappa':float(kappa)}
    for key,value in values.items():
        assert abs(value-data[key]) < 1e-12, (key,value,data[key])
    assert total == data['valid_epochs']
    assert total+data['invalid_epochs'] == data['complete_epochs']
    assert data['confirmatory'] is False
    assert data['gate_A'] == data['gate_B'] == 'NOT_RUN'
    print(json.dumps({'status':'PASS','source':'public aggregate counts',**values,
                      'macro_f1_exact':str(f1),'valid_epochs':total,'confirmation':'NOT_RUN'},indent=2))


if __name__ == '__main__':
    main()
