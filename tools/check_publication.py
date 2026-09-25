"""Check staged publication boundaries, source identities and local Markdown links."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = subprocess.check_output(['git','ls-files','-z'], cwd=ROOT).decode().split('\0')
    names = [name for name in names if name]
    errors=[]
    forbidden={'.edf','.npz','.npy','.joblib','.pt','.pth','.h5','.ckpt','.woff','.woff2','.ttf','.otf','.zip','.pyc'}
    for name in names:
        p=ROOT/name
        if p.suffix.lower() in forbidden or any(x in p.relative_to(ROOT).parts for x in ('vendor','runs','.venv','.venvs','local-data','__pycache__')):
            errors.append('Excluded payload: '+name)
        if p.suffix.lower() in ('.py','.md','.json','.csv','.txt','.toml','.log','.yml','.yaml','.cjs'):
            text=p.read_text(encoding='utf-8-sig')
            patterns=[r'ghp_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{30,}',r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----',r'C:\\Users\\',r'C:\\Program Files\\']
            if any(re.search(pattern,text) for pattern in patterns):
                errors.append('Credential or machine-path pattern: '+name)
        if p.suffix=='.md':
            for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf8')):
                target=unquote(link.split('#')[0].strip('<>'))
                if not target or ':' in target or target.startswith('//'):continue
                dest=(p.parent/target).resolve()
                if not dest.is_relative_to(ROOT) or not dest.exists():
                    errors.append('Broken or escaping link: '+name+' -> '+target)
    manifest=json.loads((ROOT/'evidence/source-copy-manifest.json').read_text())
    for item in manifest['files']:
        if hashlib.sha256((ROOT/item['public_path']).read_bytes()).hexdigest()!=item['sha256']:
            errors.append('Changed copied source: '+item['public_path'])
    figures=ROOT/'reports/publication-figures'
    figure_manifest=json.loads((figures/'manifest.json').read_text())
    if hashlib.sha256((figures/'summary.json').read_bytes()).hexdigest()!=figure_manifest['summary_sha256']:
        errors.append('Figure summary differs from bound manifest')
    if hashlib.sha256((ROOT/'tools/build_research_figures.py').read_bytes()).hexdigest()!=figure_manifest['script_sha256']:
        errors.append('Figure builder differs from bound manifest')
    extension=json.loads((ROOT/'reports/score-extension/manifest.json').read_text(encoding='utf8'))
    for item in extension['files']:
        if hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            errors.append('Score extension artifact differs from manifest: '+item['path'])
    guide=json.loads((ROOT/'reports/judge-guide/manifest.json').read_text(encoding='utf8'))
    for item in guide['files']:
        if hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            errors.append('Judge guide artifact differs from manifest: '+item['path'])
    comparison=json.loads((ROOT/'reports/comparison-extensions/manifest.json').read_text(encoding='utf8'))
    for item in comparison['files']:
        if hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            errors.append('Comparison figure input/output differs from manifest: '+item['path'])
    workbook=json.loads((ROOT/'literature/workbook-comparison.json').read_text(encoding='utf8'))
    if len(workbook['experiments'])!=56 or len(workbook['papers'])!=31:
        errors.append('Workbook coverage count changed')
    keys=[(row['source_sheet'],row['source_row']) for row in workbook['experiments']]
    if len(set(keys))!=56 or any(row['comparable_to_local'] for row in workbook['experiments']):
        errors.append('Literature row identity or comparison scope changed')
    recall=json.loads((ROOT/'literature/stage-recall-context.json').read_text(encoding='utf8'))
    if recall['source_sha256']!=workbook['source_sha256'] or recall['comparable_to_local']:
        errors.append('Stage-recall workbook identity or comparison scope changed')
    print(json.dumps({'status':'FAIL' if errors else 'PASS','tracked_files':len(names),
                      'scope':'staged/committed paths; pattern scan is not a formal privacy proof',
                      'errors':errors},indent=2))
    raise SystemExit(bool(errors))


if __name__=='__main__':main()
