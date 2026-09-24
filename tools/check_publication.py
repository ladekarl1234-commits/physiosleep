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
        if p.suffix.lower() in ('.py','.md','.json','.txt','.toml','.log','.yml','.yaml','.cjs'):
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
    print(json.dumps({'status':'FAIL' if errors else 'PASS','tracked_files':len(names),
                      'scope':'staged/committed paths; pattern scan is not a formal privacy proof',
                      'errors':errors},indent=2))
    raise SystemExit(bool(errors))


if __name__=='__main__':main()
