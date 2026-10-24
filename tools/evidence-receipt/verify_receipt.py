#!/usr/bin/env python3
"""Verify an evidence-receipt/1.0 receipt.json. Standard library only.

Usage (from the repository root):
    python3 tools/evidence-receipt/verify_receipt.py runs/cetacean-005/receipt.json

Checks:
  1. Structure against tools/evidence-receipt/schema.json (uses `jsonschema` if
     installed, otherwise a built-in check of required keys, the format const
     and no unknown top-level keys).
  2. No absolute paths in the receipt.
  3. Every path in code.scripts, datasets[].files_manifest, datasets[].annotations
     and artifacts[] exists relative to --root (default: current dir) and its
     SHA-256 matches.
  4. If code.commit is set and --root is a git checkout, reports HEAD vs commit.
Does NOT rerun the analysis; see README_RECEIPT.md for the rerun command.
Exit 0 = all checks pass, 1 = any failure.
"""
import argparse, hashlib, json, os, subprocess, sys

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('receipt')
    ap.add_argument('--root', default='.')
    ap.add_argument('--schema', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'schema.json'))
    a = ap.parse_args()
    ok = True
    def res(good, msg):
        nonlocal ok
        ok &= good
        print(('PASS ' if good else 'FAIL ') + msg)
    r = json.load(open(a.receipt))
    S = json.load(open(a.schema))
    try:
        import jsonschema
        try:
            jsonschema.validate(r, S); res(True, 'schema (jsonschema)')
        except jsonschema.ValidationError as e:
            res(False, 'schema (jsonschema): ' + e.message)
    except ImportError:
        miss = [k for k in S.get('required', []) if k not in r]
        extra = [k for k in r if k not in S.get('properties', {})]
        res(not miss and not extra and r.get('receipt_format') == 'evidence-receipt/1.0',
            f'schema (built-in minimal check; install jsonschema for full) missing={miss} extra={extra}')
    entries = [('code', s) for s in r['code'].get('scripts', [])]
    for d in r.get('datasets', []):
        if d.get('files_manifest'): entries.append(('manifest', d['files_manifest']))
        entries += [('annotation', x) for x in d.get('annotations', [])]
    entries += [('artifact', x) for x in r.get('artifacts', [])]
    blob = json.dumps(r)
    res('"/workspace' not in blob and not any(os.path.isabs(e['path']) for _, e in entries), 'no absolute paths')
    for kind, e in entries:
        p = os.path.join(a.root, e['path'])
        if not os.path.exists(p):
            res(False, f'{kind} missing: {e["path"]}'); continue
        got = sha256(p)
        res(got == e['sha256'], f'{kind} sha256 {e["path"]}' + ('' if got == e['sha256'] else f' got {got}'))
    c = r['code']
    res(bool(c.get('repo')) and bool(c.get('commit')), f'code.repo/commit set: {c.get("repo")} @ {c.get("commit")}')
    try:
        head = subprocess.run(['git', '-C', a.root, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        if head:
            print(f'INFO git HEAD {head} (receipt commit {c.get("commit")}; files are checked by hash, so a later commit is fine if hashes pass)')
    except FileNotFoundError:
        pass
    print(f'INFO reproduction.status = {r["reproduction"]["status"]} (author-recorded; not verified by this tool)')
    print('RESULT ' + ('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
