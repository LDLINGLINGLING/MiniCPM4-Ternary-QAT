"""List Ultra-FineWeb parquet shards (path, size, sha256) at a pinned revision."""
import json, sys, requests
REPO, REV = 'openbmb/Ultra-FineWeb', '02c85641e3d19a854be2e09139c25adaa9518063'
out = {}
s = requests.Session()
for lang in ['zh', 'en']:
    url = f'https://hf-mirror.com/api/datasets/{REPO}/tree/{REV}/data/ultrafineweb_{lang}'
    files = []
    while url:
        r = s.get(url, timeout=(20, 60)); r.raise_for_status()
        files += [{'path': f['path'], 'size': f['size'], 'sha256': f.get('lfs', {}).get('oid')}
                  for f in r.json() if f['path'].endswith('.parquet')]
        url = r.links.get('next', {}).get('url')
    out[lang] = sorted(files, key=lambda f: f['path'])
    print(lang, len(files), 'files', round(sum(f['size'] for f in files) / 1e9, 1), 'GB', file=sys.stderr)
json.dump({'repo': REPO, 'revision': REV, 'shards': out}, open(sys.argv[1], 'w'), indent=1)
