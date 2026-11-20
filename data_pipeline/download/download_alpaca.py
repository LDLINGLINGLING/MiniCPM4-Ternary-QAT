"""Download upstream Alpaca JSON and convert it to the trainer's CSV schema.

The English source is tatsu-lab/stanford_alpaca and the Chinese source is
ymcui/Chinese-LLaMA-Alpaca. Raw JSON is kept beside the generated CSV. Existing
final files are never overwritten; an interrupted .partial download is retried
from byte zero. The manifest records actual SHA256 values for later freezing.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import requests


SOURCES = {
    'en': 'https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json',
    'zh': 'https://raw.githubusercontent.com/ymcui/Chinese-LLaMA-Alpaca/main/data/alpaca_data_zh_51k.json',
}


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output', type=Path, required=True,
                        help='Dedicated Alpaca download directory; do not point at the source checkout.')
    parser.add_argument('--languages', nargs='+', choices=sorted(SOURCES), default=['en', 'zh'])
    parser.add_argument('--timeout', type=int, default=120, help='Per-response read timeout in seconds.')
    return parser


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def validate_records(raw_path):
    records = json.loads(Path(raw_path).read_text(encoding='utf-8'))
    if not isinstance(records, list) or not records:
        raise ValueError(f'Expected a nonempty JSON list: {raw_path}')
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f'Record {index} is not an object')
        if not isinstance(record.get('instruction'), str) or not record['instruction'].strip():
            raise ValueError(f'Record {index} lacks instruction')
        if not isinstance(record.get('output'), str) or not record['output'].strip():
            raise ValueError(f'Record {index} lacks output')
        if not isinstance(record.get('input', ''), str):
            raise ValueError(f'Record {index} has non-string input')
    return records


def fetch(url, target, timeout):
    target = Path(target)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    for attempt in range(6):
        try:
            with requests.get(url, stream=True, timeout=(20, timeout)) as response:
                response.raise_for_status()
                with partial.open('wb') as handle:
                    for block in response.iter_content(8 * 1024 * 1024):
                        if block:
                            handle.write(block)
            validate_records(partial)
            partial.replace(target)
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 20))


def convert_to_csv(raw_path, output):
    records = validate_records(raw_path)
    output = Path(output)
    if output.exists():
        return len(records)
    temporary = output.with_name(output.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['instruction', 'input', 'output'], lineterminator='\n')
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, '') for key in writer.fieldnames})
    temporary.replace(output)
    return len(records)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.timeout < 1:
        raise ValueError('--timeout must be positive')
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for language in args.languages:
        raw = output / f'alpaca_{language}.json'
        csv_path = output / f'alpaca_{language}.csv'
        fetch(SOURCES[language], raw, args.timeout)
        count = convert_to_csv(raw, csv_path)
        files.append({
            'language': language,
            'source': SOURCES[language],
            'records': count,
            'raw': {'path': raw.name, 'sha256': sha256_file(raw)},
            'csv': {'path': csv_path.name, 'sha256': sha256_file(csv_path)},
        })
        print(f'VERIFIED {language}: {count} records -> {csv_path}', flush=True)
    atomic_json(output / 'download-manifest.json', {'files': files})
    print(f'FINISHED SUCCESS: {len(files)} languages', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
