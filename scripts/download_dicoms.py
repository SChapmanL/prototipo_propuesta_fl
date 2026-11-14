"""Download a supplied manifest with the official Redivis client. No secrets in code."""
import argparse
import base64
import csv
import hashlib
from pathlib import Path, PurePosixPath
import sys

def check(path, row):
    if path.stat().st_size != int(row['size_bytes']):
        raise ValueError('Size mismatch')
    h = hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    if base64.b64encode(h.digest()).decode() != row['md5_hash']:
        raise ValueError('MD5 mismatch')
    import pydicom
    import numpy as np
    ds = pydicom.dcmread(path)
    a = ds.pixel_array
    if a.ndim != 2 or a.size == 0 or not np.isfinite(a).all() or a.max() <= a.min():
        raise ValueError('Unexpected or constant pixel array')
    if ds.PhotometricInterpretation not in ('MONOCHROME1', 'MONOCHROME2'):
        raise ValueError('Unexpected photometric interpretation')
    return f'{a.shape[0]}x{a.shape[1]}', ds.PhotometricInterpretation

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, default=Path('manifests/pilot20.csv'))
    ap.add_argument('--output-dir', type=Path, default=Path('data/raw/dicom'))
    ap.add_argument('--download', action='store_true', help='Without this flag, inspect the list only')
    args = ap.parse_args()
    with args.manifest.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    print(f'{len(rows)} files, {sum(int(r["size_bytes"]) for r in rows)/1e9:.3f} GB')
    if not args.download:
        print('Dry run. Add --download to download and validate. Requires your authorized Redivis account.')
        return
    import redivis
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = args.output_dir/'download_validation.csv'
    failures = 0
    with log.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file_name','status','shape','photometric','error'])
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            entry = dict(file_name=row['file_name'], status='failed', shape='', photometric='', error='')
            try:
                relative = PurePosixPath(row['file_name'])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Unsafe file path')
                dest = args.output_dir.joinpath(*relative.parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    temporary = dest.with_suffix('.dcm.part')
                    redivis.file(row['file_id']).download(str(temporary), overwrite=True)
                    shape, photo = check(temporary, row)
                    temporary.replace(dest)
                else:
                    shape, photo = check(dest, row)
                entry.update(status='ok', shape=shape, photometric=photo)
            except Exception as exc:
                failures += 1
                # Avoid logging authentication exceptions, URLs, or credentials.
                entry['error'] = type(exc).__name__
            writer.writerow(entry)
            f.flush()
            print(f'{i}/{len(rows)} {row["file_name"]}: {entry["status"]}')
    print(f'Failures: {failures}. Log: {log}')
    if failures:
        print('Review failures before assigning final splits. No automatic replacements.')
        sys.exit(1)

if __name__ == '__main__':
    main()
