"""Prepare candidate lists. Does not download DICOM or assign final splits."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

SEED = 42

def rank(*parts):
    return hashlib.sha256((str(SEED) + '|' + '|'.join(parts)).encode()).hexdigest()

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    source_names = ['df_chexpert_plus_240401.csv', 'report_fixed.json', 'dicom_train.csv']
    sources = {n: args.input_dir / n for n in source_names}
    labels = {}
    with sources['report_fixed.json'].open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            key = r['path_to_image']
            if key in labels:
                raise ValueError('Duplicate label image: ' + key)
            y = r['Pneumonia']
            if y not in (None, -1, 0, 1):
                raise ValueError('Unexpected Pneumonia code')
            labels[key] = y
    index = {}
    with sources['dicom_train.csv'].open(newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['file_name'] in index:
                raise ValueError('Duplicate DICOM path: ' + r['file_name'])
            index[r['file_name']] = r
    groups = collections.defaultdict(list)
    exclusions = []
    seen = set()
    counts = collections.Counter()
    with sources['df_chexpert_plus_240401.csv'].open(newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            key = r['path_to_image']
            if key in seen:
                raise ValueError('Duplicate metadata image: ' + key)
            seen.add(key)
            reason = None
            counts['metadata_rows'] += 1
            if r['split'] != 'train':
                reason = 'not_original_train'
            else:
                try:
                    age = float(r['age'])
                except ValueError:
                    age = float('nan')
                if not (age >= 18 and age < float('inf')):
                    reason = 'age_missing_invalid_or_under18'
                elif r['frontal_lateral'] != 'Frontal' or r['ap_pa'] not in ('AP', 'PA'):
                    reason = 'not_frontal_AP_PA'
                elif key not in labels:
                    reason = 'missing_label_record'
                elif labels[key] is None:
                    reason = 'pneumonia_null'
                elif labels[key] == -1:
                    reason = 'pneumonia_uncertain'
            file_name = r['path_to_dcm'].removeprefix('train/')
            if not reason and file_name not in index:
                reason = 'dicom_not_in_index'
            if reason:
                counts[reason] += 1
                exclusions.append({'image_id': key, 'reason': reason})
                continue
            ir = index[file_name]
            patient = r['deid_patient_id']
            assert file_name.split('/')[0] == patient
            study = file_name.split('/')[1]
            groups[patient].append(dict(
                image_id=key, patient_id=patient, study_id=study,
                file_id=ir['file_id'], file_name=file_name,
                dicom_path='data/raw/dicom/' + file_name,
                size_bytes=int(ir['size']), md5_hash=ir['md5_hash'],
                age=r['age'], sex=r['sex'], view=r['ap_pa'],
                label_raw=int(labels[key]), label=int(labels[key]),
                label_source='report_fixed.json', label_generator='not_verified',
                original_split=r['split'], status='pending_dicom_validation'))
            counts['eligible_images_before_patient_selection'] += 1
    pool = []
    cross_class = 0
    for patient in sorted(groups):
        rows = groups[patient]
        cross_class += len({r['label'] for r in rows}) > 1
        rows.sort(key=lambda r: (rank('image', patient, r['image_id']), r['image_id']))
        pool.append(rows[0])
        for r in rows[1:]:
            exclusions.append({'image_id': r['image_id'], 'reason': 'other_eligible_image_same_patient'})
    selected, reserve = [], []
    for label in (0, 1):
        rows = sorted((r for r in pool if r['label'] == label),
                      key=lambda r: (rank('patient', r['patient_id']), r['patient_id']))
        assert len(rows) >= 450, (label, len(rows))
        selected += rows[:400]
        reserve += rows[400:450]
    selected.sort(key=lambda r: (r['label'], rank('patient', r['patient_id'])))
    smoke = [r for label in (0, 1) for r in [x for x in selected if x['label'] == label][:10]]
    fields = list(selected[0])
    for name, rows in [('download_manifest.csv', selected), ('pilot20.csv', smoke), ('reserve100.csv', reserve)]:
        write_csv(out / name, rows, fields)
    write_csv(out / 'exclusions.csv', exclusions, ['image_id', 'reason'])
    assert len(selected) == len({r['patient_id'] for r in selected}) == 800
    assert len({r['patient_id'] for r in selected + reserve}) == 900
    assert collections.Counter(r['label'] for r in selected) == {0: 400, 1: 400}
    total = sum(r['size_bytes'] for r in selected)
    summary = dict(stage_counts=dict(counts),
        patients_with_both_labels_before_image_selection=cross_class,
        patient_pool_by_label=dict(collections.Counter(r['label'] for r in pool)),
        selected_images=800, selected_patients=800, selected_by_label={0:400, 1:400},
        selected_by_view=dict(collections.Counter(r['view'] for r in selected)),
        selected_by_sex=dict(collections.Counter(r['sex'] for r in selected)),
        download_bytes=total, download_GB=total / 1e9,
        pilot20_bytes=sum(r['size_bytes'] for r in smoke),
        status='Candidates only. Pixel validation and final splits pending.')
    config = dict(seed=SEED, original_split='train', age_min=18, views=['AP','PA'],
        label_source='report_fixed.json', label_generator='not_verified',
        label_policy={'1':1,'0':0,'-1':'exclude','null':'exclude'},
        image_per_patient='lowest SHA256 of 42|image|patient_id|image_id among eligible indexed images',
        patient_selection='lowest SHA256 of 42|patient|patient_id within each class',
        target_per_class=400, reserve_per_class=50,
        final_splits_assigned=False, source_sha256={n:sha256(p) for n,p in sources.items()})
    (out/'dataset_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out/'sample_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
