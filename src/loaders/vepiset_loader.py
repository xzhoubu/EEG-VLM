import csv
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import signal


VEPISET_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz',
    'PG1', 'PG2', 'A1', 'A2', 'ECG1', 'ECG2', 'EMG1', 'EMG2', 'EMG3', 'EMG4',
]

REGION6_LABELS = {
    'Non-IED': 0,
    'Generalized-IED': 1,
    'Frontal-IED': 2,
    'Temporal-IED': 3,
    'Centro-Parietal-IED': 4,
    'Occipital-IED': 5,
}
REGION6_TO_REGION5 = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
REGION6_CLASS_NAMES = ['Non-IED', 'Generalized-IED', 'Frontal-IED', 'Temporal-IED', 'Centro-Parietal-IED', 'Occipital-IED']
REGION5_CLASS_NAMES = ['Generalized-IED', 'Frontal-IED', 'Temporal-IED', 'Centro-Parietal-IED', 'Occipital-IED']

TASK_LABEL_KEY_VEPISET = {1: 'ied_binary', 2: 'ied_region5', 3: 'ied_region6'}
TASK_TYPE_VEPISET = {1: 'classification', 2: 'classification', 3: 'classification'}
NUM_CLASSES_VEPISET = {1: 2, 2: 5, 3: 6}
VEPISET_SOURCE_SFREQ = 500
VEPISET_DEFAULT_SFREQ = 250
VEPISET_DEFAULT_WINDOW_SEC = 1.0


def parse_subjects(spec):

    if isinstance(spec, (list, tuple, set, np.ndarray)):
        return [int(value) for value in spec]
    values = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, stop = (int(value) for value in part.split('-', 1))
            values.extend(range(start, stop + 1))
        else:
            values.append(int(part))
    return values


def _default_subject_ids():
    repo_root = Path(__file__).resolve().parents[2]
    mat_dir = repo_root / 'data' / 'vepiset' / 'MAT_Files'
    if mat_dir.is_dir():
        ids = sorted(p.stem for p in mat_dir.glob('*.mat'))
        if ids:
            return ids
    return [f'S{i:03d}' for i in range(1, 85)]


SUBJECT_IDS = _default_subject_ids()
IDX_TO_ID_VEPISET = {i + 1: sid for i, sid in enumerate(SUBJECT_IDS)}


def _subject_maps(data_dir: str):
    mat_dir = Path(data_dir) / 'MAT_Files'
    ids = sorted(p.stem for p in mat_dir.glob('*.mat')) if mat_dir.is_dir() else []
    if not ids:
        ids = _default_subject_ids()
    idx_to_id = {i + 1: sid for i, sid in enumerate(ids)}
    id_to_idx = {sid: i + 1 for i, sid in enumerate(ids)}
    return idx_to_id, id_to_idx


def _format_cache_token(value) -> str:
    return str(value).replace('.', 'p')


def _cache_path(data_dir: str, feature_tag: str, sfreq: int, window_sec: float, overlap_ratio: float) -> str:
    return os.path.join(
        data_dir,
        f'cache_vepiset_feat{feature_tag}_sf{sfreq}_win{_format_cache_token(window_sec)}_ov{_format_cache_token(overlap_ratio)}.npz',
    )


def _auto_subject_split(subjects: np.ndarray, region6_labels: np.ndarray, seed: int,
                        train_ratio: float, val_ratio: float):
    unique_subjects = sorted(int(s) for s in np.unique(subjects))
    n_subjects = len(unique_subjects)
    n_train = max(1, int(round(n_subjects * train_ratio)))
    n_val = max(1, int(round(n_subjects * val_ratio)))
    n_test = max(1, n_subjects - n_train - n_val)
    while n_train + n_val + n_test > n_subjects:
        n_train = max(1, n_train - 1)
    while n_train + n_val + n_test < n_subjects:
        n_train += 1

    by_subj = {}
    total_counts = np.zeros(6, dtype=np.float64)
    for sid in unique_subjects:
        mask = subjects == sid
        counts = np.bincount(region6_labels[mask], minlength=6).astype(np.float64)
        by_subj[sid] = counts
        total_counts += counts

    target_counts = {
        'train': total_counts * train_ratio,
        'val': total_counts * val_ratio,
        'test': total_counts * max(0.0, 1.0 - train_ratio - val_ratio),
    }
    target_sizes = {'train': n_train, 'val': n_val, 'test': n_test}
    current_counts = {k: np.zeros(6, dtype=np.float64) for k in target_sizes}
    assignments = {k: [] for k in target_sizes}
    assigned_subjects = set()

    rng = np.random.RandomState(seed)
    rare_weights = np.zeros(6, dtype=np.float64)
    positive_totals = total_counts[1:]
    rare_weights[1:] = 1.0 / np.maximum(positive_totals, 1.0)

    def subject_priority(item):
        sid, counts = item
        positive = counts[1:].sum()
        rare_score = float(np.dot(counts, rare_weights))
        n_regions = int((counts[1:] > 0).sum())
        return (-rare_score, -n_regions, -positive, -counts.sum(), rng.rand())

    def choose_seed_split(cls):
        choices = []
        for split_name in ['train', 'val', 'test']:
            if len(assignments[split_name]) >= target_sizes[split_name]:
                continue
            choices.append((
                current_counts[split_name][cls],
                len(assignments[split_name]),
                current_counts[split_name][1:].sum(),
                split_name,
            ))
        if not choices:
            return None
        choices.sort(key=lambda x: (x[0], x[1], x[2]))
        return choices[0][-1]


    for cls in np.argsort(np.maximum(total_counts[1:], 1.0)) + 1:
        subj_list = [item for item in by_subj.items() if item[1][cls] > 0 and item[0] not in assigned_subjects]
        subj_list = sorted(subj_list, key=lambda item: (-item[1][cls], -item[1][1:].sum(), rng.rand()))
        for sid, counts in subj_list[:3]:
            split_name = choose_seed_split(cls)
            if split_name is None:
                break
            assignments[split_name].append(sid)
            current_counts[split_name] += counts
            assigned_subjects.add(sid)

    ordered = [item for item in sorted(by_subj.items(), key=subject_priority) if item[0] not in assigned_subjects]
    for sid, counts in ordered:
        candidates = []
        for split_name in ['train', 'val', 'test']:
            cap = target_sizes[split_name]
            if len(assignments[split_name]) >= cap:
                continue
            new_counts = current_counts[split_name] + counts
            count_loss = np.abs(new_counts - target_counts[split_name]).sum()
            size_loss = abs((len(assignments[split_name]) + 1) - cap)
            positive_missing = 0.0
            if counts[1:].sum() > 0:
                for cls in range(1, 6):
                    if counts[cls] > 0 and current_counts[split_name][cls] == 0:
                        positive_missing -= 10.0
            overflow_penalty = 1000.0 if len(assignments[split_name]) + 1 > cap else 0.0
            score = count_loss + 5.0 * size_loss + overflow_penalty + positive_missing
            candidates.append((score, split_name))
        if not candidates:
            candidates = [(len(assignments[k]), k) for k in ['train', 'val', 'test']]
        _, best_split = min(candidates, key=lambda x: x[0])
        assignments[best_split].append(sid)
        current_counts[best_split] += counts

    return (
        ','.join(str(x) for x in sorted(assignments['train'])),
        ','.join(str(x) for x in sorted(assignments['val'])),
        ','.join(str(x) for x in sorted(assignments['test'])),
    )


def _parse_source_sfreq(path: Path) -> int:
    prefix = path.stem.rsplit('__', 1)[0]
    parts = prefix.split('_')
    if len(parts) >= 4:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return VEPISET_SOURCE_SFREQ


def _load_segment(path: Path) -> np.ndarray:
    x = np.load(path)
    if x.ndim != 2:
        raise ValueError(f'Expected 2D EEG array in {path}, got shape {x.shape}')
    if x.shape[0] == len(VEPISET_CHANNELS):
        x = x
    elif x.shape[1] == len(VEPISET_CHANNELS):
        x = x.T
    else:
        raise ValueError(f'Unexpected channel dimension in {path}: {x.shape}')

    return x.astype(np.float32, copy=False)


def _downsample_segment(x: np.ndarray, source_sfreq: int, target_sfreq: int) -> np.ndarray:
    if source_sfreq <= 0 or target_sfreq <= 0:
        raise ValueError(f'Invalid sampling rate: source={source_sfreq}, target={target_sfreq}')
    if source_sfreq == target_sfreq:
        return x.astype(np.float32, copy=False)
    x_ds = signal.resample_poly(x, up=target_sfreq, down=source_sfreq, axis=1)
    return x_ds.astype(np.float32, copy=False)


def _window_segment(x: np.ndarray, window_len: int, step_len: int) -> Tuple[list, list]:
    if window_len <= 0:
        raise ValueError(f'window_len must be positive, got {window_len}')
    if step_len <= 0:
        raise ValueError(f'step_len must be positive, got {step_len}')

    total_len = x.shape[1]
    if total_len < window_len:
        pad = np.zeros((x.shape[0], window_len - total_len), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=1)
        total_len = x.shape[1]

    starts = list(range(0, total_len - window_len + 1, step_len))
    if not starts:
        starts = [0]

    windows = []
    spans = []
    for start in starts:
        end = start + window_len
        windows.append(x[:, start:end].T.astype(np.float32, copy=False))
        spans.append((start, end))
    return windows, spans


def load_all_data(data_dir, window_sec=VEPISET_DEFAULT_WINDOW_SEC, overlap_ratio=0.0, sfreq=VEPISET_DEFAULT_SFREQ,
                  use_features=False, feature_set='psd_de', use_cache=True, **_kwargs):
    feature_tag = feature_set if use_features else 'raw'
    cache_path = _cache_path(data_dir, feature_tag, sfreq, window_sec, overlap_ratio)
    if use_cache and os.path.isfile(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        return {
            'samples': cached['samples'],
            'labels': cached['labels'].item(),
            'subjects': cached['subjects'],
            'conditions': cached['conditions'],
            'ch_names': list(cached['ch_names']),
            'valid': cached['valid'].item(),
            'feature_set': str(cached['feature_set']),
            'sample_files': list(cached['sample_files']),
            'subject_id_map': cached['subject_id_map'].item(),
            'sfreq': int(cached['sfreq']),
            'window_sec': float(cached['window_sec']),
            'overlap_ratio': float(cached['overlap_ratio']),
        }

    if not 0.0 <= float(overlap_ratio) < 1.0:
        raise ValueError(f'overlap_ratio must be in [0, 1), got {overlap_ratio}')
    window_len = int(round(float(window_sec) * sfreq))
    step_len = int(round(window_len * (1.0 - float(overlap_ratio))))
    if window_len <= 0:
        raise ValueError(f'window_sec={window_sec} and sfreq={sfreq} lead to non-positive window length.')
    if step_len <= 0:
        raise ValueError(f'window_sec={window_sec} and overlap_ratio={overlap_ratio} lead to non-positive step length.')

    idx_to_id, id_to_idx = _subject_maps(data_dir)

    all_samples = []
    binary_labels = []
    region5_labels = []
    region6_labels = []
    region5_valid = []
    subjects = []
    conditions = []
    sample_files = []

    for class_name, label6 in sorted(REGION6_LABELS.items(), key=lambda kv: kv[1]):
        class_dir = Path(data_dir) / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.glob('*.npy')):
            sample = _load_segment(path)
            subj_id = path.stem.split('_')[0]
            subj_idx = id_to_idx[subj_id]
            source_sfreq = _parse_source_sfreq(path)
            sample = _downsample_segment(sample, source_sfreq=source_sfreq, target_sfreq=sfreq)
            sample_windows, sample_spans = _window_segment(sample, window_len=window_len, step_len=step_len)

            binary = 0 if label6 == 0 else 1
            region5 = 0 if label6 == 0 else REGION6_TO_REGION5[label6]

            rel_path = str(path.relative_to(data_dir))
            for window_idx, (sample_window, span) in enumerate(zip(sample_windows, sample_spans)):
                all_samples.append(sample_window)
                binary_labels.append(binary)
                region5_labels.append(region5)
                region6_labels.append(label6)
                region5_valid.append(label6 != 0)
                subjects.append(subj_idx)
                conditions.append(label6)
                sample_files.append(f'{rel_path}::ds{sfreq}_win{window_idx}_{span[0]}_{span[1]}')

    if not all_samples:
        raise RuntimeError(f'No .npy segments found in {data_dir}')

    samples = np.stack(all_samples, axis=0).astype(np.float32)
    if use_features:
        samples = extract_features(samples, sfreq=sfreq, feature_set=feature_set).astype(np.float32)

    labels = {
        'ied_binary': np.asarray(binary_labels, dtype=np.int64),
        'ied_region5': np.asarray(region5_labels, dtype=np.int64),
        'ied_region6': np.asarray(region6_labels, dtype=np.int64),
    }
    valid = {
        'ied_binary': np.ones(len(samples), dtype=bool),
        'ied_region5': np.asarray(region5_valid, dtype=bool),
        'ied_region6': np.ones(len(samples), dtype=bool),
    }
    out = {
        'samples': samples,
        'labels': labels,
        'subjects': np.asarray(subjects, dtype=np.int64),
        'conditions': np.asarray(conditions, dtype=np.int64),
        'ch_names': VEPISET_CHANNELS,
        'valid': valid,
        'feature_set': feature_tag,
        'sample_files': np.asarray(sample_files, dtype=object),
        'subject_id_map': idx_to_id,
        'sfreq': int(sfreq),
        'window_sec': float(window_sec),
        'overlap_ratio': float(overlap_ratio),
    }
    if use_cache:
        np.savez(cache_path, **out)
    out['sample_files'] = list(out['sample_files'])
    return out


def split_arrays(all_data, task, split_mode='cross-sub',
                 train_subs='auto', val_subs='auto', test_subs='auto',
                 train_ratio=0.8, val_ratio=0.1,
                 normalize=True, seed=42, split_seed=None,
                 mode='end2end', window_sec=4.0, sfreq_in=500,
                 task_label_key_map=None, task_type_map=None):
    task_label_key_map = task_label_key_map or TASK_LABEL_KEY_VEPISET
    task_type_map = task_type_map or TASK_TYPE_VEPISET
    label_key = task_label_key_map[task]

    samples = all_data['samples']
    labels = all_data['labels'][label_key]
    subjects = all_data['subjects']
    conditions = all_data['conditions']
    valid = all_data['valid'][label_key]
    sample_files = np.asarray(all_data.get('sample_files', np.arange(len(samples))), dtype=object)

    mask = valid
    samples = samples[mask]
    labels = labels[mask]
    subjects = subjects[mask]
    conditions = conditions[mask]
    sample_files = sample_files[mask]

    if split_mode == 'cross-sub':
        if str(train_subs).lower() == 'auto' or str(val_subs).lower() == 'auto' or str(test_subs).lower() == 'auto':
            train_subs, val_subs, test_subs = _auto_subject_split(
                all_data['subjects'],
                all_data['labels']['ied_region6'],
                seed=seed if split_seed is None else split_seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
            )
        tr_subs = set(parse_subjects(train_subs))
        va_subs = set(parse_subjects(val_subs))
        te_subs = set(parse_subjects(test_subs))
        tr_idx = np.isin(subjects, list(tr_subs))
        va_idx = np.isin(subjects, list(va_subs))
        te_idx = np.isin(subjects, list(te_subs))
    else:
        raise ValueError(f'Unknown split_mode: {split_mode}')

    raw = {
        'train': samples[tr_idx].astype(np.float32, copy=False),
        'val': samples[va_idx].astype(np.float32, copy=False),
        'test': samples[te_idx].astype(np.float32, copy=False),
    }
    split_meta = {
        'train': {
            'labels': labels[tr_idx].astype(np.int64),
            'subjects': subjects[tr_idx].astype(np.int64),
            'conditions': conditions[tr_idx].astype(np.int64),
            'sample_files': sample_files[tr_idx],
            'subject_spec': train_subs,
        },
        'val': {
            'labels': labels[va_idx].astype(np.int64),
            'subjects': subjects[va_idx].astype(np.int64),
            'conditions': conditions[va_idx].astype(np.int64),
            'sample_files': sample_files[va_idx],
            'subject_spec': val_subs,
        },
        'test': {
            'labels': labels[te_idx].astype(np.int64),
            'subjects': subjects[te_idx].astype(np.int64),
            'conditions': conditions[te_idx].astype(np.int64),
            'sample_files': sample_files[te_idx],
            'subject_spec': test_subs,
        },
    }

    return {
        'task': task,
        'task_label': label_key,
        'task_type': task_type_map[task],
        'raw': raw,
        'meta': split_meta,
        'sfreq_in': sfreq_in,
        'window_sec': window_sec,
        'ch_names': all_data['ch_names'],
        'dataset_min': float(samples.min()),
        'dataset_max': float(samples.max()),
    }
