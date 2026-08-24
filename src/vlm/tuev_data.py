

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import scipy.signal
from PIL import Image
from torch.utils.data import Dataset

from runtime import default_data_dir, default_output_dir, resolve_data_dir
from vlm.data import stable_sample_id
from vlm.model_config import default_qwen3vl_model_path
from vlm.render import compute_render_stats, ensure_rendered_image


TUEV_TARGET_SFREQ = 200
TUEV_CLASS_NAMES = ['SPSW', 'GPED', 'PLED', 'EYEM', 'ARTF', 'BCKG']
TUEV_CHOICES = ['spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg']
TUEV_REC_LABEL_TO_INDEX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
TUEV_INDEX_TO_REC_LABEL = {idx: rec_label for rec_label, idx in TUEV_REC_LABEL_TO_INDEX.items()}
TUEV_LABEL_DESCRIPTIONS = {
    'spsw': 'spike and slow wave',
    'gped': 'generalized periodic epileptiform discharge',
    'pled': 'periodic lateralized epileptiform discharge',
    'eyem': 'eye movement',
    'artf': 'artifact',
    'bckg': 'background',
}
TUEV_TCP_PAIRS = [
    ('FP1', 'F7'),
    ('F7', 'T3'),
    ('T3', 'T5'),
    ('T5', 'O1'),
    ('FP2', 'F8'),
    ('F8', 'T4'),
    ('T4', 'T6'),
    ('T6', 'O2'),
    ('FP1', 'F3'),
    ('F3', 'C3'),
    ('C3', 'P3'),
    ('P3', 'O1'),
    ('FP2', 'F4'),
    ('F4', 'C4'),
    ('C4', 'P4'),
    ('P4', 'O2'),
]
TUEV_TCP_LABELS = [f'{left}-{right}' for left, right in TUEV_TCP_PAIRS]
TUEV_ZERO_CHANNEL = 'ZERO'
TUEV_VLM_CHANNEL_NAMES = TUEV_TCP_LABELS + [TUEV_ZERO_CHANNEL]

DEFAULT_TUEV_PROMPT = (
    'You are given a visualization of a TUEV EEG event segment, including an '
    'ACNS TCP montage waveform and a spectrogram.\n\n'
    'Use only the EEG signal shown in the image. Classify the event into exactly '
    'one of these options:\n\n'
    'Options:\n'
    'spsw - spike and slow wave\n'
    'gped - generalized periodic epileptiform discharge\n'
    'pled - periodic lateralized epileptiform discharge\n'
    'eyem - eye movement\n'
    'artf - artifact\n'
    'bckg - background\n\n'
    'Return exactly one option code: spsw, gped, pled, eyem, artf, or bckg. '
    'Do not include explanations, punctuation, or extra words.'
)
DEFAULT_TUEV_WAVEFORM_PROMPT = (
    'You are given a waveform visualization of a TUEV EEG event segment, shown '
    'as an ACNS TCP montage.\n\n'
    'Classify the event into exactly one option code: spsw, gped, pled, eyem, '
    'artf, or bckg. Return only the option code.'
)
DEFAULT_TUEV_SPECTROGRAM_PROMPT = (
    'You are given a spectrogram visualization of a TUEV EEG event segment, '
    'shown across ACNS TCP montage channels.\n\n'
    'Classify the event into exactly one option code: spsw, gped, pled, eyem, '
    'artf, or bckg. Return only the option code.'
)
DEFAULT_TUEV_CWT_PROMPT = (
    'You are given a CWT scalogram visualization of a TUEV EEG event segment, '
    'shown across ACNS TCP montage channels.\n\n'
    'Classify the event into exactly one option code: spsw, gped, pled, eyem, '
    'artf, or bckg. Return only the option code.'
)
DEFAULT_TUEV_COMBINED_CWT_PROMPT = (
    'You are given a visualization of a TUEV EEG event segment, including an '
    'ACNS TCP montage waveform and a CWT scalogram.\n\n'
    'Classify the event into exactly one option code: spsw, gped, pled, eyem, '
    'artf, or bckg. Return only the option code.'
)
DEFAULT_TUEV_RESULT_ROOT = default_output_dir('runs', 'tuev_vlm')
DEFAULT_TUEV_RENDER_CACHE_ROOT = default_output_dir('cache', 'tuev_vlm')
DEFAULT_TUEV_MODEL_PATH = default_qwen3vl_model_path()


@dataclass
class TUEVPreparedSplitData:


    name: str
    samples: 'TUEVLazySamples'
    manifest: pd.DataFrame


@dataclass
class TUEVPreparedWorkspace:


    prep_dir: str
    channel_names: list[str]
    sfreq: int
    duration_sec: float
    prompt: str
    render_config: dict[str, Any]
    render_stats: dict[str, Any]
    subject_split: dict[str, list[str]]
    splits: dict[str, TUEVPreparedSplitData]
    class_names: list[str]
    choices: list[str]


def resolve_tuev_prompt(args) -> str:

    prompt = str(getattr(args, 'prompt', DEFAULT_TUEV_PROMPT))
    if prompt != DEFAULT_TUEV_PROMPT:
        return prompt
    single_image_kind = str(getattr(args, 'single_image_kind', 'combined') or 'combined')
    if single_image_kind == 'waveform':
        return DEFAULT_TUEV_WAVEFORM_PROMPT
    if single_image_kind == 'spectrogram':
        return DEFAULT_TUEV_SPECTROGRAM_PROMPT
    if single_image_kind == 'cwt':
        return DEFAULT_TUEV_CWT_PROMPT
    if single_image_kind == 'combined_cwt':
        return DEFAULT_TUEV_COMBINED_CWT_PROMPT
    return DEFAULT_TUEV_PROMPT


def build_tuev_render_config(args) -> dict[str, Any]:

    montage_pairs = [[label, TUEV_ZERO_CHANNEL] for label in TUEV_TCP_LABELS]
    payload = {
        'render_version': str(getattr(args, 'render_version', 'v1')).strip() or 'v1',
        'image_input_mode': str(getattr(args, 'image_input_mode', 'single_panel')).strip() or 'single_panel',
        'single_image_kind': str(getattr(args, 'single_image_kind', 'combined')).strip() or 'combined',
        'image_size': int(getattr(args, 'image_size', 896)),
        'montage_pairs': montage_pairs,
        'montage_labels': list(TUEV_TCP_LABELS),
        'normalization_mode': str(getattr(args, 'normalization_mode', 'train_global')).strip() or 'train_global',
        'spectrogram_freq_min': float(getattr(args, 'spectrogram_freq_min', 1.0)),
        'spectrogram_freq_max': float(getattr(args, 'spectrogram_freq_max', 45.0)),
        'spectrogram_nperseg': int(getattr(args, 'spectrogram_nperseg', 64)),
        'spectrogram_noverlap': int(getattr(args, 'spectrogram_noverlap', 48)),
        'spectrogram_nfft': int(getattr(args, 'spectrogram_nfft', 256)),
        'spectrogram_log_eps': 1e-8,
        'waveform_percentile': float(getattr(args, 'waveform_percentile', 0.995)),
        'spectrogram_quantile_low': float(getattr(args, 'spectrogram_quantile_low', 0.01)),
        'spectrogram_quantile_high': float(getattr(args, 'spectrogram_quantile_high', 0.99)),
        'waveform_stats_max_samples': int(getattr(args, 'waveform_stats_max_samples', 128)),
        'spectrogram_stats_max_samples': int(getattr(args, 'spectrogram_stats_max_samples', 32)),
    }
    if payload['single_image_kind'] in {'cwt', 'combined_cwt'}:
        payload.update({
            'cwt_freq_min': float(getattr(args, 'cwt_freq_min', getattr(args, 'spectrogram_freq_min', 1.0))),
            'cwt_freq_max': float(getattr(args, 'cwt_freq_max', getattr(args, 'spectrogram_freq_max', 45.0))),
            'cwt_num_freqs': int(getattr(args, 'cwt_num_freqs', 48)),
            'cwt_freq_spacing': str(getattr(args, 'cwt_freq_spacing', 'log')).strip() or 'log',
            'cwt_morlet_w': float(getattr(args, 'cwt_morlet_w', 6.0)),
            'cwt_support': float(getattr(args, 'cwt_support', 6.0)),
            'cwt_time_stride': int(getattr(args, 'cwt_time_stride', 16)),
            'cwt_log_eps': 1e-8,
            'cwt_stats_max_samples': int(getattr(args, 'cwt_stats_max_samples', 16)),
            'cwt_quantile_low': float(getattr(args, 'cwt_quantile_low', 0.01)),
            'cwt_quantile_high': float(getattr(args, 'cwt_quantile_high', 0.99)),
        })
    serial = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(',', ':'))
    config_hash = hashlib.sha1(serial.encode('utf-8')).hexdigest()[:12]
    payload['config_hash'] = config_hash
    payload['render_id'] = f'{payload["render_version"]}-{config_hash}'
    return payload


def _channel_names_json(channel_names: list[str]) -> str:
    return json.dumps(list(channel_names), ensure_ascii=True, separators=(',', ':'))


def _clean_channel_name(name: str) -> str:
    cleaned = str(name).upper().strip()
    cleaned = re.sub(r'^EEG\s+', '', cleaned)
    cleaned = re.sub(r'-(REF|LE|A1|A2)$', '', cleaned)
    cleaned = cleaned.replace('FPZ', 'FPZ')
    return cleaned


def _patient_from_path(path: str | Path) -> str:
    return Path(path).parent.name


def _official_split_from_path(path: str | Path) -> str:
    parts = Path(path).parts
    if 'train' in parts:
        return 'train'
    if 'eval' in parts:
        return 'test'
    raise ValueError(f'Cannot infer TUEV official split from path: {path}')


def _rec_path_for_edf(path: str | Path) -> Path:
    return Path(path).with_suffix('.rec')


def _read_rec_events(path: str | Path, duration_sec: float) -> list[dict[str, Any]]:
    rec_path = _rec_path_for_edf(path)
    if not rec_path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(rec_path.read_text(errors='ignore').splitlines(), start=1):
        parts = [part for part in re.split(r'[\s,]+', line.strip()) if part]
        if len(parts) < 4:
            continue
        try:
            channel = int(float(parts[0]))
            start = float(parts[1])
            stop = float(parts[2])
            rec_label = int(float(parts[3]))
        except ValueError:
            continue
        if rec_label not in TUEV_REC_LABEL_TO_INDEX:
            continue
        start = max(0.0, min(float(duration_sec), start))
        stop = max(0.0, min(float(duration_sec), stop))
        if stop <= start:
            continue
        events.append({
            'edf_path': str(Path(path).resolve()),
            'rec_path': str(rec_path.resolve()),
            'line_no': int(line_no),
            'channel': int(channel),
            'start_sec': float(start),
            'end_sec': float(stop),
            'rec_label': int(rec_label),
            'label': int(TUEV_REC_LABEL_TO_INDEX[rec_label]),
        })
    return events


def _merge_labeled_events(events: list[dict[str, Any]], gap_sec: float) -> list[dict[str, Any]]:

    merged: list[dict[str, Any]] = []
    grouped: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(int(event['rec_label']), []).append(event)

    for rec_label, group in sorted(grouped.items()):
        group = sorted(group, key=lambda item: (float(item['start_sec']), float(item['end_sec']), int(item['channel'])))
        current: dict[str, Any] | None = None
        for event in group:
            if current is None:
                current = {
                    'edf_path': event['edf_path'],
                    'rec_path': event['rec_path'],
                    'start_sec': float(event['start_sec']),
                    'end_sec': float(event['end_sec']),
                    'rec_label': int(rec_label),
                    'label': int(event['label']),
                    'channels': {int(event['channel'])},
                    'annotation_count': 1,
                    'source_lines': [int(event['line_no'])],
                }
                continue
            if float(event['start_sec']) <= float(current['end_sec']) + float(gap_sec):
                current['end_sec'] = max(float(current['end_sec']), float(event['end_sec']))
                current['start_sec'] = min(float(current['start_sec']), float(event['start_sec']))
                current['channels'].add(int(event['channel']))
                current['annotation_count'] += 1
                current['source_lines'].append(int(event['line_no']))
            else:
                merged.append(current)
                current = {
                    'edf_path': event['edf_path'],
                    'rec_path': event['rec_path'],
                    'start_sec': float(event['start_sec']),
                    'end_sec': float(event['end_sec']),
                    'rec_label': int(rec_label),
                    'label': int(event['label']),
                    'channels': {int(event['channel'])},
                    'annotation_count': 1,
                    'source_lines': [int(event['line_no'])],
                }
        if current is not None:
            merged.append(current)

    for item in merged:
        item['channels'] = sorted(item['channels'])
        item['source_lines'] = sorted(item['source_lines'])
    return sorted(merged, key=lambda item: (int(item['rec_label']), float(item['start_sec']), float(item['end_sec'])))


def _merge_intervals(intervals: list[tuple[float, float]], gap_sec: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted((float(s), float(e)) for s, e in intervals if float(e) > float(s))
    merged = [intervals[0]]
    for start, stop in intervals[1:]:
        last_start, last_stop = merged[-1]
        if start <= last_stop + float(gap_sec):
            merged[-1] = (last_start, max(last_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def _subtract_intervals(base: list[tuple[float, float]],
                        blockers: list[tuple[float, float]],
                        margin_sec: float,
                        duration_sec: float) -> list[tuple[float, float]]:
    blockers = [
        (max(0.0, float(start) - float(margin_sec)), min(float(duration_sec), float(stop) + float(margin_sec)))
        for start, stop in blockers
        if float(stop) > float(start)
    ]
    blockers = _merge_intervals(blockers, gap_sec=0.0)
    out: list[tuple[float, float]] = []
    for base_start, base_stop in base:
        segments = [(float(base_start), float(base_stop))]
        for block_start, block_stop in blockers:
            next_segments: list[tuple[float, float]] = []
            for seg_start, seg_stop in segments:
                if block_stop <= seg_start or block_start >= seg_stop:
                    next_segments.append((seg_start, seg_stop))
                    continue
                if block_start > seg_start:
                    next_segments.append((seg_start, min(block_start, seg_stop)))
                if block_stop < seg_stop:
                    next_segments.append((max(block_stop, seg_start), seg_stop))
            segments = next_segments
            if not segments:
                break
        out.extend((s, e) for s, e in segments if e > s)
    return out


def _window_start_from_center(start_sec: float, end_sec: float, duration_sec: float, window_sec: float) -> float:
    if duration_sec <= window_sec:
        return 0.0
    center = 0.5 * (float(start_sec) + float(end_sec))
    start = center - 0.5 * float(window_sec)
    start = max(0.0, min(start, float(duration_sec) - float(window_sec)))
    return float(start)


def _background_candidates(safe_intervals: list[tuple[float, float]], window_sec: float) -> list[tuple[float, float]]:
    candidates: list[tuple[float, float]] = []
    for start, stop in safe_intervals:
        length = float(stop) - float(start)
        if length < float(window_sec):
            continue
        n_windows = int(math.floor(length / float(window_sec)))
        for idx in range(max(1, n_windows)):
            win_start = float(start) + idx * float(window_sec)
            if win_start + float(window_sec) <= float(stop) + 1e-6:
                candidates.append((win_start, win_start + float(window_sec)))
    return candidates


def _choose_limit_indices(labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if limit <= 0 or n <= limit:
        return np.arange(n, dtype=np.int64)

    rng = np.random.RandomState(seed)
    selected: list[int] = []
    for label in sorted(np.unique(labels).tolist()):
        indices = np.where(labels == label)[0]
        if len(indices) > 0 and len(selected) < limit:
            selected.append(int(indices[0]))

    remaining = limit - len(selected)
    if remaining > 0:
        leftovers = np.setdiff1d(
            np.arange(n, dtype=np.int64),
            np.asarray(selected, dtype=np.int64),
            assume_unique=False,
        )
        if len(leftovers) > 0:
            extra = rng.choice(leftovers, size=min(remaining, len(leftovers)), replace=False)
            selected.extend(int(item) for item in extra.tolist())
    return np.asarray(sorted(selected[:limit]), dtype=np.int64)


def _scan_file_metadata(data_dir: str) -> pd.DataFrame:
    root = Path(data_dir).resolve() / 'edf'
    edf_files = sorted(root.glob('train/*/*.edf')) + sorted(root.glob('eval/*/*.edf'))
    if not edf_files:
        raise FileNotFoundError(f'No TUEV EDF files found under {root}')

    rows = []
    missing_rec = 0
    for idx, path in enumerate(edf_files):
        if idx > 0 and idx % 100 == 0:
            print(f'  TUEV header scan: {idx}/{len(edf_files)} EDF files')
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        clean_to_raw = {_clean_channel_name(ch): ch for ch in raw.ch_names}
        required = sorted({name for pair in TUEV_TCP_PAIRS for name in pair})
        missing = [name for name in required if name not in clean_to_raw]
        if missing:
            print(f'  [skip] missing TUEV channels {missing}: {path}')
            continue
        rec_path = _rec_path_for_edf(path)
        if not rec_path.is_file():
            missing_rec += 1
            continue
        rows.append({
            'file_idx': len(rows),
            'edf_path': str(path.resolve()),
            'rec_path': str(rec_path.resolve()),
            'official_split': _official_split_from_path(path),
            'patient': _patient_from_path(path),
            'orig_sfreq': float(raw.info['sfreq']),
            'duration_sec': float(raw.n_times / raw.info['sfreq']),
            'raw_channels': json.dumps({name: clean_to_raw[name] for name in required}, ensure_ascii=True),
        })
    if missing_rec:
        print(f'  [warn] skipped {missing_rec} EDF files without matching .rec annotations')
    if not rows:
        raise RuntimeError(f'No usable TUEV EDF/REC pairs found under {root}')
    return pd.DataFrame(rows)


def _split_patients(files: pd.DataFrame, val_fraction: float, seed: int) -> dict[str, list[str]]:
    train_patients = sorted(files.loc[files['official_split'] == 'train', 'patient'].unique().tolist())
    test_patients = sorted(files.loc[files['official_split'] == 'test', 'patient'].unique().tolist())
    rng = np.random.RandomState(seed)
    shuffled = np.asarray(train_patients, dtype=object)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * float(val_fraction)))
    n_val = min(max(n_val, 1), max(len(shuffled) - 1, 1))
    val_patients = sorted(str(item) for item in shuffled[:n_val].tolist())
    train_keep = sorted(str(item) for item in shuffled[n_val:].tolist())
    return {'train': train_keep, 'val': val_patients, 'test': test_patients}


def _files_for_split(files: pd.DataFrame, split_name: str, subject_split: dict[str, list[str]]) -> pd.DataFrame:
    if split_name == 'test':
        return files[files['official_split'] == 'test'].copy()
    return files[
        (files['official_split'] == 'train')
        & (files['patient'].isin(subject_split[split_name]))
    ].copy()


def _build_split_events(files: pd.DataFrame,
                        split_name: str,
                        window_sec: float,
                        merge_gap_sec: float,
                        bckg_exclusion_margin_sec: float,
                        bckg_sample_ratio: float,
                        max_bckg_per_split: int,
                        seed: int) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []
    del merge_gap_sec, bckg_exclusion_margin_sec, bckg_sample_ratio, max_bckg_per_split, seed

    for file_row in files.itertuples(index=False):
        events = _read_rec_events(file_row.edf_path, duration_sec=float(file_row.duration_sec))
        if not events:
            continue
        for event in events:
            win_start = float(event['start_sec']) - 2.0
            rows.append({
                'split': split_name,
                'label': int(event['label']),
                'rec_label': int(event['rec_label']),
                'label_code': TUEV_CHOICES[int(event['label'])],
                'label_name': TUEV_CLASS_NAMES[int(event['label'])],
                'subject': str(file_row.patient),
                'official_split': str(file_row.official_split),
                'condition': int(event['label']),
                'condition_name': TUEV_CLASS_NAMES[int(event['label'])],
                'edf_path': str(file_row.edf_path),
                'rec_path': str(file_row.rec_path),
                'event_start_sec': float(event['start_sec']),
                'event_end_sec': float(event['end_sec']),
                'window_start_sec': float(win_start),
                'window_end_sec': float(win_start + window_sec),
                'duration_sec': float(window_sec),
                'source_channels': json.dumps([int(event['channel'])], ensure_ascii=True),
                'source_annotation_count': 1,
                'source_lines': json.dumps([int(event['line_no'])], ensure_ascii=True),
                'is_background_sampled': 0,
                'processing_style': 'official_rec_event',
            })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(['subject', 'edf_path', 'event_start_sec', 'source_lines', 'label']).reset_index(drop=True)
    return frame


def _attach_manifest_columns(frame: pd.DataFrame,
                             split_name: str,
                             channel_names: list[str],
                             sfreq: int,
                             render_root: Path,
                             render_config: dict[str, Any]) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows = []
    channel_names_json = _channel_names_json(channel_names)
    for idx, row in enumerate(frame.itertuples(index=False)):
        sample_key = (
            f'tuev::{row.edf_path}::label={int(row.label)}::'
            f'event={float(row.event_start_sec):.3f}-{float(row.event_end_sec):.3f}::'
            f'window={float(row.window_start_sec):.3f}-{float(row.window_end_sec):.3f}::'
            f'channels={row.source_channels}::lines={row.source_lines}'
        )
        sample_id = stable_sample_id(sample_key)
        item = row._asdict()
        item.update({
            'sample_id': sample_id,
            'sample_index': int(idx),
            'split': split_name,
            'sample_file': sample_key,
            'image_path': str((render_root / f'{sample_id}.png').resolve()),
            'fs': int(sfreq),
            'channel_names': channel_names_json,
            'n_channels': int(len(channel_names)),
            'render_config_id': render_config['render_id'],
        })
        rows.append(item)
    return pd.DataFrame(rows)


def _load_tcp_file(path: str, target_sfreq: int, max_len_sec: float | None = None) -> np.ndarray:
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    clean_to_raw = {_clean_channel_name(ch): ch for ch in raw.ch_names}
    required = sorted({name for pair in TUEV_TCP_PAIRS for name in pair})
    missing = [name for name in required if name not in clean_to_raw]
    if missing:
        raise RuntimeError(f'TUEV file is missing required channels {missing}: {path}')
    picks = [clean_to_raw[name] for name in required]
    data = raw.get_data(picks=picks) * 1e6
    orig_sfreq = int(round(float(raw.info['sfreq'])))
    if orig_sfreq != int(target_sfreq):
        gcd = math.gcd(orig_sfreq, int(target_sfreq))
        data = scipy.signal.resample_poly(
            data,
            up=int(target_sfreq) // gcd,
            down=orig_sfreq // gcd,
            axis=-1,
        )
    raw_map = {name: data[idx] for idx, name in enumerate(required)}
    tcp = np.stack(
        [raw_map[left] - raw_map[right] for left, right in TUEV_TCP_PAIRS],
        axis=0,
    )
    if max_len_sec is not None:
        max_len = int(math.ceil(float(max_len_sec) * int(target_sfreq)))
        tcp = tcp[:, :max_len]
    tcp = tcp.T.astype(np.float32, copy=False)
    zero = np.zeros((tcp.shape[0], 1), dtype=np.float32)
    return np.concatenate([tcp, zero], axis=1).astype(np.float32, copy=False)


class TUEVLazySamples:


    def __init__(self, manifest: pd.DataFrame, sfreq: int, window_sec: float, max_cached_files: int = 2):
        self.manifest = manifest.reset_index(drop=True)
        self.sfreq = int(sfreq)
        self.window_sec = float(window_sec)
        self.seq_len = int(round(self.window_sec * self.sfreq))
        self.max_cached_files = int(max_cached_files)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, item):
        if isinstance(item, slice):
            local_indices = np.arange(len(self), dtype=np.int64)[item]
            return self._stack_local(local_indices)
        if np.isscalar(item):
            return self.raw_window(int(item))
        local_indices = np.asarray(item, dtype=np.int64)
        return self._stack_local(local_indices)

    def _stack_local(self, local_indices: np.ndarray) -> np.ndarray:
        if local_indices.size == 0:
            return np.empty((0, self.seq_len, len(TUEV_VLM_CHANNEL_NAMES)), dtype=np.float32)
        return np.stack([self.raw_window(int(idx)) for idx in local_indices], axis=0).astype(np.float32, copy=False)

    def _file_array(self, edf_path: str) -> np.ndarray:
        edf_path = str(edf_path)
        if edf_path in self._cache:
            arr = self._cache.pop(edf_path)
            self._cache[edf_path] = arr
            return arr
        arr = _load_tcp_file(edf_path, target_sfreq=self.sfreq)
        self._cache[edf_path] = arr
        while len(self._cache) > self.max_cached_files:
            self._cache.popitem(last=False)
        return arr

    def raw_window(self, local_index: int) -> np.ndarray:
        row = self.manifest.iloc[int(local_index)]
        arr = self._file_array(str(row['edf_path']))
        window_start = row['window_start_sec'] if 'window_start_sec' in row else row['event_window_start_sec']
        start = int(round(float(window_start) * self.sfreq))
        stop = start + self.seq_len
        if start < 0 or stop > arr.shape[0]:



            offset = arr.shape[0]
            tiled = np.concatenate([arr, arr, arr], axis=0)
            window = tiled[offset + start:offset + stop]
        else:
            window = arr[start:stop]
        if window.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - window.shape[0], window.shape[1]), dtype=np.float32)
            window = np.concatenate([window, pad], axis=0)
        elif window.shape[0] > self.seq_len:
            window = window[:self.seq_len]
        return window.astype(np.float32, copy=False)


def prepare_tuev_workspace(args, prep_dir: str) -> TUEVPreparedWorkspace:

    prep_path = Path(prep_dir).resolve()
    prep_path.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_data_dir(getattr(args, 'data_dir', '') or default_data_dir('TUEV'), 'tuev')
    sfreq = int(getattr(args, 'target_sfreq', TUEV_TARGET_SFREQ))
    window_sec = float(getattr(args, 'window_size', 5.0))
    files = _scan_file_metadata(data_dir)
    split_mode = str(getattr(args, 'split_mode', 'cross-sub'))
    subject_split = _split_patients(
        files,
        val_fraction=float(getattr(args, 'official_val_fraction', 0.20)),
        seed=int(getattr(args, 'split_seed', getattr(args, 'seed', 42))),
    )

    render_config = build_tuev_render_config(args)
    prompt = resolve_tuev_prompt(args)
    render_root = Path(getattr(args, 'render_cache_dir', DEFAULT_TUEV_RENDER_CACHE_ROOT)).resolve() / render_config['render_id']
    render_root.mkdir(parents=True, exist_ok=True)

    split_frames = {}
    split_samples = {}
    split_counts = {}
    limit = int(getattr(args, 'max_samples_per_split', 0))
    split_source_frames: dict[str, pd.DataFrame] = {}
    for split_name in ['train', 'val', 'test']:
        split_files = _files_for_split(files, split_name, subject_split)
        split_source_frames[split_name] = _build_split_events(
            files=split_files,
            split_name=split_name,
            window_sec=window_sec,
            merge_gap_sec=float(getattr(args, 'event_merge_gap_sec', 0.5)),
            bckg_exclusion_margin_sec=float(getattr(args, 'bckg_exclusion_margin_sec', 0.5)),
            bckg_sample_ratio=float(getattr(args, 'bckg_sample_ratio', 1.0)),
            max_bckg_per_split=int(getattr(args, 'max_bckg_per_split', 0)),
            seed=int(getattr(args, 'seed', 42)) + {'train': 0, 'val': 1, 'test': 2}[split_name],
        )

    for split_name in ['train', 'val', 'test']:
        frame = split_source_frames[split_name]
        if frame.empty:
            raise RuntimeError(f'TUEV split={split_name} has no prepared samples.')
        chosen_idx = _choose_limit_indices(
            frame['label'].to_numpy(dtype=np.int64, copy=False),
            limit=limit,
            seed=int(getattr(args, 'seed', 42)) + {'train': 10, 'val': 11, 'test': 12}[split_name],
        )
        frame = frame.iloc[chosen_idx].reset_index(drop=True)
        frame = _attach_manifest_columns(
            frame=frame,
            split_name=split_name,
            channel_names=TUEV_VLM_CHANNEL_NAMES,
            sfreq=sfreq,
            render_root=render_root,
            render_config=render_config,
        )
        split_frames[split_name] = frame
        split_samples[split_name] = TUEVLazySamples(frame, sfreq=sfreq, window_sec=window_sec)
        split_counts[split_name] = frame['label'].value_counts().sort_index().to_dict()
        frame.to_csv(prep_path / f'manifest_{split_name}.csv', index=False)

    render_stats = compute_render_stats(
        train_samples=split_samples['train'],
        channel_names=TUEV_VLM_CHANNEL_NAMES,
        sfreq=sfreq,
        render_config=render_config,
        seed=int(getattr(args, 'seed', 42)),
    )

    prep_payload = {
        'dataset': 'tuev',
        'task': 1,
        'task_label': 'event_class',
        'data_dir': data_dir,
        'split_mode': split_mode,
        'seed': int(getattr(args, 'seed', 42)),
        'split_seed': int(getattr(args, 'split_seed', getattr(args, 'seed', 42))),
        'official_val_fraction': float(getattr(args, 'official_val_fraction', 0.20)),
        'target_sfreq': sfreq,
        'window_size': window_sec,
        'processing_style': 'official_rec_event',
        'event_window_rule': 'window_start = event_start_sec - 2.0; fixed window_size seconds',
        'event_merge_gap_sec': 'ignored_for_official_rec_event',
        'bckg_exclusion_margin_sec': 'ignored_for_official_rec_event',
        'bckg_sample_ratio': 'ignored_for_official_rec_event',
        'max_bckg_per_split': 'ignored_for_official_rec_event',
        'prompt': prompt,
        'model_path': str(getattr(args, 'model_path', DEFAULT_TUEV_MODEL_PATH)),
        'qwen_model_size': str(getattr(args, 'qwen_model_size', '4b')),
        'render_root': str(render_root),
        'class_names': TUEV_CLASS_NAMES,
        'choices': TUEV_CHOICES,
        'samples_per_split': {name: int(len(frame)) for name, frame in split_frames.items()},
        'label_counts_per_split': {
            name: {TUEV_CLASS_NAMES[int(label)]: int(count) for label, count in counts.items()}
            for name, counts in split_counts.items()
        },
        'n_files': int(len(files)),
    }
    (prep_path / 'prepare_args.json').write_text(json.dumps(prep_payload, indent=2), encoding='utf-8')
    (prep_path / 'render_config.json').write_text(json.dumps(render_config, indent=2), encoding='utf-8')
    (prep_path / 'render_stats.json').write_text(json.dumps(render_stats, indent=2), encoding='utf-8')
    (prep_path / 'subject_split.json').write_text(json.dumps(subject_split, indent=2), encoding='utf-8')
    files.to_csv(prep_path / 'edf_files.csv', index=False)

    return TUEVPreparedWorkspace(
        prep_dir=str(prep_path),
        channel_names=TUEV_VLM_CHANNEL_NAMES,
        sfreq=sfreq,
        duration_sec=window_sec,
        prompt=prompt,
        render_config=render_config,
        render_stats=render_stats,
        subject_split=subject_split,
        splits={
            name: TUEVPreparedSplitData(name=name, samples=split_samples[name], manifest=split_frames[name])
            for name in ['train', 'val', 'test']
        },
        class_names=TUEV_CLASS_NAMES,
        choices=TUEV_CHOICES,
    )


def build_tuev_workspace_from_manifests(args, prep_dir: str) -> TUEVPreparedWorkspace:

    prep_path = Path(prep_dir).resolve()
    if not prep_path.is_dir():
        raise FileNotFoundError(f'Prep directory not found: {prep_dir}')

    prepare_args = json.loads((prep_path / 'prepare_args.json').read_text(encoding='utf-8'))
    sfreq = int(prepare_args.get('target_sfreq', getattr(args, 'target_sfreq', TUEV_TARGET_SFREQ)))
    window_sec = float(prepare_args.get('window_size', getattr(args, 'window_size', 5.0)))
    render_config = json.loads((prep_path / 'render_config.json').read_text(encoding='utf-8'))
    render_stats = json.loads((prep_path / 'render_stats.json').read_text(encoding='utf-8'))
    subject_split = json.loads((prep_path / 'subject_split.json').read_text(encoding='utf-8'))
    splits = {}
    for split_name in ['train', 'val', 'test']:
        manifest = pd.read_csv(prep_path / f'manifest_{split_name}.csv')
        splits[split_name] = TUEVPreparedSplitData(
            name=split_name,
            samples=TUEVLazySamples(manifest, sfreq=sfreq, window_sec=window_sec),
            manifest=manifest.reset_index(drop=True),
        )

    return TUEVPreparedWorkspace(
        prep_dir=str(prep_path),
        channel_names=TUEV_VLM_CHANNEL_NAMES,
        sfreq=sfreq,
        duration_sec=window_sec,
        prompt=str(prepare_args.get('prompt', getattr(args, 'prompt', DEFAULT_TUEV_PROMPT))),
        render_config=render_config,
        render_stats=render_stats,
        subject_split=subject_split,
        splits=splits,
        class_names=list(prepare_args.get('class_names', TUEV_CLASS_NAMES)),
        choices=list(prepare_args.get('choices', TUEV_CHOICES)),
    )


class TUEVVLMDataset(Dataset):


    def __init__(self,
                 split: TUEVPreparedSplitData,
                 channel_names: list[str],
                 sfreq: int,
                 duration_sec: float,
                 render_config: dict[str, Any],
                 render_stats: dict[str, Any],
                 choices: list[str],
                 base_prompt: str | None = None):
        self.split = split
        self.channel_names = list(channel_names)
        self.sfreq = int(sfreq)
        self.duration_sec = float(duration_sec)
        self.render_config = dict(render_config)
        self.render_stats = dict(render_stats)
        self.choices = list(choices)
        self.base_prompt = str(base_prompt or DEFAULT_TUEV_PROMPT)

    def __len__(self) -> int:
        return len(self.split.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.split.manifest.iloc[index]
        sample = self.split.samples[index]
        label = int(row['label'])
        item = {
            'sample_id': str(row['sample_id']),
            'split': str(row['split']),
            'label': label,
            'answer_text': self.choices[label],
            'subject': str(row['subject']),
            'condition': int(row['condition']),
            'sample_file': str(row['sample_file']),
            'image_path': str(row.get('image_path', '')),
            'prompt': self.base_prompt,
        }
        image_path = ensure_rendered_image(
            sample=sample,
            channel_names=self.channel_names,
            sfreq=self.sfreq,
            duration_sec=self.duration_sec,
            render_config=self.render_config,
            render_stats=self.render_stats,
            output_path=str(row['image_path']),
        )
        with Image.open(image_path) as handle:
            item['image'] = handle.convert('RGB').copy()
        item['image_path'] = str(image_path)
        return item
