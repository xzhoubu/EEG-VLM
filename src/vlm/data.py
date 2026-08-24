

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from catalog import get_dataset_adapter
from loaders.vepiset_loader import split_arrays
from runtime import default_output_dir, resolve_data_dir
from vlm.model_config import default_qwen3vl_model_path
from vlm.render import compute_render_stats, ensure_rendered_image


DEFAULT_QWEN_PROMPT = (
    'You are given a visualization of a 1-second EEG segment, including a '
    'longitudinal bipolar montage and a spectrogram.\n'
    'Question: Does this EEG segment contain epileptiform activity?\n'
    'Answer only with yes or no.'
)

DOUBLE_BANANA_PAIRS = [
    ('Fp1', 'F7'),
    ('F7', 'T3'),
    ('T3', 'T5'),
    ('T5', 'O1'),
    ('Fp2', 'F8'),
    ('F8', 'T4'),
    ('T4', 'T6'),
    ('T6', 'O2'),
    ('Fp1', 'F3'),
    ('F3', 'C3'),
    ('C3', 'P3'),
    ('P3', 'O1'),
    ('Fp2', 'F4'),
    ('F4', 'C4'),
    ('C4', 'P4'),
    ('P4', 'O2'),
    ('Fz', 'Cz'),
    ('Cz', 'Pz'),
]

DEFAULT_RESULT_ROOT = default_output_dir('runs', 'vepiset_vlm')
DEFAULT_RENDER_CACHE_ROOT = default_output_dir('cache', 'vepiset_vlm')
DEFAULT_MODEL_PATH = default_qwen3vl_model_path()


@dataclass
class PreparedSplitData:


    name: str
    samples: np.ndarray
    manifest: pd.DataFrame


@dataclass
class PreparedWorkspace:


    prep_dir: str
    channel_names: list[str]
    sfreq: int
    duration_sec: float
    prompt: str
    render_config: dict[str, Any]
    render_stats: dict[str, Any]
    subject_split: dict[str, list[str]]
    splits: dict[str, PreparedSplitData]


def build_render_config(args) -> dict[str, Any]:

    payload = {
        'render_version': str(getattr(args, 'render_version', 'v1')).strip() or 'v1',
        'image_input_mode': str(getattr(args, 'image_input_mode', 'single_panel')).strip() or 'single_panel',
        'single_image_kind': str(getattr(args, 'single_image_kind', 'combined')).strip() or 'combined',
        'image_size': int(getattr(args, 'image_size', 896)),
        'montage_pairs': [list(pair) for pair in DOUBLE_BANANA_PAIRS],
        'montage_labels': [f'{left}-{right}' for left, right in DOUBLE_BANANA_PAIRS],
        'normalization_mode': str(getattr(args, 'normalization_mode', 'train_global')).strip() or 'train_global',
        'spectrogram_freq_min': float(getattr(args, 'spectrogram_freq_min', 1.0)),
        'spectrogram_freq_max': float(getattr(args, 'spectrogram_freq_max', 50.0)),
        'spectrogram_nperseg': int(getattr(args, 'spectrogram_nperseg', 64)),
        'spectrogram_noverlap': int(getattr(args, 'spectrogram_noverlap', 48)),
        'spectrogram_nfft': int(getattr(args, 'spectrogram_nfft', 256)),
        'spectrogram_log_eps': 1e-8,
        'waveform_percentile': float(getattr(args, 'waveform_percentile', 0.995)),
        'spectrogram_quantile_low': float(getattr(args, 'spectrogram_quantile_low', 0.01)),
        'spectrogram_quantile_high': float(getattr(args, 'spectrogram_quantile_high', 0.99)),
        'waveform_stats_max_samples': int(getattr(args, 'waveform_stats_max_samples', 2048)),
        'spectrogram_stats_max_samples': int(getattr(args, 'spectrogram_stats_max_samples', 512)),
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


def stable_sample_id(sample_file: str) -> str:
    return hashlib.sha1(str(sample_file).encode('utf-8')).hexdigest()[:16]


def _channel_names_json(channel_names: list[str]) -> str:
    return json.dumps(list(channel_names), ensure_ascii=True, separators=(',', ':'))


def _resolve_subject_name(subject_map: dict[int, str], subject_idx: int) -> str:
    return str(subject_map.get(int(subject_idx), f'unknown_{int(subject_idx)}'))


def _required_channels() -> list[str]:
    names = sorted({name for pair in DOUBLE_BANANA_PAIRS for name in pair})
    return names


def _validate_required_channels(channel_names: list[str]) -> None:
    missing = [name for name in _required_channels() if name not in channel_names]
    if missing:
        raise RuntimeError(
            'vEpiSet VLM rendering requires all double-banana scalp channels to be present. '
            f'Missing: {missing}'
        )


def _finite_required_mask(samples: np.ndarray, channel_names: list[str]) -> np.ndarray:
    idx_map = {name: idx for idx, name in enumerate(channel_names)}
    keep_idx = [idx_map[name] for name in _required_channels()]
    return np.isfinite(samples[:, :, keep_idx]).all(axis=(1, 2))


def _choose_limit_indices(labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if limit <= 0 or n <= limit:
        return np.arange(n, dtype=np.int64)

    rng = np.random.RandomState(seed)
    by_label = {
        label: np.where(labels == label)[0]
        for label in sorted(np.unique(labels).tolist())
    }
    selected: list[int] = []


    for label in [0, 1]:
        indices = by_label.get(label)
        if indices is not None and len(indices) > 0 and len(selected) < limit:
            selected.append(int(indices[0]))

    remaining = limit - len(selected)
    if remaining > 0:
        leftovers = np.setdiff1d(np.arange(n, dtype=np.int64), np.asarray(selected, dtype=np.int64), assume_unique=False)
        if len(leftovers) > 0:
            extra = rng.choice(leftovers, size=min(remaining, len(leftovers)), replace=False)
            selected.extend(int(item) for item in extra.tolist())

    selected = sorted(set(selected))
    if len(selected) < limit:
        leftovers = [idx for idx in range(n) if idx not in selected]
        selected.extend(leftovers[:limit - len(selected)])
    return np.asarray(sorted(selected[:limit]), dtype=np.int64)


def _load_all_data(args):
    adapter = get_dataset_adapter('vepiset')
    data_dir = resolve_data_dir(getattr(args, 'data_dir', '') or adapter.default_data_dir, adapter.name)
    all_data = adapter.load_all_data(
        data_dir=data_dir,
        window_sec=float(getattr(args, 'window_size', 1.0)),
        overlap_ratio=float(getattr(args, 'overlap_ratio', 0.0)),
        use_features=False,
        feature_set='raw',
        use_cache=bool(int(getattr(args, 'use_cache', 1))),
    )
    return adapter, data_dir, all_data


def _build_manifest(split_name: str,
                    split_samples: np.ndarray,
                    split_meta: dict[str, np.ndarray | str],
                    channel_names: list[str],
                    sfreq: int,
                    duration_sec: float,
                    render_root: Path,
                    render_config: dict[str, Any],
                    subject_map: dict[int, str]) -> tuple[pd.DataFrame, np.ndarray]:
    valid_mask = _finite_required_mask(split_samples, channel_names)
    samples = split_samples[valid_mask]
    labels = split_meta['labels'][valid_mask].astype(np.int64, copy=False)
    subjects = split_meta['subjects'][valid_mask].astype(np.int64, copy=False)
    conditions = split_meta['conditions'][valid_mask].astype(np.int64, copy=False)
    sample_files = np.asarray(split_meta['sample_files'], dtype=object)[valid_mask]

    channel_names_json = _channel_names_json(channel_names)
    rows = []
    for label, subject, condition, sample_file in zip(labels, subjects, conditions, sample_files):
        sample_id = stable_sample_id(str(sample_file))
        rows.append({
            'sample_id': sample_id,
            'split': split_name,
            'label': int(label),
            'subject': _resolve_subject_name(subject_map, int(subject)),
            'condition': int(condition),
            'sample_file': str(sample_file),
            'image_path': str((render_root / f'{sample_id}.png').resolve()),
            'fs': int(sfreq),
            'channel_names': channel_names_json,
            'n_channels': int(len(channel_names)),
            'duration_sec': float(duration_sec),
            'render_config_id': render_config['render_id'],
        })
    frame = pd.DataFrame(rows)
    return frame, samples


def prepare_workspace(args, prep_dir: str) -> PreparedWorkspace:

    prep_path = Path(prep_dir).resolve()
    prep_path.mkdir(parents=True, exist_ok=True)

    adapter, data_dir, all_data = _load_all_data(args)
    _validate_required_channels(list(all_data['ch_names']))

    split = split_arrays(
        all_data,
        task=1,
        split_mode=getattr(args, 'split_mode', 'cross-sub'),
        train_subs=getattr(args, 'train_subs', adapter.default_train_subs),
        val_subs=getattr(args, 'val_subs', adapter.default_val_subs),
        test_subs=getattr(args, 'test_subs', adapter.default_test_subs),
        train_ratio=float(getattr(args, 'train_ratio', 0.8)),
        val_ratio=float(getattr(args, 'val_ratio', 0.1)),
        normalize=False,
        seed=int(getattr(args, 'seed', 42)),
        split_seed=getattr(args, 'split_seed', None),
        mode='end2end',
        window_sec=float(getattr(args, 'window_size', 1.0)),
        sfreq_in=int(adapter.sfreq),
    )

    render_config = build_render_config(args)
    render_root = Path(getattr(args, 'render_cache_dir', DEFAULT_RENDER_CACHE_ROOT)).resolve() / render_config['render_id']
    render_root.mkdir(parents=True, exist_ok=True)

    sfreq = int(all_data['sfreq'])
    duration_sec = float(all_data['window_sec'])
    limit = int(getattr(args, 'max_samples_per_split', 0))

    split_frames = {}
    split_samples = {}
    split_subjects = {}
    filtered_summary = {}
    for split_name in ['train', 'val', 'test']:
        frame, samples = _build_manifest(
            split_name=split_name,
            split_samples=split['raw'][split_name],
            split_meta=split['meta'][split_name],
            channel_names=list(all_data['ch_names']),
            sfreq=sfreq,
            duration_sec=duration_sec,
            render_root=render_root,
            render_config=render_config,
            subject_map=all_data['subject_id_map'],
        )
        filtered_summary[split_name] = int(len(split['raw'][split_name]) - len(samples))
        chosen_idx = _choose_limit_indices(
            frame['label'].to_numpy(dtype=np.int64, copy=False),
            limit=limit,
            seed=int(getattr(args, 'seed', 42)) + {'train': 0, 'val': 1, 'test': 2}[split_name],
        )
        frame = frame.iloc[chosen_idx].reset_index(drop=True)
        samples = samples[chosen_idx]
        split_frames[split_name] = frame
        split_samples[split_name] = samples
        split_subjects[split_name] = sorted(frame['subject'].unique().tolist())
        frame.to_csv(prep_path / f'manifest_{split_name}.csv', index=False)

    render_stats = compute_render_stats(
        train_samples=split_samples['train'],
        channel_names=list(all_data['ch_names']),
        sfreq=sfreq,
        render_config=render_config,
        seed=int(getattr(args, 'seed', 42)),
    )

    prep_payload = {
        'dataset': 'vepiset',
        'task': 1,
        'task_label': 'ied_binary',
        'data_dir': data_dir,
        'split_mode': getattr(args, 'split_mode', 'cross-sub'),
        'seed': int(getattr(args, 'seed', 42)),
        'split_seed': int(
            getattr(args, 'split_seed', None)
            if getattr(args, 'split_seed', None) is not None
            else getattr(args, 'seed', 42)
        ),
        'window_size': float(getattr(args, 'window_size', 1.0)),
        'overlap_ratio': float(getattr(args, 'overlap_ratio', 0.0)),
        'prompt': getattr(args, 'prompt', DEFAULT_QWEN_PROMPT),
        'model_path': str(getattr(args, 'model_path', DEFAULT_MODEL_PATH)),
        'qwen_model_size': str(getattr(args, 'qwen_model_size', '4b')),
        'render_root': str(render_root),
        'filtered_missing_or_invalid_samples': filtered_summary,
        'samples_per_split': {name: int(len(frame)) for name, frame in split_frames.items()},
    }
    (prep_path / 'prepare_args.json').write_text(json.dumps(prep_payload, indent=2), encoding='utf-8')
    (prep_path / 'render_config.json').write_text(json.dumps(render_config, indent=2), encoding='utf-8')
    (prep_path / 'render_stats.json').write_text(json.dumps(render_stats, indent=2), encoding='utf-8')
    (prep_path / 'subject_split.json').write_text(json.dumps(split_subjects, indent=2), encoding='utf-8')

    return PreparedWorkspace(
        prep_dir=str(prep_path),
        channel_names=list(all_data['ch_names']),
        sfreq=sfreq,
        duration_sec=duration_sec,
        prompt=str(getattr(args, 'prompt', DEFAULT_QWEN_PROMPT)),
        render_config=render_config,
        render_stats=render_stats,
        subject_split=split_subjects,
        splits={
            name: PreparedSplitData(name=name, samples=split_samples[name], manifest=split_frames[name].reset_index(drop=True))
            for name in ['train', 'val', 'test']
        },
    )


def build_workspace_from_manifests(args, prep_dir: str) -> PreparedWorkspace:

    prep_path = Path(prep_dir).resolve()
    if not prep_path.is_dir():
        raise FileNotFoundError(f'Prep directory not found: {prep_dir}')

    adapter, _data_dir, all_data = _load_all_data(args)
    _validate_required_channels(list(all_data['ch_names']))

    render_config = json.loads((prep_path / 'render_config.json').read_text(encoding='utf-8'))
    render_stats = json.loads((prep_path / 'render_stats.json').read_text(encoding='utf-8'))
    prepare_args = json.loads((prep_path / 'prepare_args.json').read_text(encoding='utf-8'))
    subject_split = json.loads((prep_path / 'subject_split.json').read_text(encoding='utf-8'))

    sample_files = [str(item) for item in all_data['sample_files']]
    sample_lookup = {sample_file: idx for idx, sample_file in enumerate(sample_files)}
    binary_labels = all_data['labels']['ied_binary'].astype(np.int64, copy=False)

    splits = {}
    for split_name in ['train', 'val', 'test']:
        manifest = pd.read_csv(prep_path / f'manifest_{split_name}.csv')
        indices = []
        for sample_file in manifest['sample_file'].tolist():
            if sample_file not in sample_lookup:
                raise KeyError(f'Sample file {sample_file} from manifest_{split_name}.csv not found in raw cache.')
            indices.append(sample_lookup[sample_file])
        indices_arr = np.asarray(indices, dtype=np.int64)
        samples = all_data['samples'][indices_arr].astype(np.float32, copy=False)
        labels = binary_labels[indices_arr]
        if not np.array_equal(labels, manifest['label'].to_numpy(dtype=np.int64, copy=False)):
            raise RuntimeError(f'Manifest labels for split={split_name} do not match the raw vEpiSet cache.')
        splits[split_name] = PreparedSplitData(
            name=split_name,
            samples=samples,
            manifest=manifest.reset_index(drop=True),
        )

    return PreparedWorkspace(
        prep_dir=str(prep_path),
        channel_names=list(all_data['ch_names']),
        sfreq=int(all_data['sfreq']),
        duration_sec=float(all_data['window_sec']),
        prompt=str(prepare_args.get('prompt', getattr(args, 'prompt', DEFAULT_QWEN_PROMPT))),
        render_config=render_config,
        render_stats=render_stats,
        subject_split=subject_split,
        splits=splits,
    )


class VEPiSetVLMDataset(Dataset):


    def __init__(self,
                 split: PreparedSplitData,
                 channel_names: list[str],
                 sfreq: int,
                 duration_sec: float,
                 render_config: dict[str, Any],
                 render_stats: dict[str, Any]):
        self.split = split
        self.channel_names = list(channel_names)
        self.sfreq = int(sfreq)
        self.duration_sec = float(duration_sec)
        self.render_config = dict(render_config)
        self.render_stats = dict(render_stats)

    def __len__(self) -> int:
        return len(self.split.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.split.manifest.iloc[index]
        sample = self.split.samples[index]
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
            image = handle.convert('RGB').copy()
        label = int(row['label'])
        return {
            'sample_id': str(row['sample_id']),
            'split': str(row['split']),
            'label': label,
            'answer_text': 'yes' if label == 1 else 'no',
            'subject': str(row['subject']),
            'condition': int(row['condition']),
            'sample_file': str(row['sample_file']),
            'image_path': str(row['image_path']),
            'image': image,
        }


class DistributedWeightedSampler(Sampler[int]):


    def __init__(self,
                 weights: list[float] | np.ndarray | torch.Tensor,
                 num_replicas: int = 1,
                 rank: int = 0,
                 seed: int = 42):
        if num_replicas <= 0:
            raise ValueError(f'num_replicas must be positive, got {num_replicas}')
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f'rank must be in [0, {num_replicas}), got {rank}')

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.weights) / float(self.num_replicas)))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=generator).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
