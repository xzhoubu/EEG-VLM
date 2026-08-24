

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from catalog import get_dataset_adapter
from loaders.tuab_loader import (
    TUABWindowDataset,
    build_split_indices,
)
from runtime import default_output_dir, resolve_data_dir
from vlm.data import DOUBLE_BANANA_PAIRS, build_render_config, stable_sample_id
from vlm.model_config import default_qwen3vl_model_path
from vlm.render import compute_render_stats, ensure_rendered_image


TUAB_CLASS_NAMES = ['Normal', 'Abnormal']
TUAB_CHOICES = TUAB_CLASS_NAMES
TUAB_VLM_CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2', 'A1', 'A2',
]
DEFAULT_TUAB_PROMPT = (
    'You are given a visualization of a TUAB EEG segment, '
    'including a longitudinal bipolar montage and a spectrogram.\n\n'
    'Use only the EEG signal shown in the image. Classify the segment into '
    'exactly one of these options:\n\n'
    'Options:\n'
    'Normal\n'
    'Abnormal\n\n'
    'Return exactly one class name from the options. Do not include explanations, '
    'punctuation, or extra words.'
)
DEFAULT_TUAB_WAVEFORM_PROMPT = (
    'You are given a waveform visualization of a TUAB EEG segment, '
    'shown as a longitudinal bipolar montage.\n\n'
    'Use only the EEG waveform shown in the image. Classify the segment into '
    'exactly one of these options:\n\n'
    'Options:\n'
    'Normal\n'
    'Abnormal\n\n'
    'Return exactly one class name from the options. Do not include explanations, '
    'punctuation, or extra words.'
)
DEFAULT_TUAB_SPECTROGRAM_PROMPT = (
    'You are given a spectrogram visualization of a TUAB EEG segment, '
    'shown across longitudinal bipolar montage channels.\n\n'
    'Use only the EEG spectrogram shown in the image. Classify the segment into '
    'exactly one of these options:\n\n'
    'Options:\n'
    'Normal\n'
    'Abnormal\n\n'
    'Return exactly one class name from the options. Do not include explanations, '
    'punctuation, or extra words.'
)
DEFAULT_TUAB_CWT_PROMPT = (
    'You are given a CWT scalogram visualization of a TUAB EEG segment, '
    'shown across longitudinal bipolar montage channels.\n\n'
    'Use only the EEG scalogram shown in the image. Classify the segment into '
    'exactly one of these options:\n\n'
    'Options:\n'
    'Normal\n'
    'Abnormal\n\n'
    'Return exactly one class name from the options. Do not include explanations, '
    'punctuation, or extra words.'
)
DEFAULT_TUAB_COMBINED_CWT_PROMPT = (
    'You are given a visualization of a TUAB EEG segment, including a '
    'longitudinal bipolar montage and a CWT scalogram.\n\n'
    'Use only the EEG signal shown in the image. Classify the segment into '
    'exactly one of these options:\n\n'
    'Options:\n'
    'Normal\n'
    'Abnormal\n\n'
    'Return exactly one class name from the options. Do not include explanations, '
    'punctuation, or extra words.'
)
DEFAULT_TUAB_RESULT_ROOT = default_output_dir('runs', 'tuab_vlm')
DEFAULT_TUAB_RENDER_CACHE_ROOT = default_output_dir('cache', 'tuab_vlm')
DEFAULT_TUAB_MODEL_PATH = default_qwen3vl_model_path()


def resolve_tuab_prompt(args) -> str:

    prompt = str(getattr(args, 'prompt', DEFAULT_TUAB_PROMPT))
    if prompt != DEFAULT_TUAB_PROMPT:
        return prompt
    single_image_kind = str(getattr(args, 'single_image_kind', 'combined') or 'combined')
    if single_image_kind == 'waveform':
        return DEFAULT_TUAB_WAVEFORM_PROMPT
    if single_image_kind == 'spectrogram':
        return DEFAULT_TUAB_SPECTROGRAM_PROMPT
    if single_image_kind == 'cwt':
        return DEFAULT_TUAB_CWT_PROMPT
    if single_image_kind == 'combined_cwt':
        return DEFAULT_TUAB_COMBINED_CWT_PROMPT
    return DEFAULT_TUAB_PROMPT


@dataclass
class TUABPreparedSplitData:


    name: str
    samples: 'TUABLazySamples'
    manifest: pd.DataFrame


@dataclass
class TUABPreparedWorkspace:


    prep_dir: str
    channel_names: list[str]
    sfreq: int
    duration_sec: float
    prompt: str
    render_config: dict[str, Any]
    render_stats: dict[str, Any]
    subject_split: dict[str, list[str]]
    splits: dict[str, TUABPreparedSplitData]
    class_names: list[str]
    choices: list[str]


class TUABLazySamples:


    def __init__(self, all_data: dict[str, Any], global_indices: np.ndarray):
        self.all_data = all_data
        self.global_indices = np.asarray(global_indices, dtype=np.int64)
        zeros = np.zeros(len(self.global_indices), dtype=np.int64)
        self._dataset = TUABWindowDataset(
            all_data=all_data,
            indices=self.global_indices,
            labels=zeros,
            subjects=zeros,
            conditions=zeros,
            mode='end2end',
            use_features=False,
            normalizer=None,
            max_cached_files=2,
        )

    def __len__(self) -> int:
        return len(self.global_indices)

    def __getitem__(self, item):
        if isinstance(item, slice):
            local_indices = np.arange(len(self), dtype=np.int64)[item]
            return self._stack_local(local_indices)
        if np.isscalar(item):
            return self._dataset.raw_window(int(self.global_indices[int(item)]))
        local_indices = np.asarray(item, dtype=np.int64)
        return self._stack_local(local_indices)

    def _stack_local(self, local_indices: np.ndarray) -> np.ndarray:
        if local_indices.size == 0:
            seq_len = int(round(float(self.all_data['window_sec']) * int(self.all_data['sfreq'])))
            return np.empty((0, seq_len, len(self.all_data['ch_names'])), dtype=np.float32)
        return np.stack(
            [self._dataset.raw_window(int(self.global_indices[int(idx)])) for idx in local_indices],
            axis=0,
        ).astype(np.float32, copy=False)


def _channel_names_json(channel_names: list[str]) -> str:
    return json.dumps(list(channel_names), ensure_ascii=True, separators=(',', ':'))


def _required_channels() -> list[str]:
    return sorted({name for pair in DOUBLE_BANANA_PAIRS for name in pair})


def _validate_required_channels(channel_names: list[str]) -> None:
    missing = [name for name in _required_channels() if name not in channel_names]
    if missing:
        raise RuntimeError(
            'TUAB VLM rendering requires all double-banana scalp channels to be present. '
            f'Missing: {missing}'
        )


def _choose_limit_indices(labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if limit <= 0 or n <= limit:
        return np.arange(n, dtype=np.int64)

    rng = np.random.RandomState(seed)
    selected: list[int] = []
    for label in [0, 1]:
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


def _load_all_data(args, window_size: float | None = None, overlap_ratio: float | None = None):
    adapter = get_dataset_adapter('tuab')
    data_dir = resolve_data_dir(getattr(args, 'data_dir', '') or adapter.default_data_dir, adapter.name)
    all_data = adapter.load_all_data(
        data_dir=data_dir,
        window_sec=float(window_size if window_size is not None else getattr(args, 'window_size', 1.0)),
        overlap_ratio=float(overlap_ratio if overlap_ratio is not None else getattr(args, 'overlap_ratio', 0.0)),
        sfreq=int(adapter.sfreq),
        use_features=False,
        feature_set='raw',
        use_cache=bool(int(getattr(args, 'use_cache', 1))),
        max_windows_per_file=int(getattr(args, 'max_windows_per_file', 0)),
        tokenize=bool(int(getattr(args, 'tuab_tokenize', 1))),
        token_sec=float(getattr(args, 'tuab_token_sec', 1.0)),
        token_stride_sec=float(getattr(args, 'tuab_token_stride_sec', 1.0)),
    )
    return adapter, data_dir, all_data


def _subject_name(subject_map: dict[int, str], subject_idx: int) -> str:
    return str(subject_map.get(int(subject_idx), f'unknown_{int(subject_idx)}'))


def _build_manifest(split_name: str,
                    all_data: dict[str, Any],
                    global_indices: np.ndarray,
                    channel_names: list[str],
                    sfreq: int,
                    duration_sec: float,
                    render_root: Path,
                    render_config: dict[str, Any],
                    subject_map: dict[int, str]) -> pd.DataFrame:
    labels = all_data['labels']['abnormal_binary'][global_indices].astype(np.int64, copy=False)
    subjects = all_data['subjects'][global_indices].astype(np.int64, copy=False)
    file_idx = all_data['window_file_idx'][global_indices].astype(np.int64, copy=False)
    start_sec = all_data['window_start_sec'][global_indices].astype(np.float32, copy=False)

    channel_names_json = _channel_names_json(channel_names)
    rows = []
    for label, subject, fidx, start in zip(labels, subjects, file_idx, start_sec):
        sample_file = f'{all_data["file_paths"][int(fidx)]}::start={float(start):.3f}'
        sample_id = stable_sample_id(sample_file)
        item = {
            'sample_id': sample_id,
            'split': split_name,
            'label': int(label),
            'subject': _subject_name(subject_map, int(subject)),
            'condition': int(label),
            'condition_name': TUAB_CLASS_NAMES[int(label)],
            'global_index': int(global_indices[len(rows)]),
            'sample_file': sample_file,
            'image_path': str((render_root / f'{sample_id}.png').resolve()),
            'fs': int(sfreq),
            'channel_names': channel_names_json,
            'n_channels': int(len(channel_names)),
            'duration_sec': float(duration_sec),
            'render_config_id': render_config['render_id'],
        }
        rows.append(item)
    return pd.DataFrame(rows)


def _split_indices(all_data: dict[str, Any], args) -> dict[str, np.ndarray]:
    split_mode = str(getattr(args, 'split_mode', 'cross-sub'))
    return build_split_indices(
        all_data,
        split_mode=split_mode,
        seed=int(getattr(args, 'seed', 42)),
        split_seed=getattr(args, 'split_seed', None),
        val_fraction=float(getattr(args, 'official_val_fraction', 0.20)),
    )


def prepare_tuab_workspace(args, prep_dir: str) -> TUABPreparedWorkspace:

    prep_path = Path(prep_dir).resolve()
    prep_path.mkdir(parents=True, exist_ok=True)

    adapter, data_dir, all_data = _load_all_data(args)
    channel_names = list(TUAB_VLM_CHANNEL_NAMES)
    _validate_required_channels(channel_names)

    sfreq = int(adapter.sfreq)
    duration_sec = float(all_data['window_sec'])
    split = _split_indices(all_data, args)

    render_config = build_render_config(args)
    prompt = resolve_tuab_prompt(args)
    render_root = Path(getattr(args, 'render_cache_dir', DEFAULT_TUAB_RENDER_CACHE_ROOT)).resolve() / render_config['render_id']
    render_root.mkdir(parents=True, exist_ok=True)

    split_frames = {}
    split_samples = {}
    split_subjects = {}
    limit = int(getattr(args, 'max_samples_per_split', 0))
    for split_name in ['train', 'val', 'test']:
        indices = np.asarray(split[split_name], dtype=np.int64)
        labels = all_data['labels']['abnormal_binary'][indices].astype(np.int64, copy=False)
        chosen_idx = _choose_limit_indices(
            labels,
            limit=limit,
            seed=int(getattr(args, 'seed', 42)) + {'train': 0, 'val': 1, 'test': 2}[split_name],
        )
        indices = indices[chosen_idx]
        frame = _build_manifest(
            split_name=split_name,
            all_data=all_data,
            global_indices=indices,
            channel_names=channel_names,
            sfreq=sfreq,
            duration_sec=duration_sec,
            render_root=render_root,
            render_config=render_config,
            subject_map=all_data['subject_id_map'],
        )
        split_frames[split_name] = frame.reset_index(drop=True)
        split_samples[split_name] = TUABLazySamples(all_data, indices)
        split_subjects[split_name] = sorted(frame['subject'].unique().tolist())
        frame.to_csv(prep_path / f'manifest_{split_name}.csv', index=False)

    render_stats = compute_render_stats(
        train_samples=split_samples['train'],
        channel_names=channel_names,
        sfreq=sfreq,
        render_config=render_config,
        seed=int(getattr(args, 'seed', 42)),
    )
    prep_payload = {
        'dataset': 'tuab',
        'task': 1,
        'task_label': 'abnormal_binary',
        'data_dir': data_dir,
        'split_mode': getattr(args, 'split_mode', 'cross-sub'),
        'seed': int(getattr(args, 'seed', 42)),
        'split_seed': int(
            getattr(args, 'split_seed', None)
            if getattr(args, 'split_seed', None) is not None
            else getattr(args, 'seed', 42)
        ),
        'official_val_fraction': float(getattr(args, 'official_val_fraction', 0.20)),
        'window_size': duration_sec,
        'overlap_ratio': float(getattr(args, 'overlap_ratio', 0.0)),
        'max_windows_per_file': int(getattr(args, 'max_windows_per_file', 0)),
        'tuab_tokenize': int(bool(getattr(args, 'tuab_tokenize', 1))),
        'tuab_token_sec': float(getattr(args, 'tuab_token_sec', 1.0)),
        'tuab_token_stride_sec': float(getattr(args, 'tuab_token_stride_sec', 1.0)),
        'prompt': prompt,
        'model_path': str(getattr(args, 'model_path', DEFAULT_TUAB_MODEL_PATH)),
        'qwen_model_size': str(getattr(args, 'qwen_model_size', '4b')),
        'render_root': str(render_root),
        'class_names': TUAB_CLASS_NAMES,
        'choices': TUAB_CHOICES,
        'samples_per_split': {name: int(len(frame)) for name, frame in split_frames.items()},
    }
    (prep_path / 'prepare_args.json').write_text(json.dumps(prep_payload, indent=2), encoding='utf-8')
    (prep_path / 'render_config.json').write_text(json.dumps(render_config, indent=2), encoding='utf-8')
    (prep_path / 'render_stats.json').write_text(json.dumps(render_stats, indent=2), encoding='utf-8')
    (prep_path / 'subject_split.json').write_text(json.dumps(split_subjects, indent=2), encoding='utf-8')
    return TUABPreparedWorkspace(
        prep_dir=str(prep_path),
        channel_names=channel_names,
        sfreq=sfreq,
        duration_sec=duration_sec,
        prompt=prompt,
        render_config=render_config,
        render_stats=render_stats,
        subject_split=split_subjects,
        splits={
            name: TUABPreparedSplitData(name=name, samples=split_samples[name], manifest=split_frames[name])
            for name in ['train', 'val', 'test']
        },
        class_names=TUAB_CLASS_NAMES,
        choices=TUAB_CHOICES,
    )


def build_tuab_workspace_from_manifests(args, prep_dir: str) -> TUABPreparedWorkspace:

    prep_path = Path(prep_dir).resolve()
    if not prep_path.is_dir():
        raise FileNotFoundError(f'Prep directory not found: {prep_dir}')

    prepare_args = json.loads((prep_path / 'prepare_args.json').read_text(encoding='utf-8'))
    setattr(args, 'max_windows_per_file', int(prepare_args.get('max_windows_per_file', getattr(args, 'max_windows_per_file', 0))))
    setattr(args, 'tuab_tokenize', int(prepare_args.get('tuab_tokenize', getattr(args, 'tuab_tokenize', 1))))
    setattr(args, 'tuab_token_sec', float(prepare_args.get('tuab_token_sec', getattr(args, 'tuab_token_sec', 1.0))))
    setattr(args, 'tuab_token_stride_sec', float(prepare_args.get('tuab_token_stride_sec', getattr(args, 'tuab_token_stride_sec', 1.0))))
    adapter, _data_dir, all_data = _load_all_data(
        args,
        window_size=float(prepare_args.get('window_size', getattr(args, 'window_size', 1.0))),
        overlap_ratio=float(prepare_args.get('overlap_ratio', getattr(args, 'overlap_ratio', 0.0))),
    )
    channel_names = list(TUAB_VLM_CHANNEL_NAMES)
    _validate_required_channels(channel_names)

    render_config = json.loads((prep_path / 'render_config.json').read_text(encoding='utf-8'))
    render_stats = json.loads((prep_path / 'render_stats.json').read_text(encoding='utf-8'))
    subject_split = json.loads((prep_path / 'subject_split.json').read_text(encoding='utf-8'))
    labels = all_data['labels']['abnormal_binary'].astype(np.int64, copy=False)

    splits = {}
    for split_name in ['train', 'val', 'test']:
        manifest = pd.read_csv(prep_path / f'manifest_{split_name}.csv')
        indices_arr = manifest['global_index'].to_numpy(dtype=np.int64, copy=False)
        split_labels = labels[indices_arr]
        if not np.array_equal(split_labels, manifest['label'].to_numpy(dtype=np.int64, copy=False)):
            raise RuntimeError(f'Manifest labels for split={split_name} do not match the raw TUAB cache.')
        splits[split_name] = TUABPreparedSplitData(
            name=split_name,
            samples=TUABLazySamples(all_data, indices_arr),
            manifest=manifest.reset_index(drop=True),
        )

    return TUABPreparedWorkspace(
        prep_dir=str(prep_path),
        channel_names=channel_names,
        sfreq=int(adapter.sfreq),
        duration_sec=float(prepare_args.get('window_size', getattr(args, 'window_size', 1.0))),
        prompt=str(prepare_args.get('prompt', getattr(args, 'prompt', DEFAULT_TUAB_PROMPT))),
        render_config=render_config,
        render_stats=render_stats,
        subject_split=subject_split,
        splits=splits,
        class_names=list(prepare_args.get('class_names', TUAB_CLASS_NAMES)),
        choices=list(prepare_args.get('choices', TUAB_CHOICES)),
    )


class TUABVLMDataset(Dataset):


    def __init__(self,
                 split: TUABPreparedSplitData,
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
        self.base_prompt = str(base_prompt or DEFAULT_TUAB_PROMPT)

    def __len__(self) -> int:
        return len(self.split.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.split.manifest.iloc[index]
        sample = None

        def get_sample() -> np.ndarray:
            nonlocal sample
            if sample is None:
                sample = self.split.samples[index]
            return sample

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
        image_path = str(row['image_path'])
        if not Path(image_path).is_file():
            image_path = ensure_rendered_image(
                sample=get_sample(),
                channel_names=self.channel_names,
                sfreq=self.sfreq,
                duration_sec=self.duration_sec,
                render_config=self.render_config,
                render_stats=self.render_stats,
                output_path=image_path,
            )
        with Image.open(image_path) as handle:
            item['image'] = handle.convert('RGB').copy()
        item['image_path'] = str(image_path)
        return item
