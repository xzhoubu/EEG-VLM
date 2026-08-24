

from __future__ import annotations

import math
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import mne
import numpy as np
import scipy.signal


TUAB_TARGET_SFREQ = 250
TUAB_DEFAULT_WINDOW_SEC = 20.0
TUAB_DEFAULT_OVERLAP = 0.0
TUAB_DEFAULT_TOKEN_SEC = 1.0
TUAB_DEFAULT_TOKEN_STRIDE_SEC = 1.0
TUAB_DEFAULT_VAL_FRACTION = 0.20
TUAB_STATS_MAX_WINDOWS = 128

TUAB_CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T3", "C3", "CZ", "C4", "T4",
    "T5", "P3", "PZ", "P4", "T6",
    "O1", "O2", "A1", "A2",
]

TASK_TYPE_TUAB = {1: "classification"}
TASK_LABEL_KEY_TUAB = {1: "abnormal_binary"}
NUM_CLASSES_TUAB = {1: 2}
IDX_TO_ID_TUAB: dict[int, str] = {}


def _clean_channel_name(name: str) -> str:
    cleaned = str(name).upper().strip()
    cleaned = re.sub(r"^EEG\s+", "", cleaned)
    cleaned = re.sub(r"-(REF|LE|A1|A2)$", "", cleaned)
    return cleaned


def _subject_from_path(path: str | Path) -> str:
    return Path(path).name.split("_")[0]


def _label_from_path(path: str | Path) -> int:
    parts = Path(path).parts
    if "abnormal" in parts:
        return 1
    if "normal" in parts:
        return 0
    raise ValueError(f"Cannot infer TUAB label from path: {path}")


def _official_split_from_path(path: str | Path) -> str:
    parts = Path(path).parts
    if "train" in parts:
        return "train"
    if "eval" in parts:
        return "eval"
    raise ValueError(f"Cannot infer TUAB official split from path: {path}")


def _scan_edf_header(path: str | Path) -> dict[str, Any]:
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    clean_to_raw = {_clean_channel_name(ch): ch for ch in raw.ch_names}
    missing = [ch for ch in TUAB_CHANNELS if ch not in clean_to_raw]
    if missing:
        raise RuntimeError(f"TUAB file is missing required EEG channels {missing}: {path}")
    return {
        "sfreq": float(raw.info["sfreq"]),
        "duration_sec": float(raw.n_times / raw.info["sfreq"]),
        "raw_channels": [clean_to_raw[ch] for ch in TUAB_CHANNELS],
    }


def _window_starts(duration_sec: float, window_sec: float, overlap_ratio: float,
                   max_windows_per_file: int = 0) -> np.ndarray:
    if duration_sec < window_sec:
        return np.asarray([], dtype=np.float32)
    step_sec = max(1e-6, window_sec * (1.0 - float(overlap_ratio)))
    n_windows = int(math.floor((duration_sec - window_sec) / step_sec)) + 1
    starts = np.arange(n_windows, dtype=np.float32) * np.float32(step_sec)
    if max_windows_per_file and n_windows > int(max_windows_per_file):
        chosen = np.linspace(0, n_windows - 1, int(max_windows_per_file), dtype=np.int64)
        starts = starts[chosen]
    return starts.astype(np.float32, copy=False)


def _n_tokens_for_window(window_sec: float, token_sec: float, token_stride_sec: float) -> int:
    if token_sec <= 0 or token_stride_sec <= 0:
        return 1
    if window_sec < token_sec:
        return 1
    return int(math.floor((window_sec - token_sec) / token_stride_sec)) + 1


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


def _feature_dim(feature_set: str, n_channels: int) -> int:
    n_bands = 5 if feature_set in {"psd", "de"} else 10
    return int(n_bands * n_channels)


def _sample_shape(window_sec: float, sfreq: int, use_features: bool, feature_set: str,
                  tokenize: bool, token_sec: float, token_stride_sec: float) -> tuple[int, ...]:
    n_channels = len(TUAB_CHANNELS)
    if tokenize:
        n_tokens = _n_tokens_for_window(window_sec, token_sec, token_stride_sec)
        if use_features:
            return (1, n_tokens, _feature_dim(feature_set, n_channels))
        token_len = int(round(token_sec * sfreq))
        return (1, token_len, n_channels)
    seq_len = int(round(window_sec * sfreq))
    if use_features:
        return (1, _feature_dim(feature_set, n_channels) // n_channels, n_channels)
    return (1, seq_len, n_channels)


def _cache_path(data_dir: str, window_sec: float, overlap_ratio: float, sfreq: int,
                max_windows_per_file: int, tokenize: bool, token_sec: float,
                token_stride_sec: float) -> str:
    token_tag = ""
    if tokenize:
        token_tag = f"_tok{token_sec:g}_stride{token_stride_sec:g}"
    tag = (
        f"cache_tuab_w{window_sec:g}_o{overlap_ratio:g}_sf{int(sfreq)}"
        f"{token_tag}_max{int(max_windows_per_file)}.npz"
    )
    return os.path.join(data_dir, tag)


def _as_str_list(arr) -> list[str]:
    return [str(item) for item in list(arr)]


def load_all_data(data_dir, window_sec=TUAB_DEFAULT_WINDOW_SEC, overlap_ratio=TUAB_DEFAULT_OVERLAP,
                  sfreq=TUAB_TARGET_SFREQ, nan_strategy="max_rt", nan_fill_value=0.0,
                  use_features=False, feature_set="de", use_cache=True,
                  max_windows_per_file=0, tokenize=True,
                  token_sec=TUAB_DEFAULT_TOKEN_SEC,
                  token_stride_sec=TUAB_DEFAULT_TOKEN_STRIDE_SEC, **_kwargs):

    global IDX_TO_ID_TUAB
    data_dir = str(Path(data_dir).resolve())
    window_sec = float(window_sec)
    overlap_ratio = float(overlap_ratio)
    sfreq = int(sfreq)
    max_windows_per_file = int(max_windows_per_file or 0)
    tokenize = bool(int(tokenize)) if isinstance(tokenize, (int, np.integer, str)) else bool(tokenize)
    token_sec = float(token_sec)
    token_stride_sec = float(token_stride_sec)
    cache_file = _cache_path(
        data_dir,
        window_sec,
        overlap_ratio,
        sfreq,
        max_windows_per_file,
        tokenize=tokenize,
        token_sec=token_sec,
        token_stride_sec=token_stride_sec,
    )

    if use_cache and os.path.exists(cache_file):
        d = np.load(cache_file, allow_pickle=True)
        subject_ids = _as_str_list(d["subject_ids"])
        IDX_TO_ID_TUAB = {idx + 1: subject for idx, subject in enumerate(subject_ids)}
        cached_tokenize = bool(int(d["tokenize"])) if "tokenize" in d else tokenize
        cached_token_sec = float(d["token_sec"]) if "token_sec" in d else token_sec
        cached_token_stride_sec = float(d["token_stride_sec"]) if "token_stride_sec" in d else token_stride_sec
        shape = _sample_shape(
            float(d["window_sec"]),
            int(d["sfreq"]),
            bool(use_features),
            feature_set,
            cached_tokenize,
            cached_token_sec,
            cached_token_stride_sec,
        )
        all_data = {
            "samples": np.empty(shape, dtype=np.float32),
            "labels": {"abnormal_binary": d["labels"].astype(np.int64)},
            "subjects": d["subjects"].astype(np.int64),
            "conditions": d["conditions"].astype(np.int64),
            "valid": {"abnormal_binary": np.ones(len(d["labels"]), dtype=bool)},
            "ch_names": list(TUAB_CHANNELS),
            "sfreq": int(d["sfreq"]),
            "window_sec": float(d["window_sec"]),
            "tokenize": cached_tokenize,
            "token_sec": cached_token_sec,
            "token_stride_sec": cached_token_stride_sec,
            "n_tokens": _n_tokens_for_window(float(d["window_sec"]), cached_token_sec, cached_token_stride_sec),
            "tokenized_raw": bool(cached_tokenize and not use_features),
            "feature_set": feature_set if use_features else "raw",
            "file_paths": _as_str_list(d["file_paths"]),
            "file_sfreqs": d["file_sfreqs"].astype(np.float32),
            "file_raw_channels": [_as_str_list(row) for row in d["file_raw_channels"]],
            "file_subjects": d["file_subjects"].astype(np.int64),
            "file_labels": d["file_labels"].astype(np.int64),
            "file_official_splits": _as_str_list(d["file_official_splits"]),
            "window_file_idx": d["window_file_idx"].astype(np.int32),
            "window_start_sec": d["window_start_sec"].astype(np.float32),
            "subject_id_map": IDX_TO_ID_TUAB.copy(),
            "use_features": bool(use_features),
            "feature_set_requested": feature_set,
        }
        return all_data

    root = Path(data_dir) / "edf"
    edf_files = sorted(root.glob("*/*/01_tcp_ar/*.edf"))
    if not edf_files:
        raise FileNotFoundError(f"No TUAB EDF files found under {root}")

    subject_names = sorted({_subject_from_path(path) for path in edf_files})
    subject_to_idx = {subject: idx + 1 for idx, subject in enumerate(subject_names)}
    IDX_TO_ID_TUAB = {idx: subject for subject, idx in subject_to_idx.items()}

    file_paths: list[str] = []
    file_sfreqs: list[float] = []
    file_raw_channels: list[list[str]] = []
    file_subjects: list[int] = []
    file_labels: list[int] = []
    file_official_splits: list[str] = []
    window_file_idx: list[np.ndarray] = []
    window_start_sec: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    subjects: list[np.ndarray] = []
    conditions: list[np.ndarray] = []

    for file_idx, path in enumerate(edf_files):
        if file_idx > 0 and file_idx % 250 == 0:
            print(f"  TUAB header scan: {file_idx}/{len(edf_files)} EDF files")
        header = _scan_edf_header(path)
        starts = _window_starts(
            duration_sec=header["duration_sec"],
            window_sec=window_sec,
            overlap_ratio=overlap_ratio,
            max_windows_per_file=max_windows_per_file,
        )
        if starts.size == 0:
            continue
        label = _label_from_path(path)
        subject = subject_to_idx[_subject_from_path(path)]
        official_split = _official_split_from_path(path)

        file_paths.append(str(path.resolve()))
        file_sfreqs.append(float(header["sfreq"]))
        file_raw_channels.append(list(header["raw_channels"]))
        file_subjects.append(subject)
        file_labels.append(label)
        file_official_splits.append(official_split)

        manifest_file_idx = len(file_paths) - 1
        n = len(starts)
        window_file_idx.append(np.full(n, manifest_file_idx, dtype=np.int32))
        window_start_sec.append(starts)
        labels.append(np.full(n, label, dtype=np.int64))
        subjects.append(np.full(n, subject, dtype=np.int64))
        conditions.append(np.full(n, label, dtype=np.int64))

    if not window_file_idx:
        raise RuntimeError(f"No TUAB windows could be created from {data_dir}")

    window_file_idx_arr = np.concatenate(window_file_idx)
    window_start_sec_arr = np.concatenate(window_start_sec)
    labels_arr = np.concatenate(labels)
    subjects_arr = np.concatenate(subjects)
    conditions_arr = np.concatenate(conditions)
    seq_len = int(round(window_sec * sfreq))
    n_channels = len(TUAB_CHANNELS)

    if use_cache:
        np.savez(
            cache_file,
            file_paths=np.asarray(file_paths, dtype=object),
            file_sfreqs=np.asarray(file_sfreqs, dtype=np.float32),
            file_raw_channels=np.asarray(file_raw_channels, dtype=object),
            file_subjects=np.asarray(file_subjects, dtype=np.int64),
            file_labels=np.asarray(file_labels, dtype=np.int64),
            file_official_splits=np.asarray(file_official_splits, dtype=object),
            window_file_idx=window_file_idx_arr,
            window_start_sec=window_start_sec_arr,
            labels=labels_arr,
            subjects=subjects_arr,
            conditions=conditions_arr,
            subject_ids=np.asarray(subject_names, dtype=object),
            sfreq=np.asarray(sfreq, dtype=np.int64),
            window_sec=np.asarray(window_sec, dtype=np.float32),
            seq_len=np.asarray(seq_len, dtype=np.int64),
            tokenize=np.asarray(int(tokenize), dtype=np.int64),
            token_sec=np.asarray(token_sec, dtype=np.float32),
            token_stride_sec=np.asarray(token_stride_sec, dtype=np.float32),
        )

    shape = _sample_shape(
        window_sec,
        sfreq,
        bool(use_features),
        feature_set,
        tokenize,
        token_sec,
        token_stride_sec,
    )

    print(
        f"  TUAB manifest: files={len(file_paths)}, windows={len(labels_arr)}, "
        f"window_sec={window_sec:g}, sfreq={sfreq}, tokenize={int(tokenize)}, "
        f"token_sec={token_sec:g}, max_windows_per_file={max_windows_per_file}"
    )
    all_data = {
        "samples": np.empty(shape, dtype=np.float32),
        "labels": {"abnormal_binary": labels_arr},
        "subjects": subjects_arr,
        "conditions": conditions_arr,
        "valid": {"abnormal_binary": np.ones(len(labels_arr), dtype=bool)},
        "ch_names": list(TUAB_CHANNELS),
        "sfreq": sfreq,
        "window_sec": window_sec,
        "tokenize": tokenize,
        "token_sec": token_sec,
        "token_stride_sec": token_stride_sec,
        "n_tokens": _n_tokens_for_window(window_sec, token_sec, token_stride_sec),
        "tokenized_raw": bool(tokenize and not use_features),
        "feature_set": feature_set if use_features else "raw",
        "file_paths": file_paths,
        "file_sfreqs": np.asarray(file_sfreqs, dtype=np.float32),
        "file_raw_channels": file_raw_channels,
        "file_subjects": np.asarray(file_subjects, dtype=np.int64),
        "file_labels": np.asarray(file_labels, dtype=np.int64),
        "file_official_splits": file_official_splits,
        "window_file_idx": window_file_idx_arr,
        "window_start_sec": window_start_sec_arr,
        "subject_id_map": IDX_TO_ID_TUAB.copy(),
        "use_features": bool(use_features),
        "feature_set_requested": feature_set,
    }
    return all_data


def _val_subjects_from_train(all_data, val_fraction: float, seed: int) -> set[int]:
    file_subjects = np.asarray(all_data["file_subjects"], dtype=np.int64)
    file_labels = np.asarray(all_data["file_labels"], dtype=np.int64)
    file_splits = np.asarray(all_data["file_official_splits"], dtype=object)
    train_mask = file_splits == "train"
    train_subjects = sorted(set(file_subjects[train_mask].tolist()))
    subject_label = {}
    for subject in train_subjects:
        labels = file_labels[train_mask & (file_subjects == subject)]
        subject_label[int(subject)] = int(np.max(labels)) if labels.size else 0

    rng = np.random.RandomState(seed)
    val_subjects: set[int] = set()
    for label in [0, 1]:
        group = [subject for subject in train_subjects if subject_label[subject] == label]
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * float(val_fraction)))) if group else 0
        val_subjects.update(int(subject) for subject in group[:n_val])
    return val_subjects


def build_split_indices(all_data, split_mode="cross-sub", seed=42, split_seed=None,
                        val_fraction=TUAB_DEFAULT_VAL_FRACTION):
    if split_mode == "cross-sub":
        val_subjects = _val_subjects_from_train(all_data, val_fraction=val_fraction, seed=seed if split_seed is None else split_seed)
        file_splits = np.asarray(all_data["file_official_splits"], dtype=object)
        file_subjects = np.asarray(all_data["file_subjects"], dtype=np.int64)
        win_file_idx = np.asarray(all_data["window_file_idx"], dtype=np.int64)
        win_file_splits = file_splits[win_file_idx]
        win_subjects = file_subjects[win_file_idx]
        test_idx = np.where(win_file_splits == "eval")[0]
        train_pool = win_file_splits == "train"
        val_mask = train_pool & np.isin(win_subjects, list(val_subjects))
        train_mask = train_pool & ~val_mask
        return {
            "train": np.where(train_mask)[0],
            "val": np.where(val_mask)[0],
            "test": test_idx,
        }
    raise ValueError(f"Unknown TUAB split_mode: {split_mode}")


def _load_resampled_file(path: str, raw_channels: list[str], target_sfreq: int) -> np.ndarray:
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    data = raw.get_data(picks=raw_channels) * 1e6
    orig_sfreq = int(round(float(raw.info["sfreq"])))
    if orig_sfreq != int(target_sfreq):
        data = scipy.signal.resample_poly(data, up=int(target_sfreq), down=orig_sfreq, axis=-1)
    return data.T.astype(np.float32, copy=False)


class TUABWindowDataset:


    def __init__(self, all_data: dict[str, Any], indices: np.ndarray,
                 labels=None, subjects=None, conditions=None,
                 mode: str = "end2end", use_features: bool = False,
                 normalizer=None, max_cached_files: int = 2, **_kwargs):
        if mode != "end2end" or use_features or normalizer is not None:
            raise ValueError("The EEG-VLM release exposes raw image rendering only.")
        self.all_data = all_data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.target_sfreq = int(all_data["sfreq"])
        self.window_sec = float(all_data["window_sec"])
        self.seq_len = int(round(self.window_sec * self.target_sfreq))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.max_cached_files = int(max_cached_files)

    def __len__(self) -> int:
        return len(self.indices)

    def _file_array(self, file_index: int) -> np.ndarray:
        file_index = int(file_index)
        if file_index in self._cache:
            array = self._cache.pop(file_index)
            self._cache[file_index] = array
            return array
        array = _load_resampled_file(
            self.all_data["file_paths"][file_index],
            self.all_data["file_raw_channels"][file_index],
            self.target_sfreq,
        )
        self._cache[file_index] = array
        while len(self._cache) > self.max_cached_files:
            self._cache.popitem(last=False)
        return array

    def raw_window(self, global_index: int) -> np.ndarray:
        file_index = int(self.all_data["window_file_idx"][global_index])
        start = int(round(float(self.all_data["window_start_sec"][global_index]) * self.target_sfreq))
        window = self._file_array(file_index)[start:start + self.seq_len]
        if window.shape[0] < self.seq_len:
            padding = np.zeros((self.seq_len - window.shape[0], window.shape[1]), dtype=np.float32)
            window = np.concatenate([window, padding], axis=0)
        return window[:self.seq_len].astype(np.float32, copy=False)
