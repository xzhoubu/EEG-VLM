

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_DIR.parent


def _env_or_repo_path(names: tuple[str, ...], default: Path) -> Path:
    for name in names:
        value = os.environ.get(name)
        if value:
            path = Path(value).expanduser()
            return path if path.is_absolute() else REPO_ROOT / path
    return default


DATA_DIR = _env_or_repo_path(('EEG_VLM_DATA_DIR',), REPO_ROOT / 'data')
CHECKPOINT_DIR = _env_or_repo_path(('EEG_VLM_CHECKPOINT_DIR',), REPO_ROOT / 'checkpoints')
LEGACY_CHECKPOINT_DIR = DATA_DIR / 'checkpoints'
OUTPUTS_DIR = _env_or_repo_path(('EEG_VLM_OUTPUT_DIR',), REPO_ROOT / 'outputs')


def repo_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


def default_data_dir(dataset_name: str) -> str:
    return str(DATA_DIR / dataset_name)


def default_output_dir(*parts: str) -> str:
    return str(OUTPUTS_DIR.joinpath(*parts))


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen = set()
    out = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path.resolve()
    return None


def _is_within(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _should_relocate_project_path(path: Path) -> bool:
    return path.is_absolute() and not _is_within(path, REPO_ROOT) and REPO_ROOT.name in path.parts


def _resolve_relative_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return repo_candidate.resolve()
    return repo_candidate


def _relocate_from_anchor(path: Path, anchor: str, *bases: Path) -> list[Path]:
    matches = [idx for idx, part in enumerate(path.parts) if part == anchor]
    candidates = []
    for idx in reversed(matches):
        tail = path.parts[idx + 1:]
        for base in bases:
            candidates.append(base.joinpath(*tail))
    return _dedupe_paths(candidates)


def default_checkpoint_path(*candidate_names: str) -> str:
    model_override = os.environ.get('QWEN3VL_MODEL_PATH')
    if model_override and any('Qwen3-VL' in name for name in candidate_names):
        path = Path(model_override).expanduser()
        return str(path if path.is_absolute() else REPO_ROOT / path)

    candidates = []
    for name in candidate_names:
        candidates.append(CHECKPOINT_DIR / name)
        candidates.append(LEGACY_CHECKPOINT_DIR / name)
    candidates = _dedupe_paths(candidates)
    resolved = _first_existing(candidates)
    if resolved is not None:
        return str(resolved)
    if not candidates:
        return str(CHECKPOINT_DIR)
    return str(candidates[0])


def resolve_data_dir(path_value: str, dataset_name: str) -> str:
    if not path_value:
        return default_data_dir(dataset_name)

    path = _resolve_relative_path(path_value)
    candidates = _relocate_from_anchor(path, 'data', DATA_DIR)
    candidates.append(DATA_DIR / dataset_name)
    candidates = _dedupe_paths(candidates)
    if _should_relocate_project_path(path):
        resolved = _first_existing(candidates)
        if resolved is not None:
            return str(resolved)
    if path.exists():
        return str(path)
    resolved = _first_existing(candidates)
    if resolved is not None:
        return str(resolved)
    if candidates:
        return str(candidates[0])
    return str(path)


def resolve_output_dir(path_value: str) -> str:
    if not path_value:
        return str(default_output_dir())

    path = _resolve_relative_path(path_value)
    candidates = _relocate_from_anchor(path, 'outputs', OUTPUTS_DIR)
    if _should_relocate_project_path(path) and candidates:
        return str(candidates[0])
    if path.exists():
        return str(path)
    resolved = _first_existing(candidates)
    if resolved is not None:
        return str(resolved)
    if candidates:
        return str(candidates[0])
    return str(path)


def resolve_optional_checkpoint_path(path_value: str, *candidate_names: str) -> str:
    if not path_value:
        return ''

    path = _resolve_relative_path(path_value)
    candidates = _relocate_from_anchor(path, 'checkpoints', CHECKPOINT_DIR, LEGACY_CHECKPOINT_DIR)
    if path.name:
        candidates.extend([
            CHECKPOINT_DIR / path.name,
            LEGACY_CHECKPOINT_DIR / path.name,
        ])
    for name in candidate_names:
        candidates.extend([
            CHECKPOINT_DIR / name,
            LEGACY_CHECKPOINT_DIR / name,
        ])
    candidates = _dedupe_paths(candidates)
    if _should_relocate_project_path(path):
        resolved = _first_existing(candidates)
        if resolved is not None:
            return str(resolved)
    if path.exists():
        return str(path)
    resolved = _first_existing(candidates)
    if resolved is not None:
        return str(resolved)
    if candidates:
        return str(candidates[0])
    return str(path)
