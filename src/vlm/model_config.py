

from __future__ import annotations

from pathlib import Path
from typing import Any

from runtime import default_checkpoint_path


DEFAULT_QWEN3VL_MODEL_SIZE = '4b'
QWEN3VL_MODEL_SIZES = ('2b', '4b', '8b')
QWEN3VL_MODEL_REGISTRY = {
    '2b': {
        'checkpoint_name': 'Qwen3-VL-2B-Instruct',
        'hf_repo_id': 'Qwen/Qwen3-VL-2B-Instruct',
        'slug': 'qwen3-vl-2b-instruct',
    },
    '4b': {
        'checkpoint_name': 'Qwen3-VL-4B-Instruct',
        'hf_repo_id': 'Qwen/Qwen3-VL-4B-Instruct',
        'slug': 'qwen3-vl-4b-instruct',
    },
    '8b': {
        'checkpoint_name': 'Qwen3-VL-8B-Instruct',
        'hf_repo_id': 'Qwen/Qwen3-VL-8B-Instruct',
        'slug': 'qwen3-vl-8b-instruct',
    },
}


def normalize_qwen3vl_model_size(model_size: str | int | None = DEFAULT_QWEN3VL_MODEL_SIZE) -> str:

    value = str(model_size or DEFAULT_QWEN3VL_MODEL_SIZE).strip().lower()
    value = value.replace('_', '-')
    aliases = {
        '2': '2b',
        '2b': '2b',
        'qwen2b': '2b',
        'qwen-2b': '2b',
        'qwen3vl-2b': '2b',
        'qwen3-vl-2b': '2b',
        'qwen3-vl-2b-instruct': '2b',
        'qwen/qwen3-vl-2b-instruct': '2b',
        '4': '4b',
        '4b': '4b',
        'qwen4b': '4b',
        'qwen-4b': '4b',
        'qwen3vl-4b': '4b',
        'qwen3-vl-4b': '4b',
        'qwen3-vl-4b-instruct': '4b',
        'qwen/qwen3-vl-4b-instruct': '4b',
        '8': '8b',
        '8b': '8b',
        'qwen8b': '8b',
        'qwen-8b': '8b',
        'qwen3vl-8b': '8b',
        'qwen3-vl-8b': '8b',
        'qwen3-vl-8b-instruct': '8b',
        'qwen/qwen3-vl-8b-instruct': '8b',
    }
    normalized = aliases.get(value, value)
    if normalized not in QWEN3VL_MODEL_REGISTRY:
        choices = ', '.join(QWEN3VL_MODEL_SIZES)
        raise ValueError(f'Unknown Qwen3-VL model size {model_size!r}; expected one of: {choices}.')
    return normalized


def qwen3vl_checkpoint_name(model_size: str | int | None = DEFAULT_QWEN3VL_MODEL_SIZE) -> str:
    size = normalize_qwen3vl_model_size(model_size)
    return str(QWEN3VL_MODEL_REGISTRY[size]['checkpoint_name'])


def qwen3vl_hf_repo_id(model_size: str | int | None = DEFAULT_QWEN3VL_MODEL_SIZE) -> str:
    size = normalize_qwen3vl_model_size(model_size)
    return str(QWEN3VL_MODEL_REGISTRY[size]['hf_repo_id'])


def qwen3vl_model_slug(model_or_args: Any = DEFAULT_QWEN3VL_MODEL_SIZE) -> str:
    model_size = getattr(model_or_args, 'qwen_model_size', model_or_args)
    size = normalize_qwen3vl_model_size(model_size)
    return str(QWEN3VL_MODEL_REGISTRY[size]['slug'])


def infer_qwen3vl_model_size(model_path_or_id: str | None) -> str | None:
    value = str(model_path_or_id or '').strip().lower().replace('_', '-')
    if not value:
        return None
    for size, entry in QWEN3VL_MODEL_REGISTRY.items():
        markers = {
            str(entry['checkpoint_name']).lower(),
            str(entry['hf_repo_id']).lower(),
            str(entry['slug']).lower(),
        }
        if value in markers or any(marker in value for marker in markers):
            return size
    return None


def default_qwen3vl_model_path(model_size: str | int | None = DEFAULT_QWEN3VL_MODEL_SIZE) -> str:
    return default_checkpoint_path(qwen3vl_checkpoint_name(model_size))


def resolve_qwen3vl_model_path(model_path: str | None = '',
                               model_size: str | int | None = DEFAULT_QWEN3VL_MODEL_SIZE,
                               hf_fallback: bool = True) -> str:

    explicit = str(model_path or '').strip()
    if explicit:
        return explicit
    local_path = default_qwen3vl_model_path(model_size)
    if Path(local_path).exists() or not hf_fallback:
        return local_path
    return qwen3vl_hf_repo_id(model_size)


def add_qwen3vl_model_args(parser, default_size: str = DEFAULT_QWEN3VL_MODEL_SIZE) -> None:
    parser.add_argument(
        '--qwen_model_size',
        type=str,
        default=default_size,
        help='Qwen3-VL checkpoint size to use when --model_path is not set; accepts 2b, 4b, or 8b.',
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='',
        help='Explicit local checkpoint path or Hugging Face repo id. Overrides --qwen_model_size.',
    )


def finalize_qwen3vl_args(args):
    inferred_size = infer_qwen3vl_model_size(getattr(args, 'model_path', ''))
    args.qwen_model_size = inferred_size or normalize_qwen3vl_model_size(
        getattr(args, 'qwen_model_size', DEFAULT_QWEN3VL_MODEL_SIZE)
    )
    args.model_path = resolve_qwen3vl_model_path(getattr(args, 'model_path', ''), args.qwen_model_size)
    args.qwen_model_slug = qwen3vl_model_slug(args.qwen_model_size)
    return args
