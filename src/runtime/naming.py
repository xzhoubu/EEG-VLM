

import datetime
import re


def _slug(text: str) -> str:
    text = (text or '').strip().lower()
    text = re.sub(r'[^a-z0-9._-]+', '-', text)
    text = re.sub(r'-{2,}', '-', text).strip('-')
    return text or 'na'


def build_experiment_name(dataset: str, task_id: int, task_label: str, model: str,
                          split_mode: str, seed: int, timestamp: str = '') -> str:
    ts = timestamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    parts = [
        _slug(dataset),
        f"task{task_id}",
        _slug(task_label),
        _slug(model),
        _slug(split_mode),
        f"seed{int(seed)}",
        ts,
    ]
    return '_'.join(parts)


def build_log_name(dataset: str, task_id: str, task_label: str, model: str,
                   split_mode: str, seed: int, timestamp: str = '') -> str:
    ts = timestamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    parts = [
        _slug(dataset),
        f"task{_slug(str(task_id))}",
        _slug(task_label),
        _slug(model),
        _slug(split_mode),
        f"seed{int(seed)}",
        ts,
    ]
    return '_'.join(parts) + '.log'
