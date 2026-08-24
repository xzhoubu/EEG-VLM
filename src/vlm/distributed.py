

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, TypeVar

import pandas as pd
import torch
import torch.distributed as dist


T = TypeVar('T')


def gather_rows(local_rows: list[dict], accelerator=None, chunk_size: int = 4096) -> list[dict]:

    if accelerator is None:
        return local_rows
    if accelerator.num_processes <= 1 or not dist.is_initialized():
        return local_rows if accelerator.is_main_process else []

    chunk_size = max(1, int(chunk_size))
    local_chunk_count = (len(local_rows) + chunk_size - 1) // chunk_size
    chunk_count = torch.tensor(local_chunk_count, device=accelerator.device, dtype=torch.int64)
    dist.all_reduce(chunk_count, op=dist.ReduceOp.MAX)

    flat_rows: list[dict] = []
    for chunk_idx in range(int(chunk_count.item())):
        start = chunk_idx * chunk_size
        local_chunk = local_rows[start:start + chunk_size]
        gathered_chunks: list[list[dict] | None] = [None] * accelerator.num_processes
        dist.all_gather_object(gathered_chunks, local_chunk)
        if accelerator.is_main_process:
            for gathered_chunk in gathered_chunks:
                if gathered_chunk:
                    flat_rows.extend(gathered_chunk)
    return flat_rows if accelerator.is_main_process else []


def gather_frame_via_filesystem(local_frame: pd.DataFrame,
                                accelerator,
                                shard_dir: str | Path,
                                prefix: str,
                                timeout_sec: float = 14400.0) -> pd.DataFrame:

    if accelerator is None or accelerator.num_processes <= 1:
        return local_frame

    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    rank = int(accelerator.process_index)
    world_size = int(accelerator.num_processes)
    shard_path = shard_dir / f'{prefix}_rank{rank}.csv'
    temporary_path = shard_dir / f'{prefix}_rank{rank}.csv.tmp'
    complete_path = shard_dir / f'{prefix}.complete'




    if accelerator.is_main_process:
        complete_path.unlink(missing_ok=True)
        for item in range(world_size):
            (shard_dir / f'{prefix}_rank{item}.csv').unlink(missing_ok=True)
            (shard_dir / f'{prefix}_rank{item}.csv.tmp').unlink(missing_ok=True)
    accelerator.wait_for_everyone()

    local_frame.to_csv(temporary_path, index=False)
    temporary_path.replace(shard_path)

    started = time.monotonic()
    if accelerator.is_main_process:
        expected = [shard_dir / f'{prefix}_rank{item}.csv' for item in range(world_size)]
        while not all(path.is_file() for path in expected):
            if time.monotonic() - started > timeout_sec:
                present = sum(path.is_file() for path in expected)
                raise TimeoutError(f'Waiting for {prefix} rank shards timed out: {present}/{world_size}')
            time.sleep(2.0)
        frames = [pd.read_csv(path) for path in expected]
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        complete_path.write_text(f'rows={len(merged)}\n', encoding='utf-8')
        return merged

    while not complete_path.is_file():
        if time.monotonic() - started > timeout_sec:
            raise TimeoutError(f'Waiting for {prefix} completion marker timed out on rank {rank}')
        time.sleep(2.0)
    return pd.DataFrame()


def run_main_process_with_filesystem_sync(
        action: Callable[[], T],
        accelerator,
        marker_dir: str | Path,
        prefix: str,
        timeout_sec: float = 14400.0) -> T | None:

    if accelerator is None or accelerator.num_processes <= 1:
        return action()

    marker_dir = Path(marker_dir)
    marker_dir.mkdir(parents=True, exist_ok=True)
    success_path = marker_dir / f'{prefix}.success'
    failure_path = marker_dir / f'{prefix}.failed'

    if accelerator.is_main_process:
        success_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        try:
            result = action()
        except BaseException as exc:
            failure_path.write_text(
                f'{type(exc).__name__}: {exc}\n', encoding='utf-8'
            )
            raise
        success_path.write_text('done\n', encoding='utf-8')
        return result

    started = time.monotonic()
    while True:
        if success_path.is_file():
            return None
        if failure_path.is_file():
            detail = failure_path.read_text(encoding='utf-8').strip()
            raise RuntimeError(f'Main-process action {prefix!r} failed: {detail}')
        if time.monotonic() - started > timeout_sec:
            raise TimeoutError(f'Waiting for main-process action {prefix!r} timed out')
        time.sleep(2.0)
