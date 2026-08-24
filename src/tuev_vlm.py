

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DistributedType, InitProcessGroupKwargs
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from metric import classification_metrics
from runtime import build_experiment_name, resolve_output_dir
from vlm.data import DistributedWeightedSampler
from vlm.distributed import (
    gather_frame_via_filesystem,
    run_main_process_with_filesystem_sync,
)
from vlm.eval import finalize_multiclass_eval, save_training_curves
from vlm.qwen3vl import (
    Qwen3VLInferenceCollator as VLMInferenceCollator,
    Qwen3VLTrainCollator as VLMTrainCollator,
    apply_lora_adapters as apply_vlm_lora_adapters,
    collect_choice_predictions,
    load_qwen3vl_for_inference as load_vlm_for_inference,
    load_qwen3vl_for_training as load_vlm_for_training,
)
from vlm.model_config import (
    add_qwen3vl_model_args as add_vlm_model_args,
    finalize_qwen3vl_args as finalize_vlm_args,
    qwen3vl_model_slug as vlm_model_slug,
)
from vlm.tuev_data import (
    DEFAULT_TUEV_MODEL_PATH,
    DEFAULT_TUEV_PROMPT,
    DEFAULT_TUEV_RENDER_CACHE_ROOT,
    DEFAULT_TUEV_RESULT_ROOT,
    TUEVVLMDataset,
    build_tuev_workspace_from_manifests,
    prepare_tuev_workspace,
)

TUEV_DISTRIBUTED_TIMEOUT = timedelta(hours=6)


def _resolve_run_dir(result_root: str, mode_slug: str, args, timestamp: str = '') -> str:
    name = build_experiment_name(
        dataset='tuev',
        task_id=1,
        task_label='event_multiclass',
        model=f'{vlm_model_slug(args)}-{mode_slug}',
        split_mode=args.split_mode,
        seed=args.seed,
        timestamp=timestamp,
    )
    exp_tag = str(getattr(args, 'exp_tag', '')).strip()
    if exp_tag:
        name = f'{name}_{exp_tag}'
    return str(Path(result_root).resolve() / name)


def _broadcast_object(value, accelerator: Accelerator | None = None):
    payload = [value]
    if accelerator is not None:
        if accelerator.num_processes > 1 and dist.is_initialized():
            dist.broadcast_object_list(payload, src=0)
        return payload[0]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _save_json(path: str, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _restore_training_vlm_args_if_needed(args, run_dir: str):
    train_args_path = Path(run_dir) / 'args.json'
    if not train_args_path.is_file():
        return args
    try:
        train_args = json.loads(train_args_path.read_text(encoding='utf-8'))
    except Exception:
        return args
    for key in ['qwen_model_size', 'model_path']:
        if key in train_args:
            setattr(args, key, train_args[key])
    return finalize_vlm_args(args)


def _choice_first_token_ids(processor, choices: list[str]) -> list[int]:
    token_ids = []
    observed = {}
    for choice in choices:
        ids = processor.tokenizer(str(choice), add_special_tokens=False).input_ids
        observed[str(choice)] = ids
        if not ids:
            raise RuntimeError(f'Choice labels must tokenize to at least one token for train metrics: {observed}')
        token_ids.append(int(ids[0]))
    if len(set(token_ids)) != len(token_ids):
        raise RuntimeError(f'TUEV choices must have unique first tokens for train metrics: {observed}')
    return token_ids


def _distributed_average(total: float, count: int, accelerator: Accelerator) -> float:
    payload = torch.tensor([float(total), float(count)], device=accelerator.device, dtype=torch.float64)
    if accelerator.num_processes > 1 and dist.is_initialized():
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    return float(payload[0].item() / max(payload[1].item(), 1.0))


def _gather_metric_rows(local_rows: list[dict],
                        accelerator: Accelerator,
                        shard_dir: str | Path,
                        prefix: str) -> list[dict]:
    frame = gather_frame_via_filesystem(
        pd.DataFrame(local_rows), accelerator, shard_dir=shard_dir, prefix=prefix,
    )
    return frame.to_dict(orient='records') if accelerator.is_main_process else []


def _ensure_workspace(args, prep_dir: str, create_if_missing: bool = True):
    prep_path = Path(prep_dir)
    if _prep_complete(prep_path, args):
        return build_tuev_workspace_from_manifests(args, str(prep_path))
    if not create_if_missing:
        raise FileNotFoundError(f'Prepared workspace not found: {prep_dir}')
    return prepare_tuev_workspace(args, str(prep_path))


def _prep_complete(prep_path: Path, args=None) -> bool:
    required = [
        'prepare_args.json',
        'manifest_train.csv',
        'manifest_val.csv',
        'manifest_test.csv',
        'render_config.json',
        'render_stats.json',
        'subject_split.json',
    ]
    return prep_path.is_dir() and all((prep_path / name).is_file() for name in required)


def _wait_for_prep_complete(args,
                            prep_dir: str,
                            failed_path: Path,
                            timeout_sec: float = 12 * 60 * 60,
                            poll_interval_sec: float = 15.0) -> None:
    prep_path = Path(prep_dir)
    start = time.time()
    while not _prep_complete(prep_path, args):
        if failed_path.is_file():
            detail = failed_path.read_text(encoding='utf-8', errors='replace')
            raise RuntimeError(f'Workspace preparation failed on the main process:\n{detail}')
        if time.time() - start > timeout_sec:
            raise TimeoutError(f'Timed out waiting for prepared workspace to finish: {prep_dir}')
        time.sleep(poll_interval_sec)


def _make_dataset(workspace, split_name: str):
    return TUEVVLMDataset(
        split=workspace.splits[split_name],
        channel_names=workspace.channel_names,
        sfreq=workspace.sfreq,
        duration_sec=workspace.duration_sec,
        render_config=workspace.render_config,
        render_stats=workspace.render_stats,
        choices=workspace.choices,
        base_prompt=workspace.prompt,
    )


def _build_eval_loaders(workspace, processor, batch_size: int, num_workers: int, model_family: str = 'qwen3vl'):
    collator = VLMInferenceCollator(processor, prompt=workspace.prompt)
    loaders = {}
    for split_name in ['val', 'test']:
        loaders[split_name] = DataLoader(
            _make_dataset(workspace, split_name),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def _run_prepare(args):
    result_root = resolve_output_dir(args.result_dir or DEFAULT_TUEV_RESULT_ROOT)
    prep_dir = args.prep_dir or _resolve_run_dir(result_root, 'prepare', args)
    workspace = prepare_tuev_workspace(args, prep_dir)
    _save_json(os.path.join(prep_dir, 'args.json'), vars(args))
    print(f'Prepared workspace -> {prep_dir}')
    for split_name, split in workspace.splits.items():
        label_col = 'label_name' if 'label_name' in split.manifest.columns else 'condition_name'
        counts = split.manifest[label_col].value_counts().reindex(workspace.class_names, fill_value=0).astype(int).to_dict()
        print(f'  {split_name}: {len(split.manifest)} samples | counts={counts}')


def _run_zero_shot(args):
    result_root = resolve_output_dir(args.result_dir or DEFAULT_TUEV_RESULT_ROOT)
    run_dir = args.run_dir or _resolve_run_dir(result_root, 'zero-shot', args)
    os.makedirs(run_dir, exist_ok=True)
    prep_dir = args.prep_dir or os.path.join(run_dir, 'prepared')
    workspace = _ensure_workspace(args, prep_dir, create_if_missing=True)
    _save_json(os.path.join(run_dir, 'args.json'), vars(args))

    model, processor, device = load_vlm_for_inference(
        model_path=args.model_path,
        use_bf16=bool(args.use_bf16),
        device_map=args.device_map,
        gpu_id=args.gpu_id,
        enable_flash_attention=bool(args.flash_attention_2),
    )
    loaders = _build_eval_loaders(
        workspace,
        processor,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
    )
    val_frame = collect_choice_predictions(
        model,
        processor,
        loaders['val'],
        choices=workspace.choices,
        class_names=workspace.class_names,
        device=device,
        max_new_tokens=args.max_new_tokens,
        save_generated_text=bool(args.save_generated_text),
        sequence_scoring=bool(args.sequence_scoring),
    )
    test_frame = collect_choice_predictions(
        model,
        processor,
        loaders['test'],
        choices=workspace.choices,
        class_names=workspace.class_names,
        device=device,
        max_new_tokens=args.max_new_tokens,
        save_generated_text=bool(args.save_generated_text),
        sequence_scoring=bool(args.sequence_scoring),
    )
    metrics = finalize_multiclass_eval(val_frame, test_frame, run_dir, workspace.class_names)
    print(f'Zero-shot results -> {run_dir}')
    for key, value in metrics.items():
        print(f'  {key}: {value:.4f}')


def _train_sampler_from_manifest(manifest: pd.DataFrame, num_replicas: int, rank: int, seed: int):
    labels = manifest['label'].to_numpy(dtype=np.int64, copy=False)
    counts = np.bincount(labels, minlength=6).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = np.asarray([1.0 / counts[label] for label in labels], dtype=np.float64)
    return DistributedWeightedSampler(weights=weights, num_replicas=num_replicas, rank=rank, seed=seed)


def _normalize_fsdp_parameter_dtype(model, accelerator: Accelerator, use_bf16: bool):

    if accelerator.distributed_type != DistributedType.FSDP:
        return model
    target_dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = model.to(dtype=target_dtype)
    floating_dtypes = {
        param.dtype for param in model.parameters()
        if param.is_floating_point()
    }
    if floating_dtypes != {target_dtype}:
        raise RuntimeError(
            f'FSDP parameter dtype normalization failed: expected {target_dtype}, got {floating_dtypes}'
        )
    if accelerator.is_main_process:
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f'[FSDP] normalized all floating parameters to {target_dtype}; trainable={trainable:,}')
    return model


def _limit_processor_image_pixels(processor, max_pixels: int) -> None:

    max_pixels = int(max_pixels)
    if max_pixels <= 0:
        return
    image_processor = getattr(processor, 'image_processor', None)
    size = getattr(image_processor, 'size', None)
    if size is None or not hasattr(size, 'longest_edge'):
        raise RuntimeError('processor_max_pixels requires an image processor with a longest_edge size limit')
    size.longest_edge = max_pixels
    if getattr(size, 'shortest_edge', None) is not None:
        size.shortest_edge = min(int(size.shortest_edge), max_pixels)


def _run_train_lora(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
    timeout_kwargs = InitProcessGroupKwargs(timeout=TUEV_DISTRIBUTED_TIMEOUT)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision='bf16' if bool(args.use_bf16) else 'no',
        kwargs_handlers=[ddp_kwargs, timeout_kwargs],
    )
    result_root = resolve_output_dir(args.result_dir or DEFAULT_TUEV_RESULT_ROOT)

    timestamp = None
    if accelerator.is_main_process:
        if args.run_dir:
            run_dir = str(Path(args.run_dir).resolve())
        else:
            run_dir = _resolve_run_dir(result_root, 'lora', args)
        timestamp = run_dir
    run_dir = _broadcast_object(timestamp, accelerator=accelerator)
    if accelerator.is_main_process:
        os.makedirs(run_dir, exist_ok=True)
        _save_json(os.path.join(run_dir, 'args.json'), vars(args))
    accelerator.wait_for_everyone()

    prep_dir = args.prep_dir or os.path.join(run_dir, 'prepared')
    prep_failed_path = Path(prep_dir) / '.prepare_failed'
    if accelerator.is_main_process:
        prep_failed_path.parent.mkdir(parents=True, exist_ok=True)
        if prep_failed_path.is_file():
            prep_failed_path.unlink()
        try:
            if not _prep_complete(Path(prep_dir), args):
                prepare_tuev_workspace(args, prep_dir)
        except Exception as exc:
            prep_failed_path.write_text(f'{type(exc).__name__}: {exc}\n', encoding='utf-8')
            raise
    else:
        _wait_for_prep_complete(args, prep_dir, prep_failed_path)
    accelerator.wait_for_everyone()
    workspace = build_tuev_workspace_from_manifests(args, prep_dir)

    model, processor = load_vlm_for_training(
        model_path=args.model_path,
        use_bf16=bool(args.use_bf16),
        enable_flash_attention=bool(args.flash_attention_2),
    )
    _limit_processor_image_pixels(processor, args.processor_max_pixels)
    model = apply_vlm_lora_adapters(model, args)
    model = _normalize_fsdp_parameter_dtype(model, accelerator, bool(args.use_bf16))

    train_dataset = _make_dataset(workspace, 'train')
    val_dataset = _make_dataset(workspace, 'val')
    train_collator = VLMTrainCollator(
        processor,
        prompt=workspace.prompt,
        include_label_metadata=True,
    )
    val_loss_collator = VLMTrainCollator(processor, prompt=workspace.prompt)
    eval_collator = VLMInferenceCollator(processor, prompt=workspace.prompt)
    train_sampler = None
    if bool(int(getattr(args, 'balanced_sampling', 1))):
        train_sampler = _train_sampler_from_manifest(
            workspace.splits['train'].manifest,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=args.seed,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=train_collator,
        pin_memory=torch.cuda.is_available(),
        drop_last=accelerator.num_processes > 1,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=eval_collator,
        pin_memory=torch.cuda.is_available(),
    )
    val_loss_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_loss_collator,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps)))
    total_steps = steps_per_epoch * max(1, args.max_epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    model, optimizer, train_loader, val_loader, val_loss_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
        val_loss_loader,
        scheduler,
    )
    candidate_token_ids = _choice_first_token_ids(processor, workspace.choices)

    best_metric = -float('inf')
    best_epoch = 0
    wait = 0
    history_rows = []
    best_adapter_dir = os.path.join(run_dir, 'best_adapter')

    for epoch in range(1, args.max_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss_total = 0.0
        train_loss_steps = 0
        train_metric_rows = []
        for batch in train_loader:
            choice_labels = batch.pop('_choice_labels')
            answer_positions = batch.pop('_answer_positions')
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            train_loss_total += float(loss.detach().item())
            train_loss_steps += 1
            with torch.no_grad():
                logits = outputs.logits.detach()
                next_token_positions = (answer_positions.to(logits.device) - 1).clamp_min(0)
                batch_idx = torch.arange(logits.size(0), device=logits.device)
                next_token_logits = logits[batch_idx, next_token_positions]
                choice_logits = next_token_logits[:, candidate_token_ids]
                train_pred = torch.argmax(choice_logits.float(), dim=-1).detach().cpu().tolist()
                train_true = choice_labels.detach().cpu().tolist()
                train_metric_rows.extend(
                    {'ground_truth': int(true), 'prediction': int(pred)}
                    for true, pred in zip(train_true, train_pred)
                )

        epoch_loss = _distributed_average(train_loss_total, train_loss_steps, accelerator)
        train_metric_rows = _gather_metric_rows(
            train_metric_rows, accelerator,
            shard_dir=Path(run_dir) / '.distributed_rows', prefix=f'train_epoch{epoch}',
        )

        model.eval()
        val_loss_total = 0.0
        val_loss_steps = 0
        with torch.no_grad():
            for batch in val_loss_loader:
                outputs = model(**batch)
                val_loss_total += float(outputs.loss.detach().item())
                val_loss_steps += 1
        epoch_val_loss = _distributed_average(val_loss_total, val_loss_steps, accelerator)

        val_frame = collect_choice_predictions(
            model,
            processor,
            val_loader,
            choices=workspace.choices,
            class_names=workspace.class_names,
            device=None,
            max_new_tokens=args.max_new_tokens,
            save_generated_text=False,
            accelerator=accelerator,
            sequence_scoring=bool(args.sequence_scoring),
            partial_output_dir=str(Path(run_dir) / '.distributed_rows' / f'val_epoch{epoch}'),
            progress_label='val',
        )
        should_save = False
        should_stop = False
        if accelerator.is_main_process:
            train_frame = pd.DataFrame(train_metric_rows)
            train_metrics = classification_metrics(
                train_frame['ground_truth'].to_numpy(dtype='int64', copy=False),
                train_frame['prediction'].to_numpy(dtype='int64', copy=False),
            )
            val_true = val_frame['ground_truth'].to_numpy(dtype='int64', copy=False)
            val_pred = val_frame['prediction'].to_numpy(dtype='int64', copy=False)
            val_metrics = classification_metrics(val_true, val_pred)
            history_rows.append({
                'epoch': epoch,
                'train_loss': epoch_loss,
                'val_loss': epoch_val_loss,
                'train_balanced_acc': train_metrics['balanced_acc'],
                'val_balanced_acc': val_metrics['balanced_acc'],
                'train_cohen_kappa': train_metrics['cohen_kappa'],
                'val_cohen_kappa': val_metrics['cohen_kappa'],
                'train_weighted_f1': train_metrics['weighted_f1'],
                'val_weighted_f1': val_metrics['weighted_f1'],
                **val_metrics,
            })
            history_frame = pd.DataFrame(history_rows)
            history_frame.to_csv(os.path.join(run_dir, 'history.csv'), index=False)
            save_training_curves(history_frame, os.path.join(run_dir, 'training_curves.png'))

            current = float(val_metrics['balanced_acc'])
            if current > best_metric:
                best_metric = current
                best_epoch = epoch
                wait = 0
                should_save = True
                _save_json(
                    os.path.join(run_dir, 'best_val_metrics.json'),
                    {'best_epoch': best_epoch, 'best_balanced_acc': best_metric},
                )
            else:
                wait += 1
                if wait >= args.patience:
                    should_stop = True

        save_flag = torch.tensor([1 if should_save else 0], device=accelerator.device, dtype=torch.int64)
        if accelerator.num_processes > 1 and dist.is_initialized():
            dist.broadcast(save_flag, src=0)
        if int(save_flag.item()) == 1:


            state_dict = accelerator.get_state_dict(model)
            if accelerator.is_main_process:
                accelerator.unwrap_model(model).save_pretrained(best_adapter_dir, state_dict=state_dict)
                processor.save_pretrained(best_adapter_dir)
        accelerator.wait_for_everyone()

        stop_flag = torch.tensor([1 if should_stop else 0], device=accelerator.device, dtype=torch.int64)
        if accelerator.num_processes > 1 and dist.is_initialized():
            dist.broadcast(stop_flag, src=0)
        accelerator.wait_for_everyone()
        if int(stop_flag.item()) == 1:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f'LoRA training finished -> {run_dir}')
        print(f'  Best epoch: {best_epoch}')
        print(f'  Best val balanced_acc: {best_metric:.4f}')
        print('  Test evaluation is not run automatically; use eval_lora when you are ready.')


def _run_eval_lora(args):
    accelerator = Accelerator(
        mixed_precision='bf16' if bool(args.use_bf16) else 'no',
        kwargs_handlers=[
            DistributedDataParallelKwargs(broadcast_buffers=False),
            InitProcessGroupKwargs(timeout=TUEV_DISTRIBUTED_TIMEOUT),
        ],
    )
    run_dir = str(Path(args.run_dir).resolve())
    args = _restore_training_vlm_args_if_needed(args, run_dir)
    adapter_dir = str(Path(args.adapter_dir or os.path.join(run_dir, 'best_adapter')).resolve())
    output_dir = str(Path(args.output_dir or os.path.join(run_dir, 'test_eval')).resolve())
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        _save_json(os.path.join(output_dir, 'args.json'), vars(args))
    accelerator.wait_for_everyone()

    prep_dir = args.prep_dir or os.path.join(run_dir, 'prepared')
    workspace = _ensure_workspace(args, prep_dir, create_if_missing=False)

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError('LoRA evaluation requires the `peft` package in the active environment.') from exc

    model, processor = load_vlm_for_training(
        model_path=args.model_path,
        use_bf16=bool(args.use_bf16),
        enable_flash_attention=bool(args.flash_attention_2),
    )
    _limit_processor_image_pixels(processor, args.processor_max_pixels)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    loaders = _build_eval_loaders(
        workspace,
        processor,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
    )
    model, val_loader, test_loader = accelerator.prepare(model, loaders['val'], loaders['test'])
    val_frame = collect_choice_predictions(
        model,
        processor,
        val_loader,
        choices=workspace.choices,
        class_names=workspace.class_names,
        device=None,
        max_new_tokens=args.max_new_tokens,
        save_generated_text=bool(args.save_generated_text),
        accelerator=accelerator,
        sequence_scoring=bool(args.sequence_scoring),
        partial_output_dir=str(Path(output_dir) / '.distributed_rows' / 'val'),
        progress_label='val',
    )
    test_frame = collect_choice_predictions(
        model,
        processor,
        test_loader,
        choices=workspace.choices,
        class_names=workspace.class_names,
        device=None,
        max_new_tokens=args.max_new_tokens,
        save_generated_text=bool(args.save_generated_text),
        accelerator=accelerator,
        sequence_scoring=bool(args.sequence_scoring),
        partial_output_dir=str(Path(output_dir) / '.distributed_rows' / 'test'),
        progress_label='test',
    )
    def finalize():
        metrics = finalize_multiclass_eval(val_frame, test_frame, output_dir, workspace.class_names)
        print(f'LoRA evaluation results -> {output_dir}')
        for key, value in metrics.items():
            print(f'  {key}: {value:.4f}')
        return metrics

    run_main_process_with_filesystem_sync(
        finalize,
        accelerator=accelerator,
        marker_dir=Path(output_dir) / '.distributed_rows',
        prefix='finalize_eval',
    )


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--result_dir', type=str, default=DEFAULT_TUEV_RESULT_ROOT)
    parser.add_argument('--render_cache_dir', type=str, default=DEFAULT_TUEV_RENDER_CACHE_ROOT)
    parser.add_argument('--run_dir', type=str, default='')
    parser.add_argument('--prep_dir', type=str, default='')
    parser.add_argument('--exp_tag', type=str, default='')
    add_vlm_model_args(parser)
    parser.add_argument('--prompt', type=str, default=DEFAULT_TUEV_PROMPT)

    parser.add_argument('--split_mode', type=str, default='cross-sub', choices=['cross-sub'])
    parser.add_argument('--official_val_fraction', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--window_size', type=float, default=5.0)
    parser.add_argument('--target_sfreq', type=int, default=200)
    parser.add_argument('--event_pre_context_sec', type=float, default=2.0)
    parser.add_argument('--event_post_context_sec', type=float, default=2.0)
    parser.add_argument('--event_merge_gap_sec', type=float, default=0.5)
    parser.add_argument('--bckg_exclusion_margin_sec', type=float, default=0.5)
    parser.add_argument('--bckg_sample_ratio', type=float, default=1.0)
    parser.add_argument('--max_bckg_per_split', type=int, default=0)
    parser.add_argument('--use_cache', type=int, default=1)
    parser.add_argument('--max_samples_per_split', type=int, default=0)
    parser.add_argument('--render_version', type=str, default='v1')
    parser.add_argument('--image_input_mode', type=str, default='single_panel',
                        choices=['single_panel', 'waveform_only', 'dual_image'])
    parser.add_argument('--single_image_kind', type=str, default='combined',
                        choices=['combined', 'waveform', 'spectrogram', 'cwt', 'combined_cwt'])
    parser.add_argument('--image_size', type=int, default=896)
    parser.add_argument('--processor_max_pixels', type=int, default=0,
                        help='Optional processor-side image pixel cap; 0 keeps the checkpoint default.')
    parser.add_argument('--normalization_mode', type=str, default='train_global',
                        choices=['train_global', 'sample_global_robust'])
    parser.add_argument('--spectrogram_freq_min', type=float, default=1.0)
    parser.add_argument('--spectrogram_freq_max', type=float, default=45.0)
    parser.add_argument('--spectrogram_nperseg', type=int, default=64)
    parser.add_argument('--spectrogram_noverlap', type=int, default=48)
    parser.add_argument('--spectrogram_nfft', type=int, default=256)
    parser.add_argument('--waveform_percentile', type=float, default=0.995)
    parser.add_argument('--waveform_stats_max_samples', type=int, default=2048)
    parser.add_argument('--spectrogram_stats_max_samples', type=int, default=512)
    parser.add_argument('--spectrogram_quantile_low', type=float, default=0.01)
    parser.add_argument('--spectrogram_quantile_high', type=float, default=0.99)
    parser.add_argument('--cwt_freq_min', type=float, default=1.0)
    parser.add_argument('--cwt_freq_max', type=float, default=45.0)
    parser.add_argument('--cwt_num_freqs', type=int, default=48)
    parser.add_argument('--cwt_freq_spacing', type=str, default='log', choices=['log', 'linear'])
    parser.add_argument('--cwt_morlet_w', type=float, default=6.0)
    parser.add_argument('--cwt_support', type=float, default=6.0)
    parser.add_argument('--cwt_time_stride', type=int, default=16)
    parser.add_argument('--cwt_stats_max_samples', type=int, default=16)
    parser.add_argument('--cwt_quantile_low', type=float, default=0.01)
    parser.add_argument('--cwt_quantile_high', type=float, default=0.99)

    parser.add_argument('--use_bf16', type=int, default=1)
    parser.add_argument('--flash_attention_2', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--device_map', type=str, default='auto')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=4)
    parser.add_argument('--save_generated_text', type=int, default=0)
    parser.add_argument('--sequence_scoring', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)


def parse_args():
    parser = argparse.ArgumentParser(description='TUEV event-level VLM pipeline')
    subparsers = parser.add_subparsers(dest='command', required=True)

    prepare_parser = subparsers.add_parser('prepare', help='Prepare TUEV image manifests and render statistics')
    _add_shared_args(prepare_parser)

    zero_parser = subparsers.add_parser('zero_shot', help='Run zero-shot TUEV scoring')
    _add_shared_args(zero_parser)

    train_parser = subparsers.add_parser('train_lora', help='Train LoRA adapters with accelerate')
    _add_shared_args(train_parser)
    train_parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    train_parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    train_parser.add_argument('--max_epochs', type=int, default=8)
    train_parser.add_argument('--patience', type=int, default=2)
    train_parser.add_argument('--lr', type=float, default=2e-5)
    train_parser.add_argument('--weight_decay', type=float, default=0.0)
    train_parser.add_argument('--warmup_ratio', type=float, default=0.03)
    train_parser.add_argument('--max_grad_norm', type=float, default=1.0)
    train_parser.add_argument('--lora_rank', type=int, default=8)
    train_parser.add_argument('--lora_alpha', type=int, default=16)
    train_parser.add_argument('--lora_dropout', type=float, default=0.1)
    train_parser.add_argument('--lora_target_modules', type=str, default='auto',
                              help='Comma-separated LoRA target module names, regex:<pattern>, or auto.')
    train_parser.add_argument('--freeze_vision_tower', type=int, default=1)
    train_parser.add_argument('--gradient_checkpointing', type=int, default=1)
    train_parser.add_argument('--balanced_sampling', type=int, default=0)

    eval_parser = subparsers.add_parser('eval_lora', help='Evaluate a trained LoRA adapter with distributed inference')
    _add_shared_args(eval_parser)
    eval_parser.add_argument('--adapter_dir', type=str, default='')
    eval_parser.add_argument('--output_dir', type=str, default='')

    return finalize_vlm_args(parser.parse_args())


def main():
    args = parse_args()
    args.result_dir = resolve_output_dir(args.result_dir or DEFAULT_TUEV_RESULT_ROOT)
    args.render_cache_dir = resolve_output_dir(args.render_cache_dir or DEFAULT_TUEV_RENDER_CACHE_ROOT)
    if args.command == 'prepare':
        _run_prepare(args)
    elif args.command == 'zero_shot':
        _run_zero_shot(args)
    elif args.command == 'train_lora':
        _run_train_lora(args)
    elif args.command == 'eval_lora':
        if not args.run_dir:
            raise ValueError('eval_lora requires --run_dir to point to a LoRA training directory.')
        _run_eval_lora(args)
    else:
        raise ValueError(f'Unknown command: {args.command}')


if __name__ == '__main__':
    main()
