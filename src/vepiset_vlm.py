

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from metric import binary_paper_metrics
from runtime import build_experiment_name, resolve_output_dir
from vlm.data import (
    DEFAULT_MODEL_PATH,
    DEFAULT_QWEN_PROMPT,
    DEFAULT_RENDER_CACHE_ROOT,
    DEFAULT_RESULT_ROOT,
    DistributedWeightedSampler,
    VEPiSetVLMDataset,
    build_workspace_from_manifests,
    prepare_workspace,
)
from vlm.eval import finalize_binary_eval, save_training_curves
from vlm.model_config import add_qwen3vl_model_args, finalize_qwen3vl_args, qwen3vl_model_slug
from vlm.qwen3vl import (
    Qwen3VLInferenceCollator,
    Qwen3VLTrainCollator,
    apply_lora_adapters,
    collect_predictions,
    load_qwen3vl_for_inference,
    load_qwen3vl_for_training,
)


def _resolve_run_dir(result_root: str, mode_slug: str, args, timestamp: str = '') -> str:
    name = build_experiment_name(
        dataset='vepiset',
        task_id=1,
        task_label='ied_binary',
        model=f'{qwen3vl_model_slug(args)}-{mode_slug}',
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


def _ensure_workspace(args, prep_dir: str, create_if_missing: bool = True):
    prep_path = Path(prep_dir)
    if prep_path.is_dir() and (prep_path / 'manifest_train.csv').is_file():
        return build_workspace_from_manifests(args, str(prep_path))
    if not create_if_missing:
        raise FileNotFoundError(f'Prepared workspace not found: {prep_dir}')
    return prepare_workspace(args, str(prep_path))


def _build_eval_loaders(workspace, processor, prompt: str, batch_size: int, num_workers: int):
    collator = Qwen3VLInferenceCollator(processor, prompt=prompt)
    loaders = {}
    for split_name in ['val', 'test']:
        dataset = VEPiSetVLMDataset(
            split=workspace.splits[split_name],
            channel_names=workspace.channel_names,
            sfreq=workspace.sfreq,
            duration_sec=workspace.duration_sec,
            render_config=workspace.render_config,
            render_stats=workspace.render_stats,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def _run_prepare(args):
    result_root = resolve_output_dir(args.result_dir or DEFAULT_RESULT_ROOT)
    prep_dir = args.prep_dir or _resolve_run_dir(result_root, 'prepare', args)
    workspace = prepare_workspace(args, prep_dir)
    _save_json(os.path.join(prep_dir, 'args.json'), vars(args))
    print(f'Prepared workspace -> {prep_dir}')
    for split_name, split in workspace.splits.items():
        print(f'  {split_name}: {len(split.manifest)} samples')


def _run_zero_shot(args):
    result_root = resolve_output_dir(args.result_dir or DEFAULT_RESULT_ROOT)
    run_dir = args.run_dir or _resolve_run_dir(result_root, 'zero-shot', args)
    os.makedirs(run_dir, exist_ok=True)
    prep_dir = args.prep_dir or os.path.join(run_dir, 'prepared')
    workspace = _ensure_workspace(args, prep_dir, create_if_missing=True)
    _save_json(os.path.join(run_dir, 'args.json'), vars(args))

    model, processor, device = load_qwen3vl_for_inference(
        model_path=args.model_path,
        use_bf16=bool(args.use_bf16),
        device_map=args.device_map,
        gpu_id=args.gpu_id,
        enable_flash_attention=bool(args.flash_attention_2),
    )
    loaders = _build_eval_loaders(
        workspace,
        processor,
        prompt=workspace.prompt,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
    )
    val_frame = collect_predictions(
        model,
        processor,
        loaders['val'],
        device=device,
        max_new_tokens=args.max_new_tokens,
        save_generated_text=bool(args.save_generated_text),
    )
    test_frame = collect_predictions(
        model,
        processor,
        loaders['test'],
        device=device,
        max_new_tokens=args.max_new_tokens,
        save_generated_text=bool(args.save_generated_text),
    )
    metrics = finalize_binary_eval(val_frame, test_frame, run_dir)
    print(f'Zero-shot results -> {run_dir}')
    for key, value in metrics.items():
        print(f'  {key}: {value:.4f}')


def _train_sampler_from_manifest(manifest: pd.DataFrame, num_replicas: int, rank: int, seed: int):
    labels = manifest['label'].to_numpy(dtype=np.int64, copy=False)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = np.asarray([1.0 / counts[label] for label in labels], dtype=np.float64)
    return DistributedWeightedSampler(weights=weights, num_replicas=num_replicas, rank=rank, seed=seed)


def _run_train_lora(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision='bf16' if bool(args.use_bf16) else 'no',
    )
    result_root = resolve_output_dir(args.result_dir or DEFAULT_RESULT_ROOT)

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
    if accelerator.is_main_process and not Path(prep_dir).is_dir():
        prepare_workspace(args, prep_dir)
    accelerator.wait_for_everyone()
    workspace = build_workspace_from_manifests(args, prep_dir)

    model, processor = load_qwen3vl_for_training(
        model_path=args.model_path,
        use_bf16=bool(args.use_bf16),
        enable_flash_attention=bool(args.flash_attention_2),
    )
    model = apply_lora_adapters(model, args)

    train_dataset = VEPiSetVLMDataset(
        split=workspace.splits['train'],
        channel_names=workspace.channel_names,
        sfreq=workspace.sfreq,
        duration_sec=workspace.duration_sec,
        render_config=workspace.render_config,
        render_stats=workspace.render_stats,
    )
    val_dataset = VEPiSetVLMDataset(
        split=workspace.splits['val'],
        channel_names=workspace.channel_names,
        sfreq=workspace.sfreq,
        duration_sec=workspace.duration_sec,
        render_config=workspace.render_config,
        render_stats=workspace.render_stats,
    )

    train_collator = Qwen3VLTrainCollator(processor, prompt=workspace.prompt)
    eval_collator = Qwen3VLInferenceCollator(processor, prompt=workspace.prompt)
    train_sampler = _train_sampler_from_manifest(
        workspace.splits['train'].manifest,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=train_collator,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=eval_collator,
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

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
    )

    best_metric = -float('inf')
    best_epoch = 0
    wait = 0
    history_rows = []
    best_adapter_dir = os.path.join(run_dir, 'best_adapter')

    for epoch in range(1, args.max_epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss_total = 0.0
        train_loss_steps = 0
        for batch in train_loader:
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

        val_frame = collect_predictions(
            model,
            processor,
            val_loader,
            device=None,
            max_new_tokens=args.max_new_tokens,
            save_generated_text=False,
            accelerator=accelerator,
        )
        should_stop = False
        if accelerator.is_main_process:
            val_true = val_frame['ground_truth'].to_numpy(dtype=np.int64, copy=False)
            val_score = val_frame['prob_positive'].to_numpy(dtype=np.float64, copy=False)
            val_metrics = binary_paper_metrics(val_true, val_score)
            epoch_loss = train_loss_total / max(train_loss_steps, 1)
            history_rows.append({
                'epoch': epoch,
                'train_loss': epoch_loss,
                **val_metrics,
            })
            history_frame = pd.DataFrame(history_rows)
            history_frame.to_csv(os.path.join(run_dir, 'history.csv'), index=False)
            save_training_curves(history_frame, os.path.join(run_dir, 'training_curves.png'))

            current = float(val_metrics['auprc'])
            if current > best_metric:
                best_metric = current
                best_epoch = epoch
                wait = 0
                accelerator.unwrap_model(model).save_pretrained(best_adapter_dir)
                processor.save_pretrained(best_adapter_dir)
                _save_json(
                    os.path.join(run_dir, 'best_val_metrics.json'),
                    {'best_epoch': best_epoch, 'best_auprc': best_metric},
                )
            else:
                wait += 1
                if wait >= args.patience:
                    should_stop = True

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
        print(f'  Best val AUPRC: {best_metric:.4f}')
        print('  Test evaluation is not run automatically; use eval_lora when you are ready.')


def _run_eval_lora(args):
    run_dir = str(Path(args.run_dir).resolve())
    adapter_dir = str(Path(args.adapter_dir or os.path.join(run_dir, 'best_adapter')).resolve())
    output_dir = str(Path(args.output_dir or os.path.join(run_dir, 'test_eval')).resolve())
    os.makedirs(output_dir, exist_ok=True)
    _save_json(os.path.join(output_dir, 'args.json'), vars(args))

    prep_dir = args.prep_dir or os.path.join(run_dir, 'prepared')
    workspace = _ensure_workspace(args, prep_dir, create_if_missing=False)

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError('LoRA evaluation requires the `peft` package in the active environment.') from exc

    model, processor, device = load_qwen3vl_for_inference(
        model_path=args.model_path,
        use_bf16=bool(args.use_bf16),
        device_map=args.device_map,
        gpu_id=args.gpu_id,
        enable_flash_attention=bool(args.flash_attention_2),
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    loaders = _build_eval_loaders(
        workspace,
        processor,
        prompt=workspace.prompt,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
    )
    val_frame = collect_predictions(
        model,
        processor,
        loaders['val'],
        device=device,
        max_new_tokens=args.max_new_tokens,
        save_generated_text=bool(args.save_generated_text),
    )
    test_frame = collect_predictions(
        model,
        processor,
        loaders['test'],
        device=device,
        max_new_tokens=args.max_new_tokens,
        save_generated_text=bool(args.save_generated_text),
    )
    metrics = finalize_binary_eval(val_frame, test_frame, output_dir)
    print(f'LoRA evaluation results -> {output_dir}')
    for key, value in metrics.items():
        print(f'  {key}: {value:.4f}')


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--result_dir', type=str, default=DEFAULT_RESULT_ROOT)
    parser.add_argument('--render_cache_dir', type=str, default=DEFAULT_RENDER_CACHE_ROOT)
    parser.add_argument('--run_dir', type=str, default='')
    parser.add_argument('--prep_dir', type=str, default='')
    parser.add_argument('--exp_tag', type=str, default='')
    add_qwen3vl_model_args(parser)
    parser.add_argument('--prompt', type=str, default=DEFAULT_QWEN_PROMPT)

    parser.add_argument('--split_mode', type=str, default='cross-sub', choices=['cross-sub'])
    parser.add_argument('--train_subs', type=str, default='auto')
    parser.add_argument('--val_subs', type=str, default='auto')
    parser.add_argument('--test_subs', type=str, default='auto')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--window_size', type=float, default=1.0)
    parser.add_argument('--overlap_ratio', type=float, default=0.0)
    parser.add_argument('--use_cache', type=int, default=1)
    parser.add_argument('--max_samples_per_split', type=int, default=0)

    parser.add_argument('--render_version', type=str, default='v1')
    parser.add_argument(
        '--single_image_kind',
        type=str,
        default='combined',
        choices=['waveform', 'spectrogram', 'cwt', 'combined', 'combined_cwt'],
    )
    parser.add_argument('--image_size', type=int, default=896)
    parser.add_argument('--spectrogram_freq_min', type=float, default=1.0)
    parser.add_argument('--spectrogram_freq_max', type=float, default=50.0)
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
    parser.add_argument('--save_generated_text', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)


def parse_args():
    parser = argparse.ArgumentParser(description='vEpiSet VLM pipeline for Qwen3-VL')
    subparsers = parser.add_subparsers(dest='command', required=True)

    prepare_parser = subparsers.add_parser('prepare', help='Prepare manifests and render stats')
    _add_shared_args(prepare_parser)

    zero_parser = subparsers.add_parser('zero_shot', help='Run zero-shot yes/no scoring')
    _add_shared_args(zero_parser)

    train_parser = subparsers.add_parser('train_lora', help='Train LoRA adapters with accelerate')
    _add_shared_args(train_parser)
    train_parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    train_parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    train_parser.add_argument('--max_epochs', type=int, default=3)
    train_parser.add_argument('--patience', type=int, default=2)
    train_parser.add_argument('--lr', type=float, default=1e-4)
    train_parser.add_argument('--weight_decay', type=float, default=0.0)
    train_parser.add_argument('--warmup_ratio', type=float, default=0.03)
    train_parser.add_argument('--max_grad_norm', type=float, default=1.0)
    train_parser.add_argument('--lora_rank', type=int, default=8)
    train_parser.add_argument('--lora_alpha', type=int, default=16)
    train_parser.add_argument('--lora_dropout', type=float, default=0.1)
    train_parser.add_argument('--freeze_vision_tower', type=int, default=1)
    train_parser.add_argument('--gradient_checkpointing', type=int, default=1)

    eval_parser = subparsers.add_parser('eval_lora', help='Evaluate a trained LoRA adapter')
    _add_shared_args(eval_parser)
    eval_parser.add_argument('--adapter_dir', type=str, default='')
    eval_parser.add_argument('--output_dir', type=str, default='')

    return finalize_qwen3vl_args(parser.parse_args())


def main():
    args = parse_args()
    args.result_dir = resolve_output_dir(args.result_dir or DEFAULT_RESULT_ROOT)
    args.render_cache_dir = resolve_output_dir(args.render_cache_dir or DEFAULT_RENDER_CACHE_ROOT)
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
