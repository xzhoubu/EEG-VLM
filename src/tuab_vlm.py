

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from metric import classification_metrics
from runtime import build_experiment_name, resolve_output_dir
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
from vlm.tuab_data import (
    DEFAULT_TUAB_MODEL_PATH,
    DEFAULT_TUAB_PROMPT,
    DEFAULT_TUAB_RENDER_CACHE_ROOT,
    DEFAULT_TUAB_RESULT_ROOT,
    TUABVLMDataset,
    build_tuab_workspace_from_manifests,
    prepare_tuab_workspace,
)


def _resolve_run_dir(result_root: str, mode_slug: str, args, timestamp: str = '') -> str:
    name = build_experiment_name(
        dataset='tuab',
        task_id=1,
        task_label='abnormal_binary',
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
    return token_ids


def _distributed_average(total: float, count: int, accelerator: Accelerator) -> float:
    payload = torch.tensor([float(total), float(count)], device=accelerator.device, dtype=torch.float64)
    if accelerator.num_processes > 1 and dist.is_initialized():
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    return float(payload[0].item() / max(payload[1].item(), 1.0))


def _gather_metric_rows(local_rows: list[dict], accelerator: Accelerator) -> list[dict]:
    gathered = gather_object(local_rows)
    if not accelerator.is_main_process:
        return []
    flat_rows: list[dict] = []
    for chunk in gathered:
        if isinstance(chunk, list):
            flat_rows.extend(chunk)
        elif chunk:
            flat_rows.append(chunk)
    return flat_rows


def _ensure_workspace(args, prep_dir: str, create_if_missing: bool = True):
    prep_path = Path(prep_dir)
    if _prep_complete(prep_path):
        return build_tuab_workspace_from_manifests(args, str(prep_path))
    if not create_if_missing:
        raise FileNotFoundError(f'Prepared workspace not found: {prep_dir}')
    return prepare_tuab_workspace(args, str(prep_path))


def _prep_complete(prep_path: Path) -> bool:
    return (
        prep_path.is_dir()
        and (prep_path / 'prepare_args.json').is_file()
        and (prep_path / 'manifest_train.csv').is_file()
        and (prep_path / 'manifest_val.csv').is_file()
        and (prep_path / 'manifest_test.csv').is_file()
        and (prep_path / 'render_config.json').is_file()
        and (prep_path / 'render_stats.json').is_file()
    )


def _make_dataset(workspace, split_name: str):
    return TUABVLMDataset(
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
    result_root = resolve_output_dir(args.result_dir or DEFAULT_TUAB_RESULT_ROOT)
    prep_dir = args.prep_dir or _resolve_run_dir(result_root, 'prepare', args)
    workspace = prepare_tuab_workspace(args, prep_dir)
    _save_json(os.path.join(prep_dir, 'args.json'), vars(args))
    print(f'Prepared workspace -> {prep_dir}')
    for split_name, split in workspace.splits.items():
        counts = split.manifest['label'].value_counts().sort_index().to_dict()
        print(f'  {split_name}: {len(split.manifest)} samples | counts={counts}')


def _collect_eval_frames(model, processor, loaders, workspace, device, args, accelerator=None, progress_dir=None):
    progress_interval = int(getattr(args, 'eval_progress_interval', 0))
    rank = int(accelerator.process_index) if accelerator is not None else 0
    progress_root = Path(progress_dir) if progress_dir else None
    val_frame = collect_choice_predictions(
        model,
        processor,
        loaders['val'],
        choices=workspace.choices,
        class_names=workspace.class_names,
        device=device,
        max_new_tokens=args.max_new_tokens,
        save_generated_text=bool(args.save_generated_text),
        accelerator=accelerator,
        sequence_scoring=bool(args.sequence_scoring),
        progress_interval=progress_interval,
        progress_label='val',
        progress_status_path=str(progress_root / f'val_rank{rank}.status.json') if progress_root else None,
        partial_output_dir=str(progress_root / 'partial') if progress_root else None,
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
        accelerator=accelerator,
        sequence_scoring=bool(args.sequence_scoring),
        progress_interval=progress_interval,
        progress_label='test',
        progress_status_path=str(progress_root / f'test_rank{rank}.status.json') if progress_root else None,
        partial_output_dir=str(progress_root / 'partial') if progress_root else None,
    )
    return val_frame, test_frame


def _run_zero_shot(args):
    accelerator = Accelerator(mixed_precision='bf16' if bool(args.use_bf16) else 'no')
    result_root = resolve_output_dir(args.result_dir or DEFAULT_TUAB_RESULT_ROOT)
    run_dir = args.run_dir or _resolve_run_dir(result_root, 'zero-shot', args)
    if accelerator.is_main_process:
        os.makedirs(run_dir, exist_ok=True)
        setattr(args, 'distributed_processes', int(accelerator.num_processes))
        setattr(
            args,
            'effective_eval_batch_size',
            int(args.per_device_eval_batch_size) * int(accelerator.num_processes),
        )
        _save_json(os.path.join(run_dir, 'args.json'), vars(args))

    prep_dir = args.prep_dir or os.path.join(run_dir, 'prepared')
    if accelerator.is_main_process and not _prep_complete(Path(prep_dir)):
        prepare_tuab_workspace(args, prep_dir)
    accelerator.wait_for_everyone()
    workspace = build_tuab_workspace_from_manifests(args, prep_dir)

    distributed = accelerator.num_processes > 1
    model, processor, device = load_vlm_for_inference(
        model_path=args.model_path,
        use_bf16=bool(args.use_bf16),
        device_map='none' if distributed else args.device_map,
        gpu_id=-1 if distributed else args.gpu_id,
        enable_flash_attention=bool(args.flash_attention_2),
    )
    if distributed:
        model = model.to(accelerator.device)
        device = accelerator.device
    model.eval()
    loaders = _build_eval_loaders(
        workspace,
        processor,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
    )
    val_loader, test_loader = accelerator.prepare(loaders['val'], loaders['test'])
    if accelerator.is_main_process:
        print(
            'Zero-shot evaluation: '
            f'{accelerator.num_processes} process(es), '
            f'per-device batch={args.per_device_eval_batch_size}, '
            f'effective batch={args.per_device_eval_batch_size * accelerator.num_processes}, '
            f'val={len(workspace.splits["val"].manifest)}, '
            f'test={len(workspace.splits["test"].manifest)}',
            flush=True,
        )
    val_frame, test_frame = _collect_eval_frames(
        model,
        processor,
        {'val': val_loader, 'test': test_loader},
        workspace,
        device,
        args,
        accelerator=accelerator,
        progress_dir=os.path.join(run_dir, 'progress'),
    )
    if accelerator.is_main_process:
        metrics = finalize_multiclass_eval(val_frame, test_frame, run_dir, workspace.class_names)
        print(f'Zero-shot results -> {run_dir}')
        for key, value in metrics.items():
            print(f'  {key}: {value:.4f}')
def _run_train_lora(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision='bf16' if bool(args.use_bf16) else 'no',
        kwargs_handlers=[ddp_kwargs],
    )
    result_root = resolve_output_dir(args.result_dir or DEFAULT_TUAB_RESULT_ROOT)

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
    if accelerator.is_main_process and not _prep_complete(Path(prep_dir)):
        prepare_tuab_workspace(args, prep_dir)
    accelerator.wait_for_everyone()
    workspace = build_tuab_workspace_from_manifests(args, prep_dir)

    model, processor = load_vlm_for_training(
        model_path=args.model_path,
        use_bf16=bool(args.use_bf16),
        enable_flash_attention=bool(args.flash_attention_2),
    )
    model = apply_vlm_lora_adapters(model, args)

    train_dataset = _make_dataset(workspace, 'train')
    val_dataset = _make_dataset(workspace, 'val')

    train_collator = VLMTrainCollator(
        processor,
        prompt=workspace.prompt,
        include_label_metadata=True,
    )
    val_loss_collator = VLMTrainCollator(processor, prompt=workspace.prompt)
    eval_collator = VLMInferenceCollator(processor, prompt=workspace.prompt)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
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
        train_metric_rows = _gather_metric_rows(train_metric_rows, accelerator)

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
        )
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
                'balanced_acc': val_metrics['balanced_acc'],
            })
            history_frame = pd.DataFrame(history_rows)
            history_frame.to_csv(os.path.join(run_dir, 'history.csv'), index=False)
            save_training_curves(history_frame, os.path.join(run_dir, 'training_curves.png'))

            current = float(val_metrics['balanced_acc'])
            if current > best_metric:
                best_metric = current
                best_epoch = epoch
                wait = 0
                accelerator.unwrap_model(model).save_pretrained(best_adapter_dir)
                processor.save_pretrained(best_adapter_dir)
                _save_json(
                    os.path.join(run_dir, 'best_val_metrics.json'),
                    {'best_epoch': best_epoch, 'best_balanced_acc': best_metric},
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
        print(f'  Best val balanced_acc: {best_metric:.4f}')
        print('  Test evaluation is not run automatically; use eval_lora when you are ready.')


def _run_eval_lora(args):
    accelerator = Accelerator(mixed_precision='bf16' if bool(args.use_bf16) else 'no')
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
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    loaders = _build_eval_loaders(
        workspace,
        processor,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
    )
    model, val_loader, test_loader = accelerator.prepare(model, loaders['val'], loaders['test'])
    val_frame, test_frame = _collect_eval_frames(
        model,
        processor,
        {'val': val_loader, 'test': test_loader},
        workspace,
        None,
        args,
        accelerator=accelerator,
        progress_dir=os.path.join(output_dir, 'progress'),
    )
    if accelerator.is_main_process:
        metrics = finalize_multiclass_eval(val_frame, test_frame, output_dir, workspace.class_names)
        print(f'LoRA evaluation results -> {output_dir}')
        for key, value in metrics.items():
            print(f'  {key}: {value:.4f}')
    accelerator.wait_for_everyone()


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--result_dir', type=str, default=DEFAULT_TUAB_RESULT_ROOT)
    parser.add_argument('--render_cache_dir', type=str, default=DEFAULT_TUAB_RENDER_CACHE_ROOT)
    parser.add_argument('--run_dir', type=str, default='')
    parser.add_argument('--prep_dir', type=str, default='')
    parser.add_argument('--exp_tag', type=str, default='')
    add_vlm_model_args(parser)
    parser.add_argument('--prompt', type=str, default=DEFAULT_TUAB_PROMPT)

    parser.add_argument(
        '--split_mode',
        type=str,
        default='cross-sub',
        choices=['cross-sub'],
    )
    parser.add_argument('--official_val_fraction', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--window_size', type=float, default=20.0)
    parser.add_argument('--overlap_ratio', type=float, default=0.0)
    parser.add_argument('--use_cache', type=int, default=1)
    parser.add_argument('--max_windows_per_file', type=int, default=0)
    parser.add_argument('--max_samples_per_split', type=int, default=0)
    parser.add_argument('--tuab_tokenize', type=int, default=1)
    parser.add_argument('--tuab_token_sec', type=float, default=1.0)
    parser.add_argument('--tuab_token_stride_sec', type=float, default=1.0)

    parser.add_argument('--render_version', type=str, default='v1')
    parser.add_argument('--image_input_mode', type=str, default='single_panel',
                        choices=['single_panel', 'waveform_only', 'dual_image'])
    parser.add_argument('--single_image_kind', type=str, default='combined',
                        choices=['combined', 'waveform', 'spectrogram', 'cwt', 'combined_cwt'])
    parser.add_argument('--image_size', type=int, default=896)
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
    parser.add_argument('--save_generated_text', type=int, default=1)
    parser.add_argument('--sequence_scoring', type=int, default=1)
    parser.add_argument('--eval_progress_interval', type=int, default=250)
    parser.add_argument('--num_workers', type=int, default=0)


def parse_args():
    parser = argparse.ArgumentParser(description='TUAB task-1 VLM pipeline')
    subparsers = parser.add_subparsers(dest='command', required=True)

    prepare_parser = subparsers.add_parser('prepare', help='Prepare manifests and render stats')
    _add_shared_args(prepare_parser)

    zero_parser = subparsers.add_parser('zero_shot', help='Run zero-shot normal/abnormal scoring')
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
    train_parser.add_argument('--lora_target_modules', type=str, default='auto',
                              help='Comma-separated LoRA target module names, regex:<pattern>, or auto.')
    train_parser.add_argument('--freeze_vision_tower', type=int, default=1)
    train_parser.add_argument('--gradient_checkpointing', type=int, default=1)

    eval_parser = subparsers.add_parser('eval_lora', help='Evaluate a trained LoRA adapter')
    _add_shared_args(eval_parser)
    eval_parser.add_argument('--adapter_dir', type=str, default='')
    eval_parser.add_argument('--output_dir', type=str, default='')

    return finalize_vlm_args(parser.parse_args())


def main():
    args = parse_args()
    args.result_dir = resolve_output_dir(args.result_dir or DEFAULT_TUAB_RESULT_ROOT)
    args.render_cache_dir = resolve_output_dir(args.render_cache_dir or DEFAULT_TUAB_RENDER_CACHE_ROOT)
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
