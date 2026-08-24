

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch import nn
from torch.utils.data import DataLoader

from metric import classification_metrics
from runtime import build_experiment_name, resolve_output_dir
from vlm.classifier import FrozenQwen3VLClassifier
from vlm.data import (
    DEFAULT_MODEL_PATH,
    DEFAULT_QWEN_PROMPT,
    DEFAULT_RENDER_CACHE_ROOT,
    DEFAULT_RESULT_ROOT,
    VEPiSetVLMDataset,
    build_workspace_from_manifests,
    prepare_workspace,
)
from vlm.eval import finalize_binary_eval, save_training_curves
from vlm.model_config import add_qwen3vl_model_args, finalize_qwen3vl_args, qwen3vl_model_slug
from vlm.qwen3vl import Qwen3VLInferenceCollator, load_qwen3vl_for_inference, load_qwen3vl_for_training


CLASS_NAMES = ['Non-IED', 'IED']


def _slug(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')


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


def _gather_rows(local_rows: list[dict], accelerator: Accelerator | None = None) -> list[dict]:
    if accelerator is None:
        return local_rows
    gathered = gather_object(local_rows)
    if not accelerator.is_main_process:
        return []
    rows: list[dict] = []
    for chunk in gathered:
        if isinstance(chunk, list):
            rows.extend(chunk)
        elif chunk:
            rows.append(chunk)
    return rows


def _save_json(path: str, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _move_model_inputs(model_inputs: dict[str, torch.Tensor], device: torch.device | None) -> dict[str, torch.Tensor]:
    if device is None:
        return model_inputs
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in model_inputs.items()
    }


def _distributed_average(total: float, count: int, accelerator: Accelerator) -> float:
    payload = torch.tensor([float(total), float(count)], device=accelerator.device, dtype=torch.float64)
    if accelerator.num_processes > 1 and dist.is_initialized():
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    return float(payload[0].item() / max(payload[1].item(), 1.0))


def _ensure_workspace(args, prep_dir: str, create_if_missing: bool = True):
    prep_path = Path(prep_dir)
    if prep_path.is_dir() and (prep_path / 'manifest_train.csv').is_file():
        return build_workspace_from_manifests(args, str(prep_path))
    if not create_if_missing:
        raise FileNotFoundError(f'Prepared workspace not found: {prep_dir}')
    return prepare_workspace(args, str(prep_path))


def _load_optional_adapter(backbone, args):
    adapter_dir = str(getattr(args, 'adapter_dir', '') or '').strip()
    if not adapter_dir:
        return backbone
    adapter_path = Path(adapter_dir).resolve()
    if not adapter_path.is_dir():
        raise FileNotFoundError(f'Adapter directory not found: {adapter_path}')
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError('Loading a LoRA adapter requires the `peft` package in the active environment.') from exc
    model = PeftModel.from_pretrained(backbone, str(adapter_path))
    model.eval()
    return model


def _build_dataloaders(workspace,
                       processor,
                       batch_size: int,
                       num_workers: int):
    collator = Qwen3VLInferenceCollator(processor, prompt=workspace.prompt)
    datasets = {
        split_name: VEPiSetVLMDataset(
            split=workspace.splits[split_name],
            channel_names=workspace.channel_names,
            sfreq=workspace.sfreq,
            duration_sec=workspace.duration_sec,
            render_config=workspace.render_config,
            render_stats=workspace.render_stats,
        )
        for split_name in ['train', 'val', 'test']
    }
    train_loader = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loaders = {
        split_name: DataLoader(
            datasets[split_name],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
        for split_name in ['train', 'val', 'test']
    }
    return train_loader, eval_loaders


def _collect_cls_predictions(model,
                             dataloader,
                             device: torch.device | None = None,
                             accelerator: Accelerator | None = None) -> pd.DataFrame:
    local_rows = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**_move_model_inputs(batch['model_inputs'], device))
            probs = outputs['class_probs'].detach().cpu().numpy()
            pred = np.argmax(probs, axis=1)
            for idx, sample_id in enumerate(batch['sample_id']):
                row = {
                    'sample_id': str(sample_id),
                    'split': str(batch['split'][idx]),
                    'subject': str(batch['subject'][idx]),
                    'condition': int(batch['condition'][idx]) if not torch.is_tensor(batch['condition'][idx]) else int(batch['condition'][idx].item()),
                    'sample_file': str(batch['sample_file'][idx]),
                    'image_path': str(batch['image_path'][idx]),
                    'ground_truth': int(batch['ground_truth'][idx]) if not torch.is_tensor(batch['ground_truth'][idx]) else int(batch['ground_truth'][idx].item()),
                    'prediction': int(pred[idx]),
                    'predicted_class': CLASS_NAMES[int(pred[idx])],
                    'prob_positive': float(probs[idx, 1]),
                }
                for class_idx, class_name in enumerate(CLASS_NAMES):
                    row[f'prob_{_slug(class_name)}'] = float(probs[idx, class_idx])
                local_rows.append(row)
    rows = _gather_rows(local_rows, accelerator=accelerator)
    return pd.DataFrame(rows)


def _save_classifier_checkpoint(run_dir: str,
                                model: FrozenQwen3VLClassifier,
                                processor,
                                args,
                                best_epoch: int,
                                best_metric: float) -> None:
    ckpt_dir = Path(run_dir) / 'best_head'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'classifier_state_dict': {
                'norm': model.norm.state_dict(),
                'classifier': model.classifier.state_dict(),
            },
            'head_dropout': float(args.head_dropout),
            'class_names': list(CLASS_NAMES),
            'prompt': str(getattr(args, 'prompt', DEFAULT_QWEN_PROMPT)),
            'pooling_mode': str(model.pooling_mode),
            'mlp_hidden_size': int(model.mlp_hidden_size),
            'task_mode': str(model.task_mode),
            'qwen_model_size': str(getattr(args, 'qwen_model_size', '4b')),
            'qwen_model_slug': str(getattr(args, 'qwen_model_slug', 'qwen3-vl-4b-instruct')),
            'adapter_dir': str(getattr(args, 'adapter_dir', '') or ''),
            'best_epoch': int(best_epoch),
            'best_val_balanced_acc': float(best_metric),
        },
        ckpt_dir / 'classifier.pt',
    )
    processor.save_pretrained(ckpt_dir)


def _load_classifier_checkpoint(model: FrozenQwen3VLClassifier, checkpoint_path: str, payload: dict | None = None) -> dict:
    if payload is None:
        payload = torch.load(checkpoint_path, map_location='cpu')
    model.norm.load_state_dict(payload['classifier_state_dict']['norm'])
    model.classifier.load_state_dict(payload['classifier_state_dict']['classifier'])
    return payload


def _run_prepare(args):
    result_root = resolve_output_dir(args.result_dir or DEFAULT_RESULT_ROOT)
    prep_dir = args.prep_dir or _resolve_run_dir(result_root, 'cls-head-prepare', args)
    workspace = prepare_workspace(args, prep_dir)
    _save_json(os.path.join(prep_dir, 'args.json'), vars(args))
    print(f'Prepared workspace -> {prep_dir}')
    for split_name, split in workspace.splits.items():
        counts = split.manifest['label'].value_counts().sort_index().to_dict()
        named_counts = {CLASS_NAMES[int(key)]: int(value) for key, value in counts.items()}
        print(f'  {split_name}: {len(split.manifest)} samples | counts={named_counts}')


def _run_train_head(args):
    accelerator = Accelerator(mixed_precision='bf16' if bool(args.use_bf16) else 'no')
    result_root = resolve_output_dir(args.result_dir or DEFAULT_RESULT_ROOT)

    timestamp = None
    if accelerator.is_main_process:
        run_dir = str(Path(args.run_dir).resolve()) if args.run_dir else _resolve_run_dir(result_root, 'cls-head', args)
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

    backbone, processor = load_qwen3vl_for_training(
        model_path=args.model_path,
        use_bf16=bool(args.use_bf16),
        enable_flash_attention=bool(args.flash_attention_2),
    )
    hidden_size = int(backbone.model.language_model.config.hidden_size)
    backbone = _load_optional_adapter(backbone, args)
    model = FrozenQwen3VLClassifier(
        backbone=backbone,
        hidden_size=hidden_size,
        num_classes=len(CLASS_NAMES),
        dropout=args.head_dropout,
        pooling_mode=args.pooling_mode,
        mlp_hidden_size=args.mlp_hidden_size,
        task_mode='flat',
    )

    train_loader, eval_loaders = _build_dataloaders(
        workspace,
        processor,
        batch_size=args.per_device_batch_size,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    eval_loaders = {name: accelerator.prepare(loader) for name, loader in eval_loaders.items()}

    best_metric = -float('inf')
    best_epoch = 0
    wait = 0
    history_rows = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_loss_steps = 0
        train_rows = []
        for batch in train_loader:
            outputs = model(**batch['model_inputs'])
            labels = batch['ground_truth']
            loss = criterion(outputs['class_logits'].float(), labels)
            pred = torch.argmax(outputs['class_probs'].float(), dim=-1)
            accelerator.backward(loss)
            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss_total += float(loss.detach().item())
            train_loss_steps += 1
            train_rows.extend(
                {'ground_truth': int(t), 'prediction': int(p)}
                for t, p in zip(labels.detach().cpu().tolist(), pred.detach().cpu().tolist())
            )

        train_rows = _gather_rows(train_rows, accelerator=accelerator)
        epoch_train_loss = _distributed_average(train_loss_total, train_loss_steps, accelerator)

        val_loss_total = 0.0
        val_loss_steps = 0
        val_rows = []
        model.eval()
        with torch.no_grad():
            for batch in eval_loaders['val']:
                outputs = model(**batch['model_inputs'])
                labels = batch['ground_truth']
                loss = criterion(outputs['class_logits'].float(), labels)
                pred = torch.argmax(outputs['class_probs'].float(), dim=-1)
                val_loss_total += float(loss.detach().item())
                val_loss_steps += 1
                val_rows.extend(
                    {'ground_truth': int(t), 'prediction': int(p)}
                    for t, p in zip(labels.detach().cpu().tolist(), pred.detach().cpu().tolist())
                )

        val_rows = _gather_rows(val_rows, accelerator=accelerator)
        epoch_val_loss = _distributed_average(val_loss_total, val_loss_steps, accelerator)

        should_stop = False
        if accelerator.is_main_process:
            train_frame = pd.DataFrame(train_rows)
            val_frame = pd.DataFrame(val_rows)
            train_metrics = classification_metrics(
                train_frame['ground_truth'].to_numpy(dtype=np.int64, copy=False),
                train_frame['prediction'].to_numpy(dtype=np.int64, copy=False),
            )
            val_metrics = classification_metrics(
                val_frame['ground_truth'].to_numpy(dtype=np.int64, copy=False),
                val_frame['prediction'].to_numpy(dtype=np.int64, copy=False),
            )
            history_rows.append({
                'epoch': epoch,
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'train_balanced_acc': train_metrics['balanced_acc'],
                'val_balanced_acc': val_metrics['balanced_acc'],
                'balanced_acc': val_metrics['balanced_acc'],
            })
            history_frame = pd.DataFrame(history_rows)
            history_frame.to_csv(os.path.join(run_dir, 'history.csv'), index=False)
            save_training_curves(history_frame, os.path.join(run_dir, 'training_curves.png'))
            print(
                f'Epoch {epoch}: train_loss={epoch_train_loss:.4f}, '
                f'val_loss={epoch_val_loss:.4f}, '
                f'train_bal_acc={train_metrics["balanced_acc"]:.4f}, '
                f'val_bal_acc={val_metrics["balanced_acc"]:.4f}'
            )

            current = float(val_metrics['balanced_acc'])
            if current > best_metric:
                best_metric = current
                best_epoch = epoch
                wait = 0
                _save_classifier_checkpoint(
                    run_dir,
                    accelerator.unwrap_model(model),
                    processor,
                    args,
                    best_epoch,
                    best_metric,
                )
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
        print(f'Frozen-head training finished -> {run_dir}')
        print(f'  Best epoch: {best_epoch}')
        print(f'  Best val balanced_acc: {best_metric:.4f}')


def _run_eval_head(args):
    accelerator = Accelerator(mixed_precision='bf16' if bool(args.use_bf16) else 'no')
    run_dir = str(Path(args.run_dir).resolve())
    output_dir = str(Path(args.output_dir or os.path.join(run_dir, 'test_eval')).resolve())
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        _save_json(os.path.join(output_dir, 'args.json'), vars(args))
    accelerator.wait_for_everyone()

    prep_dir = args.prep_dir or os.path.join(run_dir, 'prepared')
    workspace = _ensure_workspace(args, prep_dir, create_if_missing=False)

    checkpoint_path = os.path.join(run_dir, 'best_head', 'classifier.pt')
    checkpoint_meta = torch.load(checkpoint_path, map_location='cpu')
    if not str(getattr(args, 'adapter_dir', '') or '').strip():
        args.adapter_dir = str(checkpoint_meta.get('adapter_dir', '') or '')

    backbone, processor = load_qwen3vl_for_training(
        model_path=args.model_path,
        use_bf16=bool(args.use_bf16),
        enable_flash_attention=bool(args.flash_attention_2),
    )
    hidden_size = int(backbone.model.language_model.config.hidden_size)
    backbone = _load_optional_adapter(backbone, args)
    args.head_dropout = float(checkpoint_meta.get('head_dropout', args.head_dropout))
    args.pooling_mode = str(checkpoint_meta.get('pooling_mode', args.pooling_mode))
    args.mlp_hidden_size = int(checkpoint_meta.get('mlp_hidden_size', args.mlp_hidden_size))
    model = FrozenQwen3VLClassifier(
        backbone=backbone,
        hidden_size=hidden_size,
        num_classes=len(checkpoint_meta['class_names']),
        dropout=args.head_dropout,
        pooling_mode=args.pooling_mode,
        mlp_hidden_size=args.mlp_hidden_size,
        task_mode='flat',
    )
    checkpoint = _load_classifier_checkpoint(model, checkpoint_path, payload=checkpoint_meta)
    model.eval()

    _, eval_loaders = _build_dataloaders(
        workspace,
        processor,
        batch_size=args.per_device_batch_size,
        num_workers=args.num_workers,
    )
    model, val_loader, test_loader = accelerator.prepare(model, eval_loaders['val'], eval_loaders['test'])
    val_frame = _collect_cls_predictions(model, val_loader, accelerator=accelerator)
    test_frame = _collect_cls_predictions(model, test_loader, accelerator=accelerator)
    if accelerator.is_main_process:
        metrics = finalize_binary_eval(val_frame, test_frame, output_dir)
        print(f'Frozen-head evaluation results -> {output_dir}')
        for key, value in metrics.items():
            print(f'  {key}: {value:.4f}')
    accelerator.wait_for_everyone()


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--result_dir', type=str, default=DEFAULT_RESULT_ROOT)
    parser.add_argument('--render_cache_dir', type=str, default=DEFAULT_RENDER_CACHE_ROOT)
    parser.add_argument('--run_dir', type=str, default='')
    parser.add_argument('--prep_dir', type=str, default='')
    parser.add_argument('--exp_tag', type=str, default='')
    parser.add_argument('--adapter_dir', type=str, default='')
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
    parser.add_argument('--image_input_mode', type=str, default='single_panel', choices=['single_panel', 'dual_image'])
    parser.add_argument(
        '--single_image_kind',
        type=str,
        default='combined',
        choices=['waveform', 'spectrogram', 'cwt', 'combined', 'combined_cwt'],
    )
    parser.add_argument('--image_size', type=int, default=896)
    parser.add_argument('--normalization_mode', type=str, default='train_global', choices=['train_global', 'sample_global_robust'])
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
    parser.add_argument('--device_map', type=str, default='none')
    parser.add_argument('--per_device_batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--pooling_mode', type=str, default='image_mean', choices=['image_mean', 'dual_branch_meanmax', 'dual_image_meanmax'])
    parser.add_argument('--mlp_hidden_size', type=int, default=1024)


def parse_args():
    parser = argparse.ArgumentParser(description='vEpiSet task-1 frozen-backbone VLM classifier')
    subparsers = parser.add_subparsers(dest='command', required=True)

    prepare_parser = subparsers.add_parser('prepare', help='Prepare manifests and render stats')
    _add_shared_args(prepare_parser)

    train_parser = subparsers.add_parser('train_head', help='Train a classification head on frozen Qwen3-VL features')
    _add_shared_args(train_parser)
    train_parser.add_argument('--lr', type=float, default=1e-3)
    train_parser.add_argument('--weight_decay', type=float, default=1e-4)
    train_parser.add_argument('--head_dropout', type=float, default=0.0)
    train_parser.add_argument('--max_epochs', type=int, default=20)
    train_parser.add_argument('--patience', type=int, default=5)
    train_parser.add_argument('--max_grad_norm', type=float, default=1.0)

    eval_parser = subparsers.add_parser('eval_head', help='Evaluate a trained frozen-backbone classifier head')
    _add_shared_args(eval_parser)
    eval_parser.add_argument('--output_dir', type=str, default='')
    eval_parser.add_argument('--head_dropout', type=float, default=0.0)

    return finalize_qwen3vl_args(parser.parse_args())


def main():
    args = parse_args()
    args.result_dir = resolve_output_dir(args.result_dir or DEFAULT_RESULT_ROOT)
    args.render_cache_dir = resolve_output_dir(args.render_cache_dir or DEFAULT_RENDER_CACHE_ROOT)
    if args.command == 'prepare':
        _run_prepare(args)
    elif args.command == 'train_head':
        _run_train_head(args)
    elif args.command == 'eval_head':
        if not args.run_dir:
            raise ValueError('eval_head requires --run_dir to point to a training directory.')
        _run_eval_head(args)
    else:
        raise ValueError(f'Unknown command: {args.command}')


if __name__ == '__main__':
    main()
