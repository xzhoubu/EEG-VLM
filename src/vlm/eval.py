

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metric import binary_paper_metrics, classification_metrics, safe_average_precision, safe_roc_auc


def save_training_curves(history: pd.DataFrame, fig_path: str) -> None:

    if history.empty:
        return

    history = history.sort_values('epoch').reset_index(drop=True)
    epochs = history['epoch'].to_numpy(dtype=np.int64, copy=False)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)

    axes[0].plot(epochs, history['train_loss'], marker='o', linewidth=1.8, color='#1f4e79', label='Train Loss')
    if 'val_loss' in history.columns:
        axes[0].plot(epochs, history['val_loss'], marker='o', linewidth=1.8, color='#d62828', label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle='--', alpha=0.35)
    axes[0].legend(loc='best')

    if {'train_balanced_acc', 'val_balanced_acc'}.issubset(history.columns):
        metric_specs = [
            ('train_balanced_acc', 'Train Balanced Accuracy', '#1f4e79'),
            ('val_balanced_acc', 'Val Balanced Accuracy', '#2a9d8f'),
        ]
        for column, label, color in [
            ('val_cohen_kappa', "Val Cohen's Kappa", '#e76f51'),
            ('val_weighted_f1', 'Val Weighted F1', '#6a4c93'),
        ]:
            if column in history.columns:
                metric_specs.append((column, label, color))
    else:
        metric_specs = [
            ('balanced_acc', 'Val Balanced Accuracy', '#2a9d8f'),
            ('auprc', 'Val AUPRC', '#e76f51'),
            ('auroc', 'Val AUROC', '#6a4c93'),
        ]
    for column, label, color in metric_specs:
        if column in history.columns:
            axes[1].plot(epochs, history[column], marker='o', linewidth=1.6, label=label, color=color)
    axes[1].set_title('Validation Metrics')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].grid(True, linestyle='--', alpha=0.35)
    axes[1].legend(loc='best')
    if epochs.size:
        axes[1].set_xticks(epochs.tolist())

    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def _drop_distributed_padding_duplicates(frame: pd.DataFrame) -> pd.DataFrame:

    if frame.empty:
        return frame
    subset = [
        column for column in ['sample_id', 'split', 'sample_file', 'ground_truth', 'prediction']
        if column in frame.columns
    ]
    if not subset:
        return frame.drop_duplicates().reset_index(drop=True)
    return frame.drop_duplicates(subset=subset, keep='first').reset_index(drop=True)


def _round_paper_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 4) for key, value in metrics.items()}


def _write_metrics(metrics: dict[str, float], save_dir: str) -> None:
    pd.DataFrame([{'metric': key, 'value': value} for key, value in metrics.items()]).to_csv(
        os.path.join(save_dir, 'metrics.csv'),
        index=False,
    )


def finalize_binary_eval(val_frame: pd.DataFrame, test_frame: pd.DataFrame, save_dir: str) -> dict[str, float]:

    os.makedirs(save_dir, exist_ok=True)
    if val_frame.empty or test_frame.empty:
        raise ValueError('Both validation and test prediction frames are required.')

    val_frame = _drop_distributed_padding_duplicates(val_frame)
    test_frame = _drop_distributed_padding_duplicates(test_frame)
    val_frame = val_frame.sort_values('sample_id').reset_index(drop=True).copy()
    test_frame = test_frame.sort_values('sample_id').reset_index(drop=True).copy()
    test_true = test_frame['ground_truth'].to_numpy(dtype=np.int64, copy=False)
    test_score = test_frame['prob_positive'].to_numpy(dtype=np.float64, copy=False)
    val_frame['prediction'] = (val_frame['prob_positive'] >= 0.5).astype(np.int64)
    test_frame['prediction'] = (test_frame['prob_positive'] >= 0.5).astype(np.int64)

    metrics = _round_paper_metrics(binary_paper_metrics(test_true, test_score))
    val_frame.to_csv(os.path.join(save_dir, 'val_predictions.csv'), index=False)
    test_frame.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    _write_metrics(metrics, save_dir)
    return metrics


def finalize_multiclass_eval(val_frame: pd.DataFrame,
                             test_frame: pd.DataFrame,
                             save_dir: str,
                             class_names: list[str]) -> dict[str, float]:

    os.makedirs(save_dir, exist_ok=True)
    if val_frame.empty or test_frame.empty:
        raise ValueError('Both validation and test prediction frames are required.')

    val_frame = _drop_distributed_padding_duplicates(val_frame)
    test_frame = _drop_distributed_padding_duplicates(test_frame)
    val_frame = val_frame.sort_values('sample_id').reset_index(drop=True).copy()
    test_frame = test_frame.sort_values('sample_id').reset_index(drop=True).copy()
    test_true = test_frame['ground_truth'].to_numpy(dtype=np.int64, copy=False)
    test_pred = test_frame['prediction'].to_numpy(dtype=np.int64, copy=False)

    if len(class_names) == 2:
        score_column = next(
            (name for name in ['prob_abnormal', 'prob_positive', 'prob_1'] if name in test_frame.columns),
            None,
        )
        if score_column is None:
            raise ValueError('TUAB evaluation requires the abnormal-class probability.')
        test_score = test_frame[score_column].to_numpy(dtype=np.float64, copy=False)
        metrics = {
            'balanced_acc': float(classification_metrics(test_true, test_pred)['balanced_acc']),
            'auprc': safe_average_precision(test_true, test_score),
            'auroc': safe_roc_auc(test_true, test_score),
        }
    else:
        classification = classification_metrics(test_true, test_pred)
        metrics = {
            'balanced_acc': classification['balanced_acc'],
            'cohen_kappa': classification['cohen_kappa'],
            'weighted_f1': classification['weighted_f1'],
        }

    metrics = _round_paper_metrics(metrics)
    val_frame.to_csv(os.path.join(save_dir, 'val_predictions.csv'), index=False)
    test_frame.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    _write_metrics(metrics, save_dir)
    return metrics
