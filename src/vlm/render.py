

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft as scipy_fft
from scipy import signal


_CWT_BANK_CACHE: dict[tuple[Any, ...], tuple[np.ndarray, int, np.ndarray]] = {}


def _channel_index_map(channel_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(channel_names)}


def extract_bipolar_sample(sample: np.ndarray,
                           channel_names: list[str],
                           montage_pairs: list[list[str]] | list[tuple[str, str]]) -> np.ndarray:

    idx_map = _channel_index_map(channel_names)
    traces = []
    for left, right in montage_pairs:
        traces.append(sample[:, idx_map[left]] - sample[:, idx_map[right]])
    return np.stack(traces, axis=0).astype(np.float32, copy=False)


def _stack_bipolar_batch(samples: np.ndarray,
                         channel_names: list[str],
                         montage_pairs: list[list[str]] | list[tuple[str, str]]) -> np.ndarray:
    idx_map = _channel_index_map(channel_names)
    left_idx = np.asarray([idx_map[left] for left, _ in montage_pairs], dtype=np.int64)
    right_idx = np.asarray([idx_map[right] for _, right in montage_pairs], dtype=np.int64)
    traces = samples[:, :, left_idx] - samples[:, :, right_idx]
    return np.transpose(traces, (0, 2, 1)).astype(np.float32, copy=False)


def build_spectrogram_panel(bipolar_traces: np.ndarray,
                            sfreq: int,
                            render_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:

    nperseg = int(render_config['spectrogram_nperseg'])
    noverlap = int(render_config['spectrogram_noverlap'])
    nfft = int(render_config['spectrogram_nfft'])
    freq_min = float(render_config['spectrogram_freq_min'])
    freq_max = float(render_config['spectrogram_freq_max'])
    eps = float(render_config.get('spectrogram_log_eps', 1e-8))

    blocks = []
    time_axis = None
    for trace in bipolar_traces:
        freqs, times, power = signal.spectrogram(
            trace,
            fs=sfreq,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=False,
            scaling='density',
            mode='psd',
        )
        keep = (freqs >= freq_min) & (freqs <= freq_max)
        power = np.log10(power[keep] + eps)
        blocks.append(power.astype(np.float32, copy=False))
        if time_axis is None:
            time_axis = times.astype(np.float32, copy=False)

    return np.concatenate(blocks, axis=0), time_axis if time_axis is not None else np.array([], dtype=np.float32)


def _cwt_frequencies(render_config: dict[str, Any]) -> np.ndarray:
    freq_min = float(render_config.get('cwt_freq_min', render_config.get('spectrogram_freq_min', 1.0)))
    freq_max = float(render_config.get('cwt_freq_max', render_config.get('spectrogram_freq_max', 45.0)))
    n_freqs = int(render_config.get('cwt_num_freqs', 48))
    spacing = str(render_config.get('cwt_freq_spacing', 'log') or 'log')
    if n_freqs <= 1:
        return np.asarray([max(freq_min, 1e-6)], dtype=np.float32)
    if spacing == 'linear':
        return np.linspace(freq_min, freq_max, n_freqs, dtype=np.float32)
    return np.geomspace(max(freq_min, 1e-6), max(freq_max, freq_min + 1e-6), n_freqs).astype(np.float32)


def _build_morlet_bank(n_times: int,
                       sfreq: int,
                       render_config: dict[str, Any]) -> tuple[np.ndarray, int, np.ndarray]:
    freqs = _cwt_frequencies(render_config)
    omega0 = float(render_config.get('cwt_morlet_w', 6.0))
    support = float(render_config.get('cwt_support', 6.0))
    cache_key = (
        int(n_times),
        int(sfreq),
        tuple(np.round(freqs.astype(np.float64), 6).tolist()),
        round(omega0, 6),
        round(support, 6),
    )
    cached = _CWT_BANK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    scales = omega0 * float(sfreq) / (2.0 * np.pi * np.maximum(freqs.astype(np.float64), 1e-6))
    half_widths = np.maximum(1, np.ceil(support * scales).astype(np.int64))
    max_half_width = int(np.max(half_widths))
    bank_len = 2 * max_half_width + 1
    center = bank_len // 2
    bank = np.zeros((len(freqs), bank_len), dtype=np.complex64)

    for idx, (scale, half_width) in enumerate(zip(scales, half_widths)):
        t = np.arange(-int(half_width), int(half_width) + 1, dtype=np.float64) / float(scale)
        wavelet = (np.pi ** -0.25) * np.exp(1j * omega0 * t) * np.exp(-0.5 * t * t)
        wavelet = wavelet / np.sqrt(float(scale))
        kernel = np.conj(wavelet[::-1])
        start = center - int(half_width)
        bank[idx, start:start + kernel.size] = kernel.astype(np.complex64, copy=False)

    n_fft = int(scipy_fft.next_fast_len(int(n_times) + bank_len - 1))
    bank_fft = np.fft.fft(bank, n=n_fft, axis=1).astype(np.complex64, copy=False)
    cached = (bank_fft, bank_len, freqs)
    _CWT_BANK_CACHE[cache_key] = cached
    return cached


def build_cwt_panel(bipolar_traces: np.ndarray,
                    sfreq: int,
                    render_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:

    n_times = int(bipolar_traces.shape[1])
    time_stride = max(1, int(render_config.get('cwt_time_stride', 16)))
    eps = float(render_config.get('cwt_log_eps', render_config.get('spectrogram_log_eps', 1e-8)))
    bank_fft, bank_len, _freqs = _build_morlet_bank(n_times, sfreq, render_config)
    n_fft = int(bank_fft.shape[1])
    start = (bank_len - 1) // 2
    stop = start + n_times

    blocks = []
    for trace in bipolar_traces:
        trace_fft = np.fft.fft(trace.astype(np.float32, copy=False), n=n_fft)
        coeffs = np.fft.ifft(bank_fft * trace_fft[None, :], n=n_fft, axis=1)[:, start:stop]
        coeffs = coeffs[:, ::time_stride]
        power = np.log10(np.abs(coeffs) ** 2 + eps)
        blocks.append(power.astype(np.float32, copy=False))

    time_axis = (np.arange(0, n_times, time_stride, dtype=np.float32) / float(sfreq)).astype(np.float32, copy=False)
    return np.concatenate(blocks, axis=0), time_axis


def _normalization_mode(render_config: dict[str, Any]) -> str:
    return str(render_config.get('normalization_mode', 'train_global') or 'train_global')


def _single_image_kind(render_config: dict[str, Any]) -> str:
    return str(render_config.get('single_image_kind', 'combined') or 'combined')


def _uses_spectrogram(render_config: dict[str, Any]) -> bool:
    return _single_image_kind(render_config) in {'combined', 'spectrogram'}


def _uses_cwt(render_config: dict[str, Any]) -> bool:
    return _single_image_kind(render_config) in {'combined_cwt', 'cwt'}


def _sample_waveform_panel(bipolar_traces: np.ndarray,
                           render_config: dict[str, Any]) -> tuple[np.ndarray, float]:
    centered = bipolar_traces.astype(np.float32, copy=False) - float(np.median(bipolar_traces))
    percentile = float(render_config['waveform_percentile'])
    scale = float(np.quantile(np.abs(centered), percentile))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return centered / scale, scale


def _sample_spectrogram_limits(spectrogram_panel: np.ndarray,
                               render_config: dict[str, Any]) -> tuple[float, float]:
    values = spectrogram_panel.reshape(-1)
    vmin = float(np.quantile(values, float(render_config['spectrogram_quantile_low'])))
    vmax = float(np.quantile(values, float(render_config['spectrogram_quantile_high'])))
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(values)) if values.size else -6.0
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(values)) if values.size else 2.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _sample_cwt_limits(cwt_panel: np.ndarray,
                       render_config: dict[str, Any]) -> tuple[float, float]:
    values = cwt_panel.reshape(-1)
    vmin = float(np.quantile(values, float(render_config.get('cwt_quantile_low', 0.01))))
    vmax = float(np.quantile(values, float(render_config.get('cwt_quantile_high', 0.99))))
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(values)) if values.size else -6.0
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(values)) if values.size else 2.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def compute_render_stats(train_samples: np.ndarray,
                         channel_names: list[str],
                         sfreq: int,
                         render_config: dict[str, Any],
                         seed: int = 42) -> dict[str, Any]:

    if len(train_samples) == 0:
        raise ValueError('Cannot compute render stats from an empty training split.')

    rng = np.random.RandomState(seed)
    montage_pairs = render_config['montage_pairs']
    norm_mode = _normalization_mode(render_config)

    probe_bipolar = extract_bipolar_sample(train_samples[0], channel_names, montage_pairs)
    stats: dict[str, Any] = {
        'normalization_mode': norm_mode,
        'waveform_abs_max': 1.0,
        'waveform_stats_samples': 0,
    }

    if _uses_spectrogram(render_config):
        probe_panel, probe_times = build_spectrogram_panel(probe_bipolar, sfreq=sfreq, render_config=render_config)
        stats.update({
            'spectrogram_vmin': 0.0,
            'spectrogram_vmax': 1.0,
            'spectrogram_stats_samples': 0,
            'freq_bins_per_channel': int(probe_panel.shape[0] // max(1, probe_bipolar.shape[0])),
            'time_bins': int(probe_panel.shape[1]) if probe_panel.ndim == 2 else int(probe_times.size),
        })

    if _uses_cwt(render_config):
        probe_panel, probe_times = build_cwt_panel(probe_bipolar, sfreq=sfreq, render_config=render_config)
        stats.update({
            'cwt_vmin': 0.0,
            'cwt_vmax': 1.0,
            'cwt_stats_samples': 0,
            'cwt_freq_bins_per_channel': int(probe_panel.shape[0] // max(1, probe_bipolar.shape[0])),
            'cwt_time_bins': int(probe_panel.shape[1]) if probe_panel.ndim == 2 else int(probe_times.size),
        })

    if norm_mode == 'sample_global_robust':
        return stats

    wave_max_samples = min(len(train_samples), int(render_config['waveform_stats_max_samples']))
    wave_idx = np.sort(rng.choice(len(train_samples), size=max(1, wave_max_samples), replace=False))
    wave_batch = _stack_bipolar_batch(train_samples[wave_idx], channel_names, montage_pairs)
    waveform_abs_max = float(np.quantile(np.abs(wave_batch), float(render_config['waveform_percentile'])))
    if not np.isfinite(waveform_abs_max) or waveform_abs_max <= 0.0:
        waveform_abs_max = 1.0

    stats.update({
        'waveform_abs_max': waveform_abs_max,
        'waveform_stats_samples': int(len(wave_idx)),
    })

    if _uses_spectrogram(render_config):
        spec_max_samples = min(len(train_samples), int(render_config['spectrogram_stats_max_samples']))
        spec_idx = np.sort(rng.choice(len(train_samples), size=max(1, spec_max_samples), replace=False))
        spec_chunks = []
        for idx in spec_idx:
            bipolar = extract_bipolar_sample(train_samples[idx], channel_names, montage_pairs)
            panel, _times = build_spectrogram_panel(bipolar, sfreq=sfreq, render_config=render_config)
            spec_chunks.append(panel.reshape(-1))
        spec_values = np.concatenate(spec_chunks, axis=0)
        spectrogram_vmin = float(np.quantile(spec_values, float(render_config['spectrogram_quantile_low'])))
        spectrogram_vmax = float(np.quantile(spec_values, float(render_config['spectrogram_quantile_high'])))
        if not np.isfinite(spectrogram_vmin):
            spectrogram_vmin = -6.0
        if not np.isfinite(spectrogram_vmax):
            spectrogram_vmax = 2.0
        if spectrogram_vmax <= spectrogram_vmin:
            spectrogram_vmax = spectrogram_vmin + 1.0
        stats.update({
            'spectrogram_vmin': spectrogram_vmin,
            'spectrogram_vmax': spectrogram_vmax,
            'spectrogram_stats_samples': int(len(spec_idx)),
        })

    if _uses_cwt(render_config):
        cwt_max_samples = min(len(train_samples), int(render_config.get('cwt_stats_max_samples', 16)))
        cwt_idx = np.sort(rng.choice(len(train_samples), size=max(1, cwt_max_samples), replace=False))
        cwt_chunks = []
        for idx in cwt_idx:
            bipolar = extract_bipolar_sample(train_samples[idx], channel_names, montage_pairs)
            panel, _times = build_cwt_panel(bipolar, sfreq=sfreq, render_config=render_config)
            cwt_chunks.append(panel.reshape(-1))
        cwt_values = np.concatenate(cwt_chunks, axis=0)
        cwt_vmin = float(np.quantile(cwt_values, float(render_config.get('cwt_quantile_low', 0.01))))
        cwt_vmax = float(np.quantile(cwt_values, float(render_config.get('cwt_quantile_high', 0.99))))
        if not np.isfinite(cwt_vmin):
            cwt_vmin = -6.0
        if not np.isfinite(cwt_vmax):
            cwt_vmax = 2.0
        if cwt_vmax <= cwt_vmin:
            cwt_vmax = cwt_vmin + 1.0
        stats.update({
            'cwt_vmin': cwt_vmin,
            'cwt_vmax': cwt_vmax,
            'cwt_stats_samples': int(len(cwt_idx)),
        })

    return stats


def render_sample(sample: np.ndarray,
                  channel_names: list[str],
                  sfreq: int,
                  duration_sec: float,
                  render_config: dict[str, Any],
                  render_stats: dict[str, Any],
                  output_path: str) -> None:

    bipolar = extract_bipolar_sample(sample, channel_names, render_config['montage_pairs'])
    spectrogram_panel, spec_times = build_spectrogram_panel(bipolar, sfreq=sfreq, render_config=render_config)

    labels = render_config['montage_labels']
    n_pairs, seq_len = bipolar.shape
    image_size = int(render_config['image_size'])
    time_axis = np.linspace(0.0, duration_sec, seq_len, endpoint=False, dtype=np.float32)

    fig = plt.figure(figsize=(image_size / 100.0, image_size / 100.0), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.15)

    ax_wave = fig.add_subplot(gs[0, 0])
    offsets = np.arange(n_pairs - 1, -1, -1, dtype=np.float32)
    if _normalization_mode(render_config) == 'sample_global_robust':
        wave_panel, _scale = _sample_waveform_panel(bipolar, render_config)
    else:
        scale = max(float(render_stats['waveform_abs_max']), 1e-6)
        wave_panel = bipolar / scale
    for pair_idx in range(n_pairs):
        trace = wave_panel[pair_idx]
        ax_wave.plot(time_axis, offsets[pair_idx] + 0.42 * trace, color='black', linewidth=0.7)
    ax_wave.set_xlim(0.0, duration_sec)
    ax_wave.set_ylim(-0.75, n_pairs - 0.25)
    ax_wave.set_yticks(offsets)
    ax_wave.set_yticklabels(labels, fontsize=7)
    ax_wave.set_xticks([0.0, duration_sec / 2.0, duration_sec])
    ax_wave.set_xlabel('Time (s)', fontsize=8)
    ax_wave.set_title('Longitudinal Bipolar Montage', fontsize=9)
    ax_wave.grid(axis='x', alpha=0.25, linewidth=0.5)
    ax_wave.tick_params(axis='x', labelsize=7)
    ax_wave.tick_params(axis='y', pad=1)

    ax_spec = fig.add_subplot(gs[0, 1])
    if _normalization_mode(render_config) == 'sample_global_robust':
        spec_vmin, spec_vmax = _sample_spectrogram_limits(spectrogram_panel, render_config)
    else:
        spec_vmin = float(render_stats['spectrogram_vmin'])
        spec_vmax = float(render_stats['spectrogram_vmax'])
    im = ax_spec.imshow(
        spectrogram_panel,
        origin='lower',
        aspect='auto',
        interpolation='nearest',
        cmap='viridis',
        vmin=spec_vmin,
        vmax=spec_vmax,
    )
    freq_bins = max(1, int(render_stats.get('freq_bins_per_channel', 0)) or (spectrogram_panel.shape[0] // max(1, n_pairs)))
    centers = [pair_idx * freq_bins + (freq_bins / 2.0) for pair_idx in range(n_pairs)]
    ax_spec.set_yticks(centers)
    ax_spec.set_yticklabels(labels, fontsize=7)
    if spec_times.size > 0:
        xticks = [0, max(0, spec_times.size // 2), max(0, spec_times.size - 1)]
        xtick_labels = [f'{float(spec_times[idx]):.2f}' for idx in xticks]
        ax_spec.set_xticks(xticks)
        ax_spec.set_xticklabels(xtick_labels, fontsize=7)
    else:
        ax_spec.set_xticks([])
    ax_spec.set_xlabel('Time (s)', fontsize=8)
    ax_spec.set_title(f'Spectrogram {render_config["spectrogram_freq_min"]:.0f}-{render_config["spectrogram_freq_max"]:.0f} Hz', fontsize=9)
    fig.colorbar(im, ax=ax_spec, fraction=0.046, pad=0.04)

    fig.tight_layout(pad=0.6)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='render_', suffix='.png', dir=str(output.parent))
    os.close(fd)
    try:
        fig.savefig(tmp_path, dpi=100)
        os.replace(tmp_path, output_path)
    finally:
        plt.close(fig)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _waveform_panel_for_render(bipolar: np.ndarray,
                               render_config: dict[str, Any],
                               render_stats: dict[str, Any]) -> np.ndarray:
    if _normalization_mode(render_config) == 'sample_global_robust':
        wave_panel, _scale = _sample_waveform_panel(bipolar, render_config)
        return wave_panel
    scale = max(float(render_stats['waveform_abs_max']), 1e-6)
    return bipolar / scale


def _spectrogram_limits_for_render(spectrogram_panel: np.ndarray,
                                   render_config: dict[str, Any],
                                   render_stats: dict[str, Any]) -> tuple[float, float]:
    if _normalization_mode(render_config) == 'sample_global_robust':
        return _sample_spectrogram_limits(spectrogram_panel, render_config)
    return float(render_stats['spectrogram_vmin']), float(render_stats['spectrogram_vmax'])


def _cwt_limits_for_render(cwt_panel: np.ndarray,
                           render_config: dict[str, Any],
                           render_stats: dict[str, Any]) -> tuple[float, float]:
    if _normalization_mode(render_config) == 'sample_global_robust':
        return _sample_cwt_limits(cwt_panel, render_config)
    return float(render_stats['cwt_vmin']), float(render_stats['cwt_vmax'])


def _save_figure_atomic(fig: plt.Figure, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='render_', suffix='.png', dir=str(output.parent))
    os.close(fd)
    try:
        fig.savefig(tmp_path, dpi=100)
        os.replace(tmp_path, output_path)
    finally:
        plt.close(fig)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def render_waveform_image(sample: np.ndarray,
                          channel_names: list[str],
                          sfreq: int,
                          duration_sec: float,
                          render_config: dict[str, Any],
                          render_stats: dict[str, Any],
                          output_path: str) -> None:

    del sfreq
    bipolar = extract_bipolar_sample(sample, channel_names, render_config['montage_pairs'])
    labels = render_config['montage_labels']
    n_pairs, seq_len = bipolar.shape
    image_size = int(render_config['image_size'])
    time_axis = np.linspace(0.0, duration_sec, seq_len, endpoint=False, dtype=np.float32)
    wave_panel = _waveform_panel_for_render(bipolar, render_config, render_stats)

    fig, ax_wave = plt.subplots(figsize=(image_size / 100.0, image_size / 100.0), dpi=100)
    offsets = np.arange(n_pairs - 1, -1, -1, dtype=np.float32)
    for pair_idx in range(n_pairs):
        trace = wave_panel[pair_idx]
        ax_wave.plot(time_axis, offsets[pair_idx] + 0.42 * trace, color='black', linewidth=0.8)
    ax_wave.set_xlim(0.0, duration_sec)
    ax_wave.set_ylim(-0.75, n_pairs - 0.25)
    ax_wave.set_yticks(offsets)
    ax_wave.set_yticklabels(labels, fontsize=8)
    ax_wave.set_xticks([0.0, duration_sec / 2.0, duration_sec])
    ax_wave.set_xlabel('Time (s)', fontsize=9)
    ax_wave.set_title('Image 1: Longitudinal Bipolar Montage', fontsize=11)
    ax_wave.grid(axis='x', alpha=0.25, linewidth=0.5)
    ax_wave.tick_params(axis='x', labelsize=8)
    ax_wave.tick_params(axis='y', pad=1)
    fig.tight_layout(pad=0.8)
    _save_figure_atomic(fig, output_path)


def render_spectrogram_image(sample: np.ndarray,
                             channel_names: list[str],
                             sfreq: int,
                             duration_sec: float,
                             render_config: dict[str, Any],
                             render_stats: dict[str, Any],
                             output_path: str) -> None:

    del duration_sec
    bipolar = extract_bipolar_sample(sample, channel_names, render_config['montage_pairs'])
    spectrogram_panel, spec_times = build_spectrogram_panel(bipolar, sfreq=sfreq, render_config=render_config)
    labels = render_config['montage_labels']
    n_pairs = bipolar.shape[0]
    image_size = int(render_config['image_size'])
    spec_vmin, spec_vmax = _spectrogram_limits_for_render(spectrogram_panel, render_config, render_stats)

    fig, ax_spec = plt.subplots(figsize=(image_size / 100.0, image_size / 100.0), dpi=100)
    im = ax_spec.imshow(
        spectrogram_panel,
        origin='lower',
        aspect='auto',
        interpolation='nearest',
        cmap='viridis',
        vmin=spec_vmin,
        vmax=spec_vmax,
    )
    freq_bins = max(1, int(render_stats.get('freq_bins_per_channel', 0)) or (spectrogram_panel.shape[0] // max(1, n_pairs)))
    centers = [pair_idx * freq_bins + (freq_bins / 2.0) for pair_idx in range(n_pairs)]
    ax_spec.set_yticks(centers)
    ax_spec.set_yticklabels(labels, fontsize=8)
    if spec_times.size > 0:
        xticks = [0, max(0, spec_times.size // 2), max(0, spec_times.size - 1)]
        xtick_labels = [f'{float(spec_times[idx]):.2f}' for idx in xticks]
        ax_spec.set_xticks(xticks)
        ax_spec.set_xticklabels(xtick_labels, fontsize=8)
    else:
        ax_spec.set_xticks([])
    ax_spec.set_xlabel('Time (s)', fontsize=9)
    ax_spec.set_title(
        f'Image 2: Spectrogram {render_config["spectrogram_freq_min"]:.0f}-{render_config["spectrogram_freq_max"]:.0f} Hz',
        fontsize=11,
    )
    fig.colorbar(im, ax=ax_spec, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.8)
    _save_figure_atomic(fig, output_path)


def render_cwt_image(sample: np.ndarray,
                     channel_names: list[str],
                     sfreq: int,
                     duration_sec: float,
                     render_config: dict[str, Any],
                     render_stats: dict[str, Any],
                     output_path: str) -> None:

    del duration_sec
    bipolar = extract_bipolar_sample(sample, channel_names, render_config['montage_pairs'])
    cwt_panel, cwt_times = build_cwt_panel(bipolar, sfreq=sfreq, render_config=render_config)
    labels = render_config['montage_labels']
    n_pairs = bipolar.shape[0]
    image_size = int(render_config['image_size'])
    cwt_vmin, cwt_vmax = _cwt_limits_for_render(cwt_panel, render_config, render_stats)

    fig, ax_cwt = plt.subplots(figsize=(image_size / 100.0, image_size / 100.0), dpi=100)
    im = ax_cwt.imshow(
        cwt_panel,
        origin='lower',
        aspect='auto',
        interpolation='nearest',
        cmap='viridis',
        vmin=cwt_vmin,
        vmax=cwt_vmax,
    )
    freq_bins = max(1, int(render_stats.get('cwt_freq_bins_per_channel', 0)) or (cwt_panel.shape[0] // max(1, n_pairs)))
    centers = [pair_idx * freq_bins + (freq_bins / 2.0) for pair_idx in range(n_pairs)]
    ax_cwt.set_yticks(centers)
    ax_cwt.set_yticklabels(labels, fontsize=8)
    if cwt_times.size > 0:
        xticks = [0, max(0, cwt_times.size // 2), max(0, cwt_times.size - 1)]
        xtick_labels = [f'{float(cwt_times[idx]):.2f}' for idx in xticks]
        ax_cwt.set_xticks(xticks)
        ax_cwt.set_xticklabels(xtick_labels, fontsize=8)
    else:
        ax_cwt.set_xticks([])
    ax_cwt.set_xlabel('Time (s)', fontsize=9)
    ax_cwt.set_title(
        f'Image 2: CWT Scalogram {render_config.get("cwt_freq_min", 1.0):.0f}-{render_config.get("cwt_freq_max", 45.0):.0f} Hz',
        fontsize=11,
    )
    fig.colorbar(im, ax=ax_cwt, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.8)
    _save_figure_atomic(fig, output_path)


def render_sample_cwt(sample: np.ndarray,
                      channel_names: list[str],
                      sfreq: int,
                      duration_sec: float,
                      render_config: dict[str, Any],
                      render_stats: dict[str, Any],
                      output_path: str) -> None:

    bipolar = extract_bipolar_sample(sample, channel_names, render_config['montage_pairs'])
    cwt_panel, cwt_times = build_cwt_panel(bipolar, sfreq=sfreq, render_config=render_config)

    labels = render_config['montage_labels']
    n_pairs, seq_len = bipolar.shape
    image_size = int(render_config['image_size'])
    time_axis = np.linspace(0.0, duration_sec, seq_len, endpoint=False, dtype=np.float32)

    fig = plt.figure(figsize=(image_size / 100.0, image_size / 100.0), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.15)

    ax_wave = fig.add_subplot(gs[0, 0])
    offsets = np.arange(n_pairs - 1, -1, -1, dtype=np.float32)
    wave_panel = _waveform_panel_for_render(bipolar, render_config, render_stats)
    for pair_idx in range(n_pairs):
        trace = wave_panel[pair_idx]
        ax_wave.plot(time_axis, offsets[pair_idx] + 0.42 * trace, color='black', linewidth=0.7)
    ax_wave.set_xlim(0.0, duration_sec)
    ax_wave.set_ylim(-0.75, n_pairs - 0.25)
    ax_wave.set_yticks(offsets)
    ax_wave.set_yticklabels(labels, fontsize=7)
    ax_wave.set_xticks([0.0, duration_sec / 2.0, duration_sec])
    ax_wave.set_xlabel('Time (s)', fontsize=8)
    ax_wave.set_title('Longitudinal Bipolar Montage', fontsize=9)
    ax_wave.grid(axis='x', alpha=0.25, linewidth=0.5)
    ax_wave.tick_params(axis='x', labelsize=7)
    ax_wave.tick_params(axis='y', pad=1)

    ax_cwt = fig.add_subplot(gs[0, 1])
    cwt_vmin, cwt_vmax = _cwt_limits_for_render(cwt_panel, render_config, render_stats)
    im = ax_cwt.imshow(
        cwt_panel,
        origin='lower',
        aspect='auto',
        interpolation='nearest',
        cmap='viridis',
        vmin=cwt_vmin,
        vmax=cwt_vmax,
    )
    freq_bins = max(1, int(render_stats.get('cwt_freq_bins_per_channel', 0)) or (cwt_panel.shape[0] // max(1, n_pairs)))
    centers = [pair_idx * freq_bins + (freq_bins / 2.0) for pair_idx in range(n_pairs)]
    ax_cwt.set_yticks(centers)
    ax_cwt.set_yticklabels(labels, fontsize=7)
    if cwt_times.size > 0:
        xticks = [0, max(0, cwt_times.size // 2), max(0, cwt_times.size - 1)]
        xtick_labels = [f'{float(cwt_times[idx]):.2f}' for idx in xticks]
        ax_cwt.set_xticks(xticks)
        ax_cwt.set_xticklabels(xtick_labels, fontsize=7)
    else:
        ax_cwt.set_xticks([])
    ax_cwt.set_xlabel('Time (s)', fontsize=8)
    ax_cwt.set_title(
        f'CWT Scalogram {render_config.get("cwt_freq_min", 1.0):.0f}-{render_config.get("cwt_freq_max", 45.0):.0f} Hz',
        fontsize=9,
    )
    fig.colorbar(im, ax=ax_cwt, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.6)
    _save_figure_atomic(fig, output_path)


def ensure_rendered_images(sample: np.ndarray,
                           channel_names: list[str],
                           sfreq: int,
                           duration_sec: float,
                           render_config: dict[str, Any],
                           render_stats: dict[str, Any],
                           output_path: str,
                           waveform_output_path: str | None = None,
                           spectrogram_output_path: str | None = None) -> list[str]:

    image_input_mode = str(render_config.get('image_input_mode', 'single_panel') or 'single_panel')
    if image_input_mode != 'dual_image':
        return [
            ensure_rendered_image(
                sample=sample,
                channel_names=channel_names,
                sfreq=sfreq,
                duration_sec=duration_sec,
                render_config=render_config,
                render_stats=render_stats,
                output_path=output_path,
            )
        ]

    output = Path(output_path)
    waveform_path = Path(waveform_output_path) if waveform_output_path else output.with_name(f'{output.stem}_waveform{output.suffix}')
    spectrogram_path = Path(spectrogram_output_path) if spectrogram_output_path else output.with_name(f'{output.stem}_spectrogram{output.suffix}')
    if not waveform_path.is_file():
        render_waveform_image(
            sample=sample,
            channel_names=channel_names,
            sfreq=sfreq,
            duration_sec=duration_sec,
            render_config=render_config,
            render_stats=render_stats,
            output_path=str(waveform_path),
        )
    if not spectrogram_path.is_file():
        render_spectrogram_image(
            sample=sample,
            channel_names=channel_names,
            sfreq=sfreq,
            duration_sec=duration_sec,
            render_config=render_config,
            render_stats=render_stats,
            output_path=str(spectrogram_path),
        )
    return [str(waveform_path), str(spectrogram_path)]


def ensure_rendered_image(sample: np.ndarray,
                          channel_names: list[str],
                          sfreq: int,
                          duration_sec: float,
                          render_config: dict[str, Any],
                          render_stats: dict[str, Any],
                          output_path: str) -> str:

    if os.path.isfile(output_path):
        return output_path
    image_input_mode = str(render_config.get('image_input_mode', 'single_panel') or 'single_panel')
    single_image_kind = _single_image_kind(render_config)
    if image_input_mode == 'waveform_only' or single_image_kind == 'waveform':
        render_waveform_image(
            sample=sample,
            channel_names=channel_names,
            sfreq=sfreq,
            duration_sec=duration_sec,
            render_config=render_config,
            render_stats=render_stats,
            output_path=output_path,
        )
        return output_path
    if single_image_kind == 'spectrogram':
        render_spectrogram_image(
            sample=sample,
            channel_names=channel_names,
            sfreq=sfreq,
            duration_sec=duration_sec,
            render_config=render_config,
            render_stats=render_stats,
            output_path=output_path,
        )
        return output_path
    if single_image_kind == 'cwt':
        render_cwt_image(
            sample=sample,
            channel_names=channel_names,
            sfreq=sfreq,
            duration_sec=duration_sec,
            render_config=render_config,
            render_stats=render_stats,
            output_path=output_path,
        )
        return output_path
    if single_image_kind == 'combined_cwt':
        render_sample_cwt(
            sample=sample,
            channel_names=channel_names,
            sfreq=sfreq,
            duration_sec=duration_sec,
            render_config=render_config,
            render_stats=render_stats,
            output_path=output_path,
        )
        return output_path
    render_sample(
        sample=sample,
        channel_names=channel_names,
        sfreq=sfreq,
        duration_sec=duration_sec,
        render_config=render_config,
        render_stats=render_stats,
        output_path=output_path,
    )
    return output_path
