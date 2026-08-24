# EEG-VLM

Official implementation of **“Recasting Visual Language Models as EEG Foundation Models for Clinical Neurodiagnostics.”**

This release contains the EEG-VLM code used for:

- TUAB normal-versus-abnormal classification;
- TUEV six-class EEG event recognition;
- vEpiSet interictal epileptiform discharge detection.

Each EEG window is rendered as a waveform montage, a time-frequency image, or
their combination and passed to Qwen3-VL with a task instruction. The release
supports frozen-backbone prediction-head adaptation, LoRA, visual input
ablations, and adapter-based transfer.

## Repository layout

```text
EEG-VLM/
├── src/                  # Data preparation, rendering, models, and training
├── scripts/              # Training launchers
├── data/                 # Empty dataset placeholders
├── checkpoints/          # Empty model/checkpoint placeholder
└── outputs/              # Generated caches and experiment outputs
```



## Environment

Python 3.10 or newer is recommended. Create an isolated environment and install
the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for the local CUDA version if the default
wheel is not suitable.

Qwen3-VL checkpoints can be downloaded automatically from Hugging Face. A local
checkpoint can instead be placed under `checkpoints/` or supplied with
`--model_path`. Gated checkpoints, when used, require the corresponding model
provider's access approval.

## Data layout

Obtain each dataset from its official distributor and accept its applicable
terms. Arrange the files as follows:

```text
data/
├── TUAB/
│   └── edf/
│       ├── train/
│       └── eval/
├── TUEV/
│   └── edf/
│       ├── train/
│       └── eval/
└── vepiset/
    ├── MAT_Files/
    ├── Non-IED/
    ├── Generalized-IED/
    ├── Frontal-IED/
    ├── Temporal-IED/
    ├── Centro-Parietal-IED/
    └── Occipital-IED/
```

TUAB and TUEV should retain their official internal subdirectory and annotation
structure. vEpiSet `MAT_Files/` contains the subject-level MATLAB files used to
construct the subject mapping, and its class directories contain the
pre-segmented NumPy EEG arrays expected by the loader.

Paths default to the repository-local `data/`, `checkpoints/`, and
`outputs/` directories. They can be overridden without editing the source:

```bash
export EEG_VLM_DATA_DIR=/path/to/data
export EEG_VLM_CHECKPOINT_DIR=/path/to/checkpoints
export EEG_VLM_OUTPUT_DIR=/path/to/outputs
export QWEN3VL_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
```

## Training

The launchers default to the paper's combined waveform-montage and STFT image,
Qwen3-VL-4B, and LoRA settings:

```bash
bash scripts/train_tuab.sh
bash scripts/train_tuev.sh
bash scripts/train_vepiset.sh
```

Use the frozen-backbone prediction-head route with:

```bash
ADAPTATION=head bash scripts/train_tuab.sh
ADAPTATION=head bash scripts/train_tuev.sh
ADAPTATION=head bash scripts/train_vepiset.sh
```

For distributed training, set `NUM_PROCESSES` to the number of visible GPUs:

```bash
NUM_PROCESSES=8 bash scripts/train_tuev.sh
```

The Python entry points expose the visual-representation choices used in the
paper:

```text
waveform
spectrogram
cwt
combined
combined_cwt
```

Run an entry point with `--help` after its subcommand for all available
options, for example:

```bash
python src/tuev_vlm.py train_lora --help
python src/tuev_vlm_cls.py train_head --help
```

Generated manifests, render caches, adapters, prediction heads, logs, and
metrics are written under `outputs/` and are ignored by version control.

## Evaluation metrics

The paper-facing `metrics.csv` contains exactly the metrics reported in the
manuscript:

- TUAB and vEpiSet: balanced accuracy, AUPRC (average precision), and AUROC;
- TUEV: balanced accuracy, unweighted Cohen's kappa, and class-frequency-
  weighted F1.

Metrics are computed per model-input EEG window. Abnormal is the positive TUAB
class and IED is the positive vEpiSet class. Evaluation writes no additional
diagnostic metrics beyond the three measures reported for each task.

## Citation

If this implementation is useful in your work, please cite:

> Recasting Visual Language Models as EEG Foundation Models for Clinical Neurodiagnostics.

The complete bibliographic record can be added after publication.

## License

This code is released under the MIT License. Dataset and model checkpoints
remain subject to their original licenses and access terms.
