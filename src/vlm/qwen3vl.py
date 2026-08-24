

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from accelerate.utils import gather_object
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import AutoModelForMultimodalLM
except ImportError:
    AutoModelForMultimodalLM = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

from vlm.data import DEFAULT_QWEN_PROMPT


QWEN3VL_LORA_TARGET_REGEX = (
    r'^(model\.language_model\.layers\.\d+\.'
    r'(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'
    r'|model\.visual\.(merger|deepstack_merger_list\.\d+)\.linear_fc[12])$'
)


def _resolve_attention_impl(enable_flash_attention: bool) -> str | None:
    if not enable_flash_attention:
        return None
    try:
        import flash_attn
    except ImportError:
        return None
    return 'flash_attention_2'


def _primary_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _move_to_device(inputs: dict[str, torch.Tensor], device: torch.device | None) -> dict[str, torch.Tensor]:
    if device is None:
        return inputs
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def _load_qwen3vl_model(model_path: str, **kwargs):
    loaders = [
        ('Qwen3VLForConditionalGeneration', Qwen3VLForConditionalGeneration),
        ('AutoModelForMultimodalLM', AutoModelForMultimodalLM),
        ('AutoModelForImageTextToText', AutoModelForImageTextToText),
    ]
    errors: list[str] = []
    for loader_name, loader_cls in loaders:
        if loader_cls is None:
            continue
        try:
            return loader_cls.from_pretrained(model_path, **kwargs)
        except Exception as exc:
            errors.append(f'{loader_name}: {type(exc).__name__}: {exc}')
    detail = '\n'.join(errors) if errors else 'No compatible Qwen3-VL loader is available in transformers.'
    raise RuntimeError(f'Failed to load Qwen3-VL model from {model_path!r}.\n{detail}')


def load_qwen3vl_for_inference(model_path: str,
                               use_bf16: bool = True,
                               device_map: str = 'auto',
                               gpu_id: int = 0,
                               enable_flash_attention: bool = False):

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    kwargs: dict[str, Any] = {'trust_remote_code': True}
    if torch.cuda.is_available():
        kwargs['torch_dtype'] = torch.bfloat16 if use_bf16 else torch.float16
    attn_impl = _resolve_attention_impl(enable_flash_attention)
    if attn_impl:
        kwargs['attn_implementation'] = attn_impl

    use_device_map = torch.cuda.is_available() and str(device_map).lower() not in {'', 'none'}
    if use_device_map:
        kwargs['device_map'] = device_map

    model = _load_qwen3vl_model(model_path, **kwargs)
    if not use_device_map and torch.cuda.is_available() and gpu_id >= 0:
        model = model.to(torch.device(f'cuda:{gpu_id}'))
    model.eval()
    return model, processor, _primary_device(model)


def load_qwen3vl_for_training(model_path: str,
                              use_bf16: bool = True,
                              enable_flash_attention: bool = False):

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    kwargs: dict[str, Any] = {'trust_remote_code': True}
    if torch.cuda.is_available():
        kwargs['torch_dtype'] = torch.bfloat16 if use_bf16 else torch.float16
    attn_impl = _resolve_attention_impl(enable_flash_attention)
    if attn_impl:
        kwargs['attn_implementation'] = attn_impl
    model = _load_qwen3vl_model(model_path, **kwargs)
    model.config.use_cache = False
    return model, processor


def _as_image_list(image: Image.Image | list[Image.Image] | tuple[Image.Image, ...] | None) -> list[Image.Image]:
    if image is None:
        return []
    if isinstance(image, (list, tuple)):
        return list(image)
    return [image]


def _has_images(images) -> bool:
    if images is None:
        return False
    if isinstance(images, (list, tuple)):
        return any(_has_images(item) for item in images)
    return True


def _processor_call(processor, texts: list[str], images, **kwargs):
    if _has_images(images):
        return processor(text=texts, images=images, **kwargs)
    return processor(text=texts, **kwargs)


def _build_user_message(prompt: str,
                        image: Image.Image | list[Image.Image] | tuple[Image.Image, ...] | None) -> list[dict[str, Any]]:
    image_content = [{'type': 'image', 'image': item} for item in _as_image_list(image)]
    return [{
        'role': 'user',
        'content': image_content + [{'type': 'text', 'text': prompt}],
    }]


def _build_supervised_messages(prompt: str,
                               image: Image.Image | list[Image.Image] | tuple[Image.Image, ...] | None,
                               answer_text: str) -> list[dict[str, Any]]:
    image_content = [{'type': 'image', 'image': item} for item in _as_image_list(image)]
    return [
        {
            'role': 'user',
            'content': image_content + [{'type': 'text', 'text': prompt}],
        },
        {
            'role': 'assistant',
            'content': answer_text,
        },
    ]


class Qwen3VLInferenceCollator:


    def __init__(self, processor, prompt: str = DEFAULT_QWEN_PROMPT):
        self.processor = processor
        self.prompt = prompt

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [item.get('images', item.get('image')) for item in batch]
        prompts = [str(item.get('prompt', self.prompt)) for item in batch]
        texts = [
            self.processor.apply_chat_template(
                _build_user_message(prompt, image),
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt, image in zip(prompts, images)
        ]
        model_inputs = _processor_call(self.processor, texts=texts, images=images, padding=True, return_tensors='pt')
        payload = {
            'model_inputs': model_inputs,
            'prompt_texts': texts,
            'prompts': prompts,
            'prompt': prompts[0] if len(set(prompts)) == 1 else '',
            'sample_id': [item['sample_id'] for item in batch],
            'split': [item['split'] for item in batch],
            'ground_truth': torch.tensor([item['label'] for item in batch], dtype=torch.long),
            'subject': [item.get('subject', '') for item in batch],
            'condition': [item.get('condition', item['label']) for item in batch],
            'sample_file': [item.get('sample_file', '') for item in batch],
            'image_path': [item.get('image_path', '') for item in batch],
            'image_paths': [item.get('image_paths', [item.get('image_path', '')]) for item in batch],
        }
        if _has_images(images):
            payload['images'] = images
        return payload


class Qwen3VLTrainCollator:


    def __init__(self,
                 processor,
                 prompt: str = DEFAULT_QWEN_PROMPT,
                 include_label_metadata: bool = False):
        self.processor = processor
        self.prompt = prompt
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.include_label_metadata = bool(include_label_metadata)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [item.get('images', item.get('image')) for item in batch]
        prompts = [str(item.get('prompt', self.prompt)) for item in batch]
        prompt_texts = [
            self.processor.apply_chat_template(
                _build_user_message(prompt, image),
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt, image in zip(prompts, images)
        ]
        full_texts = [
            self.processor.apply_chat_template(
                _build_supervised_messages(prompt, image, item['answer_text']),
                tokenize=False,
                add_generation_prompt=False,
            )
            for prompt, image, item in zip(prompts, images, batch)
        ]
        prompt_inputs = _processor_call(self.processor, texts=prompt_texts, images=images, padding=False)
        model_inputs = _processor_call(self.processor, texts=full_texts, images=images, padding=True, return_tensors='pt')
        labels = model_inputs['input_ids'].clone()
        labels[labels == self.pad_token_id] = -100
        for row_idx, prompt_ids in enumerate(prompt_inputs['input_ids']):
            labels[row_idx, :len(prompt_ids)] = -100
        model_inputs['labels'] = labels
        if self.include_label_metadata:
            model_inputs['_choice_labels'] = torch.tensor([item['label'] for item in batch], dtype=torch.long)
            model_inputs['_answer_positions'] = torch.tensor(
                [len(prompt_ids) for prompt_ids in prompt_inputs['input_ids']],
                dtype=torch.long,
            )
        return model_inputs


def _yes_no_token_ids(processor) -> tuple[int, int]:
    yes_ids = processor.tokenizer('yes', add_special_tokens=False).input_ids
    no_ids = processor.tokenizer('no', add_special_tokens=False).input_ids
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise RuntimeError(
            'This implementation expects "yes" and "no" to each map to a single token. '
            f'Observed yes={yes_ids}, no={no_ids}.'
        )
    return int(yes_ids[0]), int(no_ids[0])


def _candidate_token_ids(processor, candidates: list[str]) -> list[int]:
    token_ids = []
    observed = {}
    for candidate in candidates:
        ids = processor.tokenizer(str(candidate), add_special_tokens=False).input_ids
        observed[str(candidate)] = ids
        if len(ids) != 1:
            raise RuntimeError(
                'Candidate probability scoring expects each answer choice to map to one token. '
                f'Observed tokenization: {observed}.'
            )
        token_ids.append(int(ids[0]))
    return token_ids


def _candidate_first_token_ids(processor, candidates: list[str]) -> list[int]:
    token_ids = []
    observed = {}
    for candidate in candidates:
        ids = processor.tokenizer(str(candidate), add_special_tokens=False).input_ids
        observed[str(candidate)] = ids
        if not ids:
            raise RuntimeError(f'Candidate labels must tokenize to at least one token. Observed: {observed}.')
        token_ids.append(int(ids[0]))
    if len(set(token_ids)) != len(token_ids):
        raise RuntimeError(
            'First-token candidate scoring requires unique first tokens. '
            f'Observed first token ids: {dict(zip(candidates, token_ids))}.'
        )
    return token_ids


def score_yes_no_from_inputs(model, processor, model_inputs: dict[str, torch.Tensor], device: torch.device | None = None):

    yes_token_id, no_token_id = _yes_no_token_ids(processor)
    inputs = _move_to_device(model_inputs, device or _primary_device(model))
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        seq_idx = torch.arange(inputs['attention_mask'].size(1), device=logits.device).unsqueeze(0)
        positions = (inputs['attention_mask'].to(logits.device) * seq_idx).max(dim=1).values
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        next_token_logits = logits[batch_idx, positions]
        yes_no_logits = torch.stack([next_token_logits[:, yes_token_id], next_token_logits[:, no_token_id]], dim=-1)
        probs = torch.softmax(yes_no_logits.float(), dim=-1)
    return probs[:, 0].detach().cpu().numpy(), probs[:, 1].detach().cpu().numpy()


def score_candidates_from_inputs(model,
                                 processor,
                                 model_inputs: dict[str, torch.Tensor],
                                 candidates: list[str],
                                 device: torch.device | None = None) -> np.ndarray:

    candidate_ids = _candidate_token_ids(processor, candidates)
    inputs = _move_to_device(model_inputs, device or _primary_device(model))
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        seq_idx = torch.arange(inputs['attention_mask'].size(1), device=logits.device).unsqueeze(0)
        positions = (inputs['attention_mask'].to(logits.device) * seq_idx).max(dim=1).values
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        next_token_logits = logits[batch_idx, positions]
        candidate_logits = torch.stack(
            [next_token_logits[:, token_id] for token_id in candidate_ids],
            dim=-1,
        )
        probs = torch.softmax(candidate_logits.float(), dim=-1)
    return probs.detach().cpu().numpy()


def score_candidate_first_tokens_from_inputs(model,
                                             processor,
                                             model_inputs: dict[str, torch.Tensor],
                                             candidates: list[str],
                                             device: torch.device | None = None) -> np.ndarray:

    candidate_ids = _candidate_first_token_ids(processor, candidates)
    inputs = _move_to_device(model_inputs, device or _primary_device(model))
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        seq_idx = torch.arange(inputs['attention_mask'].size(1), device=logits.device).unsqueeze(0)
        positions = (inputs['attention_mask'].to(logits.device) * seq_idx).max(dim=1).values
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        next_token_logits = logits[batch_idx, positions]
        candidate_logits = torch.stack(
            [next_token_logits[:, token_id] for token_id in candidate_ids],
            dim=-1,
        )
        probs = torch.softmax(candidate_logits.float(), dim=-1)
    return probs.detach().cpu().numpy()


def score_candidate_sequences(model,
                              processor,
                              prompt: str | list[str] | tuple[str, ...],
                              images: list[Image.Image] | list[list[Image.Image]] | None,
                              candidates: list[str],
                              device: torch.device | None = None,
                              length_normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:

    if not candidates:
        raise ValueError('At least one candidate sequence is required.')

    if images is None:
        n_samples = len(prompt) if isinstance(prompt, (list, tuple)) else 1
        images = [None] * n_samples
    prompts = [str(item) for item in prompt] if isinstance(prompt, (list, tuple)) else [str(prompt)] * len(images)
    if len(prompts) != len(images):
        raise ValueError(f'Number of prompts ({len(prompts)}) and images ({len(images)}) must match.')

    prompt_texts = [
        processor.apply_chat_template(
            _build_user_message(sample_prompt, image),
            tokenize=False,
            add_generation_prompt=True,
        )
        for sample_prompt, image in zip(prompts, images)
    ]
    prompt_inputs = _processor_call(processor, texts=prompt_texts, images=images, padding=False)
    prompt_lengths = [len(ids) for ids in prompt_inputs['input_ids']]

    full_texts = []
    full_images = []
    owner_rows = []
    candidate_rows = []
    for sample_idx, (sample_prompt, image) in enumerate(zip(prompts, images)):
        for candidate_idx, candidate in enumerate(candidates):
            full_texts.append(
                processor.apply_chat_template(
                    _build_supervised_messages(sample_prompt, image, str(candidate)),
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            full_images.append(image)
            owner_rows.append(sample_idx)
            candidate_rows.append(candidate_idx)

    inputs = _processor_call(processor, texts=full_texts, images=full_images, padding=True, return_tensors='pt')
    inputs = _move_to_device(inputs, device or _primary_device(model))
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    scores = torch.full(
        (len(images), len(candidates)),
        fill_value=-float('inf'),
        dtype=torch.float32,
        device=logits.device,
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    for row_idx, (sample_idx, candidate_idx) in enumerate(zip(owner_rows, candidate_rows)):
        prompt_len = int(prompt_lengths[sample_idx])
        valid_positions = torch.nonzero(attention_mask[row_idx], as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            continue
        full_len = int(valid_positions[-1].item()) + 1
        if full_len <= prompt_len:
            continue
        target_positions = torch.arange(prompt_len, full_len, device=logits.device)
        pred_positions = target_positions - 1
        target_ids = input_ids[row_idx, target_positions]
        token_logits = logits[row_idx, pred_positions].float()
        target_logits = token_logits.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        token_scores = target_logits - torch.logsumexp(token_logits, dim=-1)
        score = token_scores.mean() if length_normalize else token_scores.sum()
        scores[sample_idx, candidate_idx] = score

    probs = torch.softmax(scores, dim=-1)
    return probs.detach().cpu().numpy(), scores.detach().cpu().numpy()


def generate_from_inputs(model,
                         processor,
                         model_inputs: dict[str, torch.Tensor],
                         max_new_tokens: int = 4,
                         device: torch.device | None = None) -> list[str]:
    inputs = _move_to_device(model_inputs, device or _primary_device(model))
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    prompt_len = inputs['input_ids'].shape[1]
    texts = []
    for row_idx in range(output_ids.size(0)):
        new_tokens = output_ids[row_idx, prompt_len:]
        texts.append(processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return texts


def parse_yes_no_answer(text: str) -> str:

    normalized = re.sub(r'[^a-z]+', ' ', str(text).lower()).strip()
    if not normalized:
        return 'unknown'
    first = normalized.split()[0]
    if first in {'yes', 'no'}:
        return first
    if 'yes' in normalized and 'no' not in normalized:
        return 'yes'
    if 'no' in normalized and 'yes' not in normalized:
        return 'no'
    return 'unknown'


def parse_choice_answer(text: str, choices: list[str], class_names: list[str] | None = None) -> str:

    normalized = re.sub(r'[^a-z0-9]+', ' ', str(text).lower()).strip()
    if not normalized:
        return 'unknown'

    choice_set = {str(choice).lower(): str(choice) for choice in choices}
    first = normalized.split()[0]
    if first in choice_set:
        return choice_set[first]

    for token in normalized.split():
        if token in choice_set:
            return choice_set[token]

    if class_names:
        class_lookup = {
            re.sub(r'[^a-z0-9]+', ' ', name.lower()).strip(): choices[idx]
            for idx, name in enumerate(class_names)
        }
        for name, choice in class_lookup.items():
            if name and name in normalized:
                return str(choice)
    return 'unknown'


def collect_predictions(model,
                        processor,
                        dataloader,
                        device: torch.device | None = None,
                        max_new_tokens: int = 4,
                        save_generated_text: bool = True,
                        accelerator=None) -> pd.DataFrame:

    local_rows = []

    def _to_int(value) -> int:
        if torch.is_tensor(value):
            return int(value.detach().cpu().item())
        return int(value)

    for batch in dataloader:
        probs_yes, probs_no = score_yes_no_from_inputs(model, processor, batch['model_inputs'], device=device)
        if save_generated_text:
            generated = generate_from_inputs(
                model,
                processor,
                batch['model_inputs'],
                max_new_tokens=max_new_tokens,
                device=device,
            )
        else:
            generated = [''] * len(batch['sample_id'])

        for idx, sample_id in enumerate(batch['sample_id']):
            text = generated[idx]
            image_paths = batch['image_paths'][idx] if 'image_paths' in batch else [batch['image_path'][idx]]
            local_rows.append({
                'sample_id': str(sample_id),
                'split': str(batch['split'][idx]),
                'subject': str(batch['subject'][idx]),
                'condition': _to_int(batch['condition'][idx]),
                'sample_file': str(batch['sample_file'][idx]),
                'image_path': str(batch['image_path'][idx]),
                'image_paths': str(image_paths),
                'ground_truth': _to_int(batch['ground_truth'][idx]),
                'prob_yes': float(probs_yes[idx]),
                'prob_no': float(probs_no[idx]),
                'prob_positive': float(probs_yes[idx]),
                'generated_text': text,
                'parsed_answer': parse_yes_no_answer(text),
            })

    if accelerator is None:
        return pd.DataFrame(local_rows)

    gathered = gather_object(local_rows)
    if not accelerator.is_main_process:
        return pd.DataFrame()
    flat_rows = []
    for chunk in gathered:
        if isinstance(chunk, list):
            flat_rows.extend(chunk)
        elif chunk:
            flat_rows.append(chunk)
    return pd.DataFrame(flat_rows)


def collect_choice_predictions(model,
                               processor,
                               dataloader,
                               choices: list[str],
                               class_names: list[str],
                               device: torch.device | None = None,
                               max_new_tokens: int = 4,
                               save_generated_text: bool = True,
                               accelerator=None,
                               sequence_scoring: bool = True) -> pd.DataFrame:

    local_rows = []

    def _to_int(value) -> int:
        if torch.is_tensor(value):
            return int(value.detach().cpu().item())
        return int(value)

    for batch in dataloader:
        if bool(sequence_scoring) and ('prompts' in batch or 'images' in batch):
            probs, log_probs = score_candidate_sequences(
                model,
                processor,
                prompt=batch.get('prompts', str(batch.get('prompt', ''))),
                images=batch.get('images'),
                candidates=choices,
                device=device,
                length_normalize=True,
            )
        else:
            probs = score_candidate_first_tokens_from_inputs(
                model,
                processor,
                batch['model_inputs'],
                candidates=choices,
                device=device,
            )
            log_probs = np.log(np.clip(probs, 1e-12, 1.0))
        if save_generated_text:
            generated = generate_from_inputs(
                model,
                processor,
                batch['model_inputs'],
                max_new_tokens=max_new_tokens,
                device=device,
            )
        else:
            generated = [''] * len(batch['sample_id'])

        for idx, sample_id in enumerate(batch['sample_id']):
            prediction = int(np.argmax(probs[idx]))
            text = generated[idx]
            image_paths = batch['image_paths'][idx] if 'image_paths' in batch else [batch['image_path'][idx]]
            row = {
                'sample_id': str(sample_id),
                'split': str(batch['split'][idx]),
                'subject': str(batch['subject'][idx]),
                'condition': _to_int(batch['condition'][idx]),
                'sample_file': str(batch['sample_file'][idx]),
                'image_path': str(batch['image_path'][idx]),
                'image_paths': str(image_paths),
                'ground_truth': _to_int(batch['ground_truth'][idx]),
                'prediction': prediction,
                'predicted_choice': str(choices[prediction]),
                'predicted_class': str(class_names[prediction]),
                'generated_text': text,
                'parsed_answer': parse_choice_answer(text, choices=choices, class_names=class_names),
            }
            for choice_idx, choice in enumerate(choices):
                class_slug = re.sub(r'[^a-z0-9]+', '_', class_names[choice_idx].lower()).strip('_')
                choice_slug = re.sub(r'[^a-z0-9]+', '_', str(choice).lower()).strip('_')
                row[f'prob_{choice_slug}'] = float(probs[idx, choice_idx])
                row[f'prob_{class_slug}'] = float(probs[idx, choice_idx])
                row[f'logprob_{class_slug}'] = float(log_probs[idx, choice_idx])
            local_rows.append(row)

    if accelerator is None:
        return pd.DataFrame(local_rows)

    gathered = gather_object(local_rows)
    if not accelerator.is_main_process:
        return pd.DataFrame()
    flat_rows = []
    for chunk in gathered:
        if isinstance(chunk, list):
            flat_rows.extend(chunk)
        elif chunk:
            flat_rows.append(chunk)
    return pd.DataFrame(flat_rows)


def apply_lora_adapters(model, args):

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            'LoRA training requires the `peft` package in the active environment.'
        ) from exc

    if getattr(args, 'gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=int(getattr(args, 'lora_rank', 16)),
        lora_alpha=int(getattr(args, 'lora_alpha', 32)),
        lora_dropout=float(getattr(args, 'lora_dropout', 0.05)),
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=QWEN3VL_LORA_TARGET_REGEX,
    )
    peft_model = get_peft_model(model, lora_config)
    if int(getattr(args, 'freeze_vision_tower', 1)):
        for name, param in peft_model.named_parameters():
            if name.startswith('base_model.model.model.visual.blocks.'):
                param.requires_grad = False
    return peft_model
