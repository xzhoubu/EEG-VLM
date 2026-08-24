

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch import nn

from vlm.distributed import gather_rows as _bounded_gather_rows


def broadcast_object(value, accelerator: Accelerator | None = None):
    payload = [value]
    if accelerator is not None:
        if accelerator.num_processes > 1 and dist.is_initialized():
            dist.broadcast_object_list(payload, src=0)
        return payload[0]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    return payload[0]


def save_json(path: str, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def distributed_average(total: float, count: int, accelerator: Accelerator) -> float:
    payload = torch.tensor([float(total), float(count)], device=accelerator.device, dtype=torch.float64)
    if accelerator.num_processes > 1 and dist.is_initialized():
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    return float(payload[0].item() / max(payload[1].item(), 1.0))


def gather_rows(local_rows: list[dict], accelerator: Accelerator | None = None,
                chunk_size: int = 4096) -> list[dict]:
    return _bounded_gather_rows(local_rows, accelerator=accelerator, chunk_size=chunk_size)


def move_model_inputs(model_inputs: dict[str, torch.Tensor],
                      device: torch.device | None) -> dict[str, torch.Tensor]:
    if device is None:
        return model_inputs
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in model_inputs.items()
    }


def get_qwen_hidden_size(model) -> int:

    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        model,
    ]
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        for nested_name in (None, "text_config", "language_config"):
            nested = config if nested_name is None else getattr(config, nested_name, None)
            for attribute in ("hidden_size", "n_embd", "d_model"):
                value = getattr(nested, attribute, None)
                if value is not None:
                    return int(value)
    raise RuntimeError("Could not infer the Qwen3-VL language hidden size.")


class FrozenQwen3VLClassifier(nn.Module):


    def __init__(self, backbone: nn.Module, hidden_size: int, num_classes: int,
                 dropout: float = 0.1, pooling_mode: str = "image_mean",
                 mlp_hidden_size: int = 1024, task_mode: str = "flat"):
        super().__init__()
        if task_mode != "flat":
            raise ValueError("The public EEG-VLM release supports flat task heads only.")
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()
        self.hidden_size = int(hidden_size)
        self.pooling_mode = str(pooling_mode)
        self.mlp_hidden_size = int(mlp_hidden_size)
        self.task_mode = "flat"

        if self.pooling_mode == "image_mean":
            feature_size = self.hidden_size
        elif self.pooling_mode in {"dual_branch_meanmax", "dual_image_meanmax"}:
            feature_size = self.hidden_size * 4
        else:
            raise ValueError(f"Unsupported pooling_mode: {self.pooling_mode}")

        self.norm = nn.LayerNorm(feature_size)
        self.dropout = nn.Dropout(dropout)
        if self.pooling_mode == "image_mean":
            self.classifier = nn.Linear(feature_size, num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(feature_size, self.mlp_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.mlp_hidden_size, num_classes),
            )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def _pool_image_tokens(hidden_states, mm_token_type_ids, attention_mask):
        image_mask = (mm_token_type_ids == 1) & attention_mask.bool()
        if not bool(image_mask.any().item()):
            image_mask = attention_mask.bool()
        weights = image_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _resolve_image_token_grid(self, image_grid_thw_row, num_image_tokens):
        merge = int(getattr(self.backbone.config.vision_config, "spatial_merge_size", 1) or 1)
        grid_h = max(int(image_grid_thw_row[1].item()) // merge, 1)
        grid_w = max(int(image_grid_thw_row[2].item()) // merge, 1)
        if grid_h * grid_w == int(num_image_tokens):
            return grid_h, grid_w
        probe_h = max(int(round(float(num_image_tokens) ** 0.5)), 1)
        while probe_h > 1 and num_image_tokens % probe_h != 0:
            probe_h -= 1
        return probe_h, max(int(num_image_tokens // probe_h), 1)

    def _pool_dual_branch_tokens(self, hidden_states, mm_token_type_ids,
                                 attention_mask, image_grid_thw):
        image_mask = (mm_token_type_ids == 1) & attention_mask.bool()
        rows = []
        for batch_index in range(hidden_states.size(0)):
            tokens = hidden_states[batch_index][image_mask[batch_index]]
            if tokens.numel() == 0:
                fallback = self._pool_image_tokens(
                    hidden_states[batch_index:batch_index + 1],
                    mm_token_type_ids[batch_index:batch_index + 1],
                    attention_mask[batch_index:batch_index + 1],
                )[0]
                rows.append(torch.cat([fallback] * 4, dim=0))
                continue
            grid_h, grid_w = self._resolve_image_token_grid(image_grid_thw[batch_index], tokens.size(0))
            tokens = tokens[:grid_h * grid_w].reshape(grid_h, grid_w, self.hidden_size)
            split = max(grid_w // 2, 1)
            left = tokens[:, :split].reshape(-1, self.hidden_size)
            right = tokens[:, split:].reshape(-1, self.hidden_size)
            if right.numel() == 0:
                right = left
            rows.append(torch.cat([
                left.mean(0), left.max(0).values,
                right.mean(0), right.max(0).values,
            ], dim=0))
        return torch.stack(rows)

    def _image_token_count(self, grid_row):
        merge = int(getattr(self.backbone.config.vision_config, "spatial_merge_size", 1) or 1)
        return max(int(grid_row[0].item()), 1) * max(int(grid_row[1].item()) // merge, 1) * max(int(grid_row[2].item()) // merge, 1)

    def _pool_dual_image_tokens(self, hidden_states, mm_token_type_ids,
                                attention_mask, image_grid_thw):
        image_mask = (mm_token_type_ids == 1) & attention_mask.bool()
        batch_size = hidden_states.size(0)
        images_per_sample = max(1, image_grid_thw.size(0) // max(batch_size, 1))
        rows = []
        for batch_index in range(batch_size):
            tokens = hidden_states[batch_index][image_mask[batch_index]]
            if tokens.numel() == 0:
                fallback = self._pool_image_tokens(
                    hidden_states[batch_index:batch_index + 1],
                    mm_token_type_ids[batch_index:batch_index + 1],
                    attention_mask[batch_index:batch_index + 1],
                )[0]
                rows.append(torch.cat([fallback] * 4, dim=0))
                continue
            grids = image_grid_thw[batch_index * images_per_sample:(batch_index + 1) * images_per_sample]
            counts = [self._image_token_count(row) for row in grids]
            if len(counts) < 2 or sum(counts[:2]) > tokens.size(0):
                first = max(tokens.size(0) // 2, 1)
                counts = [first, tokens.size(0) - first]
            first = tokens[:counts[0]]
            second = tokens[counts[0]:counts[0] + counts[1]]
            if second.numel() == 0:
                second = first
            rows.append(torch.cat([
                first.mean(0), first.max(0).values,
                second.mean(0), second.max(0).values,
            ], dim=0))
        return torch.stack(rows)

    def forward(self, input_ids, attention_mask, mm_token_type_ids,
                pixel_values, image_grid_thw, **kwargs):
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
            hidden = outputs.hidden_states[-1]
        if self.pooling_mode == "dual_branch_meanmax":
            features = self._pool_dual_branch_tokens(hidden, mm_token_type_ids, attention_mask, image_grid_thw)
        elif self.pooling_mode == "dual_image_meanmax":
            features = self._pool_dual_image_tokens(hidden, mm_token_type_ids, attention_mask, image_grid_thw)
        else:
            features = self._pool_image_tokens(hidden, mm_token_type_ids, attention_mask)
        logits = self.classifier(self.dropout(self.norm(features.float())))
        return {
            "task_mode": "flat",
            "class_logits": logits,
            "class_probs": torch.softmax(logits.float(), dim=-1),
        }
