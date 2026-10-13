"""Image-conditioned identity training loop backed by the real Z-Image model.

This script gives a practical minibatch sampler + training loop for:

- one-image conditioned samples (source_count=1)
- two-image conditioned samples (source_count=2)
- mixed batches that combine both conditions across many identities

Training uses the full Z-Image stack (transformer, vae, tokenizer, text encoder, scheduler)
loaded from disk via the same local layout used by inference scripts.
"""

from __future__ import annotations

import argparse
import math
import random
import os
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
from transformers import PreTrainedTokenizerBase, PreTrainedModel

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

    class _NoopTqdm:
        def __init__(self, iterable=None, total: int | None = None, *_, **__):
            self.iterable = iterable
            self.total = 0 if total is None else total

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, *_, **__):
            return None

        def update(self, *_, **__):
            return None

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    def _noop_tqdm(iterable=None, *args, **kwargs):
        return _NoopTqdm(iterable=iterable, total=kwargs.get("total", None))


if tqdm is None:
    tqdm = _noop_tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
ENV_PATH = Path(__file__).resolve().parent / ".env"

from identity_dataset.identity_pair_dataset import (
    build_identity_pairs,
    IdentityPairDataset,
    identity_collate_fn,
)
from utils import ensure_model_weights, load_from_local_dir


def _read_dotenv_value(dotenv_path: Path, key: str) -> Optional[str]:
    if not dotenv_path.exists():
        return None
    try:
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            k, v = stripped.split("=", 1)
            if k.strip() == key:
                return v.strip().strip("\"' ")
    except OSError:
        return None
    return None


def _default_data_root() -> Optional[Path]:
    env_value = (
        os.getenv("IDENTITY_TRAIN_DATA_ROOT")
        or os.getenv("ZIMAGE_IDENTITY_DATA_ROOT")
        or _read_dotenv_value(ENV_PATH, "IDENTITY_TRAIN_DATA_ROOT")
        or _read_dotenv_value(ENV_PATH, "ZIMAGE_IDENTITY_DATA_ROOT")
    )
    if not env_value:
        return None
    return Path(env_value).expanduser().resolve()


def _env_or_default_str(key: str, default: str) -> str:
    value = os.getenv(key)
    if value is None or value == "":
        value = _read_dotenv_value(ENV_PATH, key)
    if value is None or value == "":
        return default
    return value.strip().strip("\"' ")


def _env_or_default_int(key: str, default: int) -> int:
    value = _env_or_default_str(key, str(default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_or_default_optional_int(key: str, default: Optional[int]) -> Optional[int]:
    value = _env_or_default_str(key, "")
    if value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_or_default_float(key: str, default: float) -> float:
    value = _env_or_default_str(key, str(default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_or_default_torch_dtype(key: str, default: torch.dtype) -> torch.dtype:
    value = _env_or_default_str(key, str(default).replace("torch.", ""))
    value = value.lower().replace(" ", "")
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"float32", "fp32", "float"}:
        return torch.float32
    return default


def _torch_dtype_from_name(value: str, default: torch.dtype) -> torch.dtype:
    value = (value or "").strip().lower().replace("torch.", "")
    value = value.replace(" ", "")
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"float32", "fp32", "float"}:
        return torch.float32
    return default


def _env_or_default_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None or value == "":
        value = _read_dotenv_value(ENV_PATH, key)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


class IdentityConditionBatchSampler(Sampler[List[int]]):
    """Yield mini-batch index lists with controlled one/two-image source conditions."""

    def __init__(
        self,
        pairs: Sequence,
        batch_size: int,
        mode: str = "mixed",
        source_counts: Sequence[int] = (1, 2),
        source_weights: Optional[Sequence[float]] = None,
        min_identities_per_batch: int = 2,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        assert mode in {"one", "two", "mixed"}, "mode must be one|two|mixed"
        self.pairs = list(pairs)
        self.batch_size = batch_size
        self.mode = mode
        self.source_counts = tuple(sorted(set(int(x) for x in source_counts if x > 0)))
        if source_weights is None:
            self.source_weights = [1.0 for _ in self.source_counts]
        else:
            if len(source_weights) != len(self.source_counts):
                raise ValueError("source_weights length must match source_counts length.")
            self.source_weights = list(source_weights)
        self.min_identities_per_batch = max(1, min_identities_per_batch)
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        self._person_counts_by_source = self._index_pairs(pairs=self.pairs)
        self._rng = random.Random(seed)

        if self.mode == "one":
            if 1 not in self.source_counts:
                raise ValueError("one-image mode requires source_counts to include 1.")
        elif self.mode == "two":
            if 2 not in self.source_counts:
                raise ValueError("two-image mode requires source_counts to include 2.")

    def _index_pairs(self, pairs: Sequence) -> Dict[int, Dict[str, List[int]]]:
        """
        person-indexed source-count buckets:
        condition_count -> person_id -> [dataset_index...]
        """
        buckets: Dict[int, Dict[str, List[int]]] = {
            c: defaultdict(list) for c in self.source_counts
        }
        for idx, pair in enumerate(pairs):
            source_count = len(pair.source_paths)
            person = pair.person
            if source_count in buckets:
                buckets[source_count][person].append(idx)
        return buckets

    def __iter__(self) -> Iterable[List[int]]:
        # Snapshot index pools for epoch (allows fair-ish shuffling each epoch)
        pool: Dict[int, Dict[str, List[int]]] = {
            c: {person: list(indices) for person, indices in persons.items()}
            for c, persons in self._person_counts_by_source.items()
        }
        for persons in pool.values():
            for person in persons:
                if self.shuffle:
                    self._rng.shuffle(persons[person])

        if self.mode == "one":
            yield from self._iter_single_condition(pool=pool, source_count=1)
        elif self.mode == "two":
            yield from self._iter_single_condition(pool=pool, source_count=2)
        else:
            yield from self._iter_mixed(pool=pool)

    @staticmethod
    def _any_left(pool: Dict[int, Dict[str, List[int]]], source_count: int) -> bool:
        return any(len(v) > 0 for v in pool.get(source_count, {}).values())

    @classmethod
    def _pop_any_for_count(
        cls, pool: Dict[int, Dict[str, List[int]]], source_count: int
    ) -> Optional[Tuple[int, str]]:
        persons = pool.get(source_count, {})
        for person, ids in list(persons.items()):
            if ids:
                return ids.pop(), person
        return None

    @classmethod
    def _pop_for_person_count(
        cls,
        pool: Dict[int, Dict[str, List[int]]],
        source_count: int,
        person: str,
    ) -> Optional[int]:
        ids = pool.get(source_count, {}).get(person, [])
        if ids:
            return ids.pop()
        return None

    def _iter_single_condition(self, pool: Dict[int, Dict[str, List[int]]], source_count: int):
        indices: List[int] = []
        persons = pool.get(source_count, {})
        for _, ids in persons.items():
            indices.extend(ids)
        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            if batch:
                yield batch

    def _iter_mixed(self, pool: Dict[int, Dict[str, List[int]]]):
        counts = list(self.source_counts)
        weights = self.source_weights
        max_steps = self._estimate_steps(pool=pool)
        if max_steps <= 0:
            return

        for _ in range(max_steps):
            batch: List[int] = []
            if not any(self._any_left(pool, c) for c in counts):
                break

            # Preselect identities to ensure diverse minibatches (multiple people)
            people = {
                person
                for c in counts
                for person, ids in pool.get(c, {}).items()
                if ids
            }
            if not people:
                break
            identity_target = min(self.min_identities_per_batch, len(people), self.batch_size)
            selected_people = self._rng.sample(sorted(people), k=identity_target)

            for slot in range(self.batch_size):
                available_counts = [c for c in counts if self._any_left(pool, c)]
                if not available_counts:
                    break

                # Weighted source-count choice for this slot
                local_counts = [counts[i] for i, c in enumerate(counts) if c in available_counts]
                local_weights = [
                    float(weights[i]) for i, c in enumerate(counts) if c in available_counts
                ]
                if not local_counts:
                    break
                chosen_count = random.choices(local_counts, weights=local_weights, k=1)[0]

                # Try to pull from one of the selected identities first
                if selected_people:
                    start_person = selected_people[slot % len(selected_people)]
                    person_candidates = [start_person] + [
                        p for p in selected_people if p != start_person
                    ]
                else:
                    person_candidates = []
                chosen_idx: Optional[int] = None

                for candidate in person_candidates:
                    idx = self._pop_for_person_count(pool, chosen_count, candidate)
                    if idx is not None:
                        chosen_idx = idx
                        break

                if chosen_idx is None:
                    # fallback: any available person for chosen count
                    pop_result = self._pop_any_for_count(pool, chosen_count)
                    if pop_result is not None:
                        chosen_idx, _ = pop_result

                if chosen_idx is None:
                    continue
                batch.append(chosen_idx)

            if not batch:
                break
            if self.drop_last and len(batch) < self.batch_size:
                break
            if batch:
                yield batch

    def _estimate_steps(self, pool: Dict[int, Dict[str, List[int]]]) -> int:
        total = sum(len(ids) for buckets in pool.values() for ids in buckets.values())
        if total == 0:
            return 0
        return math.ceil(total / self.batch_size)

    def __len__(self) -> int:
        total = sum(len(ids) for buckets in self._person_counts_by_source.values() for ids in buckets.values())
        return math.ceil(total / self.batch_size)


def _to_dict(
    batch,
    device: torch.device,
    max_sources: int,
    non_blocking: bool = False,
) -> Dict[str, object]:
    """
    Stack variable-count source tensors into [B, K, C, H, W].
    `K` is max number of source images in the batch.
    """
    source_count = max_sources
    target = torch.stack([t.to(device, non_blocking=non_blocking) for t in batch["target"]], dim=0)
    b = len(batch["target"])
    c, h, w = target.shape[1], target.shape[2], target.shape[3]

    src = torch.zeros((b, source_count, c, h, w), dtype=target.dtype, device=device)
    src_mask = torch.zeros((b, source_count), dtype=torch.bool, device=device)

    for i, samples in enumerate(batch["sources"]):
        for j, image in enumerate(samples):
            if j >= source_count:
                break
            src[i, j] = image.to(device, non_blocking=non_blocking)
            src_mask[i, j] = True

    return {
        "person": batch["person"],
        "prompt_text": [Path(p).name for p in batch["person"]],
        "source_count": torch.as_tensor(batch["source_count"], device=device),
        "source_images": src,
        "source_mask": src_mask,
        "target": target,
    }


def _collect_prefixes(prefixes_raw: str) -> List[str]:
    return [item.strip() for item in (prefixes_raw or "").split(",") if item.strip()]


def _select_state_slice(state: dict, prefixes: Sequence[str]) -> dict:
    if not prefixes:
        return state
    return {k: v for k, v in state.items() if any(k.startswith(prefix) for prefix in prefixes)}


def _move_tensors_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_tensors_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_tensors_to_cpu(v) for v in obj)
    return obj


def _safe_torch_save(payload: dict, path: Path, *, fallback_no_zip: bool = True) -> None:
    tmp_path = Path(str(path) + ".tmp")
    last_error: Exception | None = None
    try:
        with tmp_path.open("wb") as f:
            torch.save(payload, f, _use_new_zipfile_serialization=True)
        tmp_path.replace(path)
        return
    except Exception as e:  # pragma: no cover - error-path fallback
        last_error = e
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        if not fallback_no_zip:
            raise

    if fallback_no_zip:
        try:
            with tmp_path.open("wb") as f:
                torch.save(payload, f, _use_new_zipfile_serialization=False)
            tmp_path.replace(path)
            return
        except Exception as e:  # pragma: no cover - final fallback path
            last_error = e
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    raise RuntimeError(f"Unable to write checkpoint to {path}") from last_error


def _safe_loss_inputs(prediction: torch.Tensor, target: torch.Tensor, mode: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if mode == "replace":
        return (
            torch.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0),
        )
    if mode == "skip":
        return None
    if mode == "raise":
        return prediction, target
    return None


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _safe_apply_chat_template(tokenizer: PreTrainedTokenizerBase, prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            enable_thinking=True,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)


class ZImageTrainer(nn.Module):
    """
    Identity-conditioned training wrapper around the real Z-Image stack.

    Inputs:
      source_images: [B, K, 3, H, W]
      source_mask: [B, K] boolean mask of active source slots
      target: [B, 3, H, W]
      prompt_texts: list[str] prompt conditioning per sample

    Output:
      predicted_noise: [B, 4, H, W] in latent space
      target_noise: [B, 4, H, W] used for flow-matching target
    """

    def __init__(
        self,
        model_dir: Path,
        device: torch.device,
        dtype: torch.dtype,
        max_sequence_length: int = 77,
        compile_model: bool = False,
        freeze_text_encoder: bool = True,
        freeze_vae: bool = True,
        num_train_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        model_dir = Path(model_dir).expanduser().resolve()
        model_dir = Path(ensure_model_weights(str(model_dir), verify=False))
        components = load_from_local_dir(
            model_dir=model_dir,
            device="cpu" if device.type == "cpu" else device.type,
            dtype=dtype,
            verbose=False,
            compile=compile_model,
        )
        self.transformer: nn.Module = components["transformer"]
        self.vae: nn.Module = components["vae"]
        self.text_encoder: PreTrainedModel = components["text_encoder"]
        self.tokenizer: PreTrainedTokenizerBase = components["tokenizer"]
        self.scheduler = components["scheduler"]

        self.device = device
        self.dtype = dtype
        self.max_sequence_length = max_sequence_length
        self.train_steps = 0

        # Keep tokenizer/text_encoder/vae in inference mode for memory-stable fine-tuning.
        for module in (self.text_encoder, self.vae):
            module.eval()
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        if freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad = False

        if hasattr(self.vae, "config") and hasattr(self.vae.config, "scaling_factor"):
            self.latent_scaling_factor = float(self.vae.config.scaling_factor)
        else:
            self.latent_scaling_factor = 1.0

        # Ensure scheduler has a stable timestep buffer for training.
        num_train_steps = int(num_train_timesteps or self.scheduler.config.get("num_train_timesteps", 1000))
        self.scheduler.num_train_timesteps = num_train_steps
        self.scheduler.config.num_train_timesteps = num_train_steps
        self.scheduler.set_timesteps(num_train_steps, device=device)

        self.to(device)
        try:
            self.compute_dtype = next(self.transformer.parameters()).dtype
        except StopIteration:
            self.compute_dtype = self.dtype

    def _encode_images_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB images in [0,1] to VAE latents in the latent diffusion scale.
        """
        # VAE expects roughly [-1,1] in many implementations.
        rgb = torch.clamp(images * 2.0 - 1.0, -1.0, 1.0)
        with torch.no_grad():
            moments = self.vae.encoder(rgb)
            if self.vae.quant_conv is not None:
                moments = self.vae.quant_conv(moments)
        if moments.shape[1] >= 2 * getattr(self.vae.config, "latent_channels", moments.shape[1] // 2):
            moments = moments[:, : moments.shape[1] // 2, ...]
        return moments * self.latent_scaling_factor

    def _prompt_embeddings(self, prompt_texts: Sequence[str]) -> List[torch.Tensor]:
        formatted = [_safe_apply_chat_template(self.tokenizer, str(p)) for p in prompt_texts]
        text_inputs = self.tokenizer(
            formatted,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device).bool()

        with torch.no_grad():
            hidden_states = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            ).hidden_states[-2]

        prompt_embeds: List[torch.Tensor] = []
        for i in range(len(hidden_states)):
            prompt_embeds.append(
                hidden_states[i][attention_mask[i]].to(device=self.device, dtype=self.compute_dtype)
            )
        return prompt_embeds

    def forward(
        self,
        source_images: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        prompt_texts: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build conditioning latents and predict flow-matching noise target for the target latent.
        """
        prompt_embeds = self._prompt_embeddings(prompt_texts)

        b = source_images.shape[0]
        target_latent = self._encode_images_to_latents(target).to(device=self.device, dtype=self.compute_dtype)  # [B, 4, h, w]
        latent_ref_shape = target_latent.shape

        # Precompute source latents per sample to keep source-cardinality variable support.
        source_latents: List[torch.Tensor] = []
        for i in range(b):
            source_count = int(source_mask[i].sum().item())
            if source_count <= 0:
                source_latents.append(
                    torch.zeros(
                        (latent_ref_shape[1], 1, latent_ref_shape[2], latent_ref_shape[3]),
                        device=self.device,
                        dtype=self.compute_dtype,
                    )
                )
                continue
            src = source_images[i, :source_count]  # [S, 3, H, W]
            src_latent = self._encode_images_to_latents(src).permute(1, 0, 2, 3).to(
                device=self.device, dtype=self.compute_dtype
            )  # [4, S, h, w]
            source_latents.append(src_latent)

        # Randomly pick sigma/time index per sample.
        total_steps = max(1, len(self.scheduler.timesteps) - 1)
        timesteps = self.scheduler.timesteps.to(self.device)
        step_idx = torch.randint(low=0, high=total_steps, size=(b,), device=self.device)
        t = timesteps[step_idx]
        sigma = (t / self.scheduler.config.num_train_timesteps).to(dtype=self.compute_dtype).view(b, 1, 1, 1)
        target_noise = torch.randn_like(target_latent)
        target_noisy = target_latent + sigma * target_noise

        # Model input convention follows generate() from zimage/pipeline.py:
        # latent_model_input_list are image-tensor lists with shape [C, F, H, W].
        latent_model_inputs = []
        for i in range(b):
            target_i = target_noisy[i : i + 1].permute(1, 0, 2, 3)  # [4, 1, h, w]
            latent_model_inputs.append(torch.cat([source_latents[i], target_i], dim=1))
        step_input = ((self.scheduler.config.num_train_timesteps - t) / self.scheduler.config.num_train_timesteps).to(
            dtype=self.compute_dtype
        )

        noise_pred_list = self.transformer(latent_model_inputs, step_input, prompt_embeds)[0]
        predicted_noise = torch.stack(
            [out[:, -1].to(target_noise.dtype) for out in noise_pred_list],
            dim=0,
        )
        return predicted_noise, target_noise


def build_model(
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    compile_model: bool = False,
    max_sequence_length: int = 77,
    num_train_timesteps: int = 1000,
) -> nn.Module:
    if model_path is None:
        model_path = _env_or_default_str("IDENTITY_TRAIN_MODEL_PATH", _env_or_default_str("ZIMAGE_MODEL_PATH", "./models/Z-Image-Turbo"))
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.bfloat16
    return ZImageTrainer(
        model_dir=Path(model_path),
        device=device,
        dtype=dtype,
        max_sequence_length=max_sequence_length,
        compile_model=compile_model,
        num_train_timesteps=num_train_timesteps,
    )

def compute_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Flow-matching noise prediction MSE in latent space."""
    return torch.nn.functional.mse_loss(prediction, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train image-conditioned identity pairs with the real Z-Image model."
    )
    parser.add_argument(
        "--model-path",
        default=_env_or_default_str("IDENTITY_TRAIN_MODEL_PATH", _env_or_default_str("ZIMAGE_MODEL_PATH", "./models/Z-Image-Turbo")),
        help=(
            "Z-Image model root (transformer/vae/text_encoder/tokenizer/scheduler). "
            "Defaults to IDENTITY_TRAIN_MODEL_PATH or ZIMAGE_MODEL_PATH."
        ),
    )
    parser.add_argument(
        "--model-dtype",
        default=_dtype_name(_env_or_default_torch_dtype("IDENTITY_TRAIN_DTYPE", torch.bfloat16)),
        choices=["bf16", "bfloat16", "fp16", "float16", "float32", "fp32"],
        help="Model dtype (env: IDENTITY_TRAIN_DTYPE).",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        default=_env_or_default_bool("IDENTITY_TRAIN_COMPILE_MODEL", False),
        help="Compile heavy modules with torch.compile where supported (env: IDENTITY_TRAIN_COMPILE_MODEL).",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=_env_or_default_int("IDENTITY_TRAIN_MAX_SEQUENCE_LENGTH", 256),
        help="Tokenizer truncation length (env: IDENTITY_TRAIN_MAX_SEQUENCE_LENGTH).",
    )
    parser.add_argument(
        "--scheduler-train-steps",
        type=int,
        default=_env_or_default_int("IDENTITY_TRAIN_SCHEDULER_STEPS", 1000),
        help="Scheduler num_train_timesteps for training noise sampling.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Root folder containing page/person/image folders. "
            "Falls back to IDENTITY_TRAIN_DATA_ROOT / ZIMAGE_IDENTITY_DATA_ROOT "
            "env var (or .env) if set."
        ),
    )
    parser.add_argument(
        "--source-counts",
        default=_env_or_default_str("IDENTITY_TRAIN_SOURCE_COUNTS", "1,2"),
        help="Pairs source-count modes (default: env key IDENTITY_TRAIN_SOURCE_COUNTS).",
    )
    parser.add_argument(
        "--source-counts-one-weight",
        type=float,
        default=_env_or_default_float("IDENTITY_TRAIN_SOURCE_COUNTS_ONE_WEIGHT", 1.0),
        help="Sampling weight for source_count=1 samples (env: IDENTITY_TRAIN_SOURCE_COUNTS_ONE_WEIGHT).",
    )
    parser.add_argument(
        "--source-counts-two-weight",
        type=float,
        default=_env_or_default_float("IDENTITY_TRAIN_SOURCE_COUNTS_TWO_WEIGHT", 1.0),
        help="Sampling weight for source_count=2 samples (env: IDENTITY_TRAIN_SOURCE_COUNTS_TWO_WEIGHT).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=_env_or_default_int("IDENTITY_TRAIN_IMAGE_SIZE", 768),
        help="Training image square size (env: IDENTITY_TRAIN_IMAGE_SIZE).",
    )
    parser.add_argument(
        "--max-pairs-per-person",
        type=int,
        default=_env_or_default_int("IDENTITY_TRAIN_MAX_PAIRS_PER_PERSON", 2000),
        help="Per-person pair cap (env: IDENTITY_TRAIN_MAX_PAIRS_PER_PERSON).",
    )
    parser.add_argument(
        "--max-total-pairs",
        type=int,
        default=_env_or_default_optional_int("IDENTITY_TRAIN_MAX_TOTAL_PAIRS", None),
        help="Global cap for total identity pairs (env: IDENTITY_TRAIN_MAX_TOTAL_PAIRS).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_env_or_default_int("IDENTITY_TRAIN_SEED", 42),
        help="Random seed (env: IDENTITY_TRAIN_SEED).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_env_or_default_int("IDENTITY_TRAIN_BATCH_SIZE", 2),
        help="Batch size for identity pair batches (env: IDENTITY_TRAIN_BATCH_SIZE).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=_env_or_default_int("IDENTITY_TRAIN_EPOCHS", 1),
        help="Number of training epochs (env: IDENTITY_TRAIN_EPOCHS).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=_env_or_default_float("IDENTITY_TRAIN_LR", 1e-4),
        help="Learning rate (env: IDENTITY_TRAIN_LR).",
    )
    parser.add_argument(
        "--min-identities-per-batch",
        type=int,
        default=_env_or_default_int("IDENTITY_TRAIN_MIN_IDENTITIES_PER_BATCH", 2),
        help="Minimum distinct identities per mixed batch (env: IDENTITY_TRAIN_MIN_IDENTITIES_PER_BATCH).",
    )
    parser.add_argument(
        "--mode",
        choices=["one", "two", "mixed"],
        default=_env_or_default_str("IDENTITY_TRAIN_MODE", "mixed"),
        help="one=all source_count=1, two=all source_count=2, mixed=mixed sampling",
    )
    parser.add_argument(
        "--drop-last",
        action="store_true",
        default=_env_or_default_bool("IDENTITY_TRAIN_DROP_LAST", False),
        help="Drop last incomplete batch (env: IDENTITY_TRAIN_DROP_LAST).",
    )
    parser.add_argument(
        "--optimizer",
        default=_env_or_default_str("IDENTITY_TRAIN_OPTIMIZER", "adamw"),
        choices=["adamw", "sgd"],
        help=(
            "Optimizer type used for training (env: IDENTITY_TRAIN_OPTIMIZER). "
            "Use sgd for very-low-memory smoke-test runs."
        ),
    )
    parser.add_argument(
        "--optimizer-momentum",
        type=float,
        default=_env_or_default_float("IDENTITY_TRAIN_OPTIMIZER_MOMENTUM", 0.0),
        help="Momentum for SGD (env: IDENTITY_TRAIN_OPTIMIZER_MOMENTUM).",
    )
    parser.add_argument(
        "--trainable-prefixes",
        default=_env_or_default_str("IDENTITY_TRAIN_TRAINABLE_PREFIXES", ""),
        help=(
            "Comma-separated parameter name prefixes to keep trainable. "
            "If set, every other parameter is frozen. Empty means keep default freeze behavior."
        ),
    )
    parser.add_argument(
        "--checkpoint-scope",
        default=_env_or_default_str("IDENTITY_TRAIN_CHECKPOINT_SCOPE", "auto"),
        choices=["auto", "full", "trainable"],
        help=(
            "Which parameters to include in checkpoints: "
            "'full' = model full state_dict, 'trainable' = only parameters matching trainable-prefixes, "
            "'auto' = trainable if trainable-prefixes set else full."
        ),
    )
    parser.add_argument(
        "--save-final-checkpoint",
        action="store_true",
        default=_env_or_default_bool("IDENTITY_TRAIN_SAVE_FINAL_CHECKPOINT", True),
        help="Save checkpoint_last.pt at the end of training (env: IDENTITY_TRAIN_SAVE_FINAL_CHECKPOINT).",
    )
    parser.add_argument(
        "--save-optimizer-state",
        action="store_true",
        default=_env_or_default_bool("IDENTITY_TRAIN_SAVE_OPTIMIZER_STATE", True),
        help="Include optimizer state in checkpoints (env: IDENTITY_TRAIN_SAVE_OPTIMIZER_STATE).",
    )
    parser.add_argument(
        "--save-checkpoint-cpu",
        action="store_true",
        default=_env_or_default_bool("IDENTITY_TRAIN_SAVE_CHECKPOINT_CPU", True),
        help=(
            "Move checkpoint tensors to CPU before saving to reduce GPU serialization pressure "
            "(env: IDENTITY_TRAIN_SAVE_CHECKPOINT_CPU)."
        ),
    )
    parser.add_argument(
        "--nonfinite-loss-mode",
        default=_env_or_default_str("IDENTITY_TRAIN_NONFINITE_LOSS_MODE", "skip"),
        choices=["skip", "replace", "raise"],
        help=(
            "How to handle non-finite loss inputs: "
            "'skip' ignores the batch, 'replace' sanitizes NaN/Inf values, "
            "'raise' keeps current behavior and raises."
        ),
    )
    parser.add_argument(
        "--loss-float32",
        action="store_true",
        default=_env_or_default_bool("IDENTITY_TRAIN_LOSS_FLOAT32", True),
        help="Compute loss in float32 for improved numerical stability (env: IDENTITY_TRAIN_LOSS_FLOAT32).",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=_env_or_default_float("IDENTITY_TRAIN_GRAD_CLIP_NORM", 1.0),
        help="Gradient norm clipping value (<=0 disables).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=_env_or_default_int("IDENTITY_TRAIN_NUM_WORKERS", 4),
        help="DataLoader num_workers (env: IDENTITY_TRAIN_NUM_WORKERS).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=_env_or_default_int("IDENTITY_TRAIN_PREFETCH_FACTOR", 2),
        help="DataLoader prefetch_factor when num_workers > 0 (env: IDENTITY_TRAIN_PREFETCH_FACTOR).",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=_env_or_default_bool("IDENTITY_TRAIN_PIN_MEMORY", True),
        help="Use pinned host memory for DataLoader transfers (env: IDENTITY_TRAIN_PIN_MEMORY).",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        default=_env_or_default_bool("IDENTITY_TRAIN_PERSISTENT_WORKERS", True),
        help="Keep DataLoader worker processes alive across epochs (env: IDENTITY_TRAIN_PERSISTENT_WORKERS).",
    )
    parser.add_argument(
        "--non-blocking",
        action="store_true",
        default=_env_or_default_bool("IDENTITY_TRAIN_NON_BLOCKING", True),
        help="Use non_blocking host-to-device copies (env: IDENTITY_TRAIN_NON_BLOCKING).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=_env_or_default_int("IDENTITY_TRAIN_WARMUP_STEPS", 2),
        help="Initial no-optim-step warmup iterations per epoch to stabilize cache behavior (env: IDENTITY_TRAIN_WARMUP_STEPS).",
    )
    parser.add_argument(
        "--output-dir",
        default=_env_or_default_str("IDENTITY_TRAIN_OUTPUT_DIR", "identity_dataset/checkpoints"),
        help="Checkpoint output directory (env: IDENTITY_TRAIN_OUTPUT_DIR).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=_env_or_default_int("IDENTITY_TRAIN_CHECKPOINT_EVERY", 100),
        help="Save checkpoint every N steps (env: IDENTITY_TRAIN_CHECKPOINT_EVERY).",
    )
    parser.add_argument(
        "--max-persons-per-epoch",
        type=int,
        default=_env_or_default_optional_int("IDENTITY_TRAIN_MAX_PERSONS_PER_EPOCH", None),
        help=(
            "Randomly sample this many unique persons per epoch (env: "
            "IDENTITY_TRAIN_MAX_PERSONS_PER_EPOCH). Empty/<=0 means all persons."
        ),
    )
    parser.add_argument(
        "--validation-enabled",
        action="store_true",
        default=_env_or_default_bool("IDENTITY_TRAIN_VALIDATION_ENABLED", False),
        help="Enable person-aware validation split and pass (env: IDENTITY_TRAIN_VALIDATION_ENABLED).",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=_env_or_default_float("IDENTITY_TRAIN_VALIDATION_RATIO", 0.08),
        help="Fraction of identities reserved for validation (env: IDENTITY_TRAIN_VALIDATION_RATIO).",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=_env_or_default_int("IDENTITY_TRAIN_VALIDATE_EVERY", 1),
        help="Run validation every N training epochs (env: IDENTITY_TRAIN_VALIDATE_EVERY).",
    )
    parser.add_argument(
        "--validation-persons-per-epoch",
        type=int,
        default=_env_or_default_optional_int("IDENTITY_TRAIN_VALIDATION_PERSONS_PER_EPOCH", None),
        help="Validation cap for sampled identities each validation pass (env: IDENTITY_TRAIN_VALIDATION_PERSONS_PER_EPOCH).",
    )
    parser.add_argument(
        "--device",
        default=_env_or_default_str("IDENTITY_TRAIN_DEVICE", "auto"),
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device (env: IDENTITY_TRAIN_DEVICE). auto detects cuda/mps/cpu.",
    )
    return parser.parse_args()


def _apply_trainable_prefixes(model: nn.Module, prefixes: Sequence[str]) -> int:
    """Freeze model parameters unless their name matches one of the provided prefixes.

    Returns:
        Number of trainable parameters after filtering.
    """
    if not prefixes:
        return sum(1 for p in model.parameters() if p.requires_grad)

    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            continue
        param.requires_grad = False
    return sum(1 for p in model.parameters() if p.requires_grad)


def parse_source_counts(raw: str) -> List[int]:
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        val = int(token)
        if val < 1:
            raise ValueError("source_counts values must be >= 1")
        out.append(val)
    if not out:
        raise ValueError("source_counts cannot be empty")
    return out


def infer_source_weights(raw_counts: Sequence[int], args: argparse.Namespace) -> List[float]:
    weights: List[float] = []
    for count in raw_counts:
        if count == 1:
            weights.append(args.source_counts_one_weight)
        elif count == 2:
            weights.append(args.source_counts_two_weight)
        else:
            # default to neutral weight for custom counts
            weights.append(1.0)
    return weights


def _group_pairs_by_person(pairs: Sequence) -> Dict[str, List]:
    """Group already-built identity pairs by person id for epoch-wise person sampling."""
    grouped: Dict[str, List] = defaultdict(list)
    for pair in pairs:
        grouped[pair.person].append(pair)
    return grouped


def _select_persons_for_epoch(
    persons: List[str],
    max_persons_per_epoch: Optional[int],
    rng: random.Random,
) -> List[str]:
    if not persons:
        return []
    if max_persons_per_epoch is None or max_persons_per_epoch <= 0:
        return list(persons)
    k = min(max_persons_per_epoch, len(persons))
    if k >= len(persons):
        return list(persons)
    return rng.sample(persons, k=k)


def _split_persons_train_validation(
    persons: List[str],
    ratio: float,
    seed: int,
    enabled: bool,
) -> Tuple[List[str], List[str]]:
    if not enabled or not persons:
        return list(persons), []
    try:
        ratio = float(ratio)
    except (TypeError, ValueError):
        ratio = 0.0
    if ratio <= 0:
        return list(persons), []

    total_persons = len(persons)
    if total_persons <= 1:
        return list(persons), []

    ratio = max(0.0, min(1.0, ratio))
    shuffled = list(persons)
    random.Random(seed).shuffle(shuffled)

    if ratio >= 1.0:
        validation_count = max(1, total_persons - 1)
    else:
        validation_count = int(round(total_persons * ratio))
        if validation_count <= 0:
            validation_count = 1
        validation_count = min(validation_count, total_persons - 1)

    validation_persons = sorted(shuffled[:validation_count])
    train_persons = sorted(shuffled[validation_count:])
    if not train_persons:
        return list(validation_persons), []
    return list(train_persons), list(validation_persons)


def _build_loader_from_pairs(
    selected_pairs: List,
    source_counts: Sequence[int],
    source_weights: Sequence[float],
    args: argparse.Namespace,
    rng_seed: int,
    pin_memory: bool = False,
) -> Tuple[DataLoader, int]:
    epoch_dataset = IdentityPairDataset(
        pairs=selected_pairs,
        image_size=(args.image_size, args.image_size),
        image_transform=None,
    )
    sampler = IdentityConditionBatchSampler(
        pairs=selected_pairs,
        batch_size=args.batch_size,
        mode=args.mode,
        source_counts=source_counts,
        source_weights=source_weights,
        min_identities_per_batch=args.min_identities_per_batch,
        shuffle=True,
        seed=rng_seed,
        drop_last=args.drop_last,
    )
    loader_kwargs = {
        "batch_sampler": sampler,
        "num_workers": args.num_workers,
        "collate_fn": identity_collate_fn,
        "pin_memory": pin_memory and args.pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, args.prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
    loader = DataLoader(epoch_dataset, **loader_kwargs)
    return loader, len(epoch_dataset)


@torch.no_grad()
def _validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    max_sources: int,
) -> Dict[str, float]:
    total_loss = 0.0
    valid_losses = 0
    nonfinite_batches = 0
    for batch in loader:
        batch_tensors = _to_dict(batch, device=device, max_sources=max_sources, non_blocking=args.non_blocking)
        predicted, noise_target = model(
            source_images=batch_tensors["source_images"],
            source_mask=batch_tensors["source_mask"],
            target=batch_tensors["target"],
            prompt_texts=batch_tensors["prompt_text"],
        )
        pred = predicted.float() if args.loss_float32 else predicted
        target = noise_target.float() if args.loss_float32 else noise_target

        if not (torch.isfinite(pred).all() and torch.isfinite(target).all()):
            sanitized = _safe_loss_inputs(pred, target, args.nonfinite_loss_mode)
            if sanitized is None:
                nonfinite_batches += 1
                continue
            pred, target = sanitized

        if not (torch.isfinite(pred).all() and torch.isfinite(target).all()):
            nonfinite_batches += 1
            continue

        loss = compute_loss(pred, target)
        if not torch.isfinite(loss).all():
            nonfinite_batches += 1
            continue
        total_loss += float(loss.item())
        valid_losses += 1

    if valid_losses == 0:
        return {
            "val_loss": math.inf,
            "valid_batches": 0.0,
            "nonfinite_batches": float(nonfinite_batches),
            "is_finite": 0.0,
        }
    return {
        "val_loss": total_loss / valid_losses,
        "valid_batches": float(valid_losses),
        "nonfinite_batches": float(nonfinite_batches),
        "is_finite": 1.0,
    }


def _resolve_checkpoint_scope(requested: str, trainable_prefixes: Sequence[str]) -> str:
    if requested == "auto":
        return "trainable" if trainable_prefixes else "full"
    return requested


def _build_checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    train_step: int,
    args: argparse.Namespace,
    trainable_prefixes: Sequence[str],
    include_optimizer: bool,
) -> dict:
    scope = _resolve_checkpoint_scope(args.checkpoint_scope, trainable_prefixes)
    model_state = _select_state_slice(model.state_dict(), list(trainable_prefixes) if scope == "trainable" else [])
    payload = {
        "epoch": epoch,
        "step": step,
        "train_step": train_step,
        "model": model_state,
        "config": vars(args),
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    else:
        payload["optimizer"] = {"note": "optimizer state disabled by config"}
    return payload


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    resolved_data_root = (
        Path(args.data_root).expanduser()
        if args.data_root is not None
        else _default_data_root()
    )
    if resolved_data_root is None:
        raise FileNotFoundError(
            "No data-root provided. Set --data-root or set IDENTITY_TRAIN_DATA_ROOT "
            "in environment/.env."
        )
    data_root = resolved_data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    source_counts = parse_source_counts(args.source_counts)
    pairs = build_identity_pairs(
        data_root=data_root,
        source_counts=source_counts,
        max_pairs_per_person=args.max_pairs_per_person,
        max_pairs_total=args.max_total_pairs,
        seed=args.seed,
    )

    person_to_pairs = _group_pairs_by_person(pairs)
    all_persons = sorted(person_to_pairs.keys())
    if not all_persons:
        raise ValueError("No persons found after pair building.")

    train_persons, validation_persons = _split_persons_train_validation(
        persons=all_persons,
        ratio=args.validation_ratio,
        seed=args.seed,
        enabled=args.validation_enabled,
    )
    train_pairs = [pair for person in train_persons for pair in person_to_pairs.get(person, [])]
    validation_pairs = [pair for person in validation_persons for pair in person_to_pairs.get(person, [])]

    print(
        f"Loaded persons: {len(all_persons)} (train={len(train_persons)}, val={len(validation_persons)})"
    )
    if args.validation_enabled:
        if validation_pairs:
            print(
                f"Validation enabled: ratio={args.validation_ratio:.4f}, "
                f"every={max(1, args.validate_every)} epoch(s), "
                f"validation_persons_per_epoch={args.validation_persons_per_epoch or 'all'}"
            )
            print(f"Validation reserved pairs: {len(validation_pairs)}")
        else:
            print("Validation enabled but no validation identities were selected.")
    else:
        print("Validation split disabled.")
    if args.max_persons_per_epoch is not None and args.max_persons_per_epoch > 0:
        print(
            "Configured max_persons_per_epoch:",
            args.max_persons_per_epoch,
            f"(effective per epoch: {min(args.max_persons_per_epoch, len(all_persons))})",
        )
    else:
        print("Configured max_persons_per_epoch: all persons every epoch")

    source_weights = infer_source_weights(source_counts, args)

    print(f"Loaded pairs: {len(pairs)} (train={len(train_pairs)}, val={len(validation_pairs)})")
    print(f"Mode: {args.mode}")
    print(f"Max possible batches per epoch:  depends on selected persons.")
    print(f"Data root: {data_root}")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "IDENTITY_TRAIN_DEVICE is set to cuda, but torch.cuda.is_available() is False. "
                "Use --device auto or CPU build has no CUDA support."
            )
        device = torch.device("cuda")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "IDENTITY_TRAIN_DEVICE is set to mps, but torch.backends.mps.is_available() is False."
            )
        device = torch.device("mps")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print(
        f"Using device: {device} | torch_cuda_available={torch.cuda.is_available()} | "
        f"torch_mps_available={getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()}"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    model_dtype = _torch_dtype_from_name(args.model_dtype, _env_or_default_torch_dtype("IDENTITY_TRAIN_DTYPE", torch.bfloat16))
    max_k = max(source_counts)
    model = build_model(
        model_path=args.model_path,
        device=device,
        dtype=model_dtype,
        compile_model=args.compile_model,
        max_sequence_length=args.max_sequence_length,
        num_train_timesteps=args.scheduler_train_steps,
    ).to(device)

    trainable_prefixes = _collect_prefixes(args.trainable_prefixes)
    trainable_count = _apply_trainable_prefixes(model, trainable_prefixes)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if trainable_count == 0:
        raise ValueError(
            "No trainable parameters matched IDENTITY_TRAIN_TRAINABLE_PREFIXES. "
            "Either set it to empty (full trainable graph) or include at least one valid parameter prefix."
        )
    print(f"Trainable parameter tensors: {len(trainable_params)}")

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=args.lr,
            momentum=args.optimizer_momentum,
        )
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
        if len(trainable_params) > 100000 and args.batch_size <= 1:
            print(
                "Warning: AdamW state may exceed small-memory limits. "
                "Use IDENTITY_TRAIN_OPTIMIZER=sgd for ultra-small GPU smoke runs."
            )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    train_step = 0
    nonfinite_batches = 0
    with tqdm(range(1, args.epochs + 1), desc="Epochs", total=args.epochs) as epoch_bar, tqdm(
        total=0,
        desc="Global",
        dynamic_ncols=True,
    ) as global_bar:
        for epoch in epoch_bar:
            model.train()

            epoch_rng = random.Random(args.seed + epoch)
            selected_persons = _select_persons_for_epoch(
                persons=train_persons,
                max_persons_per_epoch=args.max_persons_per_epoch,
                rng=epoch_rng,
            )
            selected_pairs: List = []
            for person in selected_persons:
                selected_pairs.extend(person_to_pairs.get(person, []))

            if not selected_pairs:
                print(f"Epoch {epoch}: no valid pairs for selected persons, skipping.")
                continue

            loader, epoch_pairs = _build_loader_from_pairs(
                selected_pairs=selected_pairs,
                source_counts=source_counts,
                source_weights=source_weights,
                args=args,
                rng_seed=args.seed + epoch,
                pin_memory=device.type == "cuda",
            )
            epoch_batches = len(loader)
            global_bar.total += epoch_batches
            global_bar.refresh()

            print(
                f"Epoch {epoch} selected persons: {len(selected_persons)}/{len(train_persons)} "
                f"-> {epoch_pairs} pairs, {epoch_batches} batches"
            )
            epoch_nonfinite_batches = 0
            with tqdm(
                loader,
                desc=f"Epoch {epoch}/{args.epochs}",
                total=epoch_batches,
                leave=False,
                dynamic_ncols=True,
            ) as batch_bar:
                for batch in batch_bar:
                    batch_tensors = _to_dict(
                        batch, device=device, max_sources=max_k, non_blocking=args.non_blocking
                    )
                    target = batch_tensors["target"]
                    prompt_texts = batch_tensors["prompt_text"]
                    if step == 0:
                        print(
                            "Example prompts:",
                            [str(p) for p in prompt_texts[: min(3, len(prompt_texts))]],
                        )
                    try:
                        predicted, noise_target = model(
                            source_images=batch_tensors["source_images"],
                            source_mask=batch_tensors["source_mask"],
                            target=target,
                            prompt_texts=prompt_texts,
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            torch.cuda.empty_cache()
                            print(
                                "\nCUDA OOM while running model forward. Suggested smoke settings:"
                                "\n  - lower IDENTITY_TRAIN_IMAGE_SIZE to 128"
                                "\n  - set IDENTITY_TRAIN_BATCH_SIZE=1"
                                "\n  - set IDENTITY_TRAIN_SOURCE_COUNTS=1"
                                "\n  - set IDENTITY_TRAIN_MAX_PERSONS_PER_EPOCH=1"
                                "\n  - increase free GPU memory before continuing"
                            )
                        raise

                    # Warmup: skip gradient graph on first N global steps to avoid first-step spikes.
                    if step < args.warmup_steps:
                        with torch.no_grad():
                            loss = compute_loss(
                                predicted.float() if args.loss_float32 else predicted,
                                noise_target.float() if args.loss_float32 else noise_target,
                            )
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                    else:
                        loss_prediction = predicted.float() if args.loss_float32 else predicted
                        loss_target = noise_target.float() if args.loss_float32 else noise_target
                        if not (torch.isfinite(loss_prediction).all() and torch.isfinite(loss_target).all()):
                            sanitized = _safe_loss_inputs(
                                loss_prediction,
                                loss_target,
                                args.nonfinite_loss_mode,
                            )
                            if sanitized is None:
                                nonfinite_batches += 1
                                epoch_nonfinite_batches += 1
                                print(
                                    "Warning: non-finite loss inputs detected; skipping optimizer update for this batch."
                                )
                                optimizer.zero_grad(set_to_none=True)
                                step += 1
                                batch_bar.set_postfix(
                                    {
                                        "step": step,
                                        "opt_step": train_step,
                                        "loss": "nan/inf",
                                        "batch_src": int(batch_tensors["source_count"].max()),
                                    }
                                )
                                global_bar.set_postfix(
                                    {
                                        "epoch": epoch,
                                        "step": step,
                                        "opt_step": train_step,
                                        "loss": "nan/inf",
                                    }
                                )
                                global_bar.update(1)
                                continue
                            loss_prediction, loss_target = sanitized

                        if not torch.isfinite(loss_prediction).all() or not torch.isfinite(loss_target).all():
                            nonfinite_batches += 1
                            epoch_nonfinite_batches += 1
                            print(
                                "Warning: non-finite loss input values remained after sanitization; skipping optimizer update for this batch."
                            )
                            optimizer.zero_grad(set_to_none=True)
                            step += 1
                            batch_bar.set_postfix(
                                {
                                    "step": step,
                                    "opt_step": train_step,
                                    "loss": "nan/inf",
                                    "batch_src": int(batch_tensors["source_count"].max()),
                                }
                            )
                            global_bar.set_postfix(
                                {
                                    "epoch": epoch,
                                    "step": step,
                                    "opt_step": train_step,
                                    "loss": "nan/inf",
                                }
                            )
                            global_bar.update(1)
                            continue

                        loss = compute_loss(loss_prediction, loss_target)
                        if not torch.isfinite(loss).all():
                            nonfinite_batches += 1
                            epoch_nonfinite_batches += 1
                            print(
                                "Warning: non-finite loss detected; skipping optimizer update for this batch."
                            )
                            optimizer.zero_grad(set_to_none=True)
                            step += 1
                            batch_bar.set_postfix(
                                {
                                    "step": step,
                                    "opt_step": train_step,
                                    "loss": "nan/inf",
                                    "batch_src": int(batch_tensors["source_count"].max()),
                                }
                            )
                            global_bar.set_postfix(
                                {
                                    "epoch": epoch,
                                    "step": step,
                                    "opt_step": train_step,
                                    "loss": "nan/inf",
                                }
                            )
                            global_bar.update(1)
                            continue
                        loss.backward()
                        if args.grad_clip_norm and args.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
                        try:
                            optimizer.step()
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                torch.cuda.empty_cache()
                                print(
                                    "\nCUDA OOM while running optimizer.step(). Suggested smoke settings:"
                                    "\n  - reduce IDENTITY_TRAIN_TRAINABLE_PREFIXES to a tiny head like transformer.all_final_layer"
                                    "\n  - switch IDENTITY_TRAIN_OPTIMIZER=sgd"
                                    "\n  - lower IDENTITY_TRAIN_IMAGE_SIZE to 128"
                                    "\n  - set IDENTITY_TRAIN_BATCH_SIZE=1"
                                    "\n  - set IDENTITY_TRAIN_SOURCE_COUNTS=1"
                                    "\n  - set IDENTITY_TRAIN_MAX_PERSONS_PER_EPOCH=1"
                                )
                            raise
                        optimizer.zero_grad(set_to_none=True)
                        train_step += 1

                    step += 1
                    batch_bar.set_postfix(
                        {
                            "step": step,
                            "opt_step": train_step,
                            "loss": f"{loss.item():.6f}",
                            "batch_src": int(batch_tensors["source_count"].max()),
                        }
                    )
                    global_bar.set_postfix(
                        {
                            "epoch": epoch,
                            "step": step,
                            "opt_step": train_step,
                            "loss": f"{loss.item():.6f}",
                        }
                    )
                    global_bar.update(1)

                    if (
                        args.checkpoint_every > 0
                        and train_step > 0
                        and train_step % args.checkpoint_every == 0
                    ):
                        ckpt = output_dir / f"checkpoint_step_{train_step:07d}.pt"
                        checkpoint_payload = _build_checkpoint_payload(
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            step=step,
                            train_step=train_step,
                            args=args,
                            trainable_prefixes=trainable_prefixes,
                            include_optimizer=args.save_optimizer_state,
                        )
                        if args.save_checkpoint_cpu:
                            checkpoint_payload = _move_tensors_to_cpu(checkpoint_payload)
                        try:
                            _safe_torch_save(checkpoint_payload, ckpt, fallback_no_zip=True)
                            print(f"Saved checkpoint: {ckpt}")
                        except Exception as e:
                            print(
                                f"Warning: failed to save intermediate checkpoint {ckpt}: {type(e).__name__}: {e}"
                            )

            if (
                args.validation_enabled
                and args.validate_every > 0
                and (epoch % args.validate_every == 0)
                and validation_pairs
            ):
                val_selected_persons = _select_persons_for_epoch(
                    persons=validation_persons,
                    max_persons_per_epoch=args.validation_persons_per_epoch,
                    rng=epoch_rng,
                )
                selected_validation_pairs: List = []
                for person in val_selected_persons:
                    selected_validation_pairs.extend(person_to_pairs.get(person, []))

                if selected_validation_pairs:
                    validation_loader, validation_pairs_for_epoch = _build_loader_from_pairs(
                        selected_pairs=selected_validation_pairs,
                        source_counts=source_counts,
                        source_weights=source_weights,
                        args=args,
                        rng_seed=args.seed + epoch + 10_000,
                        pin_memory=device.type == "cuda",
                    )
                    model.eval()
                    val_metrics = _validate_epoch(
                        model=model,
                        loader=validation_loader,
                        device=device,
                        args=args,
                        max_sources=max_k,
                    )
                    model.train()
                    val_loss = val_metrics["val_loss"]
                    val_finite = bool(val_metrics["is_finite"] > 0.5)
                    val_nonfinite = int(val_metrics["nonfinite_batches"])
                    val_batches = int(val_metrics["valid_batches"] + val_nonfinite)
                    if val_finite:
                        print(
                            f"Epoch {epoch} validation: pairs={validation_pairs_for_epoch}, "
                            f"persons={len(val_selected_persons)}, batches={val_batches}, "
                            f"loss={val_loss:.6f}, finite={val_finite}, "
                            f"nonfinite_batches={val_nonfinite}"
                        )
                    else:
                        print(
                            f"Epoch {epoch} validation: pairs={validation_pairs_for_epoch}, "
                            f"persons={len(val_selected_persons)}, batches={val_batches}, "
                            f"loss=nan, finite={val_finite}, nonfinite_batches={val_nonfinite}"
                        )
                else:
                    print(f"Epoch {epoch} validation: skipped (no validation pairs selected this pass).")
            else:
                print(f"Epoch {epoch} validation: skipped by schedule/config.")

            epoch_bar.set_postfix({"steps": step})
            print(
                f"Epoch {epoch} done | train_pairs={epoch_pairs} | epoch_nonfinite_batches={epoch_nonfinite_batches}"
            )

    if args.nonfinite_loss_mode == "skip":
        print(
            "Training summary: "
            f"steps={step}, "
            f"optimizer_updates={train_step}, "
            f"nonfinite_batches={nonfinite_batches}"
        )
    if args.save_final_checkpoint:
        final_ckpt = output_dir / "checkpoint_last.pt"
        final_payload = _build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=args.epochs,
            step=step,
            train_step=train_step,
            args=args,
            trainable_prefixes=trainable_prefixes,
            include_optimizer=args.save_optimizer_state,
        )
        if args.save_checkpoint_cpu:
            final_payload = _move_tensors_to_cpu(final_payload)
        try:
            _safe_torch_save(final_payload, final_ckpt, fallback_no_zip=True)
            print(f"Saved final checkpoint: {final_ckpt}")
        except Exception as e:
            print(f"Warning: final checkpoint save failed: {type(e).__name__}: {e}")
            fallback_path = final_ckpt.with_suffix(".last_failure.txt")
            fallback_path.write_text(
                "\n".join(
                    [
                        f"final_checkpoint: {final_ckpt}",
                        f"error: {type(e).__name__}: {e}",
                        f"epoch: {args.epochs}",
                        f"step: {step}",
                        f"opt_step: {train_step}",
                    ]
                ),
                encoding="utf-8",
            )
            print(f"Saved failure details to {fallback_path}")
    else:
        print("Final checkpoint save skipped via IDENTITY_TRAIN_SAVE_FINAL_CHECKPOINT=0")


if __name__ == "__main__":
    main()

