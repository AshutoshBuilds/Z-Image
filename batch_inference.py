"""Batch prompt inference for Z-Image."""

import os
from pathlib import Path
import sys
import time

import torch


def load_dotenv_file(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE entries from a local .env file into process environment."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _ensure_local_import_path() -> None:
    """Ensure project `src` package is importable when running scripts directly."""
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = str(src_path)


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip()


def _resolve_device(override_name: str = "BATCH_DEVICE") -> str:
    override = _get_env_str(override_name, _get_env_str("INFERENCE_DEVICE", "auto")).lower()
    if override == "auto":
        if torch.cuda.is_available():
            print("Chosen device: cuda")
            return "cuda"
        try:
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
            print("Chosen device: tpu")
            return device
        except (ImportError, RuntimeError):
            if torch.backends.mps.is_available():
                print("Chosen device: mps")
                return "mps"
            print("Chosen device: cpu")
            return "cpu"

    if override == "cuda":
        if torch.cuda.is_available():
            print("Chosen device override: cuda")
            return "cuda"
        raise RuntimeError(
            f"CUDA override requested but torch.cuda.is_available() is {torch.cuda.is_available()}"
        )
    if override == "mps":
        if torch.backends.mps.is_available():
            print("Chosen device override: mps")
            return "mps"
        raise RuntimeError(
            "MPS override requested but torch.backends.mps.is_available() is "
            f"{torch.backends.mps.is_available()}"
        )
    if override == "cpu":
        print("Chosen device override: cpu")
        return "cpu"
    if override == "tpu":
        try:
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
            print("Chosen device override: tpu")
            return device
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "TPU override requested but torch_xla is unavailable or failed to initialize"
            ) from exc

    print(f"Unknown {override_name}='{override}', falling back to auto")
    return _resolve_device_with_auto()


def _resolve_device_with_auto() -> str:
    if torch.cuda.is_available():
        print("Chosen device: cuda")
        return "cuda"
    try:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        print("Chosen device: tpu")
        return device
    except (ImportError, RuntimeError):
        if torch.backends.mps.is_available():
            print("Chosen device: mps")
            return "mps"
        print("Chosen device: cpu")
        return "cpu"


load_dotenv_file()
_ensure_local_import_path()

from utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate


def read_prompts(path: str) -> list[str]:
    """Read prompts from a text file (one per line, empty lines skipped)."""

    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with prompt_path.open("r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")
    return prompts


PROMPTS_FILE = os.environ.get("PROMPTS_FILE", "prompts/prompt1.txt")


def slugify(text: str, max_len: int = 60) -> str:
    """Create a filesystem-safe slug from the prompt."""

    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:max_len].rstrip("-") or "prompt"


def _resolve_dtype() -> torch.dtype:
    dtype_name = _get_env_str("BATCH_DTYPE", _get_env_str("INFERENCE_DTYPE", "bfloat16")).lower()
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype_name in {"float16", "fp16"}:
        return torch.float16
    if dtype_name in {"float32", "fp32"}:
        return torch.float32
    return torch.bfloat16


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _format_setting_for_path(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _build_output_path(
    output_dir: Path,
    output_prefix: str,
    idx: int,
    prompt: str,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    device: str,
    dtype: torch.dtype,
    output_ext: str,
    output_filename_suffix: str,
) -> Path:
    slug = slugify(prompt)
    base_name = f"{output_prefix}-{idx:02d}-{slug}"
    if output_filename_suffix:
        base_name = f"{base_name}-{output_filename_suffix}"
    dtype_name = _dtype_name(dtype)
    device_name = str(device).replace(":", "-")
    guidance_name = _format_setting_for_path(guidance_scale).replace(".", "p")
    filename = (
        f"{base_name}_w{width}_h{height}_s{num_inference_steps}_g{guidance_name}_"
        f"seed{seed}_{dtype_name}_{device_name}"
    )
    return output_dir / f"{filename}.{output_ext}"


def main():
    default_model_path = Path("models") / "Z-Image-Turbo"
    model_path = ensure_model_weights(
        os.environ.get("ZIMAGE_MODEL_PATH", str(default_model_path)),
        verify=False,  # set True to verify with md5
    )
    dtype = _resolve_dtype()
    compile = _get_env_bool("BATCH_COMPILE_MODEL", _get_env_bool("COMPILE_MODEL", False))
    height = _get_env_int("BATCH_HEIGHT", _get_env_int("INFERENCE_HEIGHT", 1024))
    width = _get_env_int("BATCH_WIDTH", _get_env_int("INFERENCE_WIDTH", 1024))
    num_inference_steps = _get_env_int("BATCH_STEPS", _get_env_int("INFERENCE_STEPS", 8))
    guidance_scale = _get_env_float(
        "BATCH_GUIDANCE_SCALE", _get_env_float("INFERENCE_GUIDANCE_SCALE", 0.0)
    )
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
    base_seed = _get_env_int("BATCH_SEED", _get_env_int("INFERENCE_SEED", 42))
    output_ext = _get_env_str("BATCH_OUTPUT_EXT", "png").lstrip(".").lower() or "png"
    output_dir = Path(_get_env_str("BATCH_OUTPUT_DIR", "outputs"))
    output_prefix = _get_env_str("BATCH_OUTPUT_PREFIX", "prompt")
    output_filename_suffix = _get_env_str("BATCH_OUTPUT_SUFFIX", "").strip("_")
    skip_existing = _get_env_bool("BATCH_SKIP_EXISTING", False)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device()
    prompts = read_prompts(PROMPTS_FILE)
    prompt_source = "PROMPTS_FILE (env)" if "PROMPTS_FILE" in os.environ else "default PROMPTS_FILE"

    print("Resolved batch generation settings:")
    print(f"  model_path: {model_path}")
    print(f"  device: {device}")
    print(f"  dtype: {_dtype_name(dtype)}")
    print(f"  compile: {compile}")
    print(f"  attention_backend: {attn_backend}")
    print(f"  dimensions: {width}x{height}")
    print(f"  steps: {num_inference_steps}")
    print(f"  guidance: {_format_setting_for_path(guidance_scale)}")
    print(f"  seed_base: {base_seed}")
    print(f"  prompt_source: {prompt_source} -> {PROMPTS_FILE}")
    print(f"  output_dir: {output_dir}")

    components = load_from_local_dir(model_path, device=device, dtype=dtype, compile=compile)
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    print(f"Chosen attention backend: {attn_backend}")

    for idx, prompt in enumerate(prompts, start=1):
        seed = base_seed + idx - 1
        output_path = _build_output_path(
            output_dir=output_dir,
            output_prefix=output_prefix,
            idx=idx,
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            device=device,
            dtype=dtype,
            output_ext=output_ext,
            output_filename_suffix=output_filename_suffix,
        )
        if skip_existing and output_path.exists():
            print(f"[{idx}/{len(prompts)}] Skipping existing {output_path}")
            continue
        prompt_label = prompt if len(prompt) <= 80 else prompt[:77] + "..."
        print(f"[{idx}/{len(prompts)}] Generating seed={seed}, prompt={prompt_label}")
        print(f"[{idx}/{len(prompts)}] Output target: {output_path}")
        generator = torch.Generator(device).manual_seed(seed)

        start_time = time.time()
        images = generate(
            prompt=prompt,
            **components,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        elapsed = time.time() - start_time
        images[0].save(output_path)
        print(f"[{idx}/{len(prompts)}] Saved {output_path} in {elapsed:.2f} seconds")

    print("Done.")


if __name__ == "__main__":
    main()
