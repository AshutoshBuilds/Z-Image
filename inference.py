"""Z-Image PyTorch Native Inference."""

import os
from pathlib import Path
import sys
import time
import warnings
from typing import Optional

import torch

warnings.filterwarnings("ignore")

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


def _resolve_dtype() -> torch.dtype:
    dtype_name = _get_env_str("INFERENCE_DTYPE", "bfloat16").lower()
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


def _resolve_device(prefix: str = "INFERENCE") -> str:
    override = _get_env_str(f"{prefix}_DEVICE", _get_env_str("INFERENCE_DEVICE", "auto")).lower()
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

    print(f"Unknown {prefix}_DEVICE='{override}', falling back to auto")
    return _resolve_device_with_auto()


def _supports_torch_compile() -> bool:
    try:
        import triton  # type: ignore

        return True
    except Exception:
        return False


def _resolve_compile(env_name: str, fallback_env: Optional[str] = None) -> bool:
    if fallback_env is None:
        fallback_env = ""
    requested_default = False
    if fallback_env:
        requested_default = _get_env_bool(fallback_env, False)
    enabled = _get_env_bool(env_name, requested_default)
    if not enabled:
        return False
    if _supports_torch_compile():
        return True

    print(
        "Torch compile requested but Triton is not available. Falling back to eager execution to avoid "
        "runtime failure. Install Triton for compile mode."
    )
    return False


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


def _build_output_path(
    output_path: Path,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    device: str,
    dtype: torch.dtype,
) -> Path:
    dtype_name = _dtype_name(dtype)
    device_name = str(device).replace(":", "-")
    guidance_name = _format_setting_for_path(guidance_scale).replace(".", "p")
    stem = output_path.stem or "output"
    suffix = output_path.suffix or ".png"
    filename = (
        f"{stem}_w{width}_h{height}_s{num_inference_steps}_g{guidance_name}_"
        f"seed{seed}_{dtype_name}_{device_name}"
    )
    return output_path.with_name(f"{filename}{suffix}")


load_dotenv_file()
_ensure_local_import_path()

from utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate


def main():
    default_model_path = Path("models") / "Z-Image-Turbo"
    model_path = ensure_model_weights(
        os.environ.get("ZIMAGE_MODEL_PATH", str(default_model_path)),
        verify=False,  # set True to verify with md5
    )

    base_prompt = (
        "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
        "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. "
        "Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, "
        "silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
    )
    prompt = _get_env_str("INFERENCE_PROMPT", base_prompt)
    prompt_source = (
        "INFERENCE_PROMPT (env)"
        if "INFERENCE_PROMPT" in os.environ
        else "default INFERENCE_PROMPT"
    )
    output_path = Path(_get_env_str("INFERENCE_OUTPUT_PATH", "outputs/example.png"))
    width = _get_env_int("INFERENCE_WIDTH", 1024)
    height = _get_env_int("INFERENCE_HEIGHT", 1024)
    num_inference_steps = _get_env_int("INFERENCE_STEPS", 8)
    guidance_scale = _get_env_float("INFERENCE_GUIDANCE_SCALE", 0.0)
    seed = _get_env_int("INFERENCE_SEED", 42)
    dtype = _resolve_dtype()
    device = _resolve_device()
    compile = _resolve_compile("COMPILE_MODEL")

    output_target = _build_output_path(
        output_path,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        device=device,
        dtype=dtype,
    )

    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")

    output_target_display = str(output_target)
    if len(output_target_display) > 140:
        output_target_display = output_target_display[:137] + "..."
    print("Resolved single-image generation settings:")
    print(f"  model_path: {model_path}")
    print(f"  device: {device}")
    print(f"  dtype: {_dtype_name(dtype)}")
    print(f"  compile: {compile}")
    print(f"  attention_backend: {attn_backend}")
    print(f"  dimensions: {width}x{height}")
    print(f"  steps: {num_inference_steps}")
    print(f"  guidance: {_format_setting_for_path(guidance_scale)}")
    print(f"  seed: {seed}")
    print(f"  prompt_source: {prompt_source}")
    print(f"  output_target: {output_target_display}")
    # Load models
    components = load_from_local_dir(model_path, device=device, dtype=dtype, compile=compile)
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    print(f"Chosen attention backend: {attn_backend}")

    # Gen an image
    start_time = time.time()
    images = generate(
        prompt=prompt,
        **components,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device).manual_seed(seed),
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    output_target.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_target)

    ### !! For best speed performance, recommend to use `_flash_3` backend and set `compile=True`
    ### This would give you sub-second generation speed on Hopper GPU (H100/H200/H800) after warm-up


if __name__ == "__main__":
    main()
