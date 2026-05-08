"""Run Z-Image generation using values from `.env`.

This runner mirrors the Flux style:
- all configuration is controlled from `.env`
- optional preset selectors are supported (`RUN_PRESET`)
- optional `--print-env-help` prints env values with comments from `.env`
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE entries from a `.env` file into process environment."""
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


def print_env_help(env_path: Path) -> None:
    if not env_path.exists():
        print(f".env file not found: {env_path}")
        return

    print("Z-Image runner config from .env")
    print("Values marked <unset> mean empty/not provided in .env.")
    print("=" * 60)

    comment_lines: list[str] = []
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            comment_lines = []
            continue
        if stripped.startswith("#"):
            comment = stripped.lstrip("#").strip()
            if comment:
                comment_lines.append(comment)
            continue
        if "=" not in stripped:
            continue

        key, _ = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        env_value = os.getenv(key, "<unset>")
        print(f"{key}={env_value}")
        for comment_line in comment_lines:
            print(f"  - {comment_line}")
        comment_lines = []


def _get_env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _apply_presets(mode: str, env_path: Path) -> None:
    """Apply preset values to mode-specific env vars."""
    load_env_file(env_path)

    preset = _get_env_str("RUN_PRESET", "auto")
    preset = re.sub(r"\s+", "", preset.lower())
    if not preset or preset in {"auto", "none", "off", "false"}:
        return

    if mode == "single":
        mapping = {
            "INFERENCE_WIDTH": "WIDTH",
            "INFERENCE_HEIGHT": "HEIGHT",
            "INFERENCE_STEPS": "STEPS",
            "INFERENCE_GUIDANCE_SCALE": "GUIDANCE_SCALE",
            "INFERENCE_DTYPE": "DTYPE",
            "INFERENCE_DEVICE": "DEVICE",
            "INFERENCE_SEED": "SEED",
            "INFERENCE_OUTPUT_PATH": "OUTPUT_PATH",
            "INFERENCE_PROMPT": "PROMPT",
            "COMPILE_MODEL": "COMPILE_MODEL",
        }
    elif mode == "image_edit":
        mapping = {
            "IMAGE_EDIT_WIDTH": "WIDTH",
            "IMAGE_EDIT_HEIGHT": "HEIGHT",
            "IMAGE_EDIT_STEPS": "STEPS",
            "IMAGE_EDIT_GUIDANCE_SCALE": "GUIDANCE_SCALE",
            "IMAGE_EDIT_DTYPE": "DTYPE",
            "IMAGE_EDIT_DEVICE": "DEVICE",
            "IMAGE_EDIT_SEED": "SEED",
            "IMAGE_EDIT_OUTPUT_PATH": "OUTPUT_PATH",
            "IMAGE_EDIT_STRENGTH": "STRENGTH",
            "IMAGE_EDIT_USE_SOURCE_SIZE": "USE_SOURCE_SIZE",
            "IMAGE_EDIT_SOURCE_IMAGE": "SOURCE_IMAGE",
            "IMAGE_EDIT_MASK_IMAGE": "MASK_IMAGE",
            "COMPILE_MODEL": "COMPILE_MODEL",
            "INFERENCE_PROMPT": "PROMPT",
        }
    else:
        mapping = {
            "BATCH_WIDTH": "WIDTH",
            "BATCH_HEIGHT": "HEIGHT",
            "BATCH_STEPS": "STEPS",
            "BATCH_GUIDANCE_SCALE": "GUIDANCE_SCALE",
            "BATCH_DTYPE": "DTYPE",
            "BATCH_DEVICE": "DEVICE",
            "BATCH_SEED": "SEED",
            "BATCH_OUTPUT_EXT": "OUTPUT_EXT",
            "BATCH_OUTPUT_PREFIX": "OUTPUT_PREFIX",
            "BATCH_OUTPUT_DIR": "OUTPUT_DIR",
            "BATCH_COMPILE_MODEL": "COMPILE_MODEL",
            "BATCH_OUTPUT_SUFFIX": "OUTPUT_SUFFIX",
            "BATCH_SKIP_EXISTING": "SKIP_EXISTING",
        }

    applied = []
    for target_key, preset_key in mapping.items():
        preset_candidates = [f"{preset}_{preset_key}", preset_key]
        selected_value = None
        for candidate in preset_candidates:
            value = os.getenv(candidate)
            if value is None:
                continue
            stripped = value.strip()
            if stripped:
                selected_value = stripped
                break
        if selected_value is None:
            continue
        os.environ[target_key] = selected_value
        applied.append((target_key, selected_value))

    if applied:
        print(f"Preset '{preset}' applied to {len(applied)} settings.")
        for key, value in applied:
            print(f"  {key}={value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Z-Image generation using environment configuration."
    )
    parser.add_argument(
        "--print-env-help",
        action="store_true",
        help="Print supported env values from .env and exit.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "image_edit"],
        help="Execution mode. Default: from ZIMAGE_RUN_MODE env var (batch if unset).",
    )
    parser.add_argument(
        "--source-image",
        default=None,
        help="Image used for image-edit mode (`IMAGE_EDIT_SOURCE_IMAGE`).",
    )
    parser.add_argument(
        "--mask",
        default=None,
        help="Optional mask image for image-edit mode (`IMAGE_EDIT_MASK_IMAGE`).",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Image-edit blend strength (`IMAGE_EDIT_STRENGTH`, 0..1).",
    )
    parser.add_argument(
        "--use-source-size",
        action="store_true",
        help="Use source image resolution for image-edit output (`IMAGE_EDIT_USE_SOURCE_SIZE=1`).",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Image-edit output path override (`IMAGE_EDIT_OUTPUT_PATH`).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Inline prompt override (`INFERENCE_PROMPT`).",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to env file (default: .env)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    env_path = (repo_root / args.env_file).resolve()

    if args.print_env_help:
        load_env_file(env_path)
        print_env_help(env_path)
        return

    load_env_file(env_path)

    mode = (args.mode or _get_env_str("ZIMAGE_RUN_MODE", "batch")).strip().lower()
    if mode not in {"single", "batch", "image_edit"}:
        raise ValueError(f"Invalid mode: {mode}. Use 'single', 'batch', or 'image_edit'.")

    _apply_presets(mode, env_path)

    if mode == "batch":
        target = repo_root / "batch_inference.py"
    else:
        target = repo_root / "inference.py"
    if not target.exists():
        raise FileNotFoundError(f"Cannot run target script: {target}")
    os.environ["ZIMAGE_RUN_MODE"] = mode

    if args.prompt:
        os.environ["INFERENCE_PROMPT"] = args.prompt

    if mode == "image_edit":
        if args.source_image:
            os.environ["IMAGE_EDIT_SOURCE_IMAGE"] = args.source_image
        if args.mask:
            os.environ["IMAGE_EDIT_MASK_IMAGE"] = args.mask
        if args.strength is not None:
            os.environ["IMAGE_EDIT_STRENGTH"] = str(args.strength)
        if args.use_source_size:
            os.environ["IMAGE_EDIT_USE_SOURCE_SIZE"] = "1"
        if args.output_path:
            os.environ["IMAGE_EDIT_OUTPUT_PATH"] = args.output_path

    print(f"Running Z-Image {mode} mode using values from: {env_path}")
    result = subprocess.run(
        [sys.executable, str(target)],
        cwd=str(repo_root),
        env=os.environ.copy(),
        text=True,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
