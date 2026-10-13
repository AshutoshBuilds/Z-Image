"""Identity pair dataset utilities for same-person image-to-image training.

This module builds pair/triple-style samples directly from an existing folder
layout without saving any new images to disk:

  root/
    person_01/
      img1.jpg
      img2.jpg
      ...
    person_02/
      ...

Each training sample contains one target image and one or more source images of
the same person. It supports combinations like:
- source_count=1: A->B, B->A, ...
- source_count=2: A+B->C, A+C->B, B+C->A, ...
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class IdentityPair:
    """Single sample: one target image and N source images of the same person."""

    person: str
    source_paths: Tuple[Path, ...]
    target_path: Path


def np_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image into NHWC->CHW tensor without requiring torchvision."""
    arr = torch.tensor(list(image.convert("RGB").getdata()), dtype=torch.uint8)
    # image.getdata() returns a flat list of RGB tuples
    width, height = image.size
    arr = arr.view(height, width, 3).permute(2, 0, 1).to(torch.float32) / 255.0
    return arr


def _collect_person_images(data_root: Path) -> Dict[str, List[Path]]:
    """Collect image files for each identity folder found recursively.

    The function supports both:
    - root/<person_name>/<images...>
    - root/<page_name>/<person_name>/<images...>
    """
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    groups: Dict[str, List[Path]] = {}
    for person_dir in sorted(data_root.rglob("*")):
        if not person_dir.is_dir() or person_dir == data_root:
            continue

        images = sorted(p for p in person_dir.iterdir() if _is_image_file(p))
        if not images:
            continue

        # Use the path relative to data_root as a stable person id.
        # This preserves multi-level hierarchy like page/<person>.
        person_id = str(person_dir.relative_to(data_root))
        groups[person_id] = images
    return groups


def _sample_source_target_pairs(
    images: Sequence[Path],
    source_count: int,
    rng: random.Random,
    max_pairs_per_person: Optional[int] = None,
) -> List[Tuple[Tuple[int, ...], int]]:
    """
    Generate (source_indices, target_index) pairs for one person.

    For source_count=1 and images [A,B,C]:
    - [A]->B, A->C, B->A, ...
    """
    n = len(images)
    if source_count < 1 or source_count >= n:
        return []

    pairs: List[Tuple[Tuple[int, ...], int]] = []
    for source_indices in combinations(range(n), source_count):
        source_set = set(source_indices)
        for target_index in range(n):
            if target_index in source_set:
                continue
            pairs.append((source_indices, target_index))

    if max_pairs_per_person is not None and len(pairs) > max_pairs_per_person:
        pairs = rng.sample(pairs, k=max_pairs_per_person)
    return pairs


def build_identity_pairs(
    data_root: Path,
    source_counts: Sequence[int] = (1, 2),
    max_pairs_per_person: Optional[int] = None,
    max_pairs_total: Optional[int] = None,
    seed: int = 42,
) -> List[IdentityPair]:
    """
    Build all requested same-person source->target samples.
    """
    rng = random.Random(seed)
    person_images = _collect_person_images(data_root)
    source_counts = sorted(set(int(x) for x in source_counts if int(x) > 0))
    if not source_counts:
        raise ValueError("source_counts must contain at least one positive integer.")

    pairs: List[IdentityPair] = []
    skipped: Dict[str, Tuple[int, List[int]]] = {}
    for person, images in person_images.items():
        person_skipped: List[int] = []
        for source_count in source_counts:
            if source_count >= len(images):
                person_skipped.append(source_count)
                continue
            index_pairs = _sample_source_target_pairs(
                images=images,
                source_count=source_count,
                rng=rng,
                max_pairs_per_person=max_pairs_per_person,
            )
            for source_idx, target_idx in index_pairs:
                source_paths = tuple(images[i] for i in source_idx)
                pairs.append(
                    IdentityPair(
                        person=person,
                        source_paths=source_paths,
                        target_path=images[target_idx],
                    )
                )
        if person_skipped:
            skipped[person] = (len(images), person_skipped)

    rng.shuffle(pairs)

    if skipped:
        print("Skipped folders due to insufficient images for requested source_counts:")
        for person, (count, source_list) in sorted(skipped.items(), key=lambda x: x[0]):
            print(f"  {person}: {count} image(s), skipped source_counts={source_list}")

    if max_pairs_total is not None and len(pairs) > max_pairs_total:
        pairs = pairs[:max_pairs_total]

    if not pairs:
        raise ValueError(
            "No valid pairs were created. Check folder structure and source_counts. "
            "Folders with 1 image cannot participate in source_count=1 or higher pairs."
        )
    return pairs


class IdentityPairDataset(Dataset):
    """Torch dataset for identity pair learning."""

    def __init__(
        self,
        pairs: Sequence[IdentityPair],
        image_size: Optional[Tuple[int, int]] = None,
        image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self.pairs = list(pairs)
        self.image_size = image_size
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, path: Path) -> torch.Tensor | Image.Image:
        img = Image.open(path).convert("RGB")
        if self.image_size is not None:
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        if self.image_transform is not None:
            return self.image_transform(img)
        return np_image_to_tensor(img)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.pairs[index]
        source_tensors: List[torch.Tensor] = [
            self._load_image(path) for path in sample.source_paths
        ]
        target_tensor = self._load_image(sample.target_path)
        return {
            "person": sample.person,
            "source_paths": tuple(str(p) for p in sample.source_paths),
            "target_path": str(sample.target_path),
            "source_count": len(sample.source_paths),
            "sources": source_tensors,
            "target": target_tensor,
        }


def identity_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """
    Collate list[dict] into dict[list]. We keep sources as list-of-lists to
    support mixed source_count values (1 source, 2 sources, etc.).
    """
    return {
        "person": [item["person"] for item in batch],
        "source_paths": [item["source_paths"] for item in batch],
        "target_path": [item["target_path"] for item in batch],
        "source_count": [item["source_count"] for item in batch],
        "sources": [item["sources"] for item in batch],
        "target": [item["target"] for item in batch],
    }


def export_pairs_csv(path: Path, pairs: Sequence[IdentityPair]) -> None:
    lines = []
    header = "person,source_count,source_paths,target_path\n"
    lines.append(header)
    for pair in pairs:
        src = " | ".join(str(p) for p in pair.source_paths)
        lines.append(f"{pair.person},{len(pair.source_paths)},\"{src}\",\"{pair.target_path}\"\n")
    path.write_text("".join(lines), encoding="utf-8")


def export_pairs_jsonl(path: Path, pairs: Sequence[IdentityPair]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(
                json.dumps(
                    {
                        "person": pair.person,
                        "source_count": len(pair.source_paths),
                        "source_paths": [str(p) for p in pair.source_paths],
                        "target_path": str(pair.target_path),
                    }
                )
                + "\n"
            )


def _parse_source_counts(raw: str) -> List[int]:
    values: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        parsed = int(part)
        if parsed < 1:
            raise ValueError("source_counts must be positive integers.")
        values.append(parsed)
    if not values:
        raise ValueError("source_counts cannot be empty.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create same-person image source/target pairs from existing folders."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root folder with one folder per person.",
    )
    parser.add_argument(
        "--source-counts",
        default="1,2",
        help="Comma separated source counts. Example: 1 or 1,2 or 2.",
    )
    parser.add_argument(
        "--image-size",
        default="768",
        help="Resize input images to square WxH where W=H. Example: 768 (or 1024).",
    )
    parser.add_argument(
        "--max-pairs-per-person",
        type=int,
        default=None,
        help="Cap pairs per person folder (optional).",
    )
    parser.add_argument(
        "--max-total-pairs",
        type=int,
        default=None,
        help="Cap total pairs across all people (optional).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size for DataLoader demo."
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="PyTorch DataLoader num_workers."
    )
    parser.add_argument(
        "--export",
        default=None,
        help="Optional path to save pair list as JSONL or CSV.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="How many samples to preview from generated pairs.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    source_counts = _parse_source_counts(args.source_counts)
    size = int(args.image_size)
    image_size = (size, size)

    pairs = build_identity_pairs(
        data_root=data_root,
        source_counts=source_counts,
        max_pairs_per_person=args.max_pairs_per_person,
        max_pairs_total=args.max_total_pairs,
        seed=args.seed,
    )

    print(f"Data root: {data_root}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Source counts: {source_counts}")
    by_person = {}
    for p in pairs:
        by_person[p.person] = by_person.get(p.person, 0) + 1
    print("Pairs per person:")
    for person, count in sorted(by_person.items(), key=lambda x: x[0]):
        print(f"  {person}: {count}")

    if args.preview > 0:
        print("\nPreview:")
        for i, pair in enumerate(pairs[: args.preview], start=1):
            print(
                f"{i:02d} | person={pair.person} | source_count={len(pair.source_paths)} "
                f"| sources={[p.name for p in pair.source_paths]} -> target={pair.target_path.name}"
            )

    if args.export:
        export_path = Path(args.export)
        if export_path.suffix.lower() == ".csv":
            export_pairs_csv(export_path, pairs)
        else:
            export_pairs_jsonl(export_path, pairs)
        print(f"Exported pair list: {export_path}")

    dataset = IdentityPairDataset(
        pairs=pairs,
        image_size=image_size,
        image_transform=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=identity_collate_fn,
    )

    print(f"\nCreated DataLoader with {len(loader)} batches.")
    print("Sample batch schema:")
    sample = next(iter(loader))
    print("  keys:", sorted(sample.keys()))
    print("  source_count:", sample["source_count"])
    print("  person:", sample["person"])


if __name__ == "__main__":
    main()

