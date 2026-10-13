# Person Identity Pair Dataset Loader

This folder contains a dataset utility that builds training pairs directly from your
existing folder structure without writing any new image files.

Expected input layout:

```
sweet_photos/
  page_001/
    person_a/
      img1.jpg
      img2.jpg
      ...
  page_002/
    person_b/
      ...
```

The loader also supports a flat layout (`sweet_photos/person_name/...`) and will auto-detect person folders recursively.

The generated person id uses the full relative folder path (for example: `page_001/person_a`) so identities remain unique even when similar names appear in multiple page folders.

What it creates (on the fly):
- `A -> B`, `A -> C` for source_count=1
- `A+B -> C`, `A+C -> B` for source_count=2
- and any other source count you pass (for example, `3` if you want 3-source input).

This is exactly the combination strategy you described: multiple source images of the same
person with a target image from the same person, no images are copied/saved.

## Quick usage

```bash
python .\identity_dataset\identity_pair_dataset.py --data-root "D:\VirtualEnvPy\tradebot\UltimateTradeBot\Miscellaneous\FsiBlog\downloads\sweet_photos" --source-counts 1,2 --batch-size 2 --preview 5
```

You can now also set a default root once and run the trainer without `--data-root` by setting it in the local env file:

```bash
notepad identity_dataset\.env
```

(`identity_dataset\.env` supports: `IDENTITY_TRAIN_DATA_ROOT` or `ZIMAGE_IDENTITY_DATA_ROOT`.)

Optional:
- `--max-pairs-per-person 200` to cap each person’s combinations.
- `--max-total-pairs 2000` to cap total dataset size.
- `--export pairs.jsonl` to save pair metadata.

## Integration

You can import and plug into your training loop:

```python
from identity_dataset.identity_pair_dataset import build_identity_pairs, IdentityPairDataset, identity_collate_fn
from torch.utils.data import DataLoader

pairs = build_identity_pairs(data_root=Path("..."), source_counts=(1, 2))
dataset = IdentityPairDataset(pairs, image_size=(768, 768))
loader = DataLoader(dataset, batch_size=4, collate_fn=identity_collate_fn)
```

## Training loop skeleton

For a matching image-conditioned skeleton that builds:
- one-image batches
- two-image batches
- mixed batches containing multiple identities in a batch

run:

```bash
python -m identity_dataset.train_image_conditioned_loop \
  --source-counts 1,2 \
  --mode mixed \
  --batch-size 2 \
  --epochs 1 \
  --max-total-pairs 2000 \
  --output-dir "identity_dataset/checkpoints"
```

`--max-persons-per-epoch` is optional:
- Omit it to train on **all identities** every epoch (full dataset per epoch).
- Set `--max-persons-per-epoch 8` to sample 8 random identities per epoch.

For low GPU utilization, tune the new DataLoader performance knobs first:

```bash
IDENTITY_TRAIN_NUM_WORKERS=4
IDENTITY_TRAIN_PREFETCH_FACTOR=2
IDENTITY_TRAIN_PIN_MEMORY=1
IDENTITY_TRAIN_PERSISTENT_WORKERS=1
IDENTITY_TRAIN_NON_BLOCKING=1
IDENTITY_TRAIN_WARMUP_STEPS=2
```

If `IDENTITY_TRAIN_DATA_ROOT` is set, you can skip `--data-root` in subsequent runs.

This trainer derives prompt text from folder names by default (for each sample, it uses the
identity folder name/path segment as a text prompt). For example:

`...\\page001\\Swimming pool in terrace with rain water for topless bath`
becomes prompt:
`Swimming pool in terrace with rain water for topless bath`

This training script now uses the real Z-Image stack:

- `ZImageTransformer2DModel` for denoising prediction
- `AutoencoderKL` for VAE encode/decode
- Text encoder + tokenizer for caption embeddings
- `FlowMatchEulerDiscreteScheduler` for noise-time pairing

Model path (default) follows your local layout:
`models/Z-Image-Turbo` and reads `transformer`, `vae`, `tokenizer`, and
`text_encoder` shards from the same directory structure used by inference.

The training script now prints live epoch + batch tqdm progress bars (with ETA), plus a global run-level bar for full-progress ETA across all epochs. If `tqdm` is not installed, it gracefully falls back to plain iteration.

All trainer CLI arguments are optional and support `.env` defaults from
`identity_dataset/.env` when CLI flags are omitted. You can set any of:
`IDENTITY_TRAIN_SOURCE_COUNTS`, `IDENTITY_TRAIN_SOURCE_COUNTS_ONE_WEIGHT`,
`IDENTITY_TRAIN_SOURCE_COUNTS_TWO_WEIGHT`, `IDENTITY_TRAIN_IMAGE_SIZE`,
`IDENTITY_TRAIN_MAX_PAIRS_PER_PERSON`, `IDENTITY_TRAIN_MAX_TOTAL_PAIRS`,
`IDENTITY_TRAIN_SEED`, `IDENTITY_TRAIN_BATCH_SIZE`, `IDENTITY_TRAIN_EPOCHS`,
`IDENTITY_TRAIN_LR`, `IDENTITY_TRAIN_MIN_IDENTITIES_PER_BATCH`, `IDENTITY_TRAIN_MODE`,
`IDENTITY_TRAIN_DROP_LAST`, `IDENTITY_TRAIN_NUM_WORKERS`, `IDENTITY_TRAIN_PREFETCH_FACTOR`,
`IDENTITY_TRAIN_PIN_MEMORY`, `IDENTITY_TRAIN_PERSISTENT_WORKERS`,
`IDENTITY_TRAIN_NON_BLOCKING`, `IDENTITY_TRAIN_WARMUP_STEPS`,
`IDENTITY_TRAIN_OUTPUT_DIR`, `IDENTITY_TRAIN_CHECKPOINT_EVERY`,
`IDENTITY_TRAIN_MAX_PERSONS_PER_EPOCH` (default: all),
`IDENTITY_TRAIN_DEVICE` (default: auto)
(and `IDENTITY_TRAIN_DATA_ROOT`/`ZIMAGE_IDENTITY_DATA_ROOT`).

