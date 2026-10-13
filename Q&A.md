# Q&A

## 1. Why add `run_zimage_from_env.py`?
**Q:** Can we make Z-Image generation easier like FLUX’s `.env` runner?
**A:** Yes. A new `run_zimage_from_env.py` script was added to load `.env`, print env/help, apply optional presets, and dispatch to single/batch inference scripts.

## 2. Why keep `inference.py` and `batch_inference.py` mostly unchanged?
**Q:** Should we refactor existing scripts to new helpers?
**A:** They already supported `.env` loading. The new runner centralizes the FLUX-like entrypoint behavior without changing existing execution paths, minimizing risk.

## 3. How are presets applied?
**Q:** How do `RUN_PRESET` and preset variables work?
**A:** `run_zimage_from_env.py` sets mode-specific values from preset keys (for example `FAST_STEPS`, `FAST_WIDTH`, etc.) only when the mode variables are not already set.
