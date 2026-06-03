# CHANGELOG

## v1.2.1

- Synced vendored `gguf.c` metadata info output updates.

## v1.1.3

- Fixed training graph crash: use `ctx->model.graph_size` instead of `GGML_DEFAULT_GRAPH_SIZE`.
- Fixed training to compute loss over all positions (proper causal LM) instead of only the last token.
- Fixed progress callback: separate lines per step with immediate flush instead of `\r` overwrite.
- Reduced default learning rate from `1e-4` to `1e-5` for stable multi-position training.
- Synced quantizer bugfixes (1D tensor skip, F32 data corruption) from vendored gguf.c.

## v1.1.2

- Removed Gemma4 support from the vendored GGUF dependency and LoRA training dispatch.
- Narrowed current supported architectures to `qwen2`, `qwen2.5`, and `gpt2`.
- Documented that broad model-family support is intentionally out of scope for now because maintaining a large model matrix is not practical.

## v1.1.1

- Fixed LoRA training updates to read live GGML gradients instead of empty accumulators.
- Fixed per-layer LoRA matrix initialization so adapters train the intended transformer layer.
- Fixed safetensors export layout and tensor shapes so trained adapters load correctly in inference.
- Updated vendored GPT-2 graph builder so downstream GPT-2 inference applies LoRA adapters.

## v1.1.0

- Added data-driven configuration lifecycle through `kc_lora_options_t`.
- Added `kc_lora_options_default()`, `kc_lora_options_load_env()`, and `kc_lora_options_free()` to the public API.
- CLI now initializes options through `kc_lora_options_default()` + `kc_lora_options_load_env()`, then overrides with flag values.
- Added signal listener lifecycle: `kc_lora_on_signal()`, `kc_lora_raise_signal()`, `kc_lora_listen_signals()`, `kc_lora_listen_signal()`, and `kc_lora_signal_listener()`.

## v1.0.0

- Published the stable baseline release.
- Provided LoRA adapter training for GGUF models through the CLI and public C API.
