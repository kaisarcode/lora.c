# CHANGELOG

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
