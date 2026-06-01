# lora.c - LoRA Adapter Training

`lora.c` trains LoRA (Low-Rank Adaptation) adapters for GGUF transformer models. It produces `.safetensors` files containing low-rank update matrices (A and B) that can be applied during inference using `llm.c`'s `--lora` flag.

---

## CLI

### Examples

Train a LoRA adapter on a text dataset:

```bash
./bin/x86_64/linux/lora model.gguf -o adapter.safetensors -d data.txt
```

With custom rank, alpha, and learning rate:

```bash
./bin/x86_64/linux/lora model.gguf -o adapter.safetensors -d data.txt \
    --rank 32 --alpha 64 --lr 5e-5 --epochs 3
```

Apply the trained adapter during inference:

```bash
echo "prompt" | ./bin/x86_64/linux/llm model.gguf \
    --lora adapter.safetensors --lora-scale 0.8
```

---

### Parameters

| Command/Flag | Description |
| :--- | :--- |
| `<model.gguf>` | Base GGUF model path |
| `-o`, `--output <path>` | Output safetensors adapter path |
| `-d`, `--data <path>` | Plain text training dataset path |
| `--rank <N>` | LoRA rank (default: 16) |
| `--alpha <F>` | LoRA alpha scaling (default: 32.0) |
| `--lr <F>` | Learning rate (default: 1e-4) |
| `--epochs <N>` | Number of training epochs (default: 1) |
| `--batch <N>` | Batch size (default: 1) |
| `--ctx <N>` | Context window in tokens (default: from model) |
| `--threads <N>` | Number of threads (default: auto) |
| `--gpu <N>` | GPU mode: -1 auto, 0 CPU, >0 require |
| `--gpu-layers <N>` | Layers to offload to GPU (default: all) |
| `--save-every <N>` | Checkpoint interval (default: 0 = off) |
| `-h`, `--help` | Show help and usage |
| `-v`, `--version` | Show version |

---

## Public API

```c
#include "lora.h"

kc_lora_options_t opts = {
    .model_path  = "model.gguf",
    .output_path = "adapter.safetensors",
    .rank        = 16,
    .alpha       = 32.0f,
    .lr          = 1e-4f,
    .epochs      = 1,
};

kc_lora_t *ctx = NULL;
kc_lora_open(&ctx, &opts);
kc_lora_run(ctx, "data.txt", progress_cb, NULL);
kc_lora_close(ctx);
```

---

## Lifecycle

- `kc_lora_open()` - loads GGUF model, allocates LoRA A/B matrices, initializes backends.
- `kc_lora_run()` - tokenizes dataset, trains A/B with gradient descent, saves safetensors.
- `kc_lora_stop()` - thread-safe training stop signal.
- `kc_lora_close()` - releases all resources.

---

## Compatibility

`lora.c` supports the GGUF model families implemented by its local graph builders:

- **Qwen-style:** `qwen2`, `qwen2.5` - SiLU activation, RoPE with Qwen freq base
- **GPT-2:** `gpt2` - LayerNorm, learned position embeddings, GELU, non-gated FFN

Unsupported architectures fail during model load with a clear error.

---

## Build

```bash
make
make clean
make all
make x86_64/linux
make x86_64/windows
make i686/linux
make i686/windows
make aarch64/linux
make aarch64/android
make armv7/linux
make armv7/android
make armv7hf/linux
make riscv64/linux
make powerpc64le/linux
make mips/linux
make mipsel/linux
make mips64el/linux
make s390x/linux
make loongarch64/linux
```

Compiled artifacts are generated under `bin/{arch}/{platform}/` for the host architecture running the build.

---

## Beta Notice

This is a beta project tested only on Debian x86_64. It was created out of a personal need for these libraries, but no guarantees are provided regarding its stability or future support. You are free to test it, use it, and modify it as you please.

If you'd like to reach out, you can send an email to kaisar@kaisarcode.com. Please note that I do not accept pull requests; the goal is to avoid long-term dependency on platforms like GitHub, and I do not maintain fixed infrastructure to guarantee long-term stability for these projects.

---

## License

[![GPLv3](https://www.gnu.org/graphics/gplv3-127x51.png)](https://www.gnu.org/licenses/gpl-3.0.html)

This project is distributed under the **GNU General Public License version 3 (GPLv3)**.
