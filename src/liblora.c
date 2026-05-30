/**
 * liblora.c - LoRA Adapter Training Library
 * Summary: Core logic for LoRA low-rank adaptation training of GGUF models.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif

#include "lora.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <float.h>
#include <stddef.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <signal.h>

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "graph.h"

#define MAX_ERR 1024
#define KC_LORA_GRAPH_BASE 512
#define KC_LORA_GRAPH_PER_LAYER 128
#define KC_LORA_TOKEN_MAX 16384
#define KC_LORA_SAFETENSORS_ALIGN 64

typedef enum {
    KC_ENV_TYPE_INT,
    KC_ENV_TYPE_FLOAT,
    KC_ENV_TYPE_STR,
    KC_ENV_TYPE_DOUBLE,
} kc_env_type_t;

typedef struct {
    const char *env_var;
    size_t offset;
    kc_env_type_t type;
} kc_env_map_t;

static const kc_env_map_t env_config_table[] = {
    { "KC_LORA_MODEL_PATH",  offsetof(kc_lora_options_t, model_path),  KC_ENV_TYPE_STR },
    { "KC_LORA_OUTPUT_PATH", offsetof(kc_lora_options_t, output_path), KC_ENV_TYPE_STR },
    { "KC_LORA_RANK",        offsetof(kc_lora_options_t, rank),        KC_ENV_TYPE_INT },
    { "KC_LORA_ALPHA",       offsetof(kc_lora_options_t, alpha),       KC_ENV_TYPE_FLOAT },
    { "KC_LORA_LR",          offsetof(kc_lora_options_t, lr),          KC_ENV_TYPE_FLOAT },
    { "KC_LORA_EPOCHS",      offsetof(kc_lora_options_t, epochs),      KC_ENV_TYPE_INT },
    { "KC_LORA_BATCH",       offsetof(kc_lora_options_t, batch),       KC_ENV_TYPE_INT },
    { "KC_LORA_CTX",         offsetof(kc_lora_options_t, ctx),         KC_ENV_TYPE_INT },
    { "KC_LORA_THREADS",     offsetof(kc_lora_options_t, threads),     KC_ENV_TYPE_INT },
    { "KC_LORA_GPU",         offsetof(kc_lora_options_t, gpu),         KC_ENV_TYPE_INT },
    { "KC_LORA_GPU_LAYERS",  offsetof(kc_lora_options_t, gpu_layers),  KC_ENV_TYPE_INT },
    { "KC_LORA_SAVE_EVERY",  offsetof(kc_lora_options_t, save_every),  KC_ENV_TYPE_INT },
};
static const int env_config_table_n =
    sizeof(env_config_table) / sizeof(env_config_table[0]);

typedef struct {
    int sig;
    kc_lora_signal_callback_t cb;
} kc_lora_signal_entry_t;

static kc_lora_t *g_signal_ctx = NULL;

typedef enum {
    KC_LORA_PROJ_Q = 0,
    KC_LORA_PROJ_K,
    KC_LORA_PROJ_V,
    KC_LORA_PROJ_O,
    KC_LORA_PROJ_GATE,
    KC_LORA_PROJ_DOWN,
    KC_LORA_PROJ_UP,
    KC_LORA_PROJ_COUNT
} kc_lora_proj_t;

typedef struct {
    struct ggml_tensor *A;
    struct ggml_tensor *B;
    struct ggml_tensor *W;
    int d_in;
    int d_out;
} kc_lora_weight_t;

typedef struct {
    kc_lora_weight_t projs[KC_LORA_PROJ_COUNT];
} kc_lora_layer_t;

struct kc_lora {
    kc_lora_options_t opts;
    char error[MAX_ERR];
    ggml_gallocr_t galloc;
    int n_ctx;
    int n_total_steps;
    kc_gguf_tokenizer_t *tokenizer;
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    ggml_backend_sched_t sched;
    ggml_backend_buffer_t model_buffer;
    ggml_backend_buffer_t cpu_weights_buffer;
    kc_gguf_model_t model;
    kc_lora_layer_t *lora_layers;
    struct ggml_tensor *(*build_graph_fn)(kc_gguf_model_t *m, int n_tokens,
        int n_past, struct ggml_cgraph **gf, struct ggml_tensor **embd_out,
        struct ggml_tensor **pos_out);
    volatile int stop_requested;
    kc_lora_signal_entry_t *signal_handlers;
    int n_signal_handlers;
    int signal_handlers_capacity;
};

static int kc_lora_set_err(kc_lora_t *ctx, const char *fmt, ...);

/**
 * Calculates compute graph capacity from model depth.
 * @param n_layer Transformer layer count.
 * @return Graph node capacity.
 */
static size_t kc_lora_graph_size(int n_layer) {
    size_t layers = n_layer > 0 ? (size_t)n_layer : 1;
    size_t graph_size = KC_LORA_GRAPH_BASE + layers * KC_LORA_GRAPH_PER_LAYER;
    return graph_size > GGML_DEFAULT_GRAPH_SIZE
        ? graph_size
        : GGML_DEFAULT_GRAPH_SIZE;
}

/**
 * Filters verbose GGML logs while preserving warnings and errors.
 * @param level Log severity.
 * @param text Log message.
 * @param user_data User data pointer.
 * @return None.
 */
static void kc_lora_log_callback(enum ggml_log_level level, const char *text,
    void *user_data)
{
    (void)user_data;
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

/**
 * Determines the target compute backend for a specific model tensor.
 * @param ctx Training context.
 * @param name Name of the tensor.
 * @return Best available backend (GPU or CPU) for the tensor.
 */
static ggml_backend_t kc_lora_get_tensor_backend(kc_lora_t *ctx,
    const char *name)
{
    if (ctx->opts.gpu_layers <= 0 || !ctx->backend ||
        ctx->backend == ctx->cpu_backend) {
        return ctx->cpu_backend;
    }
    if (strncmp(name, "blk.", 4) != 0) {
        return ctx->backend;
    }
    int block_idx = 0;
    if (sscanf(name, "blk.%d.", &block_idx) == 1) {
        if (block_idx < ctx->opts.gpu_layers) {
            return ctx->backend;
        }
    }
    return ctx->cpu_backend;
}

/**
 * Allocates model weights across backends (GPU/CPU) based on configuration.
 * @param ctx Training context.
 * @return 0 on success, or -1 on failure.
 */
static int kc_lora_alloc_ctx_weights(kc_lora_t *ctx) {
    int64_t n_tensors = gguf_get_n_tensors(ctx->model.gguf);
    size_t overhead = ggml_tensor_overhead() * (size_t)n_tensors + 4096;
    struct ggml_init_params params = { .mem_size = overhead, .no_alloc = true };
    struct ggml_context *ctx_gpu = ggml_init(params);
    struct ggml_context *ctx_cpu = ggml_init(params);
    if (!ctx_gpu || !ctx_cpu) {
        if (ctx_gpu) ggml_free(ctx_gpu);
        if (ctx_cpu) ggml_free(ctx_cpu);
        return -1;
    }

    for (int64_t i = 0; i < n_tensors; i++) {
        const char *name = gguf_get_tensor_name(ctx->model.gguf, i);
        struct ggml_tensor *t = ggml_get_tensor(ctx->model.ctx_meta, name);
        if (!t) continue;
        ggml_backend_t b = kc_lora_get_tensor_backend(ctx, t->name);
        struct ggml_context *target = (b == ctx->backend) ? ctx_gpu : ctx_cpu;
        struct ggml_tensor *clone = ggml_dup_tensor(target, t);
        ggml_set_name(clone, t->name);
    }

    if (ggml_get_first_tensor(ctx_gpu)) {
        ctx->model_buffer = ggml_backend_alloc_ctx_tensors(ctx_gpu,
            ctx->backend);
        if (!ctx->model_buffer) {
            ggml_free(ctx_gpu);
            ggml_free(ctx_cpu);
            return -1;
        }
        for (int64_t i = 0; i < n_tensors; i++) {
            const char *name = gguf_get_tensor_name(ctx->model.gguf, i);
            struct ggml_tensor *t = ggml_get_tensor(ctx->model.ctx_meta, name);
            struct ggml_tensor *c = ggml_get_tensor(ctx_gpu, name);
            if (t && c && c->data) {
                t->data = c->data;
                t->buffer = c->buffer;
            }
        }
    }

    if (ggml_get_first_tensor(ctx_cpu)) {
        ctx->cpu_weights_buffer = ggml_backend_alloc_ctx_tensors(ctx_cpu,
            ctx->cpu_backend);
        if (!ctx->cpu_weights_buffer) {
            ggml_free(ctx_gpu);
            ggml_free(ctx_cpu);
            return -1;
        }
        for (int64_t i = 0; i < n_tensors; i++) {
            const char *name = gguf_get_tensor_name(ctx->model.gguf, i);
            struct ggml_tensor *t = ggml_get_tensor(ctx->model.ctx_meta, name);
            struct ggml_tensor *c = ggml_get_tensor(ctx_cpu, name);
            if (t && c && c->data) {
                t->data = c->data;
                t->buffer = c->buffer;
            }
        }
    }

    ggml_free(ctx_gpu);
    ggml_free(ctx_cpu);
    return 0;
}

/**
 * Reads GGUF tensor bytes into tensors allocated by a backend buffer.
 * @param ctx Training context receiving the tensor data.
 * @return 0 on success, or -1 on failure.
 */
static int kc_lora_load_backend_tensors(kc_lora_t *ctx) {
    FILE *file = fopen(ctx->opts.model_path, "rb");
    uint8_t *buffer = NULL;

    if (!file) {
        return kc_lora_set_err(ctx, "failed to open model data: %s",
            strerror(errno));
    }

    buffer = malloc(16 * 1024 * 1024);
    if (!buffer) {
        fclose(file);
        return kc_lora_set_err(ctx, "failed to allocate tensor load buffer");
    }

    int64_t n_tensors = gguf_get_n_tensors(ctx->model.gguf);
    size_t data_offset = gguf_get_data_offset(ctx->model.gguf);

    for (int64_t i = 0; i < n_tensors; i++) {
        const char *name = gguf_get_tensor_name(ctx->model.gguf, i);
        struct ggml_tensor *t = ggml_get_tensor(ctx->model.ctx_meta, name);
        if (!t || !t->data) continue;

        size_t offset = data_offset +
            gguf_get_tensor_offset(ctx->model.gguf, i);
        size_t size = ggml_nbytes(t);

        if (fseek(file, (long)offset, SEEK_SET) != 0) {
            free(buffer);
            fclose(file);
            return kc_lora_set_err(ctx, "failed to seek tensor data: %s",
                strerror(errno));
        }

        size_t read = 0;
        while (read < size) {
            size_t chunk = size - read;
            if (chunk > 16 * 1024 * 1024) chunk = 16 * 1024 * 1024;
            size_t n = fread(buffer, 1, chunk, file);
            if (n == 0) {
                free(buffer);
                fclose(file);
                return kc_lora_set_err(ctx,
                    "failed to read tensor data");
            }
            if (ggml_backend_buffer_is_host(t->buffer)) {
                memcpy((char *)t->data + read, buffer, n);
            } else {
                ggml_backend_tensor_set(t, (char *)buffer, read, n);
            }
            read += n;
        }
    }

    free(buffer);
    fclose(file);
    return 0;
}

/**
 * Detects and maps model architecture from GGUF metadata.
 * @param ctx Training context.
 * @return 0 on success, or -1 on failure.
 */
static int kc_lora_detect_arch(kc_lora_t *ctx) {
    char arch[64] = {0};
    if (kc_gguf_get_arch(ctx->model.gguf, arch, sizeof(arch)) != 0) {
        return kc_lora_set_err(ctx, "failed to detect model architecture");
    }

    if (strcmp(arch, "llama") == 0 || strcmp(arch, "mistral") == 0 ||
        strcmp(arch, "mixtral") == 0 || strcmp(arch, "qwen2") == 0 ||
        strcmp(arch, "qwen2.5") == 0 || strcmp(arch, "qwen3") == 0) {
        ctx->build_graph_fn = kc_gguf_build_graph_llama;
    } else if (strcmp(arch, "gemma") == 0) {
        ctx->build_graph_fn = kc_gguf_build_graph_gemma;
    } else if (strcmp(arch, "gpt2") == 0) {
        ctx->build_graph_fn = kc_gguf_build_graph_gpt2;
    } else {
        return kc_lora_set_err(ctx, "unsupported model architecture: %s",
            arch);
    }

    ctx->model.n_vocab = kc_gguf_get_arch_u32(ctx->model.gguf, arch,
        "vocab_size", 0);
    if (ctx->model.n_vocab == 0) {
        int64_t id = gguf_find_key(ctx->model.gguf,
            "tokenizer.ggml.tokens");
        if (id >= 0) {
            ctx->model.n_vocab = (uint32_t)gguf_get_arr_n(
                ctx->model.gguf, id);
        } else {
            ctx->model.n_vocab = 32000;
        }
    }
    ctx->model.n_embd    = kc_gguf_get_arch_u32(ctx->model.gguf, arch,
        "embedding_length", 4096);
    ctx->model.n_head    = kc_gguf_get_arch_u32(ctx->model.gguf, arch,
        "attention.head_count", 32);
    ctx->model.n_head_kv = kc_gguf_get_arch_u32(ctx->model.gguf, arch,
        "attention.head_count_kv", ctx->model.n_head);
    ctx->model.n_layer   = kc_gguf_get_arch_u32(ctx->model.gguf, arch,
        "block_count", 32);
    ctx->model.n_rot     = kc_gguf_get_arch_u32(ctx->model.gguf, arch,
        "rope.dimension_count", 0);
    ctx->model.norm_eps  = kc_gguf_get_arch_f32(ctx->model.gguf, arch,
        "attention.layer_norm_rms_epsilon", 1e-5f);
    ctx->model.rope_freq_base = kc_gguf_get_arch_f32(ctx->model.gguf, arch,
        "rope.freq_base", 10000.0f);
    ctx->model.n_ctx = kc_gguf_get_arch_u32(ctx->model.gguf, arch,
        "context_length", 2048);

    ctx->model.n_head_dim = ctx->model.n_embd / ctx->model.n_head;
    if (ctx->model.n_rot == 0) ctx->model.n_rot = ctx->model.n_head_dim;
    ctx->model.graph_size = kc_lora_graph_size(ctx->model.n_layer);
    return 0;
}

/**
 * Maps GGUF tensor names to model struct fields.
 * @param ctx Training context.
 * @return 0 on success, or -1 on failure.
 */
static int kc_lora_map_tensors(kc_lora_t *ctx) {
    char arch[64] = {0};
    kc_gguf_get_arch(ctx->model.gguf, arch, sizeof(arch));

    ctx->model.tok_embeddings = ggml_get_tensor(ctx->model.ctx_meta,
        "token_embd.weight");
    ctx->model.position_embd_w = ggml_get_tensor(ctx->model.ctx_meta,
        "position_embd.weight");
    ctx->model.output_norm_w  = ggml_get_tensor(ctx->model.ctx_meta,
        "output_norm.weight");
    ctx->model.output_norm_b  = ggml_get_tensor(ctx->model.ctx_meta,
        "output_norm.bias");
    ctx->model.output_w       = ggml_get_tensor(ctx->model.ctx_meta,
        "output.weight");

    if (!ctx->model.tok_embeddings) {
        return kc_lora_set_err(ctx,
            "missing mandatory tensor: token_embd.weight");
    }
    if (!ctx->model.output_norm_w) {
        return kc_lora_set_err(ctx,
            "missing mandatory tensor: output_norm.weight");
    }
    if (!ctx->model.output_w) {
        ctx->model.output_w = ctx->model.tok_embeddings;
    }
    if (strcmp(arch, "gpt2") == 0 && !ctx->model.position_embd_w) {
        return kc_lora_set_err(ctx,
            "missing mandatory tensor: position_embd.weight");
    }

    ctx->model.layers = calloc((size_t)ctx->model.n_layer,
        sizeof(kc_gguf_layer_t));
    if (!ctx->model.layers) {
        return kc_lora_set_err(ctx, "failed to allocate layer context");
    }

    for (int i = 0; i < ctx->model.n_layer; i++) {
        char b[64];
        if (strcmp(arch, "gpt2") == 0) {
            snprintf(b, 64, "blk.%d.attn_qkv.weight", i);
            ctx->model.layers[i].attn_q_w =
                ggml_get_tensor(ctx->model.ctx_meta, b);
            snprintf(b, 64, "blk.%d.attn_qkv.bias", i);
            ctx->model.layers[i].attn_q_b =
                ggml_get_tensor(ctx->model.ctx_meta, b);
        } else {
            snprintf(b, 64, "blk.%d.attn_q.weight", i);
            ctx->model.layers[i].attn_q_w =
                ggml_get_tensor(ctx->model.ctx_meta, b);
            snprintf(b, 64, "blk.%d.attn_k.weight", i);
            ctx->model.layers[i].attn_k_w =
                ggml_get_tensor(ctx->model.ctx_meta, b);
            snprintf(b, 64, "blk.%d.attn_v.weight", i);
            ctx->model.layers[i].attn_v_w =
                ggml_get_tensor(ctx->model.ctx_meta, b);
            snprintf(b, 64, "blk.%d.attn_q.bias", i);
            ctx->model.layers[i].attn_q_b =
                ggml_get_tensor(ctx->model.ctx_meta, b);
            snprintf(b, 64, "blk.%d.attn_k.bias", i);
            ctx->model.layers[i].attn_k_b =
                ggml_get_tensor(ctx->model.ctx_meta, b);
            snprintf(b, 64, "blk.%d.attn_v.bias", i);
            ctx->model.layers[i].attn_v_b =
                ggml_get_tensor(ctx->model.ctx_meta, b);
        }
        snprintf(b, 64, "blk.%d.attn_output.weight", i);
        ctx->model.layers[i].attn_out_w =
            ggml_get_tensor(ctx->model.ctx_meta, b);
        snprintf(b, 64, "blk.%d.attn_output.bias", i);
        ctx->model.layers[i].attn_out_b =
            ggml_get_tensor(ctx->model.ctx_meta, b);
        snprintf(b, 64, "blk.%d.attn_norm.weight", i);
        ctx->model.layers[i].attn_norm_w =
            ggml_get_tensor(ctx->model.ctx_meta, b);
        snprintf(b, 64, "blk.%d.attn_norm.bias", i);
        ctx->model.layers[i].attn_norm_b =
            ggml_get_tensor(ctx->model.ctx_meta, b);
        snprintf(b, 64, "blk.%d.attn_q_norm.weight", i);
        ctx->model.layers[i].attn_q_norm_w =
            ggml_get_tensor(ctx->model.ctx_meta, b);
        snprintf(b, 64, "blk.%d.attn_k_norm.weight", i);
        ctx->model.layers[i].attn_k_norm_w =
            ggml_get_tensor(ctx->model.ctx_meta, b);
        snprintf(b, 64, "blk.%d.ffn_gate.weight", i);
        ctx->model.layers[i].ffn_gate_w =
            ggml_get_tensor(ctx->model.ctx_meta, b);
        snprintf(b, 64, "blk.%d.ffn_down.weight", i);
        ctx->model.layers[i].ffn_down_w =
            ggml_get_tensor(ctx->model.ctx_meta, b);
        snprintf(b, 64, "blk.%d.ffn_up.weight", i);
        ctx->model.layers[i].ffn_up_w =
            ggml_get_tensor(ctx->model.ctx_meta, b);
        snprintf(b, 64, "blk.%d.ffn_norm.weight", i);
        ctx->model.layers[i].ffn_norm_w =
            ggml_get_tensor(ctx->model.ctx_meta, b);
        snprintf(b, 64, "blk.%d.ffn_norm.bias", i);
        ctx->model.layers[i].ffn_norm_b =
            ggml_get_tensor(ctx->model.ctx_meta, b);

        if (strcmp(arch, "gpt2") == 0) {
            if (!ctx->model.layers[i].attn_q_w ||
                !ctx->model.layers[i].attn_out_w ||
                !ctx->model.layers[i].attn_norm_w ||
                !ctx->model.layers[i].ffn_down_w ||
                !ctx->model.layers[i].ffn_up_w ||
                !ctx->model.layers[i].ffn_norm_w) {
                return kc_lora_set_err(ctx,
                    "missing mandatory tensors for layer %d", i);
            }
        } else {
            if (!ctx->model.layers[i].attn_q_w ||
                !ctx->model.layers[i].attn_k_w ||
                !ctx->model.layers[i].attn_v_w ||
                !ctx->model.layers[i].attn_out_w ||
                !ctx->model.layers[i].attn_norm_w ||
                !ctx->model.layers[i].ffn_gate_w ||
                !ctx->model.layers[i].ffn_down_w ||
                !ctx->model.layers[i].ffn_up_w ||
                !ctx->model.layers[i].ffn_norm_w) {
                return kc_lora_set_err(ctx,
                    "missing mandatory tensor in layer %d", i);
            }
        }
    }

    return 0;
}

/**
 * Dequantize model weights to F32 for training.
 * @param ctx Training context.
 * @return 0 on success, or -1 on failure.
 */
static int kc_lora_dequantize_weights(kc_lora_t *ctx) {
    return kc_gguf_dequantize(&ctx->model, GGML_TYPE_F32);
}

/**
 * Creates LoRA A/B matrices for a single projection weight.
 * @param ctx LoRA context.
 * @param cctx GGML context for allocation.
 * @param W The base weight tensor (can be NULL).
 * @param rank LoRA rank.
 * @param d_in Input dimension of the projection.
 * @param d_out Output dimension of the projection.
 * @param lora_idx Index in lora_layers[X].projs[].
 * @return 0 on success, -1 on failure.
 */
static int kc_lora_init_weight(kc_lora_t *ctx, struct ggml_context *cctx,
    struct ggml_tensor *W, int layer_idx, int rank, int d_in, int d_out,
    int lora_idx)
{
    if (layer_idx < 0 || layer_idx >= ctx->model.n_layer) return -1;
    kc_lora_layer_t *ll = &ctx->lora_layers[layer_idx];
    kc_lora_weight_t *w = &ll->projs[lora_idx];
    (void)ctx;

    if (!W) return 0;
    if (d_in <= 0 || d_out <= 0) return -1;

    w->W    = W;
    w->d_in = d_in;
    w->d_out = d_out;

    w->A = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, d_in, rank);
    w->B = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, rank, d_out);

    if (!w->A || !w->B) return -1;

    ggml_set_param(w->A);
    ggml_set_param(w->B);

    return 0;
}

/**
 * Initialize LoRA A/B matrices across all layers and projections.
 * @param ctx Training context.
 * @param cctx GGML context for allocation.
 * @return 0 on success, -1 on failure.
 */
static int kc_lora_init_all(kc_lora_t *ctx, struct ggml_context *cctx) {
    int R = ctx->opts.rank;
    if (R < 1) R = 1;

    int is_gpt2 = (ctx->model.layers[0].attn_k_w == NULL
        && ctx->model.layers[0].attn_q_w != NULL);

    ctx->lora_layers = calloc((size_t)ctx->model.n_layer,
        sizeof(kc_lora_layer_t));
    if (!ctx->lora_layers) return -1;

    for (int i = 0; i < ctx->model.n_layer; i++) {
        kc_lora_layer_t *ll = &ctx->lora_layers[i];
        kc_gguf_layer_t *ml = &ctx->model.layers[i];
        (void)ll;

        if (is_gpt2) {
            int ne0 = (int)ml->attn_q_w->ne[0];
            if (kc_lora_init_weight(ctx, cctx, ml->attn_q_w, i,
                    R, ctx->model.n_embd, ne0,
                    KC_LORA_PROJ_Q) != 0) return -1;
        } else {
            if (ml->attn_q_w) {
                int d_in = (int)ml->attn_q_w->ne[0];
                int d_out = (int)ml->attn_q_w->ne[1];
                if (kc_lora_init_weight(ctx, cctx, ml->attn_q_w, i,
                        R, d_in, d_out,
                        KC_LORA_PROJ_Q) != 0) return -1;
            }
            if (ml->attn_k_w) {
                int d_in = (int)ml->attn_k_w->ne[0];
                int d_out = (int)ml->attn_k_w->ne[1];
                if (kc_lora_init_weight(ctx, cctx, ml->attn_k_w, i,
                        R, d_in, d_out,
                        KC_LORA_PROJ_K) != 0) return -1;
            }
            if (ml->attn_v_w) {
                int d_in = (int)ml->attn_v_w->ne[0];
                int d_out = (int)ml->attn_v_w->ne[1];
                if (kc_lora_init_weight(ctx, cctx, ml->attn_v_w, i,
                        R, d_in, d_out,
                        KC_LORA_PROJ_V) != 0) return -1;
            }
        }

        if (ml->attn_out_w) {
            int d_in = (int)ml->attn_out_w->ne[0];
            int d_out = (int)ml->attn_out_w->ne[1];
            if (kc_lora_init_weight(ctx, cctx, ml->attn_out_w, i,
                    R, d_in, d_out,
                    KC_LORA_PROJ_O) != 0) return -1;
        }
        if (ml->ffn_gate_w) {
            int d_in = (int)ml->ffn_gate_w->ne[0];
            int d_out = (int)ml->ffn_gate_w->ne[1];
            if (kc_lora_init_weight(ctx, cctx, ml->ffn_gate_w, i,
                    R, d_in, d_out,
                    KC_LORA_PROJ_GATE) != 0) return -1;
        }
        if (ml->ffn_down_w) {
            int d_in = (int)ml->ffn_down_w->ne[0];
            int d_out = (int)ml->ffn_down_w->ne[1];
            if (kc_lora_init_weight(ctx, cctx, ml->ffn_down_w, i,
                    R, d_in, d_out,
                    KC_LORA_PROJ_DOWN) != 0) return -1;
        }
        if (ml->ffn_up_w) {
            int d_in = (int)ml->ffn_up_w->ne[0];
            int d_out = (int)ml->ffn_up_w->ne[1];
            if (kc_lora_init_weight(ctx, cctx, ml->ffn_up_w, i,
                    R, d_in, d_out,
                    KC_LORA_PROJ_UP) != 0) return -1;
        }
    }

    return 0;
}

/**
 * Builds the training compute graph with LoRA adapters.
 * @param ctx Training context.
 * @param n_ctx Number of context tokens.
 * @param out_input_ids Receives the input token tensor.
 * @param out_pos_ids Receives the position tensor.
 * @param out_targets Receives the target distribution tensor.
 * @return The built compute graph, or NULL on failure.
 */
static struct ggml_cgraph *kc_lora_build_graph(kc_lora_t *ctx, int n_ctx,
    struct ggml_tensor **out_input_ids,
    struct ggml_tensor **out_pos_ids,
    struct ggml_tensor **out_targets)
{
    struct ggml_context *cctx = ctx->model.ctx_compute;
    float scale = ctx->opts.alpha / (float)ctx->opts.rank;
    int n_layer = ctx->model.n_layer;

    int is_gpt2 = (ctx->model.layers[0].attn_k_w == NULL
        && ctx->model.layers[0].attn_q_w != NULL);

    struct ggml_cgraph *gf = ggml_new_graph_custom(cctx,
        GGML_DEFAULT_GRAPH_SIZE, true);

    struct ggml_tensor *input_ids = ggml_new_tensor_1d(cctx, GGML_TYPE_I32,
        n_ctx);
    ggml_set_input(input_ids);
    if (out_input_ids) *out_input_ids = input_ids;

    struct ggml_tensor *tok_embd = ggml_get_rows(cctx,
        ctx->model.tok_embeddings, input_ids);

    struct ggml_tensor *pos_ids = ggml_new_tensor_1d(cctx, GGML_TYPE_I32,
        n_ctx);
    ggml_set_input(pos_ids);
    if (out_pos_ids) *out_pos_ids = pos_ids;
    struct ggml_tensor *pos_embd = NULL;
    if (ctx->model.position_embd_w) {
        pos_embd = ggml_get_rows(cctx, ctx->model.position_embd_w, pos_ids);
    }

    struct ggml_tensor *cur = tok_embd;
    if (pos_embd) cur = ggml_add(cctx, tok_embd, pos_embd);

    for (int i = 0; i < n_layer; i++) {
        struct ggml_tensor *inp_l = cur;
        cur = ggml_rms_norm(cctx, cur, ctx->model.norm_eps);
        if (ctx->model.layers[i].attn_norm_w) {
            cur = ggml_mul(cctx, cur, ctx->model.layers[i].attn_norm_w);
        }
        if (ctx->model.layers[i].attn_norm_b) {
            cur = ggml_add(cctx, cur, ctx->model.layers[i].attn_norm_b);
        }

        int head_size = ctx->model.n_head_dim;
        int n_head = ctx->model.n_head;
        int n_head_kv = ctx->model.n_head_kv;

        struct ggml_tensor *q_w, *k_w, *v_w;
        struct ggml_tensor *q_b = NULL, *k_b = NULL, *v_b = NULL;

        if (is_gpt2) {
            int ne0 = ctx->model.layers[i].attn_q_w->ne[0];
            int elem_sz = (int)ggml_type_size(
                ctx->model.layers[i].attn_q_w->type);
            q_w = ggml_view_2d(cctx, ctx->model.layers[i].attn_q_w,
                ne0, ctx->model.n_embd,
                ctx->model.layers[i].attn_q_w->nb[1], 0);
            k_w = ggml_view_2d(cctx, ctx->model.layers[i].attn_q_w,
                ne0, ctx->model.n_embd,
                ctx->model.layers[i].attn_q_w->nb[1],
                (size_t)ne0 * elem_sz);
            v_w = ggml_view_2d(cctx, ctx->model.layers[i].attn_q_w,
                ne0, ctx->model.n_embd,
                ctx->model.layers[i].attn_q_w->nb[1],
                (size_t)ne0 * 2 * elem_sz);
            if (ctx->model.layers[i].attn_q_b) {
                int belem = (int)ggml_type_size(
                    ctx->model.layers[i].attn_q_b->type);
                q_b = ggml_view_1d(cctx,
                    ctx->model.layers[i].attn_q_b,
                    ctx->model.n_embd, 0);
                k_b = ggml_view_1d(cctx,
                    ctx->model.layers[i].attn_q_b,
                    ctx->model.n_embd,
                    (size_t)ctx->model.n_embd * belem);
                v_b = ggml_view_1d(cctx,
                    ctx->model.layers[i].attn_q_b,
                    ctx->model.n_embd,
                    (size_t)ctx->model.n_embd * 2 * belem);
            }
        } else {
            q_w = ctx->model.layers[i].attn_q_w;
            k_w = ctx->model.layers[i].attn_k_w;
            v_w = ctx->model.layers[i].attn_v_w;
            q_b = ctx->model.layers[i].attn_q_b;
            k_b = ctx->model.layers[i].attn_k_b;
            v_b = ctx->model.layers[i].attn_v_b;
        }

        kc_lora_weight_t *w_q = &ctx->lora_layers[i]
            .projs[KC_LORA_PROJ_Q];
        kc_lora_weight_t *w_k = &ctx->lora_layers[i]
            .projs[KC_LORA_PROJ_K];
        kc_lora_weight_t *w_v = &ctx->lora_layers[i]
            .projs[KC_LORA_PROJ_V];

        struct ggml_tensor *Q = (q_w && w_q->A && w_q->B)
            ? ggml_add(cctx, ggml_mul_mat(cctx, q_w, cur),
                ggml_scale(cctx, ggml_mul_mat(cctx,
                    w_q->B, ggml_mul_mat(cctx, w_q->A, cur)), scale))
            : (q_w ? ggml_mul_mat(cctx, q_w, cur) : cur);
        if (q_b) Q = ggml_add(cctx, Q, q_b);

        struct ggml_tensor *K = (k_w && w_k->A && w_k->B)
            ? ggml_add(cctx, ggml_mul_mat(cctx, k_w, cur),
                ggml_scale(cctx, ggml_mul_mat(cctx,
                    w_k->B, ggml_mul_mat(cctx, w_k->A, cur)), scale))
            : (k_w ? ggml_mul_mat(cctx, k_w, cur) : cur);
        if (k_b) K = ggml_add(cctx, K, k_b);

        struct ggml_tensor *V = (v_w && w_v->A && w_v->B)
            ? ggml_add(cctx, ggml_mul_mat(cctx, v_w, cur),
                ggml_scale(cctx, ggml_mul_mat(cctx,
                    w_v->B, ggml_mul_mat(cctx, w_v->A, cur)), scale))
            : (v_w ? ggml_mul_mat(cctx, v_w, cur) : cur);
        if (v_b) V = ggml_add(cctx, V, v_b);

        Q = ggml_reshape_3d(cctx, ggml_cont(cctx, Q),
            head_size, n_head, n_ctx);
        K = ggml_reshape_3d(cctx, ggml_cont(cctx, K),
            head_size, n_head_kv, n_ctx);
        V = ggml_reshape_3d(cctx, ggml_cont(cctx, V),
            head_size, n_head_kv, n_ctx);

        struct ggml_tensor *kq = ggml_mul_mat(cctx,
            ggml_cont(cctx, ggml_permute(cctx, K, 0, 2, 1, 3)),
            ggml_cont(cctx, ggml_permute(cctx, Q, 0, 2, 1, 3)));

        kq = ggml_diag_mask_inf(cctx,
            ggml_scale(cctx, kq, 1.0f / sqrtf((float)head_size)), 0);
        kq = ggml_soft_max(cctx, kq);

        struct ggml_tensor *v_att = ggml_permute(cctx,
            ggml_mul_mat(cctx,
                ggml_cont(cctx,
                    ggml_permute(cctx, V, 1, 2, 0, 3)),
                ggml_cont(cctx, kq)),
            0, 2, 1, 3);

        kc_lora_weight_t *w_o = &ctx->lora_layers[i]
            .projs[KC_LORA_PROJ_O];
        struct ggml_tensor *attn_out = ggml_reshape_2d(cctx,
            ggml_cont(cctx, v_att), n_head * head_size, n_ctx);
        if (ctx->model.layers[i].attn_out_w) {
            attn_out = (w_o->A && w_o->B)
                ? ggml_add(cctx,
                    ggml_mul_mat(cctx,
                        ctx->model.layers[i].attn_out_w, attn_out),
                    ggml_scale(cctx, ggml_mul_mat(cctx,
                        w_o->B, ggml_mul_mat(cctx,
                            w_o->A, attn_out)), scale))
                : ggml_mul_mat(cctx,
                    ctx->model.layers[i].attn_out_w, attn_out);
        }
        cur = ggml_add(cctx, attn_out, inp_l);
        if (ctx->model.layers[i].attn_out_b) {
            cur = ggml_add(cctx, cur, ctx->model.layers[i].attn_out_b);
        }

        struct ggml_tensor *inp_ffn = cur;
        cur = ggml_rms_norm(cctx, cur, ctx->model.norm_eps);
        if (ctx->model.layers[i].ffn_norm_w) {
            cur = ggml_mul(cctx, cur, ctx->model.layers[i].ffn_norm_w);
        }
        if (ctx->model.layers[i].ffn_norm_b) {
            cur = ggml_add(cctx, cur, ctx->model.layers[i].ffn_norm_b);
        }

        kc_lora_weight_t *w_up = &ctx->lora_layers[i]
            .projs[KC_LORA_PROJ_UP];
        kc_lora_weight_t *w_gate = &ctx->lora_layers[i]
            .projs[KC_LORA_PROJ_GATE];
        kc_lora_weight_t *w_down = &ctx->lora_layers[i]
            .projs[KC_LORA_PROJ_DOWN];

        struct ggml_tensor *up = (ctx->model.layers[i].ffn_up_w
                && w_up->A && w_up->B)
            ? ggml_add(cctx,
                ggml_mul_mat(cctx, ctx->model.layers[i].ffn_up_w, cur),
                ggml_scale(cctx, ggml_mul_mat(cctx,
                    w_up->B, ggml_mul_mat(cctx, w_up->A, cur)), scale))
            : (ctx->model.layers[i].ffn_up_w
                ? ggml_mul_mat(cctx,
                    ctx->model.layers[i].ffn_up_w, cur) : cur);

        struct ggml_tensor *act = ggml_silu(cctx, up);

        struct ggml_tensor *gate = (ctx->model.layers[i].ffn_gate_w
                && w_gate->A && w_gate->B)
            ? ggml_add(cctx,
                ggml_mul_mat(cctx,
                    ctx->model.layers[i].ffn_gate_w, cur),
                ggml_scale(cctx, ggml_mul_mat(cctx,
                    w_gate->B, ggml_mul_mat(cctx,
                        w_gate->A, cur)), scale))
            : (ctx->model.layers[i].ffn_gate_w
                ? ggml_mul_mat(cctx,
                    ctx->model.layers[i].ffn_gate_w, cur)
                : NULL);
        if (gate) {
            act = ggml_mul(cctx, act, gate);
        }

        struct ggml_tensor *ffn_down = (ctx->model.layers[i].ffn_down_w
                && w_down->A && w_down->B)
            ? ggml_add(cctx,
                ggml_mul_mat(cctx,
                    ctx->model.layers[i].ffn_down_w, act),
                ggml_scale(cctx, ggml_mul_mat(cctx,
                    w_down->B, ggml_mul_mat(cctx,
                        w_down->A, act)), scale))
            : ggml_mul_mat(cctx,
                ctx->model.layers[i].ffn_down_w, act);
        cur = ggml_add(cctx, ffn_down, inp_ffn);
    }

        cur = ggml_rms_norm(cctx, cur, ctx->model.norm_eps);
    if (ctx->model.output_norm_w) {
        cur = ggml_mul(cctx, cur, ctx->model.output_norm_w);
    }
    if (ctx->model.output_norm_b) {
        cur = ggml_add(cctx, cur, ctx->model.output_norm_b);
    }

    struct ggml_tensor *logits = ggml_mul_mat(cctx, ctx->model.output_w,
        ggml_view_2d(cctx, cur, ctx->model.n_embd, 1,
            cur->nb[1], (n_ctx - 1) * cur->nb[1]));

    struct ggml_tensor *logits_f32 = (logits->type == GGML_TYPE_F32)
        ? logits : ggml_cast(cctx, logits, GGML_TYPE_F32);

    struct ggml_tensor *targets = ggml_dup_tensor(cctx, logits_f32);
    ggml_set_input(targets);
    if (out_targets) *out_targets = targets;

    struct ggml_tensor *loss = ggml_cross_entropy_loss(cctx,
        logits_f32, targets);
    ggml_set_loss(loss);

    ggml_build_forward_expand(gf, loss);
    ggml_build_backward_expand(cctx, gf, NULL);

    return gf;
}

/**
 * Initialize LoRA A matrix with Kaiming normal distribution.
 * @param data Buffer of floats (rank * d_in).
 * @param n Number of elements.
 * @return void
 */
static void kc_lora_init_A(float *data, int n) {
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < n; i++) {
        double u = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double v = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double z = sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
        data[i] = (float)z;
        sum += data[i];
        sum2 += (double)data[i] * data[i];
    }
    double mean = sum / n;
    double std = sqrt(sum2 / n - mean * mean);
    if (std > 0.0) {
        double inv_std = 1.0 / std;
        for (int i = 0; i < n; i++) {
            data[i] = (float)((data[i] - mean) * inv_std);
        }
    }
}

/**
 * Initialize LoRA B matrix with zeros.
 * @param data Buffer of floats (d_out * rank).
 * @param n Number of elements.
 * @return void
 */
static void kc_lora_init_B(float *data, int n) {
    for (int i = 0; i < n; i++) data[i] = 0.0f;
}

/**
 * Write safetensors header JSON string.
 * @param ctx Training context.
 * @param buf Output buffer (caller allocates).
 * @param buf_size Size of output buffer.
 * @return 0 on success, -1 on failure.
 */
static int kc_lora_build_safetensors_hdr(kc_lora_t *ctx,
    char *buf, size_t buf_size)
{
    char arch[64] = {0};
    kc_gguf_get_arch(ctx->model.gguf, arch, sizeof(arch));

    size_t pos = 0;
    int ret = snprintf(buf + pos, buf_size - pos, "{");
    if (ret < 0) return -1;
    pos += (size_t)ret;

    uint64_t data_off = 0;

    const char *proj_names[KC_LORA_PROJ_COUNT] = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"
    };

    for (int i = 0; i < ctx->model.n_layer; i++) {
        for (int p = 0; p < KC_LORA_PROJ_COUNT; p++) {
            kc_lora_weight_t *w = &ctx->lora_layers[i].projs[p];
            if (!w->A || !w->B) continue;

            int d_in = w->d_in;
            int d_out = w->d_out;
            int R = ctx->opts.rank;

            for (int ab = 0; ab < 2; ab++) {
                const char *suffix = (ab == 0) ? "lora_A" : "lora_B";
                int rows = (ab == 0) ? R : d_out;
                int cols = (ab == 0) ? d_in : R;
                int64_t n_elems = (int64_t)rows * cols;
                int64_t byte_size = n_elems * (int64_t)sizeof(float);

                if (pos > 0 && buf[pos - 1] != '{') {
                    if (pos >= buf_size) return -1;
                    buf[pos++] = ',';
                }

                ret = snprintf(buf + pos, buf_size - pos,
                    "\"%s.layers.%d.%s.%s\":{\"shape\":[%d,%d],"
                    "\"data_offsets\":[%" PRIu64 ",%" PRIu64 "]}",
                    arch, i, proj_names[p], suffix,
                    rows, cols, data_off, data_off + byte_size);
                if (ret < 0) return -1;
                pos += (size_t)ret;
                if (pos >= buf_size) return -1;

                data_off += byte_size;
            }
        }
    }

    ret = snprintf(buf + pos, buf_size - pos, ",\"__metadata__\":{"
        "\"format\":\"pt\"}}");
    if (ret < 0) return -1;
    pos += (size_t)ret;
    if (pos >= buf_size) return -1;

    return 0;
}

/**
 * Write the trained LoRA adapter to a safetensors file.
 * @param ctx Training context.
 * @param path Output file path.
 * @return 0 on success, -1 on failure.
 */
static int kc_lora_save_safetensors(kc_lora_t *ctx, const char *path) {
    char hdr_buf[262144];
    memset(hdr_buf, 0, sizeof(hdr_buf));

    if (kc_lora_build_safetensors_hdr(ctx, hdr_buf, sizeof(hdr_buf)) != 0) {
        return kc_lora_set_err(ctx,
            "failed to build safetensors header");
    }

    uint64_t hdr_len = (uint64_t)strlen(hdr_buf);

    FILE *f = fopen(path, "wb");
    if (!f) {
        return kc_lora_set_err(ctx, "failed to open output: %s",
            strerror(errno));
    }

    if (fwrite(&hdr_len, 8, 1, f) != 1) {
        fclose(f);
        return kc_lora_set_err(ctx, "failed to write header length");
    }

    if (fwrite(hdr_buf, 1, hdr_len, f) != hdr_len) {
        fclose(f);
        return kc_lora_set_err(ctx, "failed to write header");
    }

    for (int i = 0; i < ctx->model.n_layer; i++) {
        for (int p = 0; p < KC_LORA_PROJ_COUNT; p++) {
            kc_lora_weight_t *w = &ctx->lora_layers[i].projs[p];
            if (!w->A || !w->B) continue;

            int R = ctx->opts.rank;
            int n_a = w->d_in * R;
            int n_b = w->d_out * R;

            float *buf_a = malloc((size_t)n_a * sizeof(float));
            float *buf_b = malloc((size_t)n_b * sizeof(float));
            if (!buf_a || !buf_b) {
                free(buf_a); free(buf_b);
                fclose(f);
                return kc_lora_set_err(ctx, "out of memory");
            }

            ggml_backend_tensor_get(w->A, buf_a, 0,
                (size_t)n_a * sizeof(float));
            ggml_backend_tensor_get(w->B, buf_b, 0,
                (size_t)n_b * sizeof(float));

            if (fwrite(buf_a, sizeof(float), (size_t)n_a, f) != (size_t)n_a) {
                free(buf_a); free(buf_b);
                fclose(f);
                return kc_lora_set_err(ctx,
                    "failed to write tensor data");
            }

            if (fwrite(buf_b, sizeof(float), (size_t)n_b, f) != (size_t)n_b) {
                free(buf_a); free(buf_b);
                fclose(f);
                return kc_lora_set_err(ctx,
                    "failed to write tensor data");
            }

            free(buf_a);
            free(buf_b);
        }
    }

    fclose(f);
    fprintf(stderr, "lora: saved adapter to '%s'\n", path);
    return 0;
}

/**
 * Create options initialized with default values.
 * @return Default-initialized options.
 */
kc_lora_options_t kc_lora_options_default(void) {
    kc_lora_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.rank = 16;
    opts.alpha = 32.0f;
    opts.lr = 1e-4f;
    opts.epochs = 1;
    opts.batch = 1;
    opts.gpu = -1;
    opts.gpu_layers = 999;
    return opts;
}

/**
 * Load configuration from environment variables.
 * @param opts Options to update.
 * @return None.
 */
void kc_lora_options_load_env(kc_lora_options_t *opts) {
    int i;
    if (!opts) return;
    for (i = 0; i < env_config_table_n; i++) {
        const char *val = getenv(env_config_table[i].env_var);
        char *end;
        if (!val) continue;
        switch (env_config_table[i].type) {
            case KC_ENV_TYPE_INT: {
                long v = strtol(val, &end, 10);
                if (end != val && *end == '\0')
                    *(int *)((char *)opts + env_config_table[i].offset) = (int)v;
                break;
            }
            case KC_ENV_TYPE_FLOAT: {
                float v = strtof(val, &end);
                if (end != val && *end == '\0')
                    *(float *)((char *)opts + env_config_table[i].offset) = v;
                break;
            }
            case KC_ENV_TYPE_DOUBLE: {
                double v = strtod(val, &end);
                if (end != val && *end == '\0')
                    *(double *)((char *)opts + env_config_table[i].offset) = v;
                break;
            }
            case KC_ENV_TYPE_STR: {
                const char **p = (const char **)((char *)opts + env_config_table[i].offset);
                (void)p;
                break;
            }
        }
    }
}

/**
 * Free dynamically allocated resources within an options struct.
 * @param opts Options to clean up.
 * @return None.
 */
void kc_lora_options_free(kc_lora_options_t *opts) {
    (void)opts;
}

/**
 * Register a handler for a library-level signal number.
 * @param ctx LoRA training context.
 * @param sig Application-defined signal number.
 * @param cb Callback to invoke.
 * @return KC_LORA_OK on success, or KC_LORA_ERROR on failure.
 */
int kc_lora_on_signal(kc_lora_t *ctx, int sig, kc_lora_signal_callback_t cb) {
    int i;
    if (!ctx) return KC_LORA_ERROR;
    for (i = 0; i < ctx->n_signal_handlers; i++) {
        if (ctx->signal_handlers[i].sig == sig) {
            if (cb) {
                ctx->signal_handlers[i].cb = cb;
            } else {
                int tail = ctx->n_signal_handlers - i - 1;
                if (tail > 0)
                    memmove(&ctx->signal_handlers[i], &ctx->signal_handlers[i+1],
                            (size_t)tail * sizeof(kc_lora_signal_entry_t));
                ctx->n_signal_handlers--;
            }
            return KC_LORA_OK;
        }
    }
    if (!cb) return KC_LORA_OK;
    if (ctx->n_signal_handlers >= ctx->signal_handlers_capacity) {
        int nc = ctx->signal_handlers_capacity ? ctx->signal_handlers_capacity * 2 : 4;
        kc_lora_signal_entry_t *p = realloc(ctx->signal_handlers,
            (size_t)nc * sizeof(kc_lora_signal_entry_t));
        if (!p) return KC_LORA_ERROR;
        ctx->signal_handlers = p;
        ctx->signal_handlers_capacity = nc;
    }
    ctx->signal_handlers[ctx->n_signal_handlers].sig = sig;
    ctx->signal_handlers[ctx->n_signal_handlers].cb = cb;
    ctx->n_signal_handlers++;
    return KC_LORA_OK;
}

/**
 * Raise a library-level signal.
 * @param ctx LoRA training context.
 * @param sig Signal number to raise.
 * @return KC_LORA_OK if handled, or KC_LORA_ERROR if no handler.
 */
int kc_lora_raise_signal(kc_lora_t *ctx, int sig) {
    int i;
    if (!ctx) return KC_LORA_ERROR;
    for (i = 0; i < ctx->n_signal_handlers; i++)
        if (ctx->signal_handlers[i].sig == sig) {
            ctx->signal_handlers[i].cb(ctx);
            return KC_LORA_OK;
        }
    return KC_LORA_ERROR;
}

/**
 * Set the internal signal-listener context.
 * @param ctx LoRA training context.
 * @return KC_LORA_OK on success, or KC_LORA_ERROR if ctx is NULL.
 */
int kc_lora_listen_signals(kc_lora_t *ctx) {
    if (!ctx) return KC_LORA_ERROR;
    g_signal_ctx = ctx;
    return KC_LORA_OK;
}

/**
 * Wire an OS signal to the library signal listener.
 * @param ctx LoRA training context.
 * @param sig_id OS signal number.
 * @return KC_LORA_OK on success, or KC_LORA_ERROR on failure.
 */
int kc_lora_listen_signal(kc_lora_t *ctx, int sig_id) {
    if (!ctx) return KC_LORA_ERROR;
    g_signal_ctx = ctx;
#ifndef _WIN32
    signal(sig_id, kc_lora_signal_listener);
#else
    (void)sig_id;
#endif
    return KC_LORA_OK;
}

/**
 * Generic signal-listener compatible with signal() / sigaction().
 * @param sig OS signal number.
 * @return None.
 */
void kc_lora_signal_listener(int sig) {
    if (g_signal_ctx)
        kc_lora_raise_signal(g_signal_ctx, sig);
}

/**
 * Open a new training context with the specified options.
 * @param out Pointer to receive the allocated context.
 * @param opts Initialization options.
 * @return 0 on success, non-zero on failure.
 */
int kc_lora_open(kc_lora_t **out, const kc_lora_options_t *opts) {
    kc_lora_t *ctx = calloc(1, sizeof(kc_lora_t));
    if (!ctx) return KC_LORA_ERROR;

    memcpy(&ctx->opts, opts, sizeof(kc_lora_options_t));

    ggml_log_set(kc_lora_log_callback, NULL);

    ctx->cpu_backend = ggml_backend_cpu_init();
    if (!ctx->cpu_backend) {
        kc_lora_set_err(ctx, "failed to init CPU backend");
        kc_lora_close(ctx);
        return KC_LORA_ERROR;
    }

#ifdef GGML_USE_CUDA
    if (opts->gpu != 0) {
        ctx->backend = ggml_backend_cuda_init(0);
    }
#endif
    if (!ctx->backend) {
        ctx->backend = ctx->cpu_backend;
    }

    if (ctx->backend != ctx->cpu_backend) {
        ggml_backend_t backends[] = { ctx->backend, ctx->cpu_backend };
        ctx->sched = ggml_backend_sched_new(backends, NULL, 2,
            GGML_DEFAULT_GRAPH_SIZE, false, false);
    } else {
        ggml_backend_t backends[] = { ctx->cpu_backend };
        ctx->sched = ggml_backend_sched_new(backends, NULL, 1,
            GGML_DEFAULT_GRAPH_SIZE, false, false);
    }

    if (!ctx->sched) {
        kc_lora_set_err(ctx, "failed to create backend scheduler");
        kc_lora_close(ctx);
        return KC_LORA_ERROR;
    }

    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx = &ctx->model.ctx_meta
    };
    ctx->model.gguf = gguf_init_from_file(opts->model_path, params);
    if (!ctx->model.gguf) {
        kc_lora_set_err(ctx, "failed to open GGUF file");
        kc_lora_close(ctx);
        return KC_LORA_ERROR;
    }

    if (kc_lora_detect_arch(ctx) != 0) {
        fprintf(stderr, "detect_arch failed: %s\n", ctx->error);
        kc_lora_close(ctx);
        return KC_LORA_ERROR;
    }
    ctx->n_ctx = opts->ctx > 0 ? opts->ctx : ctx->model.n_ctx;

    if (kc_lora_alloc_ctx_weights(ctx) != 0) {
        fprintf(stderr, "alloc_ctx_weights failed\n");
        kc_lora_set_err(ctx, "failed to allocate model weights");
        kc_lora_close(ctx);
        return KC_LORA_ERROR;
    }

    if (kc_lora_load_backend_tensors(ctx) != 0) {
        fprintf(stderr, "load_backend_tensors failed: %s\n", ctx->error);
        kc_lora_close(ctx);
        return KC_LORA_ERROR;
    }

    if (kc_lora_map_tensors(ctx) != 0) {
        fprintf(stderr, "map_tensors failed: %s\n", ctx->error);
        kc_lora_close(ctx);
        return KC_LORA_ERROR;
    }

    char err[256] = {0};
    if (kc_gguf_tokenizer_load(&ctx->tokenizer, ctx->model.gguf,
        ctx->model.n_vocab, err, sizeof(err)) != 0)
    {
        fprintf(stderr, "tokenizer_load failed: %s\n", err);
        kc_lora_set_err(ctx, "failed to load tokenizer: %s", err);
        kc_lora_close(ctx);
        return KC_LORA_ERROR;
    }

    *out = ctx;
    fprintf(stderr, "model loaded: %ld tensors, n_vocab=%u, n_embd=%u, "
        "n_layer=%d\n",
        (long)gguf_get_n_tensors(ctx->model.gguf), ctx->model.n_vocab,
        ctx->model.n_embd, ctx->model.n_layer);
    return KC_LORA_OK;
}

/**
 * Close and release the training context.
 * @param ctx Context to close.
 * @return 0 on success, non-zero on failure.
 */
int kc_lora_close(kc_lora_t *ctx) {
    if (!ctx) return KC_LORA_OK;

    if (ctx->tokenizer) kc_gguf_tokenizer_free(ctx->tokenizer);
    if (ctx->model.layers) free(ctx->model.layers);
    if (ctx->lora_layers) {
        free(ctx->lora_layers);
    }
    if (ctx->model.gguf) gguf_free(ctx->model.gguf);
    if (ctx->model.ctx_meta) ggml_free(ctx->model.ctx_meta);
    if (ctx->model_buffer) ggml_backend_buffer_free(ctx->model_buffer);
    if (ctx->cpu_weights_buffer) {
        ggml_backend_buffer_free(ctx->cpu_weights_buffer);
    }
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->galloc) ggml_gallocr_free(ctx->galloc);
    if (ctx->backend && ctx->backend != ctx->cpu_backend)
        ggml_backend_free(ctx->backend);
    if (ctx->cpu_backend) ggml_backend_free(ctx->cpu_backend);

    free(ctx->signal_handlers);
    free(ctx);
    return KC_LORA_OK;
}

/**
 * Run LoRA training on the provided dataset.
 * @param ctx Training context.
 * @param data_path Path to the plain text dataset file.
 * @param progress Callback function called with progress updates.
 * @param user User-supplied pointer passed back to the callback.
 * @return 0 on success, or -1 on failure.
 */
int kc_lora_run(kc_lora_t *ctx, const char *data_path,
    kc_lora_progress_fn progress, void *user)
{
    if (!ctx || !data_path) return KC_LORA_ERROR;
    fprintf(stderr, "kc_lora_run start\n");

    FILE *f = fopen(data_path, "r");
    if (!f) {
        return kc_lora_set_err(ctx, "failed to open dataset: %s",
            strerror(errno));
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *data = malloc((size_t)fsize + 1);
    if (!data) {
        fclose(f);
        return kc_lora_set_err(ctx, "failed to allocate dataset buffer");
    }
    size_t nread = fread(data, 1, (size_t)fsize, f);
    data[nread] = '\0';
    fclose(f);

    int *tokens = malloc((size_t)ctx->n_ctx * sizeof(int));
    if (!tokens) {
        free(data);
        return kc_lora_set_err(ctx, "failed to allocate token buffer");
    }

    char err[256] = {0};
    int n_tokens = kc_gguf_tokenizer_encode(ctx->tokenizer, data, tokens,
        ctx->n_ctx, err, sizeof(err));
    free(data);
    fprintf(stderr, "tokens: %d\n", n_tokens);

    if (n_tokens <= 1) {
        free(tokens);
        return kc_lora_set_err(ctx,
            "dataset produced no tokens: %s", err);
    }

    int n_ctx = ctx->n_ctx;
    if (n_tokens < n_ctx) n_ctx = n_tokens;
    if (n_ctx < 2) {
        free(tokens);
        return kc_lora_set_err(ctx,
            "context too small for training (need >= 2)");
    }

    if (kc_lora_dequantize_weights(ctx) != 0) {
        free(tokens);
        return kc_lora_set_err(ctx,
            "failed to dequantize weights to F32");
    }
    fprintf(stderr, "dequantize ok\n");

    size_t lora_ctx_size = (size_t)(ctx->model.n_layer * KC_LORA_PROJ_COUNT
        * 2 + 64) * ggml_tensor_overhead();
    lora_ctx_size += (size_t)512 * 1024 * 1024;

    struct ggml_init_params params_compute = {
        .mem_size = lora_ctx_size,
        .no_alloc = true
    };
    struct ggml_context *ctx_compute = ggml_init(params_compute);
    if (!ctx_compute) {
        free(tokens);
        return kc_lora_set_err(ctx,
            "failed to allocate compute context");
    }

    ctx->model.ctx_compute = ctx_compute;

    int head_size = ctx->model.n_head_dim;
    for (int i = 0; i < ctx->model.n_layer; i++) {
        ctx->model.layers[i].k_cache = ggml_new_tensor_3d(ctx_compute,
            GGML_TYPE_F32, head_size, ctx->model.n_head_kv, n_ctx);
        ctx->model.layers[i].v_cache = ggml_new_tensor_3d(ctx_compute,
            GGML_TYPE_F32, head_size, ctx->model.n_head_kv, n_ctx);
    }

    if (kc_lora_init_all(ctx, ctx_compute) != 0) {
        free(tokens);
        ggml_free(ctx_compute);
        return kc_lora_set_err(ctx, "failed to init LoRA matrices");
    }

    struct ggml_tensor *t_input_ids = NULL;
    struct ggml_tensor *t_pos_ids = NULL;
    struct ggml_tensor *t_targets = NULL;

    struct ggml_cgraph *gf = kc_lora_build_graph(ctx, n_ctx,
        &t_input_ids, &t_pos_ids, &t_targets);
    if (!gf || !t_input_ids || !t_targets) {
        free(tokens);
        ggml_free(ctx_compute);
        return kc_lora_set_err(ctx, "failed to build training graph");
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_compute,
        ctx->cpu_backend);
    if (!buf) {
        free(tokens);
        ggml_free(ctx_compute);
        return kc_lora_set_err(ctx,
            "failed to allocate compute buffer");
    }

    for (int i = 0; i < ctx->model.n_layer; i++) {
        for (int p = 0; p < KC_LORA_PROJ_COUNT; p++) {
            kc_lora_weight_t *w = &ctx->lora_layers[i].projs[p];
            if (!w->A || !w->B) continue;
            int n_a = w->d_in * ctx->opts.rank;
            int n_b = w->d_out * ctx->opts.rank;
            if (n_a > 0) {
                float *tmp = malloc((size_t)n_a * sizeof(float));
                if (tmp) {
                    kc_lora_init_A(tmp, n_a);
                    ggml_backend_tensor_set(w->A, tmp, 0,
                        (size_t)n_a * sizeof(float));
                    free(tmp);
                }
            }
            if (n_b > 0) {
                float *tmp = malloc((size_t)n_b * sizeof(float));
                if (tmp) {
                    kc_lora_init_B(tmp, n_b);
                    ggml_backend_tensor_set(w->B, tmp, 0,
                        (size_t)n_b * sizeof(float));
                    free(tmp);
                }
            }
        }
    }

    int n_steps = ctx->opts.epochs;
    float lr = ctx->opts.lr;
    int total_steps = n_steps;

    int *pos_ids_data = malloc((size_t)n_ctx * sizeof(int));
    for (int i = 0; i < n_ctx; i++) pos_ids_data[i] = i;

    int n_logits = 1;
    size_t target_sz = (size_t)ctx->model.n_vocab * n_logits *
        sizeof(float);
    float *target_data = calloc(
        (size_t)ctx->model.n_vocab * n_logits, sizeof(float));

    kc_lora_weight_t *all_weights[2048];
    int n_all_weights = 0;
    for (int i = 0; i < ctx->model.n_layer; i++) {
        for (int p = 0; p < KC_LORA_PROJ_COUNT; p++) {
            kc_lora_weight_t *w = &ctx->lora_layers[i].projs[p];
            if (w->A && w->B) {
                all_weights[n_all_weights++] = w;
            }
        }
    }

    fprintf(stderr, "starting %d training steps, rank=%d, alpha=%.1f, "
        "lr=%e\n", n_steps, ctx->opts.rank, ctx->opts.alpha, lr);

    for (int step = 0; step < n_steps; step++) {
        if (ctx->stop_requested) {
            fprintf(stderr, "\nlora: training stopped by user\n");
            break;
        }

        int target_token = tokens[n_ctx - 1];
        memset(target_data, 0, target_sz);
        target_data[target_token] = 1.0f;

        ggml_backend_tensor_set(t_input_ids, tokens, 0,
            (size_t)n_ctx * sizeof(int));
        if (t_pos_ids) {
            ggml_backend_tensor_set(t_pos_ids, pos_ids_data, 0,
                (size_t)n_ctx * sizeof(int));
        }
        ggml_backend_tensor_set(t_targets, target_data, 0, target_sz);

        ggml_graph_reset(gf);
        ggml_backend_graph_compute(ctx->cpu_backend, gf);

        float loss_val = 0.0f;
        struct ggml_tensor *loss_t = NULL;
        struct ggml_tensor *scan = ggml_get_first_tensor(ctx_compute);
        while (scan) {
            if (scan->flags & GGML_TENSOR_FLAG_LOSS) {
                loss_t = scan;
                break;
            }
            if (strcmp(scan->name, "loss") == 0) {
                loss_t = scan;
                break;
            }
            scan = ggml_get_next_tensor(ctx_compute, scan);
        }

        if (loss_t) {
            ggml_backend_tensor_get(loss_t, &loss_val, 0,
                sizeof(float));
        }

        if (progress) {
            int rc = progress(0, step, total_steps, loss_val, user);
            if (rc != 0) {
                ctx->stop_requested = 1;
            }
        }

        for (int wi = 0; wi < n_all_weights; wi++) {
            kc_lora_weight_t *w = all_weights[wi];
            if (!w->A || !w->B) continue;

            size_t n_a = (size_t)ggml_nelements(w->A);
            size_t n_b = (size_t)ggml_nelements(w->B);

            float *a_data = malloc(n_a * sizeof(float));
            float *b_data = malloc(n_b * sizeof(float));
            float *a_grad = malloc(n_a * sizeof(float));
            float *b_grad = malloc(n_b * sizeof(float));

            if (!a_data || !b_data || !a_grad || !b_grad) {
                free(a_data); free(b_data);
                free(a_grad); free(b_grad);
                continue;
            }

            ggml_backend_tensor_get(w->A, a_data, 0,
                n_a * sizeof(float));
            ggml_backend_tensor_get(w->B, b_data, 0,
                n_b * sizeof(float));

            struct ggml_tensor *grad_a = ggml_graph_get_grad(gf, w->A);
            struct ggml_tensor *grad_b = ggml_graph_get_grad(gf, w->B);

            if (grad_a) {
                memset(a_grad, 0, n_a * sizeof(float));
                ggml_backend_tensor_get(grad_a, a_grad, 0,
                    n_a * sizeof(float));
            } else {
                memset(a_grad, 0, n_a * sizeof(float));
            }

            if (grad_b) {
                memset(b_grad, 0, n_b * sizeof(float));
                ggml_backend_tensor_get(grad_b, b_grad, 0,
                    n_b * sizeof(float));
            } else {
                memset(b_grad, 0, n_b * sizeof(float));
            }

            for (size_t j = 0; j < n_a; j++) {
                a_data[j] -= lr * a_grad[j];
            }
            for (size_t j = 0; j < n_b; j++) {
                b_data[j] -= lr * b_grad[j];
            }

            ggml_backend_tensor_set(w->A, a_data, 0,
                n_a * sizeof(float));
            ggml_backend_tensor_set(w->B, b_data, 0,
                n_b * sizeof(float));

            free(a_data); free(b_data);
            free(a_grad); free(b_grad);
        }
    }

    free(tokens);
    free(pos_ids_data);
    free(target_data);

    if (ctx->opts.output_path) {
        if (kc_lora_save_safetensors(ctx, ctx->opts.output_path) != 0) {
            return kc_lora_set_err(ctx,
                "failed to save LoRA adapter");
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx_compute);

    return KC_LORA_OK;
}

/**
 * Stop an ongoing training call.
 * @param ctx Training context.
 * @return 0 on success, or -1 on failure.
 */
int kc_lora_stop(kc_lora_t *ctx) {
    if (!ctx) return KC_LORA_ERROR;
    ctx->stop_requested = 1;
    return KC_LORA_OK;
}

/**
 * Get the last error message from the context.
 * @param ctx Training context.
 * @return Pointer to a human-readable error string.
 */
const char *kc_lora_error(kc_lora_t *ctx) {
    if (!ctx) return "null context";
    return ctx->error;
}

/**
 * Sets the error message in the context.
 * @param ctx Training context.
 * @param fmt Format string.
 * @return KC_LORA_ERROR.
 */
static int kc_lora_set_err(kc_lora_t *ctx, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ctx->error, MAX_ERR, fmt, ap);
    va_end(ap);
    return KC_LORA_ERROR;
}
