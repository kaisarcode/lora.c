/**
 * libgguf.c
 * Summary: GGUF model loading, tokenizer wrappers, and KV helpers.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif

#include "gguf.h"
#include "tok/tok.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <errno.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <signal.h>

typedef enum {
    KC_ENV_TYPE_INT,
    KC_ENV_TYPE_FLOAT,
    KC_ENV_TYPE_STR
} kc_env_type_t;

typedef struct {
    const char *env_var;
    size_t offset;
    kc_env_type_t type;
} kc_env_map_t;

static const kc_env_map_t env_config_table[] = {
    { "KC_GGUF_MODEL_PATH", offsetof(kc_gguf_options_t, model_path), KC_ENV_TYPE_STR },
};
static const int env_config_table_n = sizeof(env_config_table) / sizeof(env_config_table[0]);

typedef struct {
    int sig;
    kc_gguf_signal_callback_t cb;
} kc_gguf_signal_entry_t;

static kc_gguf_model_t *g_signal_ctx = NULL;

struct kc_gguf_signal_handlers {
    kc_gguf_signal_entry_t *handlers;
    int count;
    int capacity;
};

static kc_gguf_signal_entry_t *g_signal_handlers = NULL;
static int g_signal_handlers_count = 0;
static int g_signal_handlers_capacity = 0;

/**
 * Set error message on model context.
 * @param m Model context.
 * @param fmt Printf-style format string.
 * @return Always -1.
 */
static int kc_gguf_set_err(kc_gguf_model_t *m, const char *fmt, ...) {
    if (!m) return -1;
    va_list args;
    va_start(args, fmt);
    vsnprintf(m->error, sizeof(m->error), fmt, args);
    va_end(args);
    return -1;
}

/**
 * Return last error message from model context.
 * @param m Model context.
 * @return Error string.
 */
const char *kc_gguf_error(const kc_gguf_model_t *m) {
    return m ? m->error : "null model";
}

/**
 * Read a uint32 KV pair from GGUF context.
 * @param ctx GGUF context.
 * @param key Metadata key.
 * @param def Default value if key not found.
 * @return Value or default.
 */
uint32_t kc_gguf_get_kv_u32(const struct gguf_context *ctx,
    const char *key, uint32_t def)
{
    int id = gguf_find_key(ctx, key);
    return (id < 0) ? def : gguf_get_val_u32(ctx, id);
}

/**
 * Read a float32 KV pair from GGUF context.
 * @param ctx GGUF context.
 * @param key Metadata key.
 * @param def Default value if key not found.
 * @return Value or default.
 */
float kc_gguf_get_kv_f32(const struct gguf_context *ctx,
    const char *key, float def)
{
    int id = gguf_find_key(ctx, key);
    return (id < 0) ? def : gguf_get_val_f32(ctx, id);
}

/**
 * Read a uint32 arch-specific field, with fallback to llama.*.
 * @param ctx GGUF context.
 * @param arch Architecture name.
 * @param field Field name after arch prefix.
 * @param def Default if neither arch.key nor llama.key exists.
 * @return Value or default.
 */
uint32_t kc_gguf_get_arch_u32(const struct gguf_context *ctx,
    const char *arch, const char *field, uint32_t def)
{
    char key[128];
    snprintf(key, sizeof(key), "%s.%s", arch, field);
    int id = gguf_find_key(ctx, key);
    if (id >= 0) return gguf_get_val_u32(ctx, id);
    snprintf(key, sizeof(key), "llama.%s", field);
    return kc_gguf_get_kv_u32(ctx, key, def);
}

/**
 * Read a float32 arch-specific field, with fallback to llama.*.
 * @param ctx GGUF context.
 * @param arch Architecture name.
 * @param field Field name after arch prefix.
 * @param def Default if neither arch.key nor llama.key exists.
 * @return Value or default.
 */
float kc_gguf_get_arch_f32(const struct gguf_context *ctx,
    const char *arch, const char *field, float def)
{
    char key[128];
    snprintf(key, sizeof(key), "%s.%s", arch, field);
    int id = gguf_find_key(ctx, key);
    if (id >= 0) return gguf_get_val_f32(ctx, id);
    snprintf(key, sizeof(key), "llama.%s", field);
    return kc_gguf_get_kv_f32(ctx, key, def);
}

/**
 * Read architecture name from GGUF context.
 * Falls back to "llama" if general.architecture key is absent.
 * @param gguf GGUF context.
 * @param arch Output buffer for architecture name.
 * @param arch_size Size of output buffer.
 * @return 0 on success.
 */
int kc_gguf_get_arch(const struct gguf_context *gguf,
    char *arch, size_t arch_size)
{
    int id = gguf_find_key(gguf, "general.architecture");
    if (id < 0) {
        snprintf(arch, arch_size, "%s", "llama");
        return 0;
    }
    snprintf(arch, arch_size, "%s", gguf_get_val_str(gguf, id));
    return 0;
}

/**
 * Load a GGUF model from file.
 * Allocates model context, reads metadata and tensor references.
 * @param out Output pointer to allocated model.
 * @param opts Options.
 * @return 0 on success, -1 on error (check kc_gguf_error).
 */
int kc_gguf_open(kc_gguf_model_t **out, const kc_gguf_options_t *opts) {
    const char *path;
    if (!opts || !opts->model_path) return -1;
    path = opts->model_path;

    *out = calloc(1, sizeof(kc_gguf_model_t));
    if (!*out) return -1;
    kc_gguf_model_t *m = *out;

    struct gguf_init_params params = { .no_alloc = true, .ctx = &m->ctx_meta };
    m->gguf = gguf_init_from_file(path, params);
    if (!m->gguf) return kc_gguf_set_err(m, "failed to open model: %s", path);

    char arch[64];
    kc_gguf_get_arch(m->gguf, arch, sizeof(arch));

    if (strcmp(arch, "llama")   != 0 && strcmp(arch, "mistral") != 0 &&
        strcmp(arch, "mixtral") != 0 && strcmp(arch, "qwen2")   != 0 &&
        strcmp(arch, "qwen2.5") != 0 && strcmp(arch, "qwen3")   != 0 &&
        strcmp(arch, "gemma")   != 0 && strcmp(arch, "gpt2")    != 0) {
        return kc_gguf_set_err(m, "unsupported arch: %s", arch);
    }

    m->n_vocab = kc_gguf_get_arch_u32(m->gguf, arch, "vocab_size", 0);
    if (m->n_vocab == 0) {
        int id = gguf_find_key(m->gguf, "tokenizer.ggml.tokens");
        if (id >= 0) m->n_vocab = (int)gguf_get_arr_n(m->gguf, id);
        else m->n_vocab = 32000;
    }

    m->n_embd    = (int)kc_gguf_get_arch_u32(m->gguf, arch, "embedding_length", 4096);
    m->n_head    = (int)kc_gguf_get_arch_u32(m->gguf, arch, "attention.head_count", 32);
    m->n_head_kv = (int)kc_gguf_get_arch_u32(m->gguf, arch,
        "attention.head_count_kv", m->n_head);

    m->n_head_dim = (int)kc_gguf_get_arch_u32(m->gguf, arch, "attention.head_dim", 0);
    if (m->n_head_dim == 0) {
        m->n_head_dim = (int)kc_gguf_get_arch_u32(m->gguf, arch,
            "attention.key_length", 0);
    }
    if (m->n_head_dim == 0)
        m->n_head_dim = m->n_embd / m->n_head;

    m->n_layer = (int)kc_gguf_get_arch_u32(m->gguf, arch, "block_count", 32);
    m->n_rot   = (int)kc_gguf_get_arch_u32(m->gguf, arch,
        "rope.dimension_count", m->n_head_dim);
    m->norm_eps = kc_gguf_get_arch_f32(m->gguf, arch,
        "attention.layer_norm_rms_epsilon", 1e-5f);

    if (strncmp(arch, "qwen", 4) == 0) {
        m->rope_mode      = 2;
        m->rope_freq_base = kc_gguf_get_arch_f32(m->gguf, arch,
            "rope.freq_base", 1000000.0f);
    } else {
        m->rope_mode      = 0;
        m->rope_freq_base = kc_gguf_get_arch_f32(m->gguf, arch,
            "rope.freq_base", 10000.0f);
    }

    int rope_scaling = gguf_find_key(m->gguf, "llama.rope.scaling.type");
    if (rope_scaling >= 0) {
        const char *stype = gguf_get_val_str(m->gguf, rope_scaling);
        if (strcmp(stype, "yarn") == 0 || strcmp(stype, "linear") == 0) {
            float factor = kc_gguf_get_arch_f32(m->gguf, arch,
                "rope.scaling.attention_factor", 1.0f);
            if (factor > 0.0f) m->rope_freq_base /= factor;
        }
    }

    m->tok_embeddings  = ggml_get_tensor(m->ctx_meta, "token_embd.weight");
    m->position_embd_w = ggml_get_tensor(m->ctx_meta, "position_embd.weight");
    m->output_norm_w   = ggml_get_tensor(m->ctx_meta, "output_norm.weight");
    m->output_norm_b   = ggml_get_tensor(m->ctx_meta, "output_norm.bias");
    m->output_w        = ggml_get_tensor(m->ctx_meta, "output.weight");

    if (!m->tok_embeddings)
        return kc_gguf_set_err(m, "missing mandatory: token_embd.weight");
    if (!m->output_norm_w)
        return kc_gguf_set_err(m, "missing mandatory: output_norm.weight");
    if (!m->output_w) m->output_w = m->tok_embeddings;

    int is_gpt2 = (strcmp(arch, "gpt2") == 0);
    if (is_gpt2 && !m->position_embd_w)
        return kc_gguf_set_err(m, "missing mandatory: position_embd.weight");

    m->layers = calloc(m->n_layer, sizeof(kc_gguf_layer_t));
    if (!m->layers)
        return kc_gguf_set_err(m, "failed to allocate layers");

    for (int i = 0; i < m->n_layer; i++) {
        char b[64];
        if (is_gpt2) {
            snprintf(b, 64, "blk.%d.attn_qkv.weight", i);
            m->layers[i].attn_q_w = ggml_get_tensor(m->ctx_meta, b);
            snprintf(b, 64, "blk.%d.attn_qkv.bias", i);
            m->layers[i].attn_q_b = ggml_get_tensor(m->ctx_meta, b);
        } else {
            snprintf(b, 64, "blk.%d.attn_q.weight", i);
            m->layers[i].attn_q_w = ggml_get_tensor(m->ctx_meta, b);
            snprintf(b, 64, "blk.%d.attn_k.weight", i);
            m->layers[i].attn_k_w = ggml_get_tensor(m->ctx_meta, b);
            snprintf(b, 64, "blk.%d.attn_v.weight", i);
            m->layers[i].attn_v_w = ggml_get_tensor(m->ctx_meta, b);
            snprintf(b, 64, "blk.%d.attn_q.bias", i);
            m->layers[i].attn_q_b = ggml_get_tensor(m->ctx_meta, b);
            snprintf(b, 64, "blk.%d.attn_k.bias", i);
            m->layers[i].attn_k_b = ggml_get_tensor(m->ctx_meta, b);
            snprintf(b, 64, "blk.%d.attn_v.bias", i);
            m->layers[i].attn_v_b = ggml_get_tensor(m->ctx_meta, b);
        }
        snprintf(b, 64, "blk.%d.attn_output.weight", i);
        m->layers[i].attn_out_w = ggml_get_tensor(m->ctx_meta, b);
        snprintf(b, 64, "blk.%d.attn_output.bias", i);
        m->layers[i].attn_out_b = ggml_get_tensor(m->ctx_meta, b);
        snprintf(b, 64, "blk.%d.attn_norm.weight", i);
        m->layers[i].attn_norm_w = ggml_get_tensor(m->ctx_meta, b);
        snprintf(b, 64, "blk.%d.attn_norm.bias", i);
        m->layers[i].attn_norm_b = ggml_get_tensor(m->ctx_meta, b);
        snprintf(b, 64, "blk.%d.attn_q_norm.weight", i);
        m->layers[i].attn_q_norm_w = ggml_get_tensor(m->ctx_meta, b);
        snprintf(b, 64, "blk.%d.attn_k_norm.weight", i);
        m->layers[i].attn_k_norm_w = ggml_get_tensor(m->ctx_meta, b);
        snprintf(b, 64, "blk.%d.ffn_gate.weight", i);
        m->layers[i].ffn_gate_w = ggml_get_tensor(m->ctx_meta, b);
        snprintf(b, 64, "blk.%d.ffn_down.weight", i);
        m->layers[i].ffn_down_w = ggml_get_tensor(m->ctx_meta, b);
        snprintf(b, 64, "blk.%d.ffn_up.weight", i);
        m->layers[i].ffn_up_w = ggml_get_tensor(m->ctx_meta, b);
        snprintf(b, 64, "blk.%d.ffn_norm.weight", i);
        m->layers[i].ffn_norm_w = ggml_get_tensor(m->ctx_meta, b);
        snprintf(b, 64, "blk.%d.ffn_norm.bias", i);
        m->layers[i].ffn_norm_b = ggml_get_tensor(m->ctx_meta, b);

        if (is_gpt2) {
            if (!m->layers[i].attn_q_w || !m->layers[i].attn_out_w ||
                !m->layers[i].attn_norm_w ||
                !m->layers[i].ffn_down_w || !m->layers[i].ffn_up_w ||
                !m->layers[i].ffn_norm_w) {
                return kc_gguf_set_err(m,
                    "missing mandatory tensors for layer %d", i);
            }
        } else {
            if (!m->layers[i].attn_q_w || !m->layers[i].attn_k_w ||
                !m->layers[i].attn_v_w || !m->layers[i].attn_out_w ||
                !m->layers[i].attn_norm_w ||
                !m->layers[i].ffn_down_w || !m->layers[i].ffn_up_w ||
                !m->layers[i].ffn_norm_w) {
                return kc_gguf_set_err(m,
                    "missing mandatory tensors for layer %d", i);
            }
            if (!m->layers[i].ffn_gate_w) {
                return kc_gguf_set_err(m,
                    "missing mandatory ffn_gate.weight for layer %d", i);
            }
        }
        if (strcmp(arch, "qwen3") == 0 &&
            (!m->layers[i].attn_q_norm_w || !m->layers[i].attn_k_norm_w)) {
            return kc_gguf_set_err(m,
                "missing mandatory Qwen3 Q/K norm for layer %d", i);
        }
    }

    return 0;
}

/**
 * Allocates memory aligned to a specified boundary.
 * @param alignment Alignment boundary in bytes (must be power of two).
 * @param size Number of bytes to allocate.
 * @return Aligned pointer, or NULL on failure.
 */
static void *kc_aligned_alloc(size_t alignment, size_t size) {
    if (alignment < sizeof(void *)) alignment = sizeof(void *);
    void *raw = malloc(size + alignment + sizeof(void *));
    if (!raw) return NULL;
    void *aligned = (void *)(((uintptr_t)raw + alignment + sizeof(void *)) & ~(alignment - 1));
    ((void **)aligned)[-1] = raw;
    return aligned;
}

/**
 * Frees memory allocated by kc_aligned_alloc.
 * @param ptr Pointer returned by kc_aligned_alloc.
 * @return None.
 */
static void kc_aligned_free(void *ptr) {
    if (ptr) free(((void **)ptr)[-1]);
}

/**
 * Free a model context allocated by kc_gguf_model_load.
 * Safe to call with NULL.
 * @param m Model context (may be NULL).
 * @return None.
 */
void kc_gguf_close(kc_gguf_model_t *m) {
    if (!m) return;
    if (m->ctx_meta) {
        struct ggml_tensor *t = ggml_get_first_tensor(m->ctx_meta);
        while (t) {
            if (t->buffer) {
                const char *bname = ggml_backend_buffer_name(t->buffer);
                if (bname && strcmp(bname, "CPU_Mapped") == 0) {
                    kc_aligned_free(t->data);
                    t->data = NULL;
                }
            }
            t = ggml_get_next_tensor(m->ctx_meta, t);
        }
    }
    if (m->gguf) gguf_free(m->gguf);
    if (m->ctx_meta) ggml_free(m->ctx_meta);
    free(m->layers);
    free(m);
}

/**
 * Load and initialize a tokenizer from GGUF metadata.
 * @param out Output pointer to allocated tokenizer.
 * @param gguf GGUF context with tokenizer metadata.
 * @param n_vocab Vocabulary size.
 * @param error Error output buffer.
 * @param error_size Error buffer capacity.
 * @return 0 on success, -1 on error.
 */
int kc_gguf_tokenizer_load(kc_gguf_tokenizer_t **out,
    const struct gguf_context *gguf, int n_vocab,
    char *error, size_t error_size)
{
    return kc_tokenizer_load(out, gguf, n_vocab, error, error_size);
}

/**
 * Encode a text string into token IDs.
 * @param tok Tokenizer context.
 * @param input Input text (UTF-8).
 * @param tokens Output token ID array.
 * @param max_tokens Max tokens to write.
 * @param error Error output buffer.
 * @param error_size Error buffer capacity.
 * @return Number of tokens on success, -1 on error.
 */
int kc_gguf_tokenizer_encode(kc_gguf_tokenizer_t *tok,
    const char *input, int *tokens, int max_tokens,
    char *error, size_t error_size)
{
    return kc_tokenizer_encode(tok, input, tokens, max_tokens, error, error_size);
}

/**
 * Decode a token ID to its string representation.
 * @param tok Tokenizer context.
 * @param id Token ID.
 * @return String pointer (borrowed, valid until tokenizer free).
 */
const char *kc_gguf_tokenizer_decode(
    const kc_gguf_tokenizer_t *tok, int id)
{
    return kc_tokenizer_decode(tok, id);
}

/**
 * Return BOS token ID.
 * @param tok Tokenizer context.
 * @return Token ID or -1.
 */
int kc_gguf_tokenizer_bos(const kc_gguf_tokenizer_t *tok) {
    return kc_tokenizer_bos(tok);
}

/**
 * Return EOS token ID.
 * @param tok Tokenizer context.
 * @return Token ID or -1.
 */
int kc_gguf_tokenizer_eos(const kc_gguf_tokenizer_t *tok) {
    return kc_tokenizer_eos(tok);
}

/**
 * Return whether BOS is prepended by default.
 * @param tok Tokenizer context.
 * @return 1 if add_bos, 0 otherwise.
 */
int kc_gguf_tokenizer_add_bos(const kc_gguf_tokenizer_t *tok) {
    return kc_tokenizer_add_bos(tok);
}

/**
 * Return UNK token ID.
 * @param tok Tokenizer context.
 * @return Token ID or -1.
 */
int kc_gguf_tokenizer_unk(const kc_gguf_tokenizer_t *tok) {
    return kc_tokenizer_unk(tok);
}

/**
 * Check if a token ID is an end-of-generation token.
 * @param tok Tokenizer context.
 * @param id Token ID to check.
 * @return 1 if EOG, 0 otherwise.
 */
int kc_gguf_tokenizer_is_eog(const kc_gguf_tokenizer_t *tok, int id) {
    return kc_tokenizer_is_eog(tok, id);
}

/**
 * Free a tokenizer allocated by kc_gguf_tokenizer_load.
 * @param tok Tokenizer context (may be NULL).
 * @return None.
 */
void kc_gguf_tokenizer_free(kc_gguf_tokenizer_t *tok) {
    kc_tokenizer_free(tok);
}

/**
 * Re-quantize a GGUF file.
 * @param input_path Path to source GGUF file.
 * @param output_path Path for output quantized GGUF file.
 * @param target_type Target quantization type.
 * @return 0 on success, -1 on error.
 */
int kc_gguf_quantize(const char *input_path, const char *output_path, enum ggml_type target_type) {
    struct ggml_context *input_ggml = NULL;
    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx      = &input_ggml,
    };
    struct gguf_context *input = gguf_init_from_file(input_path, params);
    if (!input) {
        fprintf(stderr, "gguf: failed to load '%s'\n", input_path);
        return -1;
    }

    const int64_t n_tensors = gguf_get_n_tensors(input);

    struct gguf_context *output = gguf_init_empty();
    if (!output) {
        fprintf(stderr, "gguf: failed to create output context\n");
        gguf_free(input);
        return -1;
    }
    gguf_set_kv(output, input);

    size_t ctx_size = ggml_tensor_overhead() * (size_t)n_tensors + 1024;
    struct ggml_init_params out_params = {
        .mem_size   = ctx_size,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context *output_ggml = ggml_init(out_params);
    if (!output_ggml) {
        fprintf(stderr, "gguf: failed to init ggml context\n");
        gguf_free(output);
        gguf_free(input);
        return -1;
    }

    for (int64_t i = 0; i < n_tensors; i++) {
        const char *name = gguf_get_tensor_name(input, i);
        struct ggml_tensor *src_t = ggml_get_tensor(input_ggml, name);
        if (!src_t) {
            fprintf(stderr, "gguf: tensor '%s' not found\n", name);
            ggml_free(output_ggml);
            gguf_free(output);
            gguf_free(input);
            return -1;
        }

        int n_dims = GGML_MAX_DIMS;
        while (n_dims > 1 && src_t->ne[n_dims - 1] == 1) n_dims--;

        enum ggml_type src_type = gguf_get_tensor_type(input, i);
        enum ggml_type eff_type = target_type;
        int blck = ggml_blck_size(eff_type);
        if (src_type == GGML_TYPE_F32)
            eff_type = src_type;
        else if (eff_type != src_type && blck > 1 && src_t->ne[0] % blck != 0)
            eff_type = src_type;

        struct ggml_tensor *dst_t = ggml_new_tensor(output_ggml, eff_type, n_dims, src_t->ne);
        if (!dst_t) {
            fprintf(stderr, "gguf: failed to create output tensor '%s'\n", name);
            ggml_free(output_ggml);
            gguf_free(output);
            gguf_free(input);
            return -1;
        }
        ggml_set_name(dst_t, name);
        gguf_add_tensor(output, dst_t);
    }

    int ret = 0;

    for (int64_t i = 0; i < n_tensors; i++) {
        const char *name     = gguf_get_tensor_name(input, i);
        enum ggml_type src_type = gguf_get_tensor_type(input, i);
        struct ggml_tensor *src_t = ggml_get_tensor(input_ggml, name);
        size_t src_size = gguf_get_tensor_size(input, i);
        size_t src_off  = gguf_get_data_offset(input) + gguf_get_tensor_offset(input, i);
        int64_t n_elems   = ggml_nelements(src_t);
        int64_t n_per_row = src_t->ne[0];
        int64_t n_rows    = n_elems / n_per_row;

        enum ggml_type eff_type = target_type;
        int blck = ggml_blck_size(eff_type);
        if (src_type == GGML_TYPE_F32)
            eff_type = src_type;
        else if (eff_type != src_type && blck > 1 && n_per_row % blck != 0)
            eff_type = src_type;

        fprintf(stderr, "gguf: tensor %ld of %ld  %s  %s to %s  %ld MB\n",
            (long)(i + 1), (long)n_tensors, name,
            ggml_type_name(src_type),
            eff_type == src_type ? ggml_type_name(src_type) : ggml_type_name(target_type),
            (long)(src_size / (1024 * 1024)));

        FILE *fin = fopen(input_path, "rb");
        if (!fin) {
            fprintf(stderr, "gguf: failed to open '%s': %s\n", input_path, strerror(errno));
            ret = -1;
            break;
        }
        if (fseek(fin, (long)src_off, SEEK_SET) != 0) {
            fprintf(stderr, "gguf: seek error for '%s': %s\n", name, strerror(errno));
            fclose(fin);
            ret = -1;
            break;
        }

        void *src_data = malloc(src_size);
        if (!src_data) {
            fprintf(stderr, "gguf: alloc %zu bytes failed for '%s'\n", src_size, name);
            fclose(fin);
            ret = -1;
            break;
        }

        if (fread(src_data, 1, src_size, fin) != src_size) {
            fprintf(stderr, "gguf: read error for '%s': %s\n",
                name, feof(fin) ? "EOF" : strerror(ferror(fin)));
            free(src_data);
            fclose(fin);
            ret = -1;
            break;
        }
        fclose(fin);

        void *out_data;

        if (src_type == eff_type) {
            out_data = src_data;
        } else {
            float *f32_buf;
            size_t f32_nbytes = (size_t)n_elems * sizeof(float);

            if (src_type == GGML_TYPE_F32) {
                f32_buf = (float *)src_data;
            } else {
                const struct ggml_type_traits *src_traits =
                    ggml_get_type_traits(src_type);
                if (!src_traits || !src_traits->to_float) {
                    fprintf(stderr, "gguf: no dequant for '%s'\n", ggml_type_name(src_type));
                    free(src_data);
                    ret = -1;
                    break;
                }
                f32_buf = malloc(f32_nbytes);
                if (!f32_buf) {
                    fprintf(stderr, "gguf: alloc F32 failed (%zu bytes)\n", f32_nbytes);
                    free(src_data);
                    ret = -1;
                    break;
                }
                src_traits->to_float(src_data, f32_buf, n_elems);
                free(src_data);
            }

            if (ggml_quantize_requires_imatrix(eff_type))
                fprintf(stderr, "gguf: warning: %s needs importance matrix; using default quantization\n",
                    ggml_type_name(eff_type));

            size_t dst_size = ggml_row_size(eff_type, n_elems);
            void *quant_buf = malloc(dst_size);
            if (!quant_buf) {
                fprintf(stderr, "gguf: alloc quant buffer failed (%zu bytes)\n", dst_size);
                free(f32_buf);
                ret = -1;
                break;
            }
            ggml_quantize_chunk(eff_type, f32_buf, quant_buf,
                0, n_rows, n_per_row, NULL);
            free(f32_buf);
            out_data = quant_buf;
        }

        gguf_set_tensor_data(output, name, out_data);
    }

    if (ret == 0) {
        fprintf(stderr, "gguf: writing '%s'...\n", output_path);
        if (!gguf_write_to_file(output, output_path, false)) {
            fprintf(stderr, "gguf: failed to write '%s'\n", output_path);
            ret = -1;
        }
    }

    ggml_free(output_ggml);
    ggml_free(input_ggml);
    gguf_free(output);
    gguf_free(input);
    ggml_quantize_free();

    if (ret == 0)
        fprintf(stderr, "gguf: done\n");

    return ret;
}

/**
 * Dequantize model weights to F16 in-place.
 * @param model Loaded GGUF model.
 * @param target_type Target type (F16 or F32).
 * @return 0 on success, -1 on failure.
 */
int kc_gguf_dequantize(kc_gguf_model_t *model, enum ggml_type target_type) {
    if (target_type != GGML_TYPE_F16 && target_type != GGML_TYPE_F32)
        return -1;

    int64_t n_tensors = gguf_get_n_tensors(model->gguf);
    int n_dequant = 0;
    size_t total_f32_size = 0;

    for (int64_t i = 0; i < n_tensors; i++) {
        const char *name = gguf_get_tensor_name(model->gguf, i);
        struct ggml_tensor *t = ggml_get_tensor(model->ctx_meta, name);
        if (!t || !t->data) continue;
        if (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) continue;

        n_dequant++;
        total_f32_size += (size_t)ggml_nelements(t) * sizeof(float);
    }

    if (n_dequant == 0) return 0;

    float *f32_buf = malloc(total_f32_size);
    if (!f32_buf) return -1;

    size_t offset = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char *name = gguf_get_tensor_name(model->gguf, i);
        struct ggml_tensor *t = ggml_get_tensor(model->ctx_meta, name);
        if (!t || !t->data) continue;
        if (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) continue;

        const struct ggml_type_traits *traits = ggml_get_type_traits(t->type);
        if (!traits || !traits->to_float) {
            free(f32_buf);
            return -1;
        }

        int64_t n_elems = ggml_nelements(t);
        size_t src_size = ggml_nbytes(t);
        void *src_data = malloc(src_size);
        if (!src_data) {
            free(f32_buf);
            return -1;
        }

        if (ggml_backend_buffer_is_host(t->buffer)) {
            memcpy(src_data, t->data, src_size);
        } else {
            ggml_backend_tensor_get(t, src_data, 0, src_size);
        }

        float *tensor_f32 = f32_buf + offset;
        traits->to_float(src_data, tensor_f32, n_elems);
        free(src_data);

        size_t target_elem_size = target_type == GGML_TYPE_F16
            ? sizeof(ggml_fp16_t) : sizeof(float);
        size_t new_size = (size_t)n_elems * target_elem_size;
        void *new_data = kc_aligned_alloc(64, new_size);
        if (!new_data) {
            free(f32_buf);
            return -1;
        }

        if (target_type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(tensor_f32, new_data, n_elems);
        } else {
            memcpy(new_data, tensor_f32, new_size);
        }

        t->type = target_type;
        t->data = new_data;
        t->buffer = ggml_backend_cpu_buffer_from_ptr(new_data, new_size);
        t->nb[0] = ggml_type_size(target_type);
        t->nb[1] = t->nb[0] * (t->ne[0] / ggml_blck_size(target_type));
        for (int i = 2; i < GGML_MAX_DIMS; i++) {
            t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
        }

        offset += (size_t)n_elems;
    }

    free(f32_buf);
    return 0;
}

/**
 * Create an options struct initialized with default values.
 * @param none Unused.
 * @return Default-initialized options.
 */
kc_gguf_options_t kc_gguf_options_default(void) {
    kc_gguf_options_t opts;
    memset(&opts, 0, sizeof(opts));
    return opts;
}

/**
 * Load configuration from environment variables.
 * @param opts Options to update.
 * @return None.
 */
void kc_gguf_options_load_env(kc_gguf_options_t *opts) {
    int i;
    if (!opts) return;
    for (i = 0; i < env_config_table_n; i++) {
        const char *val = getenv(env_config_table[i].env_var);
        char *end;
        if (!val) continue;
        switch (env_config_table[i].type) {
            case KC_ENV_TYPE_INT: {
                long v = strtol(val, &end, 10);
                if (end != val && *end == '\0') {
                    *(int *)((char *)opts + env_config_table[i].offset) = (int)v;
                }
                break;
            }
            case KC_ENV_TYPE_FLOAT: {
                float v = strtof(val, &end);
                if (end != val && *end == '\0') {
                    *(float *)((char *)opts + env_config_table[i].offset) = v;
                }
                break;
            }
            case KC_ENV_TYPE_STR: {
                char **p = (char **)((char *)opts + env_config_table[i].offset);
                free(*p);
                *p = strdup(val);
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
void kc_gguf_options_free(kc_gguf_options_t *opts) {
    if (!opts) return;
    free(opts->model_path);
    opts->model_path = NULL;
}

/**
 * Register a handler for a library-level signal number.
 * @param ctx GGUF context.
 * @param sig Application-defined signal number.
 * @param cb Callback to invoke.
 * @return KC_GGUF_OK on success, or KC_GGUF_ERROR on failure.
 */
int kc_gguf_on_signal(kc_gguf_model_t *ctx, int sig, kc_gguf_signal_callback_t cb) {
    int i;
    if (!ctx) return KC_GGUF_ERROR;

    for (i = 0; i < g_signal_handlers_count; i++) {
        if (g_signal_handlers[i].sig == sig) {
            if (cb) {
                g_signal_handlers[i].cb = cb;
            } else {
                int tail = g_signal_handlers_count - i - 1;
                if (tail > 0) {
                    memmove(&g_signal_handlers[i], &g_signal_handlers[i + 1],
                            (size_t)tail * sizeof(kc_gguf_signal_entry_t));
                }
                g_signal_handlers_count--;
            }
            return KC_GGUF_OK;
        }
    }

    if (!cb) return KC_GGUF_OK;

    if (g_signal_handlers_count >= g_signal_handlers_capacity) {
        int new_cap = g_signal_handlers_capacity ? g_signal_handlers_capacity * 2 : 4;
        kc_gguf_signal_entry_t *p = (kc_gguf_signal_entry_t *)realloc(g_signal_handlers,
            (size_t)new_cap * sizeof(kc_gguf_signal_entry_t));
        if (!p) return KC_GGUF_ERROR;
        g_signal_handlers = p;
        g_signal_handlers_capacity = new_cap;
    }

    g_signal_handlers[g_signal_handlers_count].sig = sig;
    g_signal_handlers[g_signal_handlers_count].cb = cb;
    g_signal_handlers_count++;

    return KC_GGUF_OK;
}

/**
 * Raise a library-level signal.
 * @param ctx GGUF context.
 * @param sig Signal number to raise.
 * @return KC_GGUF_OK if handled, or KC_GGUF_ERROR if no handler.
 */
int kc_gguf_raise_signal(kc_gguf_model_t *ctx, int sig) {
    int i;
    if (!ctx) return KC_GGUF_ERROR;
    for (i = 0; i < g_signal_handlers_count; i++) {
        if (g_signal_handlers[i].sig == sig) {
            g_signal_handlers[i].cb(ctx);
            return KC_GGUF_OK;
        }
    }
    return KC_GGUF_ERROR;
}

/**
 * Set the internal signal-listener context.
 * @param ctx GGUF context.
 * @return KC_GGUF_OK on success, or KC_GGUF_ERROR if ctx is NULL.
 */
int kc_gguf_listen_signals(kc_gguf_model_t *ctx) {
    if (!ctx) return KC_GGUF_ERROR;
    g_signal_ctx = ctx;
    return KC_GGUF_OK;
}

/**
 * Wire an OS signal to the library signal listener.
 * @param ctx GGUF context.
 * @param sig_id OS signal number.
 * @return KC_GGUF_OK on success, or KC_GGUF_ERROR on failure.
 */
int kc_gguf_listen_signal(kc_gguf_model_t *ctx, int sig_id) {
    if (!ctx) return KC_GGUF_ERROR;
    g_signal_ctx = ctx;
#ifdef _WIN32
    (void)sig_id;
#else
    signal(sig_id, kc_gguf_signal_listener);
#endif
    return KC_GGUF_OK;
}

/**
 * Generic signal-listener compatible with signal() / sigaction().
 * @param sig OS signal number.
 * @return None.
 */
void kc_gguf_signal_listener(int sig) {
    if (g_signal_ctx) {
        kc_gguf_raise_signal(g_signal_ctx, sig);
    }
}
