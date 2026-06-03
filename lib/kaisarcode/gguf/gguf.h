/**
 * gguf.h
 * Summary: GGUF model types, tokenizer, and graph builder declarations.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef KC_GGUF_H
#define KC_GGUF_H

#include "include/gguf.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KC_GGUF_OK      0
#define KC_GGUF_ERROR  -1

#define KC_GGUF_GRAPH_BASE       1024
#define KC_GGUF_GRAPH_PER_LAYER  384

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
struct gguf_context;

typedef struct {
    struct ggml_tensor *attn_q_w, *attn_k_w, *attn_v_w;
    struct ggml_tensor *attn_q_b, *attn_k_b, *attn_v_b;
    struct ggml_tensor *attn_out_w, *attn_out_b;
    struct ggml_tensor *attn_norm_w, *attn_norm_b;
    struct ggml_tensor *attn_q_norm_w, *attn_k_norm_w;
    struct ggml_tensor *ffn_gate_w, *ffn_down_w, *ffn_up_w;
    struct ggml_tensor *ffn_norm_w, *ffn_norm_b;
    struct ggml_tensor *k_cache;
    struct ggml_tensor *v_cache;
} kc_gguf_layer_t;

typedef struct {
    struct ggml_tensor *A;
    struct ggml_tensor *B;
} kc_gguf_lora_weight_t;

typedef struct {
    kc_gguf_lora_weight_t attn_q, attn_k, attn_v, attn_o;
    kc_gguf_lora_weight_t ffn_gate, ffn_down, ffn_up;
} kc_gguf_lora_layer_t;

typedef struct {
    struct ggml_context *ctx;
    void *buf;
    kc_gguf_lora_layer_t *layers;
    int n_layers;
    float scale;
} kc_gguf_lora_t;

#define KC_GGUF_LORA_ATTN_Q      0
#define KC_GGUF_LORA_ATTN_K      1
#define KC_GGUF_LORA_ATTN_V      2
#define KC_GGUF_LORA_ATTN_O      3
#define KC_GGUF_LORA_FFN_GATE    4
#define KC_GGUF_LORA_FFN_DOWN    5
#define KC_GGUF_LORA_FFN_UP      6
#define KC_GGUF_LORA_MAX_TENSORS 512

#define KC_GGUF_PROJ_ATTN_Q      0
#define KC_GGUF_PROJ_ATTN_K      1
#define KC_GGUF_PROJ_ATTN_V      2
#define KC_GGUF_PROJ_ATTN_O      3
#define KC_GGUF_PROJ_FFN_GATE    4
#define KC_GGUF_PROJ_FFN_UP      5
#define KC_GGUF_PROJ_FFN_DOWN    6

typedef struct {
    int d_in;
    int d_out;
    int present;
} kc_gguf_projection_info_t;

typedef struct {
    struct gguf_context *gguf;
    struct ggml_context *ctx_meta;
    struct ggml_context *ctx_compute;
    char error[1024];
    int n_vocab, n_embd, n_head, n_head_kv, n_head_dim;
    int n_layer, n_rot, rope_mode;
    float norm_eps, rope_freq_base;
    int n_ctx;
    int q_head_dim;
    size_t graph_size;
    struct ggml_tensor *tok_embeddings;
    struct ggml_tensor *position_embd_w;
    struct ggml_tensor *output_norm_w, *output_norm_b;
    struct ggml_tensor *output_w;
    kc_gguf_layer_t *layers;
    kc_gguf_lora_t **loras;
    int n_loras;
    int arch_id;
} kc_gguf_model_t;

typedef struct {
    float embd_scale;
    int use_gelu;
} kc_gguf_arch_params_t;

typedef struct kc_gguf_options {
    char *model_path;
} kc_gguf_options_t;

typedef void (*kc_gguf_signal_callback_t)(kc_gguf_model_t *ctx);

int kc_gguf_open(kc_gguf_model_t **out, const kc_gguf_options_t *opts);
int kc_gguf_load_model(kc_gguf_model_t *m);
void kc_gguf_close(kc_gguf_model_t *m);

kc_gguf_options_t kc_gguf_options_default(void);
void kc_gguf_options_load_env(kc_gguf_options_t *opts);
void kc_gguf_options_free(kc_gguf_options_t *opts);

size_t kc_gguf_graph_size(int n_layer);

int kc_gguf_on_signal(kc_gguf_model_t *ctx, int sig, kc_gguf_signal_callback_t cb);
int kc_gguf_raise_signal(kc_gguf_model_t *ctx, int sig);
int kc_gguf_listen_signals(kc_gguf_model_t *ctx);
int kc_gguf_listen_signal(kc_gguf_model_t *ctx, int sig_id);
void kc_gguf_signal_listener(int sig);
const char *kc_gguf_error(const kc_gguf_model_t *m);

int kc_gguf_get_arch(const struct gguf_context *gguf, char *arch, size_t arch_size);
uint32_t kc_gguf_get_arch_u32(const struct gguf_context *ctx,
    const char *arch, const char *field, uint32_t def);
float kc_gguf_get_arch_f32(const struct gguf_context *ctx,
    const char *arch, const char *field, float def);
uint32_t kc_gguf_get_kv_u32(const struct gguf_context *ctx,
    const char *key, uint32_t def);
float kc_gguf_get_kv_f32(const struct gguf_context *ctx,
    const char *key, float def);

typedef struct {
    int n_tokens;
    int n_past;
    int training;
    int with_loss;
    int lora_training;
    float lora_scale;
} kc_gguf_graph_options_t;

struct ggml_tensor *kc_gguf_build_graph(kc_gguf_model_t *m,
    const kc_gguf_graph_options_t *opts,
    struct ggml_cgraph **gf,
    struct ggml_tensor **tokens,
    struct ggml_tensor **positions);

struct ggml_cgraph *kc_gguf_build_training_graph(kc_gguf_model_t *m,
    int n_ctx, float lora_scale,
    struct ggml_tensor **input_ids,
    struct ggml_tensor **pos_ids,
    struct ggml_tensor **target_ids);

int kc_gguf_alloc_kv(kc_gguf_model_t *m,
    struct ggml_context *ctx,
    int n_ctx,
    enum ggml_type type);

int kc_gguf_projection_info(kc_gguf_model_t *m,
    int layer,
    int projection,
    kc_gguf_projection_info_t *out);

typedef struct kc_tokenizer kc_gguf_tokenizer_t;

int kc_gguf_tokenizer_load(kc_gguf_tokenizer_t **out,
    const struct gguf_context *gguf, int n_vocab,
    char *error, size_t error_size);
int kc_gguf_tokenizer_encode(kc_gguf_tokenizer_t *tok,
    const char *input, int *tokens, int max_tokens,
    char *error, size_t error_size);
const char *kc_gguf_tokenizer_decode(
    const kc_gguf_tokenizer_t *tok, int id);
int kc_gguf_tokenizer_bos(const kc_gguf_tokenizer_t *tok);
int kc_gguf_tokenizer_eos(const kc_gguf_tokenizer_t *tok);
int kc_gguf_tokenizer_add_bos(const kc_gguf_tokenizer_t *tok);
int kc_gguf_tokenizer_unk(const kc_gguf_tokenizer_t *tok);
int kc_gguf_tokenizer_is_eog(const kc_gguf_tokenizer_t *tok, int id);
void kc_gguf_tokenizer_free(kc_gguf_tokenizer_t *tok);

int kc_gguf_quantize(const char *input_path, const char *output_path, enum ggml_type target_type);
int kc_gguf_dequantize(kc_gguf_model_t *model, enum ggml_type target_type);

#ifdef __cplusplus
}
#endif

#endif
