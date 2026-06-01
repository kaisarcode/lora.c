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
    struct ggml_tensor *attn_qkv_w;
    struct ggml_tensor *attn_gate_w;
    struct ggml_tensor *post_attn_norm_w;
    struct ggml_tensor *ssm_conv1d_w;
    struct ggml_tensor *ssm_a;
    struct ggml_tensor *ssm_alpha_w;
    struct ggml_tensor *ssm_beta_w;
    struct ggml_tensor *ssm_dt_b;
    struct ggml_tensor *ssm_norm_w;
    struct ggml_tensor *ssm_out_w;
    struct ggml_tensor *ssm_conv_state;
    struct ggml_tensor *ssm_state;
    struct ggml_tensor *post_ffn_norm_w;
    struct ggml_tensor *layer_output_scale_w;
    struct ggml_tensor *attn_v_norm_w;
    struct ggml_tensor *inp_gate_w;
    struct ggml_tensor *per_layer_proj_w;
    struct ggml_tensor *per_layer_post_norm_w;
    int is_swa;
    int kv_reuse_from;
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
    int ssm_d_conv;
    int ssm_d_state;
    int ssm_dt_rank;
    int ssm_n_group;
    int ssm_inner_size;
    int full_attention_interval;
    size_t graph_size;
    struct ggml_tensor *tok_embeddings;
    struct ggml_tensor *position_embd_w;
    struct ggml_tensor *output_norm_w, *output_norm_b;
    struct ggml_tensor *output_w;
    struct ggml_tensor *rope_freqs_w;
    struct ggml_tensor *per_layer_tok_embd_w;
    struct ggml_tensor *per_layer_model_proj_w;
    struct ggml_tensor *per_layer_proj_norm_w;
    int n_swa;
    float swa_freq_base;
    int n_shared_kv;
    float final_logit_softcapping;
    kc_gguf_layer_t *layers;
    kc_gguf_lora_t **loras;
    int n_loras;
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
void kc_gguf_close(kc_gguf_model_t *m);

kc_gguf_options_t kc_gguf_options_default(void);
void kc_gguf_options_load_env(kc_gguf_options_t *opts);
void kc_gguf_options_free(kc_gguf_options_t *opts);

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

struct ggml_tensor *kc_gguf_build_graph_impl(kc_gguf_model_t *m,
    int n_tokens, int n_past, struct ggml_cgraph **gf,
    struct ggml_tensor **embd_out, struct ggml_tensor **pos_out,
    kc_gguf_arch_params_t params);

struct ggml_tensor *kc_gguf_build_graph_llama(kc_gguf_model_t *m,
    int n_tokens, int n_past, struct ggml_cgraph **gf,
    struct ggml_tensor **embd_out, struct ggml_tensor **pos_out);

struct ggml_tensor *kc_gguf_build_graph_gemma(kc_gguf_model_t *m,
    int n_tokens, int n_past, struct ggml_cgraph **gf,
    struct ggml_tensor **embd_out, struct ggml_tensor **pos_out);

struct ggml_tensor *kc_gguf_build_graph_gpt2(kc_gguf_model_t *m,
    int n_tokens, int n_past, struct ggml_cgraph **gf,
    struct ggml_tensor **embd_out, struct ggml_tensor **pos_out);

struct ggml_tensor *kc_gguf_build_graph_gemma4(kc_gguf_model_t *m,
    int n_tokens, int n_past, struct ggml_cgraph **gf,
    struct ggml_tensor **embd_out, struct ggml_tensor **pos_out);

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
