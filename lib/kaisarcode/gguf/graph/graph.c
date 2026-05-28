/**
 * graph.c - Shared transformer graph builder for gguf.c
 * Summary: Parameterizable compute graph builder reused by all architecture backends.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#include "graph.h"
#include <math.h>
#include <stdbool.h>

#include "ggml.h"

/**
 * Applies LoRA adaptation to a matrix multiplication operation.
 * @param cctx GGML context for allocations.
 * @param m Loaded model.
 * @param W Weight tensor.
 * @param x Input tensor.
 * @param layer Transformer layer index.
 * @param proj KC_GGUF_LORA_* projection constant.
 * @return Result tensor.
 */
struct ggml_tensor *kc_lora_mm(struct ggml_context *cctx, kc_gguf_model_t *m,
    struct ggml_tensor *W, struct ggml_tensor *x, int layer, int proj)
{
    struct ggml_tensor *out = ggml_mul_mat(cctx, W, x);
    for (int li = 0; li < m->n_loras; li++) {
        kc_gguf_lora_t *lora = m->loras[li];
        if (layer >= lora->n_layers) continue;
        kc_gguf_lora_weight_t *w = NULL;
        switch (proj) {
            case KC_GGUF_LORA_ATTN_Q:   w = &lora->layers[layer].attn_q;   break;
            case KC_GGUF_LORA_ATTN_K:   w = &lora->layers[layer].attn_k;   break;
            case KC_GGUF_LORA_ATTN_V:   w = &lora->layers[layer].attn_v;   break;
            case KC_GGUF_LORA_ATTN_O:   w = &lora->layers[layer].attn_o;   break;
            case KC_GGUF_LORA_FFN_GATE: w = &lora->layers[layer].ffn_gate; break;
            case KC_GGUF_LORA_FFN_DOWN: w = &lora->layers[layer].ffn_down; break;
            case KC_GGUF_LORA_FFN_UP:   w = &lora->layers[layer].ffn_up;   break;
        }
        if (!w || !w->A || !w->B) continue;
        struct ggml_tensor *ax = ggml_mul_mat(cctx, w->A, x);
        out = ggml_add(cctx, out,
            ggml_scale(cctx, ggml_mul_mat(cctx, w->B, ax), lora->scale));
    }
    return out;
}

/**
 * Builds a transformer compute graph for one generation step.
 * @param m Loaded model.
 * @param n_tokens Number of input tokens.
 * @param n_past Number of past tokens in KV cache.
 * @param gf Output compute graph.
 * @param embd_out Output embedding input tensor.
 * @param pos_out Output position input tensor.
 * @param params Architecture-specific parameters.
 * @return Output logits tensor.
 */
struct ggml_tensor *kc_gguf_build_graph_impl(kc_gguf_model_t *m, int n_tokens, int n_past,
    struct ggml_cgraph **gf, struct ggml_tensor **embd_out, struct ggml_tensor **pos_out,
    kc_gguf_arch_params_t params)
{
    struct ggml_context *cctx = m->ctx_compute;
    *gf = ggml_new_graph_custom(cctx, m->graph_size, false);
    *embd_out = ggml_new_tensor_1d(cctx, GGML_TYPE_I32, n_tokens);
    *pos_out  = ggml_new_tensor_1d(cctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(*embd_out); ggml_set_input(*pos_out);
    struct ggml_tensor *cur = ggml_get_rows(cctx, m->tok_embeddings, *embd_out);
    if (params.embd_scale != 1.0f)
        cur = ggml_scale(cctx, cur, params.embd_scale);
    for (int i = 0; i < m->n_layer; i++) {
        struct ggml_tensor *inp_l = cur;
        cur = ggml_rms_norm(cctx, cur, m->norm_eps);
        if (m->layers[i].attn_norm_w) cur = ggml_mul(cctx, cur, m->layers[i].attn_norm_w);
        struct ggml_tensor *Q = kc_lora_mm(cctx, m, m->layers[i].attn_q_w, cur, i, KC_GGUF_LORA_ATTN_Q);
        struct ggml_tensor *K = kc_lora_mm(cctx, m, m->layers[i].attn_k_w, cur, i, KC_GGUF_LORA_ATTN_K);
        struct ggml_tensor *V = kc_lora_mm(cctx, m, m->layers[i].attn_v_w, cur, i, KC_GGUF_LORA_ATTN_V);
        if (m->layers[i].attn_q_b) Q = ggml_add(cctx, Q, m->layers[i].attn_q_b);
        if (m->layers[i].attn_k_b) K = ggml_add(cctx, K, m->layers[i].attn_k_b);
        if (m->layers[i].attn_v_b) V = ggml_add(cctx, V, m->layers[i].attn_v_b);
        int head_size = m->n_head_dim;
        Q = ggml_reshape_3d(cctx, ggml_cont(cctx, Q), head_size, m->n_head, n_tokens);
        K = ggml_reshape_3d(cctx, ggml_cont(cctx, K), head_size, m->n_head_kv, n_tokens);
        if (m->layers[i].attn_q_norm_w) {
            Q = ggml_rms_norm(cctx, Q, m->norm_eps);
            Q = ggml_mul(cctx, Q, m->layers[i].attn_q_norm_w);
        }
        if (m->layers[i].attn_k_norm_w) {
            K = ggml_rms_norm(cctx, K, m->norm_eps);
            K = ggml_mul(cctx, K, m->layers[i].attn_k_norm_w);
        }
        Q = ggml_rope_ext(cctx, Q,
            *pos_out, NULL, m->n_rot, m->rope_mode, 0, m->rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(cctx, K,
            *pos_out, NULL, m->n_rot, m->rope_mode, 0, m->rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        V = ggml_reshape_3d(cctx, ggml_cont(cctx, V), head_size, m->n_head_kv, n_tokens);
        struct ggml_tensor *k = ggml_view_3d(cctx, m->layers[i].k_cache, head_size, m->n_head_kv, n_tokens,
            m->layers[i].k_cache->nb[1], m->layers[i].k_cache->nb[2], n_past * m->layers[i].k_cache->nb[2]);
        struct ggml_tensor *v = ggml_view_3d(cctx, m->layers[i].v_cache, head_size, m->n_head_kv, n_tokens,
            m->layers[i].v_cache->nb[1], m->layers[i].v_cache->nb[2], n_past * m->layers[i].v_cache->nb[2]);
        ggml_build_forward_expand(*gf, ggml_cpy(cctx, K, k));
        ggml_build_forward_expand(*gf, ggml_cpy(cctx, V, v));
        K = ggml_view_3d(cctx, m->layers[i].k_cache, head_size, m->n_head_kv, n_past + n_tokens,
            m->layers[i].k_cache->nb[1], m->layers[i].k_cache->nb[2], 0);
        V = ggml_view_3d(cctx, m->layers[i].v_cache, head_size, m->n_head_kv, n_past + n_tokens,
            m->layers[i].v_cache->nb[1], m->layers[i].v_cache->nb[2], 0);
        struct ggml_tensor *kq = ggml_mul_mat(cctx,
            ggml_cont(cctx, ggml_permute(cctx, K, 0, 2, 1, 3)),
            ggml_cont(cctx, ggml_permute(cctx, Q, 0, 2, 1, 3)));
        kq = ggml_diag_mask_inf(cctx, ggml_scale(cctx, kq, 1.0f / sqrtf((float)head_size)), n_past);
        kq = ggml_soft_max(cctx, kq);
        struct ggml_tensor *v_att = ggml_permute(cctx,
            ggml_mul_mat(cctx,
                ggml_cont(cctx, ggml_permute(cctx, V, 1, 2, 0, 3)),
                ggml_cont(cctx, kq)),
            0, 2, 1, 3);
        cur = ggml_add(cctx,
            kc_lora_mm(cctx, m, m->layers[i].attn_out_w,
                ggml_reshape_2d(cctx, ggml_cont(cctx, v_att), m->n_head * head_size, n_tokens),
                i, KC_GGUF_LORA_ATTN_O),
            inp_l);
        struct ggml_tensor *inp_ffn = cur;
        cur = ggml_rms_norm(cctx, cur, m->norm_eps);
        if (m->layers[i].ffn_norm_w) cur = ggml_mul(cctx, cur, m->layers[i].ffn_norm_w);
        struct ggml_tensor *gate = kc_lora_mm(cctx, m, m->layers[i].ffn_gate_w, cur, i, KC_GGUF_LORA_FFN_GATE);
        struct ggml_tensor *up   = kc_lora_mm(cctx, m, m->layers[i].ffn_up_w,   cur, i, KC_GGUF_LORA_FFN_UP);
        struct ggml_tensor *act  = params.use_gelu ? ggml_gelu(cctx, gate) : ggml_silu(cctx, gate);
        cur = ggml_add(cctx,
            kc_lora_mm(cctx, m, m->layers[i].ffn_down_w, ggml_mul(cctx, act, up), i, KC_GGUF_LORA_FFN_DOWN),
            inp_ffn);
    }
    cur = ggml_rms_norm(cctx, cur, m->norm_eps);
    if (m->output_norm_w) cur = ggml_mul(cctx, cur, m->output_norm_w);
    cur = ggml_mul_mat(cctx, m->output_w,
        ggml_view_2d(cctx, cur, m->n_embd, 1, cur->nb[1], (n_tokens - 1) * cur->nb[1]));
    ggml_set_output(cur); ggml_build_forward_expand(*gf, cur); return cur;
}
