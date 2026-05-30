/**
 * gpt2.c - GPT-2 compute graph builder for gguf.c
 * Summary: Compute graph builder for the GPT-2 architecture.
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
 * Builds the compute graph for GPT-2 models (learned position embeddings,
 * LayerNorm, fused QKV, no RoPE, non-gated FFN with GELU).
 * @param m Loaded model.
 * @param n_tokens Number of input tokens.
 * @param n_past Number of past tokens in KV cache.
 * @param gf Output compute graph.
 * @param embd_out Output embedding input tensor.
 * @param pos_out Output position input tensor.
 * @return Output logits tensor.
 */
struct ggml_tensor *kc_gguf_build_graph_gpt2(kc_gguf_model_t *m, int n_tokens, int n_past,
    struct ggml_cgraph **gf, struct ggml_tensor **embd_out, struct ggml_tensor **pos_out)
{
    struct ggml_context *cctx = m->ctx_compute;
    *gf = ggml_new_graph_custom(cctx, m->graph_size, false);
    *embd_out = ggml_new_tensor_1d(cctx, GGML_TYPE_I32, n_tokens);
    *pos_out  = ggml_new_tensor_1d(cctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(*embd_out); ggml_set_input(*pos_out);

    struct ggml_tensor *tok_embd = ggml_get_rows(cctx, m->tok_embeddings, *embd_out);
    struct ggml_tensor *pos_embd = ggml_get_rows(cctx, m->position_embd_w, *pos_out);
    struct ggml_tensor *cur = ggml_add(cctx, tok_embd, pos_embd);

    for (int i = 0; i < m->n_layer; i++) {
        struct ggml_tensor *inp_l = cur;

        cur = ggml_norm(cctx, cur, m->norm_eps);
        if (m->layers[i].attn_norm_w) cur = ggml_mul(cctx, cur, m->layers[i].attn_norm_w);
        if (m->layers[i].attn_norm_b) cur = ggml_add(cctx, cur, m->layers[i].attn_norm_b);

        int head_size = m->n_head_dim;
        int ne0 = m->layers[i].attn_q_w->ne[0];
        int elem_sz = (int)ggml_type_size(m->layers[i].attn_q_w->type);
        struct ggml_tensor *q_w = ggml_view_2d(cctx, m->layers[i].attn_q_w, ne0, m->n_embd,
            m->layers[i].attn_q_w->nb[1], 0);
        struct ggml_tensor *k_w = ggml_view_2d(cctx, m->layers[i].attn_q_w, ne0, m->n_embd,
            m->layers[i].attn_q_w->nb[1], (size_t)ne0 * elem_sz);
        struct ggml_tensor *v_w = ggml_view_2d(cctx, m->layers[i].attn_q_w, ne0, m->n_embd,
            m->layers[i].attn_q_w->nb[1], (size_t)ne0 * 2 * elem_sz);

        struct ggml_tensor *Q = kc_lora_mm(cctx, m, q_w, cur, i, KC_GGUF_LORA_ATTN_Q);
        struct ggml_tensor *K = ggml_mul_mat(cctx, k_w, cur);
        struct ggml_tensor *V = ggml_mul_mat(cctx, v_w, cur);

        if (m->layers[i].attn_q_b) {
            int belem = (int)ggml_type_size(m->layers[i].attn_q_b->type);
            struct ggml_tensor *q_b = ggml_view_1d(cctx, m->layers[i].attn_q_b, m->n_embd, 0);
            struct ggml_tensor *k_b = ggml_view_1d(cctx, m->layers[i].attn_q_b, m->n_embd,
                (size_t)m->n_embd * belem);
            struct ggml_tensor *v_b = ggml_view_1d(cctx, m->layers[i].attn_q_b, m->n_embd,
                (size_t)m->n_embd * 2 * belem);
            Q = ggml_add(cctx, Q, q_b);
            K = ggml_add(cctx, K, k_b);
            V = ggml_add(cctx, V, v_b);
        }

        Q = ggml_reshape_3d(cctx, ggml_cont(cctx, Q), head_size, m->n_head, n_tokens);
        K = ggml_reshape_3d(cctx, ggml_cont(cctx, K), head_size, m->n_head_kv, n_tokens);
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
                ggml_reshape_2d(cctx, ggml_cont(cctx, v_att),
                    m->n_head * head_size, n_tokens),
                i, KC_GGUF_LORA_ATTN_O),
            inp_l);
        if (m->layers[i].attn_out_b)
            cur = ggml_add(cctx, cur, m->layers[i].attn_out_b);

        struct ggml_tensor *inp_ffn = cur;
        cur = ggml_norm(cctx, cur, m->norm_eps);
        if (m->layers[i].ffn_norm_w) cur = ggml_mul(cctx, cur, m->layers[i].ffn_norm_w);
        if (m->layers[i].ffn_norm_b) cur = ggml_add(cctx, cur, m->layers[i].ffn_norm_b);

        struct ggml_tensor *up = kc_lora_mm(cctx, m, m->layers[i].ffn_up_w, cur, i, KC_GGUF_LORA_FFN_UP);
        struct ggml_tensor *act = ggml_gelu(cctx, up);
        cur = ggml_add(cctx,
            kc_lora_mm(cctx, m, m->layers[i].ffn_down_w, act, i, KC_GGUF_LORA_FFN_DOWN),
            inp_ffn);
    }

    cur = ggml_norm(cctx, cur, m->norm_eps);
    if (m->output_norm_w) cur = ggml_mul(cctx, cur, m->output_norm_w);
    if (m->output_norm_b) cur = ggml_add(cctx, cur, m->output_norm_b);

    cur = ggml_mul_mat(cctx, m->output_w,
        ggml_view_2d(cctx, cur, m->n_embd, 1, cur->nb[1], (n_tokens - 1) * cur->nb[1]));
    ggml_set_output(cur); ggml_build_forward_expand(*gf, cur); return cur;
}
