/**
 * gemma4.c - Gemma 4 graph builder for gguf.c
 * Summary: Compute graph builder for the Gemma 4 architecture (SWA, GeGLU, per-layer emb, QK-norm).
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
 * Builds the compute graph for Gemma 4 models.
 * @param m Loaded model.
 * @param n_tokens Number of input tokens.
 * @param n_past Number of past tokens in KV cache.
 * @param gf Output compute graph.
 * @param embd_out Output embedding input tensor.
 * @param pos_out Output position input tensor.
 * @return Output logits tensor.
 */
struct ggml_tensor *kc_gguf_build_graph_gemma4(kc_gguf_model_t *m, int n_tokens, int n_past,
    struct ggml_cgraph **gf, struct ggml_tensor **embd_out, struct ggml_tensor **pos_out)
{
    struct ggml_context *cctx = m->ctx_compute;
    *gf = ggml_new_graph_custom(cctx, m->graph_size, false);
    *embd_out = ggml_new_tensor_1d(cctx, GGML_TYPE_I32, n_tokens);
    *pos_out  = ggml_new_tensor_1d(cctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(*embd_out); ggml_set_input(*pos_out);

    struct ggml_tensor *cur = ggml_get_rows(cctx, m->tok_embeddings, *embd_out);
    cur = ggml_scale(cctx, cur, sqrtf((float)m->n_embd));

    int n_kv = n_past + n_tokens;
    int n_embd_per_layer = 0;
    struct ggml_tensor *inp_per_layer = NULL;
    if (m->per_layer_tok_embd_w && m->per_layer_model_proj_w && m->per_layer_proj_norm_w) {
        n_embd_per_layer = (int)m->per_layer_proj_norm_w->ne[0];
        inp_per_layer = ggml_get_rows(cctx, m->per_layer_tok_embd_w, *embd_out);
        inp_per_layer = ggml_reshape_3d(cctx, ggml_cont(cctx, inp_per_layer),
            n_embd_per_layer, m->n_layer, n_tokens);
        inp_per_layer = ggml_scale(cctx, inp_per_layer, sqrtf((float)n_embd_per_layer));

        struct ggml_tensor *model_proj = ggml_mul_mat(cctx, m->per_layer_model_proj_w, cur);
        model_proj = ggml_scale(cctx, model_proj, 1.0f / sqrtf((float)m->n_embd));
        model_proj = ggml_reshape_3d(cctx, ggml_cont(cctx, model_proj),
            n_embd_per_layer, m->n_layer, n_tokens);
        model_proj = ggml_rms_norm(cctx, model_proj, m->norm_eps);
        model_proj = ggml_mul(cctx, model_proj, m->per_layer_proj_norm_w);
        inp_per_layer = ggml_scale(cctx,
            ggml_add(cctx, model_proj, inp_per_layer), 1.0f / sqrtf(2.0f));
    }

    for (int i = 0; i < m->n_layer; i++) {
        int n_head = m->n_head;
        int n_head_kv = m->n_head_kv;

        struct ggml_tensor *inp_l = cur;

        cur = ggml_rms_norm(cctx, cur, m->norm_eps);
        if (m->layers[i].attn_norm_w)
            cur = ggml_mul(cctx, cur, m->layers[i].attn_norm_w);

        int q_head_dim = m->layers[i].attn_q_w->ne[1] / n_head;
        int k_head_dim = m->layers[i].attn_k_w->ne[1] / n_head_kv;

        struct ggml_tensor *Q = ggml_mul_mat(cctx, m->layers[i].attn_q_w, cur);
        struct ggml_tensor *K = NULL;
        struct ggml_tensor *V = NULL;

        int kv_reuse = m->layers[i].kv_reuse_from;
        bool has_own_kv = (kv_reuse < 0);
        int kv_cache_layer = has_own_kv ? i : kv_reuse;

        if (has_own_kv) {
            K = ggml_mul_mat(cctx, m->layers[i].attn_k_w, cur);
            V = ggml_mul_mat(cctx, m->layers[i].attn_v_w, cur);
        }

        Q = ggml_reshape_3d(cctx, ggml_cont(cctx, Q), q_head_dim, n_head, n_tokens);
        if (m->layers[i].attn_q_norm_w) {
            Q = ggml_rms_norm(cctx, Q, m->norm_eps);
            Q = ggml_mul(cctx, Q, m->layers[i].attn_q_norm_w);
        }

        float freq_base = m->layers[i].is_swa ? m->swa_freq_base : m->rope_freq_base;
        struct ggml_tensor *freq_factors = m->layers[i].is_swa ? NULL : m->rope_freqs_w;
        int n_rot = q_head_dim < m->n_rot ? q_head_dim : m->n_rot;
        if (m->layers[i].is_swa) {
            int n_rot_swa = m->n_rot;
            n_rot = q_head_dim < n_rot_swa ? q_head_dim : n_rot_swa;
        }
        Q = ggml_rope_ext(cctx, Q, *pos_out, freq_factors,
            n_rot, m->rope_mode, 0, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        if (has_own_kv) {
            K = ggml_reshape_3d(cctx, ggml_cont(cctx, K), k_head_dim, n_head_kv, n_tokens);
            if (m->layers[i].attn_k_norm_w) {
                K = ggml_rms_norm(cctx, K, m->norm_eps);
                K = ggml_mul(cctx, K, m->layers[i].attn_k_norm_w);
            }
            K = ggml_rope_ext(cctx, K, *pos_out, freq_factors,
                n_rot, m->rope_mode, 0, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            V = ggml_reshape_3d(cctx, ggml_cont(cctx, V), k_head_dim, n_head_kv, n_tokens);
            if (m->layers[i].attn_v_norm_w) {
                V = ggml_rms_norm(cctx, V, m->norm_eps);
                V = ggml_mul(cctx, V, m->layers[i].attn_v_norm_w);
            }
        }

        int head_size = q_head_dim;
        struct ggml_tensor *k = ggml_view_3d(cctx, m->layers[kv_cache_layer].k_cache,
            head_size, n_head_kv, n_tokens,
            m->layers[kv_cache_layer].k_cache->nb[1],
            m->layers[kv_cache_layer].k_cache->nb[2],
            n_past * m->layers[kv_cache_layer].k_cache->nb[2]);
        struct ggml_tensor *v = ggml_view_3d(cctx, m->layers[kv_cache_layer].v_cache,
            head_size, n_head_kv, n_tokens,
            m->layers[kv_cache_layer].v_cache->nb[1],
            m->layers[kv_cache_layer].v_cache->nb[2],
            n_past * m->layers[kv_cache_layer].v_cache->nb[2]);

        if (has_own_kv) {
            ggml_build_forward_expand(*gf, ggml_cpy(cctx, K, k));
            ggml_build_forward_expand(*gf, ggml_cpy(cctx, V, v));
        }

        struct ggml_tensor *K_full = ggml_view_3d(cctx,
            m->layers[kv_cache_layer].k_cache,
            head_size, n_head_kv, n_kv,
            m->layers[kv_cache_layer].k_cache->nb[1],
            m->layers[kv_cache_layer].k_cache->nb[2], 0);
        struct ggml_tensor *V_full = ggml_view_3d(cctx,
            m->layers[kv_cache_layer].v_cache,
            head_size, n_head_kv, n_kv,
            m->layers[kv_cache_layer].v_cache->nb[1],
            m->layers[kv_cache_layer].v_cache->nb[2], 0);

        struct ggml_tensor *kq = ggml_mul_mat(cctx,
            ggml_cont(cctx, ggml_permute(cctx, K_full, 0, 2, 1, 3)),
            ggml_cont(cctx, ggml_permute(cctx, Q, 0, 2, 1, 3)));
        kq = ggml_scale(cctx, kq, 1.0f / sqrtf((float)head_size));
        kq = ggml_diag_mask_inf(cctx, kq, n_past);
        kq = ggml_soft_max(cctx, kq);

        struct ggml_tensor *v_att = ggml_permute(cctx,
            ggml_mul_mat(cctx,
                ggml_cont(cctx, ggml_permute(cctx, V_full, 1, 2, 0, 3)),
                ggml_cont(cctx, kq)),
            0, 2, 1, 3);

        struct ggml_tensor *attn = ggml_mul_mat(cctx, m->layers[i].attn_out_w,
            ggml_reshape_2d(cctx, ggml_cont(cctx, v_att),
                n_head * head_size, n_tokens));

        if (m->layers[i].post_attn_norm_w) {
            attn = ggml_rms_norm(cctx, attn, m->norm_eps);
            attn = ggml_mul(cctx, attn, m->layers[i].post_attn_norm_w);
        }
        cur = ggml_add(cctx, inp_l, attn);

        struct ggml_tensor *attn_residual = cur;

        cur = ggml_rms_norm(cctx, cur, m->norm_eps);
        if (m->layers[i].ffn_norm_w)
            cur = ggml_mul(cctx, cur, m->layers[i].ffn_norm_w);

        struct ggml_tensor *gate = ggml_mul_mat(cctx, m->layers[i].ffn_gate_w, cur);
        struct ggml_tensor *up   = ggml_mul_mat(cctx, m->layers[i].ffn_up_w, cur);
        struct ggml_tensor *act  = ggml_gelu(cctx, gate);
        struct ggml_tensor *ffn = ggml_mul_mat(cctx,
            m->layers[i].ffn_down_w, ggml_mul(cctx, act, up));

        if (m->layers[i].post_ffn_norm_w) {
            ffn = ggml_rms_norm(cctx, ffn, m->norm_eps);
            ffn = ggml_mul(cctx, ffn, m->layers[i].post_ffn_norm_w);
        }
        cur = ggml_add(cctx, attn_residual, ffn);

        if (inp_per_layer && m->layers[i].inp_gate_w &&
            m->layers[i].per_layer_proj_w && m->layers[i].per_layer_post_norm_w) {
            struct ggml_tensor *inp_layer = ggml_view_2d(cctx, inp_per_layer,
                n_embd_per_layer, n_tokens, inp_per_layer->nb[2],
                i * inp_per_layer->nb[1]);
            struct ggml_tensor *gate = ggml_gelu(cctx,
                ggml_mul_mat(cctx, m->layers[i].inp_gate_w, cur));
            struct ggml_tensor *layer_cur = ggml_mul(cctx, gate, inp_layer);
            layer_cur = ggml_mul_mat(cctx, m->layers[i].per_layer_proj_w, layer_cur);
            layer_cur = ggml_rms_norm(cctx, layer_cur, m->norm_eps);
            layer_cur = ggml_mul(cctx, layer_cur, m->layers[i].per_layer_post_norm_w);
            cur = ggml_add(cctx, cur, layer_cur);
        }

        if (m->layers[i].layer_output_scale_w)
            cur = ggml_mul(cctx, cur, m->layers[i].layer_output_scale_w);
    }

    cur = ggml_rms_norm(cctx, cur, m->norm_eps);
    if (m->output_norm_w) cur = ggml_mul(cctx, cur, m->output_norm_w);
    cur = ggml_mul_mat(cctx, m->output_w,
        ggml_view_2d(cctx, cur, m->n_embd, 1, cur->nb[1],
            (n_tokens - 1) * cur->nb[1]));

    if (m->final_logit_softcapping > 0.0f) {
        cur = ggml_scale(cctx, cur, 1.0f / m->final_logit_softcapping);
        cur = ggml_tanh(cctx, cur);
        cur = ggml_scale(cctx, cur, m->final_logit_softcapping);
    }

    ggml_set_output(cur); ggml_build_forward_expand(*gf, cur); return cur;
}
