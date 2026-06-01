/**
 * qwen35.c - Qwen3.5 hybrid SSM+attention graph builder for gguf.c
 * Summary: Compute graph builder for Qwen3.5 architecture.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#include "graph.h"
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "ggml.h"

/**
 * Builds a Qwen3.5 compute graph for one generation step.
 * @param m Loaded model.
 * @param n_tokens Number of input tokens.
 * @param n_past Number of past tokens in KV cache.
 * @param gf Output compute graph.
 * @param embd_out Output embedding input tensor.
 * @param pos_out Output position input tensor.
 * @return Output logits tensor.
 */
struct ggml_tensor *kc_gguf_build_graph_qwen35(kc_gguf_model_t *m,
    int n_tokens, int n_past, struct ggml_cgraph **gf,
    struct ggml_tensor **embd_out, struct ggml_tensor **pos_out)
{
    struct ggml_context *cctx = m->ctx_compute;
    *gf = ggml_new_graph_custom(cctx, m->graph_size, false);
    *embd_out = ggml_new_tensor_1d(cctx, GGML_TYPE_I32, n_tokens);
    *pos_out  = ggml_new_tensor_1d(cctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(*embd_out); ggml_set_input(*pos_out);
    struct ggml_tensor *cur = ggml_get_rows(cctx, m->tok_embeddings, *embd_out);

    int n_head_k = m->ssm_n_group;
    int n_head_v = m->ssm_dt_rank;
    int d_state  = m->ssm_d_state;
    int d_conv   = m->ssm_d_conv;
    int q_head_dim = m->q_head_dim;
    int head_dim   = m->n_head_dim;
    int key_dim   = d_state * n_head_k;
    int value_dim = d_state * n_head_v;
    int conv_dim  = 2 * key_dim + value_dim;

    for (int i = 0; i < m->n_layer; i++) {
        struct ggml_tensor *inp_l = cur;

        cur = ggml_rms_norm(cctx, cur, m->norm_eps);
        if (m->layers[i].attn_norm_w)
            cur = ggml_mul(cctx, cur, m->layers[i].attn_norm_w);

        if (m->layers[i].attn_gate_w) {
            struct ggml_tensor *n = cur;

            struct ggml_tensor *qkv = ggml_mul_mat(cctx, m->layers[i].attn_qkv_w, n);

            struct ggml_tensor *sx = ggml_new_tensor_2d(cctx, GGML_TYPE_F32,
                d_conv - 1 + n_tokens, conv_dim);

            if (m->layers[i].ssm_conv_state) {
                struct ggml_tensor *state_view = ggml_view_2d(cctx, sx,
                    conv_dim, d_conv - 1, sx->nb[1], 0);
                ggml_build_forward_expand(*gf,
                    ggml_cpy(cctx, m->layers[i].ssm_conv_state, state_view));
            }

            struct ggml_tensor *qkv_t = ggml_cont(cctx, ggml_transpose(cctx, qkv));
            struct ggml_tensor *qkv_view = ggml_view_2d(cctx, sx,
                conv_dim, n_tokens, sx->nb[1],
                (d_conv - 1) * ggml_element_size(sx));
            ggml_build_forward_expand(*gf, ggml_cpy(cctx, qkv_t, qkv_view));

            struct ggml_tensor *conv_out = ggml_ssm_conv(cctx, sx,
                m->layers[i].ssm_conv1d_w);
            conv_out = ggml_silu(cctx, conv_out);

            struct ggml_tensor *q_conv = ggml_reshape_3d(cctx,
                ggml_view_2d(cctx, conv_out, key_dim, n_tokens,
                    conv_out->nb[1], 0),
                d_state, n_head_k, n_tokens);
            struct ggml_tensor *k_conv = ggml_reshape_3d(cctx,
                ggml_view_2d(cctx, conv_out, key_dim, n_tokens,
                    conv_out->nb[1], key_dim * ggml_element_size(conv_out)),
                d_state, n_head_k, n_tokens);
            struct ggml_tensor *v_conv = ggml_reshape_3d(cctx,
                ggml_view_2d(cctx, conv_out, value_dim, n_tokens,
                    conv_out->nb[1], (2 * key_dim) * ggml_element_size(conv_out)),
                d_state, n_head_v, n_tokens);

            q_conv = ggml_l2_norm(cctx, q_conv, 1e-7f);
            k_conv = ggml_l2_norm(cctx, k_conv, 1e-7f);

            if (n_head_v > n_head_k) {
                struct ggml_tensor *target = ggml_new_tensor_4d(cctx,
                    GGML_TYPE_F32, d_state, n_head_v, n_tokens, 1);
                q_conv = ggml_repeat(cctx, q_conv, target);
                k_conv = ggml_repeat(cctx, k_conv, target);
            }

            struct ggml_tensor *alpha_raw = ggml_mul_mat(cctx,
                m->layers[i].ssm_alpha_w, n);
            struct ggml_tensor *alpha = ggml_reshape_4d(cctx, alpha_raw,
                1, n_head_v, n_tokens, 1);

            struct ggml_tensor *dt_bias = ggml_reshape_4d(cctx,
                m->layers[i].ssm_dt_b, 1, n_head_v, 1, 1);

            alpha = ggml_add(cctx, alpha, dt_bias);
            alpha = ggml_softplus(cctx, alpha);

            struct ggml_tensor *ssm_a = ggml_reshape_4d(cctx,
                m->layers[i].ssm_a, 1, n_head_v, 1, 1);

            struct ggml_tensor *gate = ggml_mul(cctx, alpha, ssm_a);

            struct ggml_tensor *beta_raw = ggml_mul_mat(cctx,
                m->layers[i].ssm_beta_w, n);
            struct ggml_tensor *beta = ggml_view_4d(cctx, beta_raw,
                1, n_head_v, n_tokens, 1,
                sizeof(float), sizeof(float) * n_head_v,
                sizeof(float) * n_head_v * n_tokens, 0);
            beta = ggml_sigmoid(cctx, beta);

            struct ggml_tensor *state = m->layers[i].ssm_state
                ? m->layers[i].ssm_state
                : ggml_new_tensor_4d(cctx, GGML_TYPE_F32,
                    d_state, d_state, n_head_v, 1);

            struct ggml_tensor *gdn = ggml_gated_delta_net(cctx,
                q_conv, k_conv, v_conv, gate, beta, state);

            struct ggml_tensor *gdn_cont = ggml_cont(cctx,
                ggml_permute(cctx, gdn, 1, 0, 2, 3));
            gdn_cont = ggml_reshape_4d(cctx, gdn_cont,
                value_dim, 1, n_tokens, 1);
            gdn_cont = ggml_rms_norm(cctx, gdn_cont, m->norm_eps);
            if (m->layers[i].ssm_norm_w)
                gdn_cont = ggml_mul(cctx, gdn_cont, m->layers[i].ssm_norm_w);

            struct ggml_tensor *z = ggml_mul_mat(cctx,
                m->layers[i].attn_gate_w, n);
            z = ggml_silu(cctx, z);
            z = ggml_reshape_4d(cctx, z, value_dim, 1, n_tokens, 1);
            gdn_cont = ggml_mul(cctx, gdn_cont, z);

            gdn_cont = ggml_reshape_2d(cctx, gdn_cont, value_dim, n_tokens);
            gdn_cont = ggml_mul_mat(cctx, m->layers[i].ssm_out_w, gdn_cont);
            cur = ggml_add(cctx, inp_l, gdn_cont);

            if (m->layers[i].ssm_conv_state) {
                struct ggml_tensor *new_conv = ggml_view_2d(cctx, sx,
                    conv_dim, d_conv - 1, sx->nb[1],
                    n_tokens * ggml_element_size(sx));
                ggml_build_forward_expand(*gf,
                    ggml_cpy(cctx, new_conv, m->layers[i].ssm_conv_state));
            }
        } else {
            struct ggml_tensor *Q_full = ggml_mul_mat(cctx,
                m->layers[i].attn_q_w, cur);
            struct ggml_tensor *K = ggml_mul_mat(cctx,
                m->layers[i].attn_k_w, cur);
            struct ggml_tensor *V = ggml_mul_mat(cctx,
                m->layers[i].attn_v_w, cur);

            int qk_dim = q_head_dim * m->n_head;
            struct ggml_tensor *Q = ggml_view_2d(cctx, Q_full,
                qk_dim, n_tokens, Q_full->nb[1], 0);
            Q = ggml_reshape_3d(cctx, Q, q_head_dim, m->n_head, n_tokens);

            struct ggml_tensor *gate = ggml_view_2d(cctx, Q_full,
                qk_dim, n_tokens, Q_full->nb[1],
                qk_dim * ggml_element_size(Q_full));
            gate = ggml_reshape_3d(cctx, gate, q_head_dim, m->n_head, n_tokens);

            K = ggml_reshape_3d(cctx, K, head_dim, m->n_head_kv, n_tokens);
            V = ggml_reshape_3d(cctx, V, head_dim, m->n_head_kv, n_tokens);

            if (m->layers[i].attn_q_norm_w) {
                Q = ggml_rms_norm(cctx, Q, m->norm_eps);
                Q = ggml_mul(cctx, Q, m->layers[i].attn_q_norm_w);
            }
            if (m->layers[i].attn_k_norm_w) {
                K = ggml_rms_norm(cctx, K, m->norm_eps);
                K = ggml_mul(cctx, K, m->layers[i].attn_k_norm_w);
            }

            Q = ggml_rope_ext(cctx, Q,
                *pos_out, NULL, m->n_rot, m->rope_mode, 0,
                m->rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            K = ggml_rope_ext(cctx, K,
                *pos_out, NULL, m->n_rot, m->rope_mode, 0,
                m->rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            struct ggml_tensor *k = ggml_view_3d(cctx, m->layers[i].k_cache,
                head_dim, m->n_head_kv, n_tokens,
                m->layers[i].k_cache->nb[1], m->layers[i].k_cache->nb[2],
                n_past * m->layers[i].k_cache->nb[2]);
            struct ggml_tensor *v = ggml_view_3d(cctx, m->layers[i].v_cache,
                head_dim, m->n_head_kv, n_tokens,
                m->layers[i].v_cache->nb[1], m->layers[i].v_cache->nb[2],
                n_past * m->layers[i].v_cache->nb[2]);
            ggml_build_forward_expand(*gf, ggml_cpy(cctx, K, k));
            ggml_build_forward_expand(*gf, ggml_cpy(cctx, V, v));
            K = ggml_view_3d(cctx, m->layers[i].k_cache,
                head_dim, m->n_head_kv, n_past + n_tokens,
                m->layers[i].k_cache->nb[1], m->layers[i].k_cache->nb[2], 0);
            V = ggml_view_3d(cctx, m->layers[i].v_cache,
                head_dim, m->n_head_kv, n_past + n_tokens,
                m->layers[i].v_cache->nb[1], m->layers[i].v_cache->nb[2], 0);
            struct ggml_tensor *kq = ggml_mul_mat(cctx,
                ggml_cont(cctx, ggml_permute(cctx, K, 0, 2, 1, 3)),
                ggml_cont(cctx, ggml_permute(cctx, Q, 0, 2, 1, 3)));
            kq = ggml_diag_mask_inf(cctx,
                ggml_scale(cctx, kq, 1.0f / sqrtf((float)q_head_dim)), n_past);
            kq = ggml_soft_max(cctx, kq);
            struct ggml_tensor *v_att = ggml_permute(cctx,
                ggml_mul_mat(cctx,
                    ggml_cont(cctx, ggml_permute(cctx, V, 1, 2, 0, 3)),
                    ggml_cont(cctx, kq)),
                0, 2, 1, 3);

            struct ggml_tensor *attn_out = ggml_reshape_2d(cctx,
                ggml_cont(cctx, v_att), qk_dim, n_tokens);
            gate = ggml_sigmoid(cctx, ggml_reshape_2d(cctx,
                ggml_cont(cctx, gate), qk_dim, n_tokens));
            attn_out = ggml_mul(cctx, attn_out, gate);
            attn_out = ggml_mul_mat(cctx, m->layers[i].attn_out_w, attn_out);
            cur = ggml_add(cctx, inp_l, attn_out);
        }

        struct ggml_tensor *inp_ffn = cur;
        cur = ggml_rms_norm(cctx, cur, m->norm_eps);
        if (m->layers[i].post_attn_norm_w)
            cur = ggml_mul(cctx, cur, m->layers[i].post_attn_norm_w);

        struct ggml_tensor *ffn_gate = ggml_mul_mat(cctx, m->layers[i].ffn_gate_w, cur);
        struct ggml_tensor *up   = ggml_mul_mat(cctx, m->layers[i].ffn_up_w, cur);
        struct ggml_tensor *act  = ggml_silu(cctx, ffn_gate);
        cur = ggml_add(cctx,
            ggml_mul_mat(cctx, m->layers[i].ffn_down_w, ggml_mul(cctx, act, up)),
            inp_ffn);
    }

    cur = ggml_rms_norm(cctx, cur, m->norm_eps);
    if (m->output_norm_w) cur = ggml_mul(cctx, cur, m->output_norm_w);
    cur = ggml_mul_mat(cctx, m->output_w,
        ggml_view_2d(cctx, cur, m->n_embd, 1,
            cur->nb[1], (n_tokens - 1) * cur->nb[1]));
    ggml_set_output(cur);
    ggml_build_forward_expand(*gf, cur);
    return cur;
}
