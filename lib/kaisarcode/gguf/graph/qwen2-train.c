/**
 * train_qwen2.c - Qwen2 training graph builder for gguf.c
 * Summary: Training compute graph for Qwen2-family transformer models.
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
 * Builds the training compute graph for Qwen2-family models.
 * Uses full-sequence processing with causal attention and cross-entropy loss.
 * @param m Loaded model with LoRA adapters.
 * @param n_ctx Context length in tokens.
 * @param lora_scale LoRA scale factor.
 * @param input_ids Output receives the input token tensor.
 * @param pos_ids Output receives the position tensor.
 * @param target_ids Output receives the target token tensor.
 * @return Compute graph.
 */
struct ggml_cgraph *kc_gguf_build_training_graph_qwen2(kc_gguf_model_t *m,
    int n_ctx, float lora_scale,
    struct ggml_tensor **input_ids,
    struct ggml_tensor **pos_ids,
    struct ggml_tensor **target_ids)
{
    struct ggml_context *cctx = m->ctx_compute;
    (void)lora_scale;
    int n_layer = m->n_layer;
    int head_size = m->n_head_dim;
    int n_head = m->n_head;
    int n_head_kv = m->n_head_kv;

    struct ggml_cgraph *gf = ggml_new_graph_custom(cctx,
        m->graph_size, true);

    struct ggml_tensor *cur;

    struct ggml_tensor *inp_ids = ggml_new_tensor_1d(cctx,
        GGML_TYPE_I32, n_ctx);
    ggml_set_input(inp_ids);

    cur = ggml_get_rows(cctx, m->tok_embeddings, inp_ids);

    struct ggml_tensor *p_ids = ggml_new_tensor_1d(cctx,
        GGML_TYPE_I32, n_ctx);
    ggml_set_input(p_ids);

    for (int i = 0; i < n_layer; i++) {
        struct ggml_tensor *inp_l = cur;
        cur = ggml_rms_norm(cctx, cur, m->norm_eps);
        if (m->layers[i].attn_norm_w)
            cur = ggml_mul(cctx, cur, m->layers[i].attn_norm_w);

        struct ggml_tensor *Q = kc_lora_mm(cctx, m,
            m->layers[i].attn_q_w, cur, i, KC_GGUF_LORA_ATTN_Q);
        if (m->layers[i].attn_q_b)
            Q = ggml_add(cctx, Q, m->layers[i].attn_q_b);
        struct ggml_tensor *K = kc_lora_mm(cctx, m,
            m->layers[i].attn_k_w, cur, i, KC_GGUF_LORA_ATTN_K);
        if (m->layers[i].attn_k_b)
            K = ggml_add(cctx, K, m->layers[i].attn_k_b);
        struct ggml_tensor *V = kc_lora_mm(cctx, m,
            m->layers[i].attn_v_w, cur, i, KC_GGUF_LORA_ATTN_V);
        if (m->layers[i].attn_v_b)
            V = ggml_add(cctx, V, m->layers[i].attn_v_b);

        Q = ggml_reshape_3d(cctx, ggml_cont(cctx, Q),
            head_size, n_head, n_ctx);
        K = ggml_reshape_3d(cctx, ggml_cont(cctx, K),
            head_size, n_head_kv, n_ctx);
        if (m->layers[i].attn_q_norm_w) {
            Q = ggml_rms_norm(cctx, Q, m->norm_eps);
            Q = ggml_mul(cctx, Q, m->layers[i].attn_q_norm_w);
        }
        if (m->layers[i].attn_k_norm_w) {
            K = ggml_rms_norm(cctx, K, m->norm_eps);
            K = ggml_mul(cctx, K, m->layers[i].attn_k_norm_w);
        }
        Q = ggml_rope_ext(cctx, Q,
            p_ids, NULL, m->n_rot, m->rope_mode, 0,
            m->rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(cctx, K,
            p_ids, NULL, m->n_rot, m->rope_mode, 0,
            m->rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        V = ggml_reshape_3d(cctx, ggml_cont(cctx, V),
            head_size, n_head_kv, n_ctx);

        struct ggml_tensor *kq = ggml_mul_mat(cctx,
            ggml_cont(cctx, ggml_permute(cctx, K, 0, 2, 1, 3)),
            ggml_cont(cctx, ggml_permute(cctx, Q, 0, 2, 1, 3)));
        kq = ggml_scale(cctx, kq,
            1.0f / sqrtf((float)head_size));
        kq = ggml_diag_mask_inf(cctx, kq, 0);
        kq = ggml_soft_max(cctx, kq);
        struct ggml_tensor *v_att = ggml_permute(cctx,
            ggml_mul_mat(cctx,
                ggml_cont(cctx, ggml_permute(cctx, V, 1, 2, 0, 3)),
                ggml_cont(cctx, kq)),
            0, 2, 1, 3);

        cur = ggml_add(cctx,
            kc_lora_mm(cctx, m, m->layers[i].attn_out_w,
                ggml_reshape_2d(cctx,
                    ggml_cont(cctx, v_att),
                    n_head * head_size, n_ctx),
                i, KC_GGUF_LORA_ATTN_O),
            inp_l);

        struct ggml_tensor *inp_ffn = cur;
        cur = ggml_rms_norm(cctx, cur, m->norm_eps);
        if (m->layers[i].ffn_norm_w)
            cur = ggml_mul(cctx, cur, m->layers[i].ffn_norm_w);

        struct ggml_tensor *gate = kc_lora_mm(cctx, m,
            m->layers[i].ffn_gate_w, cur, i,
            KC_GGUF_LORA_FFN_GATE);
        struct ggml_tensor *up   = kc_lora_mm(cctx, m,
            m->layers[i].ffn_up_w, cur, i,
            KC_GGUF_LORA_FFN_UP);
        struct ggml_tensor *act  = ggml_silu(cctx, gate);
        cur = ggml_add(cctx,
            kc_lora_mm(cctx, m, m->layers[i].ffn_down_w,
                ggml_mul(cctx, act, up), i,
                KC_GGUF_LORA_FFN_DOWN),
            inp_ffn);
    }

    cur = ggml_rms_norm(cctx, cur, m->norm_eps);
    if (m->output_norm_w)
        cur = ggml_mul(cctx, cur, m->output_norm_w);

    cur = ggml_mul_mat(cctx, m->output_w, cur);

    struct ggml_tensor *logits_f32 = (cur->type == GGML_TYPE_F32)
        ? cur : ggml_cast(cctx, cur, GGML_TYPE_F32);
    struct ggml_tensor *targets = ggml_dup_tensor(cctx, logits_f32);
    ggml_set_input(targets);
    struct ggml_tensor *loss = ggml_cross_entropy_loss(cctx,
        logits_f32, targets);
    ggml_set_loss(loss);

    if (input_ids)  *input_ids  = inp_ids;
    if (pos_ids)    *pos_ids    = p_ids;
    if (target_ids) *target_ids = targets;

    ggml_build_forward_expand(gf, loss);
    ggml_build_backward_expand(cctx, gf, NULL);
    return gf;
}
