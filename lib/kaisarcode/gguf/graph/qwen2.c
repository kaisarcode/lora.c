/**
 * qwen2.c - Qwen2 transformer graph builder for gguf.c
 * Summary: Compute graph builder for Qwen2-family transformer models.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#include "graph.h"
#include <stdbool.h>

#include "ggml.h"

/**
 * Builds the compute graph for Qwen2-family models.
 * @param m Loaded model.
 * @param n_tokens Number of input tokens.
 * @param n_past Number of past tokens in KV cache.
 * @param gf Output compute graph.
 * @param embd_out Output embedding input tensor.
 * @param pos_out Output position input tensor.
 * @return Output logits tensor.
 */
struct ggml_tensor *kc_gguf_build_graph_qwen2(kc_gguf_model_t *m, int n_tokens, int n_past,
    struct ggml_cgraph **gf, struct ggml_tensor **embd_out, struct ggml_tensor **pos_out)
{
    kc_gguf_arch_params_t p = { 1.0f, 0 };
    return kc_gguf_build_graph_impl(m, n_tokens, n_past, gf, embd_out, pos_out, p);
}
