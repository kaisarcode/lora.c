/**
 * gemma.c - Gemma graph builder for gguf.c
 * Summary: Compute graph builder for the Gemma architecture.
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
 * Builds the compute graph for Gemma models (GELU, embedding scale sqrt(n_embd)).
 * @param m Loaded model.
 * @param n_tokens Number of input tokens.
 * @param n_past Number of past tokens in KV cache.
 * @param gf Output compute graph.
 * @param embd_out Output embedding input tensor.
 * @param pos_out Output position input tensor.
 * @return Output logits tensor.
 */
struct ggml_tensor *kc_gguf_build_graph_gemma(kc_gguf_model_t *m, int n_tokens, int n_past,
    struct ggml_cgraph **gf, struct ggml_tensor **embd_out, struct ggml_tensor **pos_out)
{
    kc_gguf_arch_params_t p = { sqrtf((float)m->n_embd), 1 };
    return kc_gguf_build_graph_impl(m, n_tokens, n_past, gf, embd_out, pos_out, p);
}
