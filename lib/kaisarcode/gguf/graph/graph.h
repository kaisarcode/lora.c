/**
 * graph.h - Compute graph builder interface for gguf.c
 * Summary: Shared types and declarations for per-architecture compute graph builders.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef KC_GRAPH_H
#define KC_GRAPH_H

#include "gguf.h"

struct ggml_tensor *kc_lora_mm(struct ggml_context *cctx, kc_gguf_model_t *m,
    struct ggml_tensor *W, struct ggml_tensor *x, int layer, int proj);

struct ggml_tensor *kc_gguf_build_graph_impl(kc_gguf_model_t *m, int n_tokens, int n_past,
    struct ggml_cgraph **gf, struct ggml_tensor **embd_out, struct ggml_tensor **pos_out,
    kc_gguf_arch_params_t params);

struct ggml_tensor *kc_gguf_build_graph_llama(kc_gguf_model_t *m, int n_tokens, int n_past,
    struct ggml_cgraph **gf, struct ggml_tensor **embd_out, struct ggml_tensor **pos_out);

struct ggml_tensor *kc_gguf_build_graph_gemma(kc_gguf_model_t *m, int n_tokens, int n_past,
    struct ggml_cgraph **gf, struct ggml_tensor **embd_out, struct ggml_tensor **pos_out);

struct ggml_tensor *kc_gguf_build_graph_gpt2(kc_gguf_model_t *m, int n_tokens, int n_past,
    struct ggml_cgraph **gf, struct ggml_tensor **embd_out, struct ggml_tensor **pos_out);

struct ggml_tensor *kc_gguf_build_graph_qwen35(kc_gguf_model_t *m, int n_tokens, int n_past,
    struct ggml_cgraph **gf, struct ggml_tensor **embd_out, struct ggml_tensor **pos_out);

struct ggml_tensor *kc_gguf_build_graph_gemma4(kc_gguf_model_t *m, int n_tokens, int n_past,
    struct ggml_cgraph **gf, struct ggml_tensor **embd_out, struct ggml_tensor **pos_out);

#endif
