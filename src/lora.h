/**
 * lora.h - LoRA Adapter Training Library Public API
 * Summary: Interface for training LoRA adapters from GGUF models.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef KC_LORA_H
#define KC_LORA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_lora kc_lora_t;

#define KC_LORA_OK      0
#define KC_LORA_ERROR  -1

/**
 * The library copies this structure during kc_lora_open(); the caller may
 * release its copy immediately after the call.
 *
 * @param model_path Path to the base GGUF model file.
 * @param output_path Path to save the trained LoRA adapter (.safetensors).
 * @param rank LoRA rank (r).
 * @param alpha LoRA alpha scaling factor.
 * @param lr Learning rate for optimizer.
 * @param epochs Number of training epochs.
 * @param batch Micro-batch size.
 * @param ctx Context window size in tokens (0 for model default).
 * @param threads Number of threads for compute (0 for auto).
 * @param gpu GPU mode: -1 auto, 0 CPU, > 0 require GPU.
 * @param gpu_layers Number of layers to offload to GPU.
 * @param save_every Save checkpoint every N steps (0 = disabled).
 */
typedef struct kc_lora_options {
    const char *model_path;
    const char *output_path;
    int rank;
    float alpha;
    float lr;
    int epochs;
    int batch;
    int ctx;
    int threads;
    int gpu;
    int gpu_layers;
    int save_every;
} kc_lora_options_t;

/**
 * Callback function for training progress updates.
 *
 * @param epoch Current epoch number.
 * @param step Current step number.
 * @param total_steps Total steps in this epoch.
 * @param loss Current loss value.
 * @param user User-supplied pointer passed back to the callback.
 * @return 0 to continue, non-zero to stop training.
 */
typedef int (*kc_lora_progress_fn)(int epoch, int step, int total_steps,
    float loss, void *user);

/**
 * Open a new training context with the specified options.
 * @param out Pointer to receive the allocated context.
 * @param opts Initialization options.
 * @return 0 on success, non-zero on failure.
 */
int kc_lora_open(kc_lora_t **out, const kc_lora_options_t *opts);

/**
 * Close and release the training context.
 * @param ctx Context to close.
 * @return 0 on success, non-zero on failure.
 */
int kc_lora_close(kc_lora_t *ctx);

/**
 * Run LoRA training on the provided dataset.
 *
 * Reads plain text from the file, tokenizes it, and trains LoRA A/B
 * matrices using gradient descent with cross-entropy loss.
 *
 * @param ctx Training context.
 * @param data_path Path to the plain text dataset file.
 * @param progress Callback function called with progress updates.
 * @param user User-supplied pointer passed back to the callback.
 * @return 0 on success, or -1 on failure.
 */
int kc_lora_run(kc_lora_t *ctx, const char *data_path,
    kc_lora_progress_fn progress, void *user);

/**
 * Stop an ongoing training call.
 *
 * This function is thread-safe and can be called from another thread or
 * from the progress callback. Once called, the active kc_lora_run() call
 * will finish its current step and return KC_LORA_OK.
 *
 * @param ctx Training context.
 * @return 0 on success, or -1 on failure.
 */
int kc_lora_stop(kc_lora_t *ctx);

/**
 * Get the last error message from the context.
 *
 * The returned string is owned by the training context and is valid until
 * the next library call that modifies the context state.
 *
 * @param ctx Training context.
 * @return Pointer to a human-readable error string.
 */
const char *kc_lora_error(kc_lora_t *ctx);

#ifdef __cplusplus
}
#endif

#endif
