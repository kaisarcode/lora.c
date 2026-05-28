/**
 * lora.c - LoRA Adapter Training CLI
 * Summary: Command-line interface for LoRA low-rank adaptation training.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif

#include "lora.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VERSION "1.0.0"

/**
 * Prints usage information to stdout.
 * @return None.
 */
static void print_help(void) {
    printf("Usage: lora [input] -o adapter.safetensors -d data.txt [options]\n");
    printf("\n");
    printf("Required:\n");
    printf("  input                     GGUF model path (positional)\n");
    printf("  -o, --output PATH         LoRA adapter output path (.safetensors)\n");
    printf("  -d, --data PATH           Plain text dataset path\n");
    printf("\n");
    printf("Options:\n");
    printf("  --rank N                  LoRA rank (default: 16)\n");
    printf("  --alpha F                 LoRA alpha scaling (default: 32.0)\n");
    printf("  --lr F                    Learning rate (default: 1e-4)\n");
    printf("  --epochs N                Number of epochs (default: 1)\n");
    printf("  --batch N                 Batch size (default: 1)\n");
    printf("  --ctx N                   Context size in tokens (default: from model)\n");
    printf("  --threads N               Number of threads (default: auto)\n");
    printf("  --gpu N                   GPU mode: -1 auto, 0 CPU, >0 require\n");
    printf("  --gpu-layers N            Layers to offload to GPU (default: all)\n");
    printf("  --save-every N            Checkpoint every N steps, 0=off (default: 0)\n");
    printf("  -h, --help                Show help\n");
    printf("  -v, --version             Show version\n");
}

/**
 * Training progress callback.
 * @param epoch Current epoch number.
 * @param step Current step number.
 * @param total_steps Total steps in this epoch.
 * @param loss Current loss value.
 * @param user User-supplied pointer.
 * @return 0 to continue.
 */
static int progress_cb(int epoch, int step, int total_steps,
    float loss, void *user)
{
    (void)user;
    fprintf(stderr, "\rEpoch %d | Step %d/%d | Loss: %.4f",
        epoch, step + 1, total_steps, loss);
    return 0;
}

/**
 * CLI entry point for lora training tool.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Exit code.
 */
int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *output_path = NULL;
    const char *data_path = NULL;
    int rank = 16;
    float alpha = 32.0f;
    float lr = 1e-4f;
    int epochs = 1;
    int batch = 1;
    int ctx = 0;
    int threads = 0;
    int gpu = -1;
    int gpu_layers = 999;
    int save_every = 0;

    if (argc < 2) {
        print_help();
        return 1;
    }

    int i = 1;

    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
        print_help();
        return 0;
    }

    if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--version") == 0) {
        printf("lora %s\n", VERSION);
        return 0;
    }

    if (i < argc && argv[i][0] != '-') {
        model_path = argv[i];
        i++;
    }

    while (i < argc) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help();
            return 0;
        } else if (strcmp(argv[i], "-v") == 0
            || strcmp(argv[i], "--version") == 0)
        {
            printf("lora %s\n", VERSION);
            return 0;
        } else if (strcmp(argv[i], "-o") == 0
            || strcmp(argv[i], "--output") == 0)
        {
            if (i + 1 >= argc) {
                fprintf(stderr, "lora: -o requires a value\n");
                return 1;
            }
            output_path = argv[++i];
        } else if (strcmp(argv[i], "-d") == 0
            || strcmp(argv[i], "--data") == 0)
        {
            if (i + 1 >= argc) {
                fprintf(stderr, "lora: -d requires a value\n");
                return 1;
            }
            data_path = argv[++i];
        } else if (strcmp(argv[i], "--rank") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "lora: --rank requires a value\n");
                return 1;
            }
            rank = atoi(argv[++i]);
            if (rank < 1) {
                fprintf(stderr, "lora: invalid rank %d\n", rank);
                return 1;
            }
        } else if (strcmp(argv[i], "--alpha") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "lora: --alpha requires a value\n");
                return 1;
            }
            alpha = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "lora: --lr requires a value\n");
                return 1;
            }
            lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "lora: --epochs requires a value\n");
                return 1;
            }
            epochs = atoi(argv[++i]);
            if (epochs < 1) epochs = 1;
        } else if (strcmp(argv[i], "--batch") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "lora: --batch requires a value\n");
                return 1;
            }
            batch = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ctx") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "lora: --ctx requires a value\n");
                return 1;
            }
            ctx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "lora: --threads requires a value\n");
                return 1;
            }
            threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "lora: --gpu requires a value\n");
                return 1;
            }
            gpu = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu-layers") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "lora: --gpu-layers requires a value\n");
                return 1;
            }
            gpu_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--save-every") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "lora: --save-every requires a value\n");
                return 1;
            }
            save_every = atoi(argv[++i]);
        } else {
            fprintf(stderr, "lora: unknown flag '%s'\n", argv[i]);
            return 1;
        }
        i++;
    }

    if (!model_path) {
        fprintf(stderr, "lora: model path is required\n");
        return 1;
    }
    if (!output_path) {
        fprintf(stderr, "lora: -o output path is required\n");
        return 1;
    }
    if (!data_path) {
        fprintf(stderr, "lora: -d dataset path is required\n");
        return 1;
    }

    kc_lora_options_t opts = {
        .model_path  = model_path,
        .output_path = output_path,
        .rank        = rank,
        .alpha       = alpha,
        .lr          = lr,
        .epochs      = epochs,
        .batch       = batch,
        .ctx         = ctx,
        .threads     = threads,
        .gpu         = gpu,
        .gpu_layers  = gpu_layers,
        .save_every  = save_every
    };

    kc_lora_t *lora_ctx = NULL;
    int rc = kc_lora_open(&lora_ctx, &opts);
    if (rc != KC_LORA_OK) {
        fprintf(stderr, "lora: %s\n", kc_lora_error(lora_ctx));
        return 1;
    }

    fprintf(stderr, "Starting LoRA training...\n");
    int result = kc_lora_run(lora_ctx, data_path, progress_cb, NULL);
    fprintf(stderr, "\n");

    if (result != KC_LORA_OK) {
        fprintf(stderr, "lora: %s\n", kc_lora_error(lora_ctx));
        kc_lora_close(lora_ctx);
        return 1;
    }

    fprintf(stderr, "Training complete. Adapter saved to %s\n", output_path);
    kc_lora_close(lora_ctx);
    return 0;
}
