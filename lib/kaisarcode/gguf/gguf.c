/**
 * gguf.c
 * Summary: CLI for GGUF model metadata, tokenizer, graph, and quantization utilities.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#define _XOPEN_SOURCE 700

#include "gguf.h"
#include "tok/tok.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <getopt.h>
#include <errno.h>

#define KC_VERSION "1.3.2"

static const char * const kc_usage =
    "Usage: gguf MODEL [options] [INPUT]\n"
    "       gguf MODEL -q TYPE -o output.gguf\n"
    "\n"
    "GGUF model metadata, tokenizer, graph, and quantization utilities.\n"
    "\n"
    "Options:\n"
    "  -h, --help           Show this help\n"
    "  -v, --version        Show version\n"
    "  -i, --info           Print model metadata and exit\n"
    "  -t, --tokenize       Tokenize input text, print IDs\n"
    "  -d, --detokenize     Decode token IDs from stdin, print text\n"
    "  -q, --quantize TYPE  Quantize a GGUF file\n"
    "  -o, --output PATH    Output path for quantized file (required with --quantize)\n"
    "\n"
    "If no operation flag is given, --info is implied.\n"
    "INPUT: text input for --tokenize.\n";

static const char * const kc_version = "gguf v" KC_VERSION;

static const struct {
    const char *name;
    enum ggml_type type;
} kc_type_map[] = {
    { "F32",     GGML_TYPE_F32     },
    { "F16",     GGML_TYPE_F16     },
    { "Q8_0",    GGML_TYPE_Q8_0    },
    { "Q6_K",    GGML_TYPE_Q6_K    },
    { "Q5_K",    GGML_TYPE_Q5_K    },
    { "Q5_K_M",  GGML_TYPE_Q5_K    },
    { "Q5_K_S",  GGML_TYPE_Q5_K    },
    { "Q5_0",    GGML_TYPE_Q5_0    },
    { "Q5_1",    GGML_TYPE_Q5_1    },
    { "Q4_K",    GGML_TYPE_Q4_K    },
    { "Q4_K_M",  GGML_TYPE_Q4_K    },
    { "Q4_K_S",  GGML_TYPE_Q4_K    },
    { "Q4_0",    GGML_TYPE_Q4_0    },
    { "Q4_1",    GGML_TYPE_Q4_1    },
    { "Q3_K",    GGML_TYPE_Q3_K    },
    { "Q3_K_M",  GGML_TYPE_Q3_K    },
    { "Q3_K_S",  GGML_TYPE_Q3_K    },
    { "Q2_K",    GGML_TYPE_Q2_K    },
    { "TQ1_0",   GGML_TYPE_TQ1_0   },
    { "TQ2_0",   GGML_TYPE_TQ2_0   },
    { "IQ1_S",   GGML_TYPE_IQ1_S   },
    { "IQ1_M",   GGML_TYPE_IQ1_M   },
    { "IQ2_XXS", GGML_TYPE_IQ2_XXS },
    { "IQ2_XS",  GGML_TYPE_IQ2_XS  },
    { "IQ2_S",   GGML_TYPE_IQ2_S   },
    { "IQ3_XXS", GGML_TYPE_IQ3_XXS },
    { "IQ3_S",   GGML_TYPE_IQ3_S   },
    { "IQ4_NL",  GGML_TYPE_IQ4_NL  },
    { "IQ4_XS",  GGML_TYPE_IQ4_XS  },
    { NULL,      GGML_TYPE_COUNT   },
};

/**
 * Print usage information to stderr.
 * @param name Program name (argv[0]).
 * @return None.
 */
static void print_usage(const char *name) {
    fprintf(stderr, "%s", kc_usage);
    (void)name;
}

/**
 * Print version string to stdout.
 * @return None.
 */
static void print_version(void) {
    printf("%s\n", kc_version);
}

/**
 * Parse a type name string to ggml_type.
 * @param s Case-insensitive type name string.
 * @return Matching ggml_type, or GGML_TYPE_COUNT if unrecognized.
 */
static enum ggml_type parse_type(const char *s) {
    for (int i = 0; kc_type_map[i].name != NULL; i++)
        if (strcasecmp(s, kc_type_map[i].name) == 0)
            return kc_type_map[i].type;
    return GGML_TYPE_COUNT;
}

/**
 * Print model metadata to stdout.
 * @param m Loaded GGUF model.
 * @return None.
 */
static void print_info(kc_gguf_model_t *m) {
    printf("Architecture:  ");
    int id = gguf_find_key(m->gguf, "general.architecture");
    if (id >= 0) puts(gguf_get_val_str(m->gguf, id));
    else puts("(unknown)");

    printf("Vocabulary:    %d\n", m->n_vocab);
    printf("Embedding:     %d\n", m->n_embd);
    printf("Heads:         %d\n", m->n_head);
    printf("KV heads:      %d\n", m->n_head_kv);
    printf("Head dim:      %d\n", m->n_head_dim);
    printf("Layers:        %d\n", m->n_layer);
    printf("RoPE dim:      %d\n", m->n_rot);
    printf("RoPE freq:     %.0f\n", m->rope_freq_base);
    printf("Norm eps:      %g\n", m->norm_eps);
    printf("Tok embd:      %s\n", m->tok_embeddings ? "yes" : "no");
    printf("Output norm:   %s\n", m->output_norm_w ? "yes" : "no");
    printf("Output bias:   %s\n", m->output_norm_b ? "yes" : "no");
    printf("Output weight: %s%s\n", m->output_w ? "yes" : "no",
        m->output_w == m->tok_embeddings ? " (tied)" : "");
    printf("Pos embd:      %s\n", m->position_embd_w ? "yes" : "no");

    int n_tensors = gguf_get_n_tensors(m->gguf);
    printf("Tensors:       %d\n", n_tensors);

    size_t total_size = 0;
    for (int i = 0; i < n_tensors; i++)
        total_size += gguf_get_tensor_size(m->gguf, i);
    printf("Weight size:   %.2f MB\n", (double)total_size / (1024 * 1024));

}

/**
 * Tokenize input text and print token IDs to stdout.
 * @param m Loaded GGUF model.
 * @param input Null-terminated input text.
 * @return None.
 */
static void do_tokenize(kc_gguf_model_t *m, const char *input) {
    char err[1024];
    kc_gguf_tokenizer_t *tok = NULL;
    if (kc_gguf_tokenizer_load(&tok, m->gguf, m->n_vocab, err, sizeof(err)) != 0) {
        fprintf(stderr, "gguf: failed to load tokenizer: %s\n", err);
        exit(1);
    }
    int tokens[16384];
    int n = kc_gguf_tokenizer_encode(tok, input, tokens, 16384, err, sizeof(err));
    if (n < 0) {
        fprintf(stderr, "gguf: tokenization failed: %s\n", err);
        kc_gguf_tokenizer_free(tok);
        exit(1);
    }
    for (int i = 0; i < n; i++) {
        if (i > 0) putchar(' ');
        printf("%d", tokens[i]);
    }
    putchar('\n');
    kc_gguf_tokenizer_free(tok);
}

/**
 * Read token IDs from stdin and print decoded text to stdout.
 * @param m Loaded GGUF model.
 * @return None.
 */
static void do_detokenize(kc_gguf_model_t *m) {
    char err[1024];
    kc_gguf_tokenizer_t *tok = NULL;
    if (kc_gguf_tokenizer_load(&tok, m->gguf, m->n_vocab, err, sizeof(err)) != 0) {
        fprintf(stderr, "gguf: failed to load tokenizer: %s\n", err);
        exit(1);
    }
    char line[65536];
    while (fgets(line, sizeof(line), stdin)) {
        const char *p = line;
        while (*p) {
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '\n' || *p == '\0') break;
            char *end;
            long id = strtol(p, &end, 10);
            if (end == p) break;
            const char *text = kc_gguf_tokenizer_decode(tok, (int)id);
            fputs(text, stdout);
            p = end;
        }
    }
    putchar('\n');
    kc_gguf_tokenizer_free(tok);
}

/**
 * Main entry point.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success, 1 on error.
 */
int main(int argc, char **argv) {
    static struct option long_opts[] = {
        {"help",       no_argument,       0, 'h'},
        {"version",    no_argument,       0, 'v'},
        {"info",       no_argument,       0, 'i'},
        {"tokenize",   no_argument,       0, 't'},
        {"detokenize", no_argument,       0, 'd'},
        {"quantize",   required_argument, 0, 'q'},
        {"output",     required_argument, 0, 'o'},
        {0, 0, 0, 0}
    };

    kc_gguf_options_t opts = kc_gguf_options_default();
    kc_gguf_options_load_env(&opts);

    const char *output_path = NULL;
    int op_info = 0, op_tokenize = 0, op_detokenize = 0;
    int op_quantize = 0;
    enum ggml_type target_type = GGML_TYPE_Q4_K;
    int c;

    while ((c = getopt_long(argc, argv, "hvitdq:o:", long_opts, NULL)) != -1) {
        switch (c) {
            case 'h': print_usage(argv[0]); return 0;
            case 'v': print_version(); return 0;
            case 'i': op_info = 1; break;
            case 't': op_tokenize = 1; break;
            case 'd': op_detokenize = 1; break;
            case 'q':
                op_quantize = 1;
                target_type = parse_type(optarg);
                if (target_type == GGML_TYPE_COUNT) {
                    fprintf(stderr, "gguf: unknown type '%s'\n", optarg);
                    return 1;
                }
                break;
            case 'o': output_path = optarg; break;
            default:  fprintf(stderr, "Try '%s --help'\n", argv[0]); return 1;
        }
    }

    if (optind >= argc || argv[optind][0] == '-') {
        fprintf(stderr, "gguf: model path is required\n");
        fprintf(stderr, "Try '%s --help'\n", argv[0]);
        kc_gguf_options_free(&opts);
        return 1;
    }

    free(opts.model_path);
    opts.model_path = strdup(argv[optind++]);
    if (!opts.model_path) {
        fprintf(stderr, "gguf: out of memory\n");
        kc_gguf_options_free(&opts);
        return 1;
    }

    if (op_quantize) {
        if (!output_path) {
            fprintf(stderr, "gguf: output path required for quantize\n");
            kc_gguf_options_free(&opts);
            return 1;
        }
        if (optind < argc) {
            fprintf(stderr, "gguf: unexpected positional argument '%s'\n", argv[optind]);
            kc_gguf_options_free(&opts);
            return 1;
        }
        int quant_rc = kc_gguf_quantize(opts.model_path, output_path, target_type) == 0 ? 0 : 1;
        kc_gguf_options_free(&opts);
        return quant_rc;
    }

    if (!op_info && !op_tokenize && !op_detokenize)
        op_info = 1;

    kc_gguf_model_t *m = NULL;
    if (kc_gguf_open(&m, &opts) != 0) {
        fprintf(stderr, "gguf: %s\n", kc_gguf_error(m));
        kc_gguf_close(m);
        kc_gguf_options_free(&opts);
        return 1;
    }

    kc_gguf_listen_signals(m);
#ifndef _WIN32
    kc_gguf_listen_signal(m, 2);
    kc_gguf_listen_signal(m, 15);
#endif

    if (op_info) print_info(m);

    if (op_tokenize) {
        const char *input = NULL;
        if (optind < argc)
            input = argv[optind];
        else {
            static char buf[65536];
            if (fgets(buf, sizeof(buf), stdin)) {
                size_t len = strlen(buf);
                if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';
                input = buf;
            }
        }
        if (input) do_tokenize(m, input);
        else fprintf(stderr, "gguf: no input for tokenize\n");
        if (optind + 1 < argc) {
            fprintf(stderr, "gguf: unexpected positional argument '%s'\n", argv[optind + 1]);
            kc_gguf_close(m);
            kc_gguf_options_free(&opts);
            return 1;
        }
    } else if (optind < argc) {
        fprintf(stderr, "gguf: unexpected positional argument '%s'\n", argv[optind]);
        kc_gguf_close(m);
        kc_gguf_options_free(&opts);
        return 1;
    }

    if (op_detokenize) do_detokenize(m);

    kc_gguf_close(m);
    kc_gguf_options_free(&opts);
    return 0;
}
