/**
 * spm.c
 * Summary: SentencePiece BPE tokenizer with unigram score merging.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#include "tok.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int id;
    int next;
    int prev;
} spm_symbol_t;

/**
 * Decode a SentencePiece raw token string into a clean string.
 * Handles <0xNN> byte tokens and U+2581 (space) replacement.
 * @param raw The raw token string.
 * @return Allocated clean string, or NULL on failure.
 */
static char *kc_tokenizer_spm_decode_token(const char *raw) {
    unsigned int byte_val;
    if (strlen(raw) == 6 && raw[0] == '<' && raw[5] == '>' &&
        sscanf(raw, "<0x%02X>", &byte_val) == 1) {
        char *out = malloc(2);
        if (!out) return NULL;
        out[0] = (char)(unsigned char)byte_val;
        out[1] = '\0';
        return out;
    }
    size_t len = strlen(raw);
    char *out = malloc(len + 1);
    if (!out) return NULL;
    size_t r = 0, w = 0;
    while (r < len) {
        if ((unsigned char)raw[r] == 0xE2 && r + 2 < len &&
            (unsigned char)raw[r + 1] == 0x96 &&
            (unsigned char)raw[r + 2] == 0x81) {
            out[w++] = ' ';
            r += 3;
        } else {
            out[w++] = raw[r++];
        }
    }
    out[w] = '\0';
    return out;
}

/**
 * Encode an input string into token IDs using SentencePiece unigram / BPE.
 * @param tokenizer The tokenizer.
 * @param input The input string.
 * @param tokens Output token ID array.
 * @param max_tokens Maximum capacity of the token array.
 * @return Number of tokens on success, -1 on failure.
 */
static int kc_tokenizer_spm_encode(kc_tokenizer_t *tokenizer,
    const char *input, int *tokens, int max_tokens)
{
    if (!tokenizer || !input || !tokens || max_tokens <= 0) return -1;

    size_t input_len = strlen(input);
    if (input_len == 0) return 0;

    char *buf = NULL;
    if (tokenizer->add_space_prefix) {
        buf = malloc(input_len + 4);
        if (!buf) return -1;
        strcpy(buf, "\xe2\x96\x81");
        strcat(buf, input);
        input = buf;
        input_len = strlen(input);
    }

    spm_symbol_t *symbols = malloc(input_len * sizeof(spm_symbol_t));
    if (!symbols) { free(buf); return -1; }

    int n_symbols = 0;
    for (size_t i = 0; i < input_len; ) {
        unsigned char c = (unsigned char)input[i];
        size_t char_len = 1;
        if ((c & 0xE0) == 0xC0 && i + 1 < input_len) char_len = 2;
        else if ((c & 0xF0) == 0xE0 && i + 2 < input_len) char_len = 3;
        else if ((c & 0xF8) == 0xF0 && i + 3 < input_len) char_len = 4;

        char s[8] = {0};
        memcpy(s, input + i, char_len);

        const char *lookup = s;
        if (char_len == 1 && s[0] == ' ') lookup = "\xe2\x96\x81";

        int id = -1;
        if (kc_token_map_get(&tokenizer->vocab_map, lookup, &id) == 0) {
            symbols[n_symbols].id = id;
            symbols[n_symbols].prev = n_symbols - 1;
            symbols[n_symbols].next = n_symbols + 1;
            n_symbols++;
        } else if (tokenizer->byte_fallback) {
            for (size_t b = 0; b < char_len; b++) {
                char hex[10];
                snprintf(hex, sizeof(hex), "<0x%02X>",
                    (unsigned char)input[i + b]);
                id = -1;
                if (kc_token_map_get(&tokenizer->vocab_map, hex, &id) == 0) {
                    symbols[n_symbols].id = id;
                    symbols[n_symbols].prev = n_symbols - 1;
                    symbols[n_symbols].next = n_symbols + 1;
                    n_symbols++;
                }
            }
        }
        i += char_len;
    }
    if (n_symbols == 0) { free(symbols); free(buf); return 0; }
    symbols[n_symbols - 1].next = -1;

    while (1) {
        int best_p = -1;
        float best_score = -1e20f;
        int best_id = -1;

        for (int i = 0; i != -1; i = symbols[i].next) {
            int next = symbols[i].next;
            if (next == -1) break;

            const char *s1 = tokenizer->vocab_raw[symbols[i].id];
            const char *s2 = tokenizer->vocab_raw[symbols[next].id];
            char combined[512];
            snprintf(combined, sizeof(combined), "%s%s", s1, s2);

            int id = -1;
            if (kc_token_map_get(&tokenizer->vocab_map, combined, &id) == 0) {
                if (tokenizer->scores[id] > best_score) {
                    best_score = tokenizer->scores[id];
                    best_p = i;
                    best_id = id;
                }
            }
        }

        if (best_p == -1) break;

        int next = symbols[best_p].next;
        symbols[best_p].id = best_id;
        symbols[best_p].next = symbols[next].next;
        if (symbols[next].next != -1) {
            symbols[symbols[next].next].prev = best_p;
        }
    }

    int count = 0;
    for (int i = 0; i != -1; i = symbols[i].next) {
        if (count < max_tokens) tokens[count++] = symbols[i].id;
    }

    free(symbols);
    free(buf);
    return count;
}

/**
 * Free SentencePiece backend resources.
 * @param tokenizer The tokenizer.
 * @return None.
 */
static void kc_tokenizer_spm_free(kc_tokenizer_t *tokenizer) {
    if (!tokenizer) return;
    free(tokenizer->scores);
    tokenizer->scores = NULL;
}

/**
 * Load and initialize a SentencePiece tokenizer from GGUF metadata.
 * @param tokenizer The tokenizer to initialize.
 * @param error Error output buffer.
 * @param error_size Size of the error buffer.
 * @return 0 on success, -1 on failure.
 */
int kc_tokenizer_spm_load(kc_tokenizer_t *tokenizer,
    char *error, size_t error_size)
{
    const struct gguf_context *gguf = tokenizer->gguf;
    int s_id = gguf_find_key(gguf, "tokenizer.ggml.scores");
    if (s_id < 0)
        return kc_token_error(error, error_size, "missing tokenizer scores for spm");

    tokenizer->scores = calloc(tokenizer->n_vocab, sizeof(float));
    if (!tokenizer->scores)
        return kc_token_error(error, error_size, "failed to allocate spm scores");

    const float *scores = gguf_get_arr_data(gguf, s_id);
    for (int i = 0; i < tokenizer->n_vocab; i++)
        tokenizer->scores[i] = scores[i];

    tokenizer->byte_fallback = kc_gguf_bool(gguf,
        "tokenizer.ggml.byte_fallback", 0);
    int prefix_id = gguf_find_key(gguf, "tokenizer.ggml.add_space_prefix");
    tokenizer->add_space_prefix = (prefix_id >= 0)
        ? (int)gguf_get_val_bool(gguf, prefix_id) : 1;

    for (int i = 0; i < tokenizer->n_vocab; i++) {
        char *decoded = kc_tokenizer_spm_decode_token(tokenizer->vocab[i]);
        if (!decoded)
            return kc_token_error(error, error_size,
                "failed to decode spm vocab token");
        free(tokenizer->vocab[i]);
        tokenizer->vocab[i] = decoded;
    }
    tokenizer->encode_backend = kc_tokenizer_spm_encode;
    tokenizer->free_backend = kc_tokenizer_spm_free;
    return 0;
}
