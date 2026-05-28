/**
 * ugm.c
 * Summary: Unigram tokenizer with Viterbi dynamic programming decoding.
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

/**
 * Decode a unigram raw token string into a clean string.
 * Handles <0xNN> byte tokens and U+2581 (space) replacement.
 * @param raw The raw token string.
 * @return Allocated clean string, or NULL on failure.
 */
static char *kc_tokenizer_ugm_decode_token(const char *raw) {
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
 * Normalize input by replacing spaces with U+2581.
 * @param input The input string.
 * @return Allocated normalized string, or NULL on failure.
 */
static char *kc_ugm_normalize(const char *input) {
    size_t len = strlen(input);
    char *out = malloc(len * 3 + 1);
    if (!out) return NULL;
    size_t w = 0;
    for (size_t i = 0; i < len; i++) {
        if ((unsigned char)input[i] == ' ') {
            out[w++] = '\xe2';
            out[w++] = '\x96';
            out[w++] = '\x81';
        } else {
            out[w++] = input[i];
        }
    }
    out[w] = '\0';
    return out;
}

typedef struct {
    int id;
    float score;
    int prev;
} ugm_node_t;

/**
 * Encode an input string into token IDs using unigram Viterbi decoding.
 * @param tokenizer The tokenizer.
 * @param input The input string.
 * @param tokens Output token ID array.
 * @param max_tokens Maximum capacity of the token array.
 * @return Number of tokens on success, -1 on failure.
 */
static int kc_tokenizer_ugm_encode(kc_tokenizer_t *tokenizer,
    const char *input, int *tokens, int max_tokens)
{
    if (!tokenizer || !input || !tokens || max_tokens <= 0) return -1;
    if (strlen(input) == 0) return 0;

    char *norm = kc_ugm_normalize(input);
    if (!norm) return -1;

    size_t len = strlen(norm);
    ugm_node_t *dp = malloc((len + 1) * sizeof(ugm_node_t));
    if (!dp) { free(norm); return -1; }

    for (size_t i = 0; i <= len; i++) {
        dp[i].score = -1e20f;
        dp[i].id = -1;
        dp[i].prev = -1;
    }
    dp[0].score = 0;

    for (size_t i = 0; i < len; i++) {
        if (dp[i].score <= -1e19f) continue;
        for (size_t l = 1; i + l <= len && l < 256; l++) {
            char sub[256];
            memcpy(sub, norm + i, l);
            sub[l] = 0;
            int id = -1;
            if (kc_token_map_get(&tokenizer->vocab_map, sub, &id) == 0) {
                float score = dp[i].score + tokenizer->scores[id];
                if (score > dp[i + l].score) {
                    dp[i + l].score = score;
                    dp[i + l].id = id;
                    dp[i + l].prev = (int)i;
                }
            }
        }
        if (dp[i + 1].id == -1) {
            float score = dp[i].score - 10.0f;
            if (score > dp[i + 1].score) {
                dp[i + 1].score = score;
                dp[i + 1].id = tokenizer->unk_token_id;
                dp[i + 1].prev = (int)i;
            }
        }
    }

    int temp_tokens[2048];
    int n = 0;
    for (int i = (int)len; i > 0; ) {
        if (n >= 2048) break;
        temp_tokens[n++] = dp[i].id;
        i = dp[i].prev;
    }

    int count = 0;
    for (int i = n - 1; i >= 0; i--) {
        if (count < max_tokens) tokens[count++] = temp_tokens[i];
    }

    free(dp);
    free(norm);
    return count;
}

/**
 * Load and initialize a unigram tokenizer from GGUF metadata.
 * @param tokenizer The tokenizer to initialize.
 * @param error Error output buffer.
 * @param error_size Size of the error buffer.
 * @return 0 on success, -1 on failure.
 */
int kc_tokenizer_ugm_load(kc_tokenizer_t *tokenizer,
    char *error, size_t error_size)
{
    const struct gguf_context *gguf = tokenizer->gguf;
    int s_id = gguf_find_key(gguf, "tokenizer.ggml.scores");
    if (s_id < 0)
        return kc_token_error(error, error_size,
            "missing tokenizer scores for unigram");

    tokenizer->scores = calloc(tokenizer->n_vocab, sizeof(float));
    if (!tokenizer->scores)
        return kc_token_error(error, error_size,
            "failed to allocate unigram scores");

    const float *scores = gguf_get_arr_data(gguf, s_id);
    for (int i = 0; i < tokenizer->n_vocab; i++)
        tokenizer->scores[i] = scores[i];

    for (int i = 0; i < tokenizer->n_vocab; i++) {
        char *decoded = kc_tokenizer_ugm_decode_token(tokenizer->vocab[i]);
        if (!decoded)
            return kc_token_error(error, error_size,
                "failed to decode ugm vocab token");
        free(tokenizer->vocab[i]);
        tokenizer->vocab[i] = decoded;
    }
    tokenizer->encode_backend = kc_tokenizer_ugm_encode;
    return 0;
}
