/**
 * gpt2.c
 * Summary: GPT-2 byte-level BPE tokenizer with pre-tokenizer regex.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif

#include "tok.h"
#include "gguf.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Map a byte to its GPT-2 codepoint.
 * @param byte The byte value.
 * @return The corresponding codepoint.
 */
static uint32_t kc_gpt2_byte_cp(unsigned char byte) {
    if (byte >= 33 && byte <= 126) return byte;
    if (byte >= 161 && byte <= 172) return byte;
    if (byte >= 174) return byte;
    int n = 0;
    for (int b = 0; b < (int)byte; b++) {
        if (!((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174)) n++;
    }
    return 256U + (uint32_t)n;
}

/**
 * Reverse-lookup a byte from a GPT-2 codepoint.
 * @param cp The codepoint.
 * @return The corresponding byte.
 */
static unsigned char kc_gpt2_cp_byte(uint32_t cp) {
    for (int b = 0; b < 256; b++) {
        if (kc_gpt2_byte_cp((unsigned char)b) == cp) return (unsigned char)b;
    }
    return (unsigned char)(cp & 0xFF);
}

/**
 * Decode a GPT-2 raw token string into a clean byte string.
 * @param raw The raw token string.
 * @return Allocated clean string, or NULL on failure.
 */
static char *kc_gpt2_decode_token(const char *raw) {
    size_t n = strlen(raw);
    char *clean = malloc(n + 1);
    if (!clean) return NULL;
    size_t out = 0;
    const char *p = raw;
    while (*p) {
        uint32_t cp = 0;
        const unsigned char *s = (const unsigned char *)p;
        int len = 0;
        if (*s < 0x80) { cp = *s; len = 1; }
        else if ((*s & 0xE0) == 0xC0) { cp = *s & 0x1F; len = 2; }
        else if ((*s & 0xF0) == 0xE0) { cp = *s & 0x0F; len = 3; }
        else if ((*s & 0xF8) == 0xF0) { cp = *s & 0x07; len = 4; }
        else { cp = *s; len = 1; }
        for (int i = 1; i < len; i++) {
            if (!s[i] || (s[i] & 0xC0) != 0x80) { len = i; break; }
            cp = (cp << 6) | (s[i] & 0x3F);
        }
        clean[out++] = (char)kc_gpt2_cp_byte(cp);
        p += len;
    }
    clean[out] = '\0';
    return clean;
}

/**
 * Encode a Unicode codepoint into UTF-8 bytes.
 * @param cp The codepoint.
 * @param out Output buffer (must have space for at least 4 bytes).
 * @return Number of bytes written.
 */
static int kc_cp_utf8(uint32_t cp, char *out) {
    if (cp <= 0x7F) { out[0] = (char)cp; return 1; }
    if (cp <= 0x7FF) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp <= 0xFFFF) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/**
 * Append a string to a dynamically-allocated buffer.
 * @param dst Pointer to the destination buffer (reallocated as needed).
 * @param len Current length of the destination.
 * @param src Source string to append.
 * @param n Number of bytes to append.
 * @return 0 on success, -1 on allocation failure.
 */
static int kc_str_append(char **dst, size_t *len, const char *src, size_t n) {
    char *next = realloc(*dst, *len + n + 1);
    if (!next) return -1;
    memcpy(next + *len, src, n);
    *len += n;
    next[*len] = '\0';
    *dst = next;
    return 0;
}

/**
 * Encode an input byte string into GPT-2 byte-encoded UTF-8.
 * @param input The input bytes.
 * @param ilen Number of input bytes.
 * @return Allocated encoded string, or NULL on failure.
 */
static char *kc_gpt2_encode_bytes(const char *input, size_t ilen) {
    char *out = NULL;
    size_t len = 0;
    for (size_t j = 0; j < ilen; j++) {
        char buf[4];
        int n = kc_cp_utf8(kc_gpt2_byte_cp((unsigned char)input[j]), buf);
        if (kc_str_append(&out, &len, buf, (size_t)n) != 0) {
            free(out);
            return NULL;
        }
    }
    if (!out) out = strdup("");
    return out;
}

/**
 * Split an encoded byte string into UTF-8 character pieces.
 * @param input The input string.
 * @param pieces Output array of allocated piece strings.
 * @param count Output number of pieces.
 * @return 0 on success, -1 on allocation failure.
 */
static int kc_gpt2_split_bytes(const char *input, char ***pieces, int *count) {
    int cap = 64;
    int n = 0;
    char **out = calloc((size_t)cap, sizeof(char *));
    if (!out) return -1;
    for (size_t i = 0; input[i];) {
        unsigned char c = (unsigned char)input[i];
        size_t step = 1;
        if ((c & 0xE0) == 0xC0) step = 2;
        else if ((c & 0xF0) == 0xE0) step = 3;
        else if ((c & 0xF8) == 0xF0) step = 4;
        if (n == cap) {
            cap *= 2;
            char **next = realloc(out, (size_t)cap * sizeof(char *));
            if (!next) goto fail;
            out = next;
        }
        out[n] = malloc(step + 1);
        if (!out[n]) goto fail;
        memcpy(out[n], input + i, step);
        out[n][step] = '\0';
        n++;
        i += step;
    }
    *pieces = out;
    *count = n;
    return 0;
fail:
    for (int j = 0; j < n; j++) free(out[j]);
    free(out);
    return -1;
}

/**
 * Free an array of BPE piece strings.
 * @param pieces The piece array.
 * @param count Number of pieces.
 * @return None.
 */
static void kc_bpe_free_pieces(char **pieces, int count) {
    for (int i = 0; i < count; i++) free(pieces[i]);
    free(pieces);
}

/**
 * Build a merge key from two BPE pieces.
 * @param a First piece.
 * @param b Second piece.
 * @return Allocated key string ("a b"), or NULL on failure.
 */
static char *kc_bpe_merge_key(const char *a, const char *b) {
    size_t la = strlen(a);
    size_t lb = strlen(b);
    char *key = malloc(la + lb + 2);
    if (!key) return NULL;
    memcpy(key, a, la);
    key[la] = ' ';
    memcpy(key + la + 1, b, lb + 1);
    return key;
}

/**
 * Merge two adjacent BPE pieces at a given position.
 * @param pieces The piece array.
 * @param count Pointer to the piece count (updated in-place).
 * @param pos Index of the first piece to merge.
 * @return 0 on success, -1 on allocation failure.
 */
static int kc_bpe_merge_pieces(char **pieces, int *count, int pos) {
    size_t la = strlen(pieces[pos]);
    size_t lb = strlen(pieces[pos + 1]);
    char *merged = malloc(la + lb + 1);
    if (!merged) return -1;
    memcpy(merged, pieces[pos], la);
    memcpy(merged + la, pieces[pos + 1], lb + 1);
    free(pieces[pos]);
    free(pieces[pos + 1]);
    pieces[pos] = merged;
    for (int i = pos + 1; i < *count - 1; i++) pieces[i] = pieces[i + 1];
    *count -= 1;
    return 0;
}

/**
 * Apply BPE merging to a piece array using the tokenizer's merge map.
 * @param tokenizer The tokenizer.
 * @param pieces The piece array (modified in-place).
 * @param count Pointer to the piece count (updated in-place).
 * @return 0 on success, -1 on failure.
 */
static int kc_bpe_apply(kc_tokenizer_t *tokenizer, char **pieces, int *count) {
    while (*count > 1) {
        int best_pos = -1;
        int best_rank = 0;
        for (int i = 0; i < *count - 1; i++) {
            int rank = 0;
            char *key = kc_bpe_merge_key(pieces[i], pieces[i + 1]);
            if (!key) return -1;
            int found = kc_token_map_get(&tokenizer->merge_map, key, &rank);
            free(key);
            if (found == 0 && (best_pos < 0 || rank < best_rank)) {
                best_pos = i;
                best_rank = rank;
            }
        }
        if (best_pos < 0) break;
        if (kc_bpe_merge_pieces(pieces, count, best_pos) != 0) return -1;
    }
    return 0;
}

/**
 * Return the byte length of a UTF-8 character given its leading byte.
 * @param c The leading byte.
 * @return Character length in bytes (1-4).
 */
static int kc_utf8_char_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

/**
 * Check if a UTF-8 character starts a letter.
 * @param p Pointer to the start of the character.
 * @return Non-zero if a letter.
 */
static int kc_is_letter(const unsigned char *p) {
    if (*p < 0x80) return isalpha(*p);
    return *p >= 0xC2;
}

/**
 * Check if a UTF-8 character is a digit.
 * @param p Pointer to the start of the character.
 * @return Non-zero if a digit.
 */
static int kc_is_digit(const unsigned char *p) {
    return isdigit(*p);
}

/**
 * Check if a UTF-8 character is whitespace.
 * @param p Pointer to the start of the character.
 * @return Non-zero if whitespace.
 */
static int kc_is_space(const unsigned char *p) {
    return *p < 0x80 && isspace(*p);
}

/**
 * Find the length of the next pre-tokenization chunk at the given position.
 * @param p The input string position.
 * @return Length of the chunk in bytes.
 */
static size_t kc_gpt2_next_chunk(const char *p) {
    const unsigned char *u = (const unsigned char *)p;

    if (u[0] == '\'') {
        if (u[1] == 's' || u[1] == 't' || u[1] == 'm' || u[1] == 'd') return 2;
        if ((u[1] == 'r' && u[2] == 'e') || (u[1] == 'v' && u[2] == 'e') ||
            (u[1] == 'l' && u[2] == 'l')) return 3;
    }

    size_t skip = (u[0] == ' ' && u[1]) ? 1 : 0;
    const unsigned char *q = u + skip;

    if (kc_is_letter(q)) {
        size_t i = skip + (size_t)kc_utf8_char_len(*q);
        while (u[i] && kc_is_letter(u + i)) i += (size_t)kc_utf8_char_len(u[i]);
        return i;
    }
    if (kc_is_digit(q)) {
        size_t i = skip + 1;
        while (u[i] && kc_is_digit(u + i)) i++;
        return i;
    }
    if (!kc_is_space(q) && !kc_is_letter(q) && !kc_is_digit(q) && *q) {
        size_t i = skip + 1;
        while (u[i] && !kc_is_space(u + i) && !kc_is_letter(u + i) &&
            !kc_is_digit(u + i)) i++;
        return i;
    }

    if (kc_is_space(u)) {
        size_t i = 1;
        while (u[i] && kc_is_space(u + i)) i++;
        return i;
    }

    return 1;
}

/**
 * Encode a single pre-tokenization chunk into token IDs.
 * @param tokenizer The tokenizer.
 * @param chunk The chunk string.
 * @param clen Length of the chunk.
 * @param tokens Output token ID array.
 * @param n_tokens Current number of tokens in the array.
 * @param max_tokens Maximum capacity of the token array.
 * @return Updated token count on success, -1 on failure.
 */
static int kc_gpt2_encode_chunk(kc_tokenizer_t *tokenizer,
    const char *chunk, size_t clen,
    int *tokens, int n_tokens, int max_tokens)
{
    char *encoded = kc_gpt2_encode_bytes(chunk, clen);
    if (!encoded) return -1;

    char **pieces = NULL;
    int count = 0;
    if (kc_gpt2_split_bytes(encoded, &pieces, &count) != 0) {
        free(encoded);
        return -1;
    }
    free(encoded);

    if (kc_bpe_apply(tokenizer, pieces, &count) != 0) {
        kc_bpe_free_pieces(pieces, count);
        return -1;
    }

    for (int i = 0; i < count; i++) {
        int id = 0;
        if (kc_token_map_get(&tokenizer->vocab_map, pieces[i], &id) != 0) {
            fprintf(stderr, "gguf: gpt2 tokenization failed at piece '%s'\n", pieces[i]);
            kc_bpe_free_pieces(pieces, count);
            return -1;
        }
        if (n_tokens >= max_tokens) {
            kc_bpe_free_pieces(pieces, count);
            return -1;
        }
        tokens[n_tokens++] = id;
    }
    kc_bpe_free_pieces(pieces, count);
    return n_tokens;
}

/**
 * Match the longest special token at the current input position.
 * @param tokenizer The tokenizer.
 * @param input The input string.
 * @param out_id Output token ID of the matched special token.
 * @param out_len Output length of the matched special token in bytes.
 * @return 1 if a special token matched, 0 otherwise.
 */
static int kc_gpt2_match_special(kc_tokenizer_t *tokenizer,
    const char *input, int *out_id, size_t *out_len)
{
    int best_id = -1;
    size_t best_len = 0;
    for (int i = 0; i < tokenizer->n_special; i++) {
        int id = tokenizer->special_token_ids[i];
        const char *raw = tokenizer->vocab_raw[id];
        size_t len = raw ? strlen(raw) : 0;
        if (len > best_len && strncmp(input, raw, len) == 0) {
            best_id = id;
            best_len = len;
        }
    }
    if (best_id < 0) return 0;
    *out_id = best_id;
    *out_len = best_len;
    return 1;
}

/**
 * Encode an input string into token IDs using GPT-2 BPE.
 * @param tokenizer The tokenizer.
 * @param input The input string.
 * @param tokens Output token ID array.
 * @param max_tokens Maximum capacity of the token array.
 * @return Number of tokens on success, -1 on failure.
 */
static int kc_tokenizer_gpt2_encode(kc_tokenizer_t *tokenizer,
    const char *input, int *tokens, int max_tokens)
{
    int n_tokens = 0;
    const char *p = input;
    while (*p && n_tokens < max_tokens) {
        int special_id = 0;
        size_t special_len = 0;
        if (kc_gpt2_match_special(tokenizer, p, &special_id, &special_len)) {
            tokens[n_tokens++] = special_id;
            p += special_len;
            continue;
        }
        size_t clen = kc_gpt2_next_chunk(p);
        n_tokens = kc_gpt2_encode_chunk(tokenizer, p, clen, tokens, n_tokens, max_tokens);
        if (n_tokens < 0) return -1;
        p += clen;
    }
    return n_tokens;
}

/**
 * Load BPE merge pairs from a GGUF tensor.
 * @param tokenizer The tokenizer.
 * @param m_id The GGUF merge array key ID.
 * @param error Error output buffer.
 * @param error_size Size of the error buffer.
 * @return 0 on success, -1 on failure.
 */
static int kc_tokenizer_gpt2_load_merges(kc_tokenizer_t *tokenizer,
    int m_id, char *error, size_t error_size)
{
    size_t n_merges = gguf_get_arr_n(tokenizer->gguf, m_id);
    if (kc_token_map_init(&tokenizer->merge_map, n_merges) != 0)
        return kc_token_error(error, error_size, "failed to allocate tokenizer merge map");
    for (size_t i = 0; i < n_merges; i++) {
        const char *merge = gguf_get_arr_str(tokenizer->gguf, m_id, i);
        if (kc_token_map_set(&tokenizer->merge_map, merge, (int)i) != 0)
            return kc_token_error(error, error_size,
                "failed to build tokenizer merge map");
    }
    return 0;
}

/**
 * Check if a pre-tokenizer type identifier is supported by GPT-2.
 * @param pre The pre-tokenizer type string.
 * @return Non-zero if supported.
 */
static int kc_tokenizer_gpt2_supports_pre(const char *pre) {
    return !pre ||
        strcmp(pre, "") == 0 ||
        strcmp(pre, "default") == 0 ||
        strcmp(pre, "gpt2") == 0 ||
        strcmp(pre, "gpt-2") == 0 ||
        strcmp(pre, "qwen2") == 0 ||
        strcmp(pre, "smollm") == 0;
}

/**
 * Free GPT-2 backend resources.
 * @param tokenizer The tokenizer.
 * @return None.
 */
static void kc_tokenizer_gpt2_free(kc_tokenizer_t *tokenizer) {
    kc_token_map_free(&tokenizer->merge_map);
}

/**
 * Load and initialize a GPT-2 tokenizer from GGUF metadata.
 * @param tokenizer The tokenizer to initialize.
 * @param error Error output buffer.
 * @param error_size Size of the error buffer.
 * @return 0 on success, -1 on failure.
 */
int kc_tokenizer_gpt2_load(kc_tokenizer_t *tokenizer,
    char *error, size_t error_size)
{
    int m_id = gguf_find_key(tokenizer->gguf, "tokenizer.ggml.merges");
    if (!kc_tokenizer_gpt2_supports_pre(tokenizer->pre)) {
        return kc_token_error(error, error_size,
            "unsupported GPT-2 tokenizer pre-tokenizer: %s", tokenizer->pre);
    }
    if (m_id < 0) {
        return kc_token_error(error, error_size,
            "missing GPT-2 tokenizer merges");
    }
    if (kc_tokenizer_gpt2_load_merges(tokenizer, m_id, error, error_size) != 0)
        return -1;
    for (int i = 0; i < tokenizer->n_vocab; i++) {
        char *decoded = kc_gpt2_decode_token(tokenizer->vocab[i]);
        if (!decoded)
            return kc_token_error(error, error_size,
                "failed to decode gpt2 vocab token");
        free(tokenizer->vocab[i]);
        tokenizer->vocab[i] = decoded;
    }
    tokenizer->encode_backend = kc_tokenizer_gpt2_encode;
    tokenizer->free_backend = kc_tokenizer_gpt2_free;
    return 0;
}
