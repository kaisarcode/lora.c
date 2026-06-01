/**
 * tok.c
 * Summary: Tokenizer core: hash map, GGUF metadata helpers, and dispatch.
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
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Set error message in buffer with printf-style format.
 * @param error Error output buffer.
 * @param error_size Error buffer capacity.
 * @param fmt Printf-style format string.
 * @return Always negative.
 */
int kc_token_error(char *error, size_t error_size, const char *fmt, ...) {
    if (error && error_size > 0) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(error, error_size, fmt, args);
        va_end(args);
    }
    return -1;
}

/**
 * Compute 64-bit FNV-1a hash for a string.
 * @param s Null-terminated string to hash.
 * @return 64-bit hash value.
 */
static uint64_t kc_token_hash(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ULL;
    }
    return h;
}

/**
 * Initialize token map with capacity for given number of entries.
 * @param map Uninitialized token map.
 * @param count Expected number of entries to store.
 * @return 0 on success, -1 on allocation failure.
 */
int kc_token_map_init(kc_token_map_t *map, size_t count) {
    size_t capacity = 16;
    while (capacity < count * 2) capacity *= 2;
    map->entries = calloc(capacity, sizeof(kc_token_map_entry_t));
    if (!map->entries) return -1;
    map->capacity = capacity;
    return 0;
}

/**
 * Free all resources owned by a token map.
 * @param map Token map to free.
 * @return None.
 */
void kc_token_map_free(kc_token_map_t *map) {
    free(map->entries);
    map->entries = NULL;
    map->capacity = 0;
}

/**
 * Insert or update a key-value pair in the token map.
 * @param map Initialized token map.
 * @param key String key (must outlive the map).
 * @param value Integer value to associate with key.
 * @return Zero on success.
 */
int kc_token_map_set(kc_token_map_t *map, const char *key, int value) {
    size_t mask = map->capacity - 1;
    size_t pos = (size_t)kc_token_hash(key) & mask;
    while (map->entries[pos].used) {
        if (strcmp(map->entries[pos].key, key) == 0) {
            map->entries[pos].value = value;
            return 0;
        }
        pos = (pos + 1) & mask;
    }
    map->entries[pos].key = key;
    map->entries[pos].value = value;
    map->entries[pos].used = 1;
    return 0;
}

/**
 * Look up a key in the token map and write its value.
 * @param map Initialized token map.
 * @param key String key to look up.
 * @param out Receives the value if found.
 * @return Zero on success, non-zero if key is not found or map is empty.
 */
int kc_token_map_get(const kc_token_map_t *map, const char *key, int *out) {
    if (!map->entries || map->capacity == 0) return -1;
    size_t mask = map->capacity - 1;
    size_t pos = (size_t)kc_token_hash(key) & mask;
    while (map->entries[pos].used) {
        if (strcmp(map->entries[pos].key, key) == 0) {
            *out = map->entries[pos].value;
            return 0;
        }
        pos = (pos + 1) & mask;
    }
    return -1;
}

/**
 * Look up a string-valued GGUF metadata key.
 * @param gguf Open GGUF context.
 * @param key Metadata key name.
 * @return String value, or NULL if key not found.
 */
const char *kc_gguf_str(const struct gguf_context *gguf, const char *key) {
    int id = gguf_find_key(gguf, key);
    return (id < 0) ? NULL : gguf_get_val_str(gguf, id);
}

/**
 * Look up a uint32-valued GGUF metadata key with fallback default.
 * @param gguf Open GGUF context.
 * @param key Metadata key name.
 * @param def Default value when key is absent.
 * @return Stored value or default.
 */
uint32_t kc_gguf_u32(const struct gguf_context *gguf, const char *key, uint32_t def) {
    int id = gguf_find_key(gguf, key);
    return (id < 0) ? def : gguf_get_val_u32(gguf, id);
}

/**
 * Look up a boolean-valued GGUF metadata key with fallback default.
 * @param gguf Open GGUF context.
 * @param key Metadata key name.
 * @param def Default value when key is absent.
 * @return Stored value or default.
 */
int kc_gguf_bool(const struct gguf_context *gguf, const char *key, int def) {
    int id = gguf_find_key(gguf, key);
    return (id < 0) ? def : gguf_get_val_bool(gguf, id);
}

/**
 * Load vocabulary strings from GGUF into the tokenizer.
 * @param tokenizer Partially initialized tokenizer.
 * @param v_id GGUF array index for vocab tokens.
 * @param error Error output buffer.
 * @param error_size Error buffer capacity.
 * @return 0 on success, -1 and set error on failure.
 */
static int kc_tokenizer_load_vocab(kc_tokenizer_t *tokenizer,
    int v_id, char *error, size_t error_size)
{
    size_t n_tokens = gguf_get_arr_n(tokenizer->gguf, v_id);
    if ((int)n_tokens < tokenizer->n_vocab)
        return kc_token_error(error, error_size, "vocab metadata is truncated");
    tokenizer->vocab = malloc((size_t)tokenizer->n_vocab * sizeof(char *));
    tokenizer->vocab_raw = malloc((size_t)tokenizer->n_vocab * sizeof(char *));
    if (!tokenizer->vocab || !tokenizer->vocab_raw)
        return kc_token_error(error, error_size, "failed to allocate tokenizer vocab");
    if (kc_token_map_init(&tokenizer->vocab_map, (size_t)tokenizer->n_vocab) != 0)
        return kc_token_error(error, error_size, "failed to allocate tokenizer vocab map");
    for (int i = 0; i < tokenizer->n_vocab; i++) {
        const char *raw = gguf_get_arr_str(tokenizer->gguf, v_id, (size_t)i);
        char *clean = strdup(raw);
        if (!clean)
            return kc_token_error(error, error_size, "failed to allocate tokenizer token");
        tokenizer->vocab_raw[i] = raw;
        tokenizer->vocab[i] = clean;
        if (kc_token_map_set(&tokenizer->vocab_map, raw, i) != 0)
            return kc_token_error(error, error_size, "failed to build tokenizer vocab map");
    }
    return 0;
}

/**
 * Load special token type data from GGUF into the tokenizer.
 * @param tokenizer Tokenizer to populate.
 * @return 0 on success, -1 on allocation failure.
 */
static int kc_tokenizer_load_special_tokens(kc_tokenizer_t *tokenizer) {
    int t_id = gguf_find_key(tokenizer->gguf, "tokenizer.ggml.token_type");
    if (t_id < 0) return 0;
    size_t n = gguf_get_arr_n(tokenizer->gguf, t_id);
    if ((int)n < tokenizer->n_vocab) return 0;
    const int32_t *types = gguf_get_arr_data(tokenizer->gguf, t_id);
    int count = 0;
    for (int i = 0; i < tokenizer->n_vocab; i++) {
        if (types[i] == 3 || types[i] == 4) count++;
    }
    if (count == 0) return 0;
    tokenizer->special_token_ids = malloc((size_t)count * sizeof(int));
    if (!tokenizer->special_token_ids) return -1;
    tokenizer->n_special = count;
    count = 0;
    for (int i = 0; i < tokenizer->n_vocab; i++) {
        if (types[i] == 3 || types[i] == 4)
            tokenizer->special_token_ids[count++] = i;
    }
    return 0;
}

/**
 * Parse a whitespace-separated list of numeric token IDs from input.
 * @param tokenizer Tokenizer for vocab bounds checking.
 * @param input Whitespace-separated integer token IDs or "ids:..." string.
 * @param tokens Output buffer for parsed token IDs.
 * @param max_tokens Capacity of tokens buffer.
 * @return Number of tokens parsed, or -1 on error.
 */
static int kc_tokenizer_encode_ids(kc_tokenizer_t *tokenizer,
    const char *input, int *tokens, int max_tokens)
{
    int n = 0;
    while (*input) {
        while (isspace((unsigned char)*input)) input++;
        if (!*input) break;
        char *end = NULL;
        errno = 0;
        long id = strtol(input, &end, 10);
        if (input == end || errno != 0 || id < 0 || id >= tokenizer->n_vocab)
            return -1;
        if (n >= max_tokens) return -1;
        tokens[n++] = (int)id;
        input = end;
    }
    return n;
}

/**
 * Create and initialize a tokenizer from a GGUF context.
 * @param out Receives pointer to allocated tokenizer.
 * @param gguf Open GGUF context with tokenizer metadata.
 * @param n_vocab Number of vocabulary entries.
 * @param error Error output buffer.
 * @param error_size Error buffer capacity.
 * @return 0 on success, -1 and set error on failure.
 */
int kc_tokenizer_load(kc_tokenizer_t **out,
    const struct gguf_context *gguf, int n_vocab,
    char *error, size_t error_size)
{
    const char *model = kc_gguf_str(gguf, "tokenizer.ggml.model");
    const char *pre = kc_gguf_str(gguf, "tokenizer.ggml.pre");
    int v_id = gguf_find_key(gguf, "tokenizer.ggml.tokens");
    kc_tokenizer_t *tokenizer = NULL;
    if (!out || !gguf)
        return kc_token_error(error, error_size, "invalid tokenizer arguments");
    *out = NULL;
    if (!model)
        return kc_token_error(error, error_size, "missing tokenizer model");
    if (v_id < 0)
        return kc_token_error(error, error_size, "missing tokenizer vocab");
    tokenizer = calloc(1, sizeof(*tokenizer));
    if (!tokenizer)
        return kc_token_error(error, error_size, "failed to allocate tokenizer");
    tokenizer->gguf = gguf;
    tokenizer->n_vocab = n_vocab;
    snprintf(tokenizer->model, sizeof(tokenizer->model), "%s", model);
    snprintf(tokenizer->pre, sizeof(tokenizer->pre), "%s", pre ? pre : "");
    tokenizer->add_bos_token = kc_gguf_bool(gguf, "tokenizer.ggml.add_bos_token", 0);
    tokenizer->add_space_prefix = kc_gguf_bool(gguf, "tokenizer.ggml.add_space_prefix", 0);
    tokenizer->bos_token_id = (int)kc_gguf_u32(gguf, "tokenizer.ggml.bos_token_id", 1);
    tokenizer->eos_token_id = (int)kc_gguf_u32(gguf, "tokenizer.ggml.eos_token_id", 0);
    tokenizer->unk_token_id = (int)kc_gguf_u32(gguf, "tokenizer.ggml.unknown_token_id", -1);
    tokenizer->padding_token_id = (int)kc_gguf_u32(gguf, "tokenizer.ggml.padding_token_id", -1);
    int eog_id = gguf_find_key(gguf, "tokenizer.ggml.eog_token_ids");
    if (eog_id >= 0) {
        size_t n = gguf_get_arr_n(gguf, eog_id);
        if (n > 0) {
            const uint32_t *data = gguf_get_arr_data(gguf, eog_id);
            tokenizer->eog_token_ids = malloc(n * sizeof(int));
            if (tokenizer->eog_token_ids) {
                tokenizer->n_eog = (int)n;
                for (size_t i = 0; i < n; i++)
                    tokenizer->eog_token_ids[i] = (int)data[i];
            }
        }
    }
    if (kc_tokenizer_load_vocab(tokenizer, v_id, error, error_size) != 0) {
        kc_tokenizer_free(tokenizer);
        return -1;
    }
    if (kc_tokenizer_load_special_tokens(tokenizer) != 0) {
        kc_tokenizer_free(tokenizer);
        return kc_token_error(error, error_size,
            "failed to allocate tokenizer special tokens");
    }
    if (strcmp(tokenizer->model, "gpt2") == 0) {
        if (kc_tokenizer_gpt2_load(tokenizer, error, error_size) != 0) {
            kc_tokenizer_free(tokenizer);
            return -1;
        }
    } else if (strcmp(tokenizer->model, "llama") == 0 ||
        strcmp(tokenizer->model, "gemma4") == 0) {
        if (kc_tokenizer_spm_load(tokenizer, error, error_size) != 0) {
            kc_tokenizer_free(tokenizer);
            return -1;
        }
    } else if (strcmp(tokenizer->model, "unigram") == 0) {
        if (kc_tokenizer_ugm_load(tokenizer, error, error_size) != 0) {
            kc_tokenizer_free(tokenizer);
            return -1;
        }
    } else {
        kc_tokenizer_free(tokenizer);
        return kc_token_error(error, error_size,
            "unsupported tokenizer model: %s", model);
    }
    *out = tokenizer;
    return 0;
}

/**
 * Free all resources owned by a tokenizer, including backend data.
 * @param tokenizer Tokenizer to free (NULL-safe).
 * @return None.
 */
void kc_tokenizer_free(kc_tokenizer_t *tokenizer) {
    if (!tokenizer) return;
    if (tokenizer->free_backend) tokenizer->free_backend(tokenizer);
    for (int i = 0; i < tokenizer->n_vocab; i++) free(tokenizer->vocab[i]);
    free(tokenizer->vocab);
    free(tokenizer->vocab_raw);
    free(tokenizer->eog_token_ids);
    free(tokenizer->special_token_ids);
    free(tokenizer->scores);
    kc_token_map_free(&tokenizer->vocab_map);
    free(tokenizer);
}

/**
 * Encode a text string into token IDs using the tokenizer backend.
 * @param tokenizer Initialized tokenizer.
 * @param input Text to tokenize (or "ids:..." for debug direct IDs).
 * @param tokens Output buffer for token IDs.
 * @param max_tokens Capacity of tokens buffer.
 * @param error Error output buffer.
 * @param error_size Error buffer capacity.
 * @return Number of tokens on success, -1 and set error on failure.
 */
int kc_tokenizer_encode(kc_tokenizer_t *tokenizer,
    const char *input, int *tokens, int max_tokens,
    char *error, size_t error_size)
{
    const char *debug_input = input;
    if (!tokenizer || !input || !tokens || max_tokens <= 0)
        return kc_token_error(error, error_size, "invalid tokenizer encode arguments");
    while (isspace((unsigned char)*debug_input)) debug_input++;
    if (strncmp(debug_input, "ids:", 4) == 0) {
        int n = kc_tokenizer_encode_ids(tokenizer, debug_input + 4, tokens, max_tokens);
        if (n < 0)
            return kc_token_error(error, error_size, "invalid debug token id input");
        return n;
    }
    if (!tokenizer->encode_backend)
        return kc_token_error(error, error_size, "tokenizer backend is not initialized");
    int n = tokenizer->encode_backend(tokenizer, input, tokens, max_tokens);
    if (n < 0)
        return kc_token_error(error, error_size, "%s tokenization failed", tokenizer->model);
    return n;
}

/**
 * Decode a single token ID back to its text representation.
 * @param tokenizer Initialized tokenizer.
 * @param id Token ID to decode.
 * @return Decoded text, or empty string for invalid/special IDs.
 */
const char *kc_tokenizer_decode(const kc_tokenizer_t *tokenizer, int id) {
    if (!tokenizer || id < 0 || id >= tokenizer->n_vocab) return "";
    if (id == tokenizer->bos_token_id || id == tokenizer->unk_token_id ||
        id == tokenizer->padding_token_id) return "";
    return tokenizer->vocab[id] ? tokenizer->vocab[id] : "";
}

/**
 * Get the beginning-of-sequence token ID.
 * @param tokenizer Tokenizer (NULL-safe).
 * @return BOS token ID, or 0 if tokenizer is NULL.
 */
int kc_tokenizer_bos(const kc_tokenizer_t *tokenizer) {
    return tokenizer ? tokenizer->bos_token_id : 0;
}

/**
 * Get the end-of-sequence token ID.
 * @param tokenizer Tokenizer (NULL-safe).
 * @return EOS token ID, or 0 if tokenizer is NULL.
 */
int kc_tokenizer_eos(const kc_tokenizer_t *tokenizer) {
    return tokenizer ? tokenizer->eos_token_id : 0;
}

/**
 * Check whether the tokenizer expects BOS token prepended.
 * @param tokenizer Tokenizer (NULL-safe).
 * @return 1 if BOS should be added, 0 otherwise.
 */
int kc_tokenizer_add_bos(const kc_tokenizer_t *tokenizer) {
    return tokenizer ? tokenizer->add_bos_token : 0;
}

/**
 * Get the unknown token ID.
 * @param tokenizer Tokenizer (NULL-safe).
 * @return Unknown token ID, or -1 if tokenizer is NULL.
 */
int kc_tokenizer_unk(const kc_tokenizer_t *tokenizer) {
    return tokenizer ? tokenizer->unk_token_id : -1;
}

/**
 * Check whether a token ID is an end-of-generation token.
 * @param tokenizer Tokenizer (NULL-safe).
 * @param id Token ID to check.
 * @return 1 if the token is EOG, 0 otherwise.
 */
int kc_tokenizer_is_eog(const kc_tokenizer_t *tokenizer, int id) {
    if (!tokenizer) return 0;
    if (id == tokenizer->eos_token_id) return 1;
    for (int i = 0; i < tokenizer->n_eog; i++) {
        if (tokenizer->eog_token_ids[i] == id) return 1;
    }
    return 0;
}
