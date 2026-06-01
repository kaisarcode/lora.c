/**
 * tok.h
 * Summary: Tokenizer interface: hash map, GGUF metadata helpers, and dispatch.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef KC_GGUF_TOK_H
#define KC_GGUF_TOK_H

#include <stddef.h>
#include <stdint.h>

struct gguf_context;

typedef struct {
    const char *key;
    int value;
    int used;
} kc_token_map_entry_t;

typedef struct {
    kc_token_map_entry_t *entries;
    size_t capacity;
} kc_token_map_t;

typedef struct kc_tokenizer kc_tokenizer_t;

struct kc_tokenizer {
    const struct gguf_context *gguf;
    int n_vocab;
    char **vocab;
    const char **vocab_raw;
    kc_token_map_t vocab_map;
    kc_token_map_t merge_map;
    float *scores;
    int byte_fallback;
    char model[32];
    char pre[32];
    int bos_token_id;
    int eos_token_id;
    int unk_token_id;
    int padding_token_id;
    int *eog_token_ids;
    int n_eog;
    int *special_token_ids;
    int n_special;
    int add_bos_token;
    int add_space_prefix;
    int (*encode_backend)(kc_tokenizer_t *tokenizer,
        const char *input, int *tokens, int max_tokens);
    void (*free_backend)(kc_tokenizer_t *tokenizer);
};

int kc_token_error(char *error, size_t error_size, const char *fmt, ...);
int kc_token_map_init(kc_token_map_t *map, size_t count);
void kc_token_map_free(kc_token_map_t *map);
int kc_token_map_set(kc_token_map_t *map, const char *key, int value);
int kc_token_map_get(const kc_token_map_t *map, const char *key, int *out);
const char *kc_gguf_str(const struct gguf_context *gguf, const char *key);
uint32_t kc_gguf_u32(const struct gguf_context *gguf, const char *key, uint32_t def);
int kc_gguf_bool(const struct gguf_context *gguf, const char *key, int def);
int kc_tokenizer_gpt2_load(kc_tokenizer_t *tokenizer, char *error, size_t error_size);
int kc_tokenizer_spm_load(kc_tokenizer_t *tokenizer, char *error, size_t error_size);

int kc_tokenizer_load(kc_tokenizer_t **out,
    const struct gguf_context *gguf, int n_vocab,
    char *error, size_t error_size);
void kc_tokenizer_free(kc_tokenizer_t *tokenizer);
int kc_tokenizer_encode(kc_tokenizer_t *tokenizer,
    const char *input, int *tokens, int max_tokens,
    char *error, size_t error_size);
const char *kc_tokenizer_decode(const kc_tokenizer_t *tokenizer, int id);
int kc_tokenizer_bos(const kc_tokenizer_t *tokenizer);
int kc_tokenizer_eos(const kc_tokenizer_t *tokenizer);
int kc_tokenizer_add_bos(const kc_tokenizer_t *tokenizer);
int kc_tokenizer_unk(const kc_tokenizer_t *tokenizer);
int kc_tokenizer_is_eog(const kc_tokenizer_t *tokenizer, int id);

#endif
