#!/bin/sh
# test.sh
# Summary: Functional tests for lora CLI.
# Author:  KaisarCode
# Website: https://kaisarcode.com
# License: https://www.gnu.org/licenses/gpl-3.0.html

set -e

KC_TEST_ARCH=$(uname -m)
case "$KC_TEST_ARCH" in
    x86_64|amd64) KC_TEST_ARCH=x86_64 ;;
    i386|i686) KC_TEST_ARCH=i686 ;;
    aarch64|arm64) KC_TEST_ARCH=aarch64 ;;
    armv7l|armv7*) KC_TEST_ARCH=armv7 ;;
    *) ;;
esac

KC_TEST_PLATFORM=$(uname -s)
case "$KC_TEST_PLATFORM" in
    Linux) KC_TEST_PLATFORM=linux ;;
    MINGW*|MSYS*|CYGWIN*) KC_TEST_PLATFORM=windows ;;
    *) ;;
esac

KC_TEST_BINARY="./bin/${KC_TEST_ARCH}/${KC_TEST_PLATFORM}/lora"
if [ "$KC_TEST_PLATFORM" = "windows" ]; then
    KC_TEST_BINARY="${KC_TEST_BINARY}.exe"
fi

# Prints one success line.
# @param $1 Success message.
# @return 0 on success.
kc_test_pass() { printf "\033[32m[PASS]\033[0m %s\n" "$1"; }

# Prints one failure line.
# @param $1 Failure message.
# @return 1 on failure.
kc_test_fail() { printf "\033[31m[FAIL]\033[0m %s\n" "$1"; exit 1; }

# Checks that the built binary exists.
# @return Exits 1 if binary not found.
kc_test_check_binary() {
    if [ ! -f "$KC_TEST_BINARY" ]; then
        echo "Error: binary not found at $KC_TEST_BINARY" >&2
        echo "Run 'make' first." >&2
        exit 1
    fi
}

kc_test_check_binary

if $KC_TEST_BINARY --help > /dev/null 2>&1; then
    kc_test_pass "--help"
else
    kc_test_fail "--help"
fi

if $KC_TEST_BINARY --version 2>&1 | grep -q "1.0.0"; then
    kc_test_pass "--version"
else
    kc_test_fail "--version"
fi

if $KC_TEST_BINARY unknown 2>/dev/null; then
    kc_test_fail "unknown operation should fail"
else
    kc_test_pass "unknown operation fails"
fi

if $KC_TEST_BINARY train -o out.safetensors -d data.txt 2>/dev/null; then
    kc_test_fail "missing model should fail"
else
    kc_test_pass "missing model fails"
fi

if $KC_TEST_BINARY train lib/model.gguf -d data.txt 2>/dev/null; then
    kc_test_fail "missing -o should fail"
else
    kc_test_pass "missing -o fails"
fi

if $KC_TEST_BINARY train lib/model.gguf -o out.safetensors 2>/dev/null; then
    kc_test_fail "missing -d should fail"
else
    kc_test_pass "missing -d fails"
fi

if $KC_TEST_BINARY train --unknown 2>/dev/null; then
    kc_test_fail "unknown flag should fail"
else
    kc_test_pass "unknown flag fails"
fi

if $KC_TEST_BINARY train nonexistent.gguf -o out.safetensors -d data.txt 2>&1 | grep -qi "error\|failed"; then
    kc_test_pass "model file error expected"
else
    kc_test_fail "model file error not reported"
fi
