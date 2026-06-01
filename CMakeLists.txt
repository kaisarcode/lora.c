## CMakeLists.txt
## Summary: Build system for lora — LoRA adapter training.
##
## Author:  KaisarCode
## Website: https://kaisarcode.com
## License: https://www.gnu.org/licenses/gpl-3.0.html

cmake_minimum_required(VERSION 3.14)
project(lora C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

option(LORA_CUDA_ENABLED "Build with CUDA backend support" OFF)
option(LORA_TOKENIZER_GPT2_ENABLED "Build GPT-2 byte-level BPE tokenizer support" ON)
option(LORA_TOKENIZER_SPM_ENABLED "Build SentencePiece tokenizer support" ON)

if(LORA_CUDA_ENABLED)
    message(STATUS "lora CUDA backend: enabled")
else()
    message(STATUS "lora CUDA backend: disabled")
endif()

if(LORA_CUDA_ENABLED)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>")
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        message(STATUS "lora CUDA cross-compilation: target=${CMAKE_SYSTEM_PROCESSOR}")
        set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "sbsa-linux")
        set(CMAKE_CUDA_FLAGS "-Xcompiler -U__GNUC__ -Xcompiler -D__GNUC__=8")
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" AND CUDAToolkit_ROOT)
        set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "${CUDAToolkit_ROOT}"
            CACHE PATH "CUDA toolkit installation root" FORCE)
        set(CUDAToolkit_INCLUDE_DIR "${CUDAToolkit_ROOT}/include"
            CACHE PATH "" FORCE)
        set(CUDAToolkit_CUDA_LIBRARY "${CUDAToolkit_ROOT}/targets/sbsa-linux/lib/libcuda.so"
            CACHE FILEPATH "" FORCE)
    endif()
    enable_language(CUDA)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" AND CUDAToolkit_ROOT)
        set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "${CUDAToolkit_ROOT}"
            CACHE PATH "CUDA toolkit installation root" FORCE)
    endif()
    find_package(CUDAToolkit REQUIRED)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" AND CUDAToolkit_ROOT)
        set(CUDAToolkit_INCLUDE_DIRS "${CUDAToolkit_ROOT}/include")
        if(TARGET CUDA::cuda_driver)
            set_target_properties(CUDA::cuda_driver PROPERTIES
                IMPORTED_LOCATION "${CUDAToolkit_ROOT}/targets/sbsa-linux/lib/libcuda.so")
        endif()
    endif()
    add_definitions(-DGGML_USE_CUDA)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(CMAKE_LIBRARY_ARCHITECTURE "aarch64-linux-gnu")
    endif()
endif()

## Compiler flags

if(NOT MSVC)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:C,CXX>:-O3>"
        "$<$<COMPILE_LANGUAGE:C,CXX>:-fno-finite-math-only>"
        "$<$<COMPILE_LANGUAGE:C,CXX>:-Wall>"
        "$<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>"
        "$<$<COMPILE_LANGUAGE:C,CXX>:-Werror>"
    )
    add_definitions(-D_GNU_SOURCE -DGGML_USE_CPU -DGGML_VERSION="1" -DGGML_COMMIT="custom")
else()
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:C,CXX>:/O2>"
        "$<$<COMPILE_LANGUAGE:C,CXX>:/arch:AVX2>"
        "$<$<COMPILE_LANGUAGE:C,CXX>:/W4>"
        "$<$<COMPILE_LANGUAGE:C,CXX>:/WX>"
    )
    add_definitions(-DGGML_USE_CPU -DGGML_VERSION="1" -DGGML_COMMIT="custom")
endif()

## GGML sources

include_directories(
    src
    lib/kaisarcode/gguf
    lib/kaisarcode/gguf/graph
    lib/kaisarcode/gguf/tok
    lib/ggml
    lib/ggml/include
    lib/ggml/src
    lib/ggml/src/ggml-cpu
)

file(GLOB GGML_BASE_SOURCES
    "lib/ggml/src/*.c"
    "lib/ggml/src/*.cpp"
    "lib/ggml/src/ggml-cpu/*.c"
    "lib/ggml/src/ggml-cpu/*.cpp"
)

if(LORA_CUDA_ENABLED)
    file(GLOB_RECURSE GGML_CUDA_SOURCES "lib/ggml/src/ggml-cuda/*.cu")
    list(APPEND GGML_BASE_SOURCES ${GGML_CUDA_SOURCES})
endif()

set(GGML_ARCH_SOURCES "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    file(GLOB GGML_ARCH_SOURCES
        "lib/ggml/src/ggml-cpu/arch/x86/*.c"
        "lib/ggml/src/ggml-cpu/arch/x86/*.cpp"
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i686|i386")
    file(GLOB GGML_ARCH_SOURCES
        "lib/ggml/src/ggml-cpu/arch/x86/*.c"
        "lib/ggml/src/ggml-cpu/arch/x86/*.cpp"
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv7|arm")
    file(GLOB GGML_ARCH_SOURCES
        "lib/ggml/src/ggml-cpu/arch/arm/*.c"
        "lib/ggml/src/ggml-cpu/arch/arm/*.cpp"
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    file(GLOB GGML_ARCH_SOURCES
        "lib/ggml/src/ggml-cpu/arch/riscv/*.c"
        "lib/ggml/src/ggml-cpu/arch/riscv/*.cpp"
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "powerpc64le|ppc64le")
    file(GLOB GGML_ARCH_SOURCES
        "lib/ggml/src/ggml-cpu/arch/powerpc/*.c"
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "mips")
    set(GGML_ARCH_SOURCES "")
    add_definitions(-DGGML_CPU_GENERIC)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "s390x")
    file(GLOB GGML_ARCH_SOURCES
        "lib/ggml/src/ggml-cpu/arch/s390/*.c"
        "lib/ggml/src/ggml-cpu/arch/s390/*.cpp"
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "loongarch64")
    file(GLOB GGML_ARCH_SOURCES
        "lib/ggml/src/ggml-cpu/arch/loongarch/*.c"
    )
endif()

if(MINGW)
    file(WRITE "${CMAKE_BINARY_DIR}/ggml-backend-reg-mingw.cpp"
"#include \"ggml-backend-impl.h\"
#include \"ggml-backend.h\"
#include \"ggml-impl.h\"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

#ifdef GGML_USE_CPU
#include \"ggml-cpu.h\"
#endif

using dl_handle = void;
struct dl_handle_deleter {
    void operator()(void *) {}
};
dl_handle * dl_load_library(const char *) { return nullptr; }
void * dl_get_sym(dl_handle *, const char *) { return nullptr; }
const char * dl_error() { return \"dynamic loading not supported\"; }

struct ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle * handle;
};

struct ggml_backend_registry {
    std::vector<ggml_backend_reg_entry> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#ifdef GGML_USE_CUDA
        register_backend(ggml_backend_cuda_reg());
#endif
#ifdef GGML_USE_CPU
        register_backend(ggml_backend_cpu_reg());
#endif
    }

    ~ggml_backend_registry() {}

    void register_backend(ggml_backend_reg_t reg, dl_handle * handle = nullptr) {
        if (!reg) return;
        backends.push_back({ reg, handle });
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    void register_device(ggml_backend_dev_t device) {
        devices.push_back(device);
    }
};

static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    return reg;
}

void ggml_backend_register(ggml_backend_reg_t reg) { get_reg().register_backend(reg); }
void ggml_backend_device_register(ggml_backend_dev_t device) { get_reg().register_device(device); }

static bool striequals(const char * a, const char * b) {
    for (; *a && *b; a++, b++) {
        if (std::tolower(*a) != std::tolower(*b)) return false;
    }
    return *a == *b;
}

size_t ggml_backend_reg_count() { return get_reg().backends.size(); }
ggml_backend_reg_t ggml_backend_reg_get(size_t index) { return get_reg().backends[index].reg; }
ggml_backend_reg_t ggml_backend_reg_by_name(const char * name) {
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto reg = ggml_backend_reg_get(i);
        if (striequals(ggml_backend_reg_name(reg), name)) return reg;
    }
    return nullptr;
}

size_t ggml_backend_dev_count() { return get_reg().devices.size(); }
ggml_backend_dev_t ggml_backend_dev_get(size_t index) { return get_reg().devices[index]; }
ggml_backend_dev_t ggml_backend_dev_by_name(const char * name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto dev = ggml_backend_dev_get(i);
        if (striequals(ggml_backend_dev_name(dev), name)) return dev;
    }
    return nullptr;
}
ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == type) return ggml_backend_dev_get(i);
    }
    return nullptr;
}

ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params) {
    auto dev = ggml_backend_dev_by_name(name);
    return dev ? ggml_backend_dev_init(dev, params) : nullptr;
}
ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params) {
    auto dev = ggml_backend_dev_by_type(type);
    return dev ? ggml_backend_dev_init(dev, params) : nullptr;
}
ggml_backend_t ggml_backend_init_best() {
    auto dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    return dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
}

ggml_backend_reg_t ggml_backend_load(const char *) { return nullptr; }
void ggml_backend_unload(ggml_backend_reg_t) {}
void ggml_backend_load_all() {}
void ggml_backend_load_all_from_path(const char *) {}
"
    )
    foreach(_src ${GGML_BASE_SOURCES})
        if(_src MATCHES "ggml-backend-dl\\.cpp$" OR _src MATCHES "ggml-backend-reg\\.cpp$")
            list(REMOVE_ITEM GGML_BASE_SOURCES "${_src}")
        endif()
    endforeach()
    list(APPEND GGML_BASE_SOURCES "${CMAKE_BINARY_DIR}/ggml-backend-reg-mingw.cpp")
    message(STATUS "MinGW backend-reg replacement active, removed vendored ggml-backend-dl/reg")
endif()

set(GGML_SOURCES ${GGML_BASE_SOURCES} ${GGML_ARCH_SOURCES})
set(GGML_C_SOURCES "")
set(GGML_CXX_SOURCES "")

foreach(GGML_SOURCE ${GGML_SOURCES})
    get_filename_component(GGML_SOURCE_EXT "${GGML_SOURCE}" EXT)
    if(GGML_SOURCE_EXT STREQUAL ".c")
        list(APPEND GGML_C_SOURCES "${GGML_SOURCE}")
    elseif(GGML_SOURCE_EXT STREQUAL ".cpp")
        list(APPEND GGML_CXX_SOURCES "${GGML_SOURCE}")
    endif()
endforeach()

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
        ${GGML_C_SOURCES}
        PROPERTIES COMPILE_OPTIONS "-Wno-pointer-to-int-cast;-Wno-psabi;-Wno-unused-function;-Wno-unused-variable;-Wno-unused-but-set-variable"
    )
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
        ${GGML_CXX_SOURCES}
        PROPERTIES COMPILE_OPTIONS "-Wno-psabi;-Wno-unused-function;-Wno-unused-variable;-Wno-unused-but-set-variable"
    )
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set_property(
        SOURCE "lib/ggml/src/ggml-quants.c"
        APPEND PROPERTY COMPILE_OPTIONS "-O2"
    )
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set_property(
        SOURCE "lib/kaisarcode/gguf/tok/spm.c"
        APPEND PROPERTY COMPILE_OPTIONS "-O2"
    )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_property(
        SOURCE ${GGML_CXX_SOURCES}
        APPEND PROPERTY COMPILE_OPTIONS "$<IF:$<BOOL:${MINGW}>,-O0,-O1>"
    )
endif()

set(LORA_SYSTEM_LIBS "")
if(UNIX AND NOT ANDROID)
    list(APPEND LORA_SYSTEM_LIBS pthread dl m)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|arm|mips")
        list(APPEND LORA_SYSTEM_LIBS atomic)
    endif()
elseif(ANDROID)
    list(APPEND LORA_SYSTEM_LIBS dl m)
endif()

if(LORA_CUDA_ENABLED)
    if(TARGET CUDA::cudart_static AND TARGET CUDA::cublas_static)
        list(APPEND LORA_SYSTEM_LIBS CUDA::cuda_driver CUDA::cudart_static CUDA::cublas_static)
        if(TARGET CUDA::cublasLt_static)
            list(APPEND LORA_SYSTEM_LIBS CUDA::cublasLt_static)
        endif()
    else()
        list(APPEND LORA_SYSTEM_LIBS CUDA::cuda_driver CUDA::cudart CUDA::cublas)
        if(TARGET CUDA::cublasLt)
            list(APPEND LORA_SYSTEM_LIBS CUDA::cublasLt)
        endif()
    endif()
endif()

## Library targets

set(LORA_TOKENIZER_SOURCES lib/kaisarcode/gguf/tok/tok.c)

if(LORA_TOKENIZER_GPT2_ENABLED)
    add_definitions(-DLORA_TOKENIZER_GPT2_ENABLED)
    list(APPEND LORA_TOKENIZER_SOURCES lib/kaisarcode/gguf/tok/gpt2.c)
endif()

if(LORA_TOKENIZER_SPM_ENABLED)
    add_definitions(-DLORA_TOKENIZER_SPM_ENABLED)
    list(APPEND LORA_TOKENIZER_SOURCES lib/kaisarcode/gguf/tok/spm.c)
endif()

set(LIB_SOURCES
    src/liblora.c
    lib/kaisarcode/gguf/graph/graph.c
    lib/kaisarcode/gguf/graph/qwen2.c
    lib/kaisarcode/gguf/graph/gpt2.c
    lib/kaisarcode/gguf/libgguf.c
    ${LORA_TOKENIZER_SOURCES}
    ${GGML_SOURCES}
)

add_library(lora_static STATIC ${LIB_SOURCES})
set_target_properties(lora_static PROPERTIES OUTPUT_NAME "lora" POSITION_INDEPENDENT_CODE ON)
target_include_directories(lora_static PRIVATE src lib/kaisarcode/gguf lib/kaisarcode/gguf/graph lib/kaisarcode/gguf/tok lib/ggml lib/ggml/include lib/ggml/src lib/ggml/src/ggml-cpu)

if(UNIX AND NOT ANDROID)
    target_link_libraries(lora_static ${LORA_SYSTEM_LIBS})
elseif(ANDROID)
    target_link_libraries(lora_static ${LORA_SYSTEM_LIBS})
endif()

add_library(lora_shared SHARED ${LIB_SOURCES})
set_target_properties(lora_shared PROPERTIES OUTPUT_NAME "lora")
target_include_directories(lora_shared PRIVATE src lib/kaisarcode/gguf lib/ggml lib/ggml/include lib/ggml/src lib/ggml/src/ggml-cpu)

if(UNIX AND NOT ANDROID)
    target_link_libraries(lora_shared ${LORA_SYSTEM_LIBS})
elseif(ANDROID)
    target_link_libraries(lora_shared ${LORA_SYSTEM_LIBS})
endif()

## Executable

add_executable(lora_bin src/lora.c)
target_include_directories(lora_bin PRIVATE src)
target_link_libraries(lora_bin lora_shared)
set_target_properties(lora_bin PROPERTIES OUTPUT_NAME "lora")

if(UNIX AND NOT APPLE)
    set_target_properties(lora_bin PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
    set_target_properties(lora_shared PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
endif()

if(UNIX AND NOT ANDROID)
    target_link_libraries(lora_bin ${LORA_SYSTEM_LIBS})
elseif(ANDROID)
    target_link_libraries(lora_bin ${LORA_SYSTEM_LIBS})
endif()
