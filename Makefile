## Makefile
## Summary: Cross-compilation builder for lora.
##
## Author:  KaisarCode
## Website: https://kaisarcode.com
## License: https://www.gnu.org/licenses/gpl-3.0.html

ANDROID_HOME  ?= $(HOME)/.local/share/android-sdk
NDK_VERSION   ?= 27.2.12479018
NDK_DIR       := $(ANDROID_HOME)/ndk/$(NDK_VERSION)
NDK_TOOLCHAIN := $(NDK_DIR)/build/cmake/android.toolchain.cmake

CUDA      ?= 0
## Architectures that can support CUDA (used to decide LORA_CUDA_ENABLED flag)
CUDA_CAPABLE_ARCHS := x86_64 aarch64
## Auto-detected nvcc host compiler for CUDA cross-compilation.
## CUDA 12.4 requires GCC <= 13; tries g++-13, -12, -11, then g++.
## Override with CUDA_HOST_CXX=<path> if auto-detection is wrong.
_cuda_host_cxx   = $(or $(CUDA_HOST_CXX),$(firstword $(wildcard /usr/bin/aarch64-linux-gnu-g++-13 /usr/bin/aarch64-linux-gnu-g++-12 /usr/bin/aarch64-linux-gnu-g++-11 /usr/bin/aarch64-linux-gnu-g++)))
## Auto-detected CUDA cross-toolkit root (SBSA runfile extracted to /opt/cuda-sbsa).
## Override with CROSS_CUDA_ROOT=<path> if needed.
_cuda_cross_root = $(or $(CROSS_CUDA_ROOT),$(firstword $(wildcard /opt/cuda-sbsa/cuda-12.4)))
## nvcc wrapper for aarch64 CUDA cross-compilation. Adds -Xcompiler flags to
## work around glibc ARM vector header issues and -L for aarch64 CUDA libs.
_cuda_cross_nvcc = $(or $(CUDA_CROSS_NVCC),/tmp/nvcc-aarch64-wrapper)

BUILD_DIR := .build
BIN_DIR   := bin
CMAKE     ?= cmake

HOST_ARCH       := $(shell uname -m)
HOST_SYSTEM     := $(shell uname -s)
NATIVE_ARCH     := unsupported
NATIVE_PLATFORM := unsupported

ifneq ($(filter x86_64 amd64,$(HOST_ARCH)),)
NATIVE_ARCH := x86_64
endif

ifneq ($(filter i386 i686,$(HOST_ARCH)),)
NATIVE_ARCH := i686
endif

ifneq ($(filter aarch64 arm64,$(HOST_ARCH)),)
NATIVE_ARCH := aarch64
endif

ifneq ($(filter armv7l armv7%,$(HOST_ARCH)),)
NATIVE_ARCH := armv7
endif

ifneq ($(filter ppc64le powerpc64le,$(HOST_ARCH)),)
NATIVE_ARCH := powerpc64le
endif

ifneq ($(filter riscv64 s390x loongarch64 mips64el mipsel mips,$(HOST_ARCH)),)
NATIVE_ARCH := $(HOST_ARCH)
endif

ifeq ($(HOST_SYSTEM),Linux)
NATIVE_PLATFORM := linux
endif

ifneq ($(filter MINGW% MSYS% CYGWIN%,$(HOST_SYSTEM)),)
NATIVE_PLATFORM := windows
endif

NATIVE_TARGET := $(NATIVE_ARCH)/$(NATIVE_PLATFORM)

.DEFAULT_GOAL := native

check_tools:
	@command -v $(CMAKE) > /dev/null || { echo "Error: cmake is required to build lora." >&2; exit 1; }
	@command -v ninja > /dev/null || { echo "Error: ninja is required to build lora." >&2; exit 1; }

.PHONY: native all test clean \
	check_tools \
	x86_64/linux x86_64/windows \
	i686/linux i686/windows \
	aarch64/linux aarch64/android \
	armv7/linux armv7/android \
	armv7hf/linux \
	riscv64/linux \
	powerpc64le/linux \
	mips/linux mipsel/linux mips64el/linux \
	s390x/linux \
	loongarch64/linux

native:
	@if [ "$(NATIVE_ARCH)" = "unsupported" ] || [ "$(NATIVE_PLATFORM)" = "unsupported" ]; then \
		echo "Unsupported native target $(HOST_ARCH)/$(HOST_SYSTEM)" >&2; \
		exit 1; \
	fi
	@$(MAKE) $(NATIVE_TARGET)

all: \
	x86_64/linux x86_64/windows \
	i686/linux i686/windows \
	aarch64/linux aarch64/android \
	armv7/linux armv7/android \
	armv7hf/linux \
	riscv64/linux \
	powerpc64le/linux \
	mips/linux mipsel/linux mips64el/linux \
	s390x/linux \
	loongarch64/linux

## Linux
## $(1)=target  $(2)=processor  $(3)=CC  $(4)=CXX
##
## CUDA output subdir — when CUDA=1, artifacts go into cuda/ to keep
## CPU and GPU builds separate without overwriting each other.
cuda_subdir = $(if $(and $(filter 1,$(CUDA)),$(filter $(1),$(CUDA_CAPABLE_ARCHS))),/cuda)
## CUDA build dir suffix (separate cache to avoid CMakeCache conflicts)
cuda_build  = $(if $(and $(filter 1,$(CUDA)),$(filter $(1),$(CUDA_CAPABLE_ARCHS))),-cuda)
## Whether CUDA is enabled for the given arch
cuda_enabled = $(and $(filter 1,$(CUDA)),$(filter $(1),$(CUDA_CAPABLE_ARCHS)))
## Whether to warn that the arch does not support CUDA
cuda_warn    = $(and $(filter 1,$(CUDA)),$(if $(filter $(1),$(CUDA_CAPABLE_ARCHS)),,warn))

define linux_target
	@mkdir -p $(BIN_DIR)/$(1)/linux$(cuda_subdir)
	@rm -f $(BUILD_DIR)/$(subst /,-,$(1))-linux$(cuda_build)/CMakeCache.txt
	$(if $(call cuda_warn,$(1)),@echo "  $(1)/linux: CUDA not supported - building CPU-only")
	@$(if $(and $(call cuda_enabled,$(1)),$(filter-out $(NATIVE_ARCH),$(1)),$(_cuda_cross_root)), \
	    if [ ! -x $(_cuda_cross_nvcc) ]; then \
	        printf '#!/bin/bash\nexec /usr/bin/nvcc -Xcompiler "-U__GNUC__" -Xcompiler "-D__GNUC__=8" -L"%s/targets/sbsa-linux/lib" "$$@"\n' "$(_cuda_cross_root)" > $(_cuda_cross_nvcc); \
	        chmod +x $(_cuda_cross_nvcc); \
	    fi; \
	)
	@$(if $(and $(call cuda_enabled,$(1)),$(filter-out $(NATIVE_ARCH),$(1))),CUDAHOSTCXX=$(_cuda_host_cxx) )$(CMAKE) -S . -B $(BUILD_DIR)/$(subst /,-,$(1))-linux$(cuda_build) \
		-U CMAKE_CUDA_COMPILER \
		-U CUDAToolkit_* \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_SYSTEM_NAME=Linux \
		-DCMAKE_SYSTEM_PROCESSOR=$(2) \
		-DCMAKE_C_COMPILER=$(3) \
		-DCMAKE_CXX_COMPILER=$(4) \
		-DLORA_CUDA_ENABLED=$(if $(call cuda_enabled,$(1)),ON,OFF) \
		$(if $(and $(call cuda_enabled,$(1)),$(filter-out $(NATIVE_ARCH),$(1)),$(_cuda_cross_root)),-DCUDAToolkit_ROOT=$(_cuda_cross_root),) \
		$(if $(and $(call cuda_enabled,$(1)),$(filter-out $(NATIVE_ARCH),$(1)),$(_cuda_cross_nvcc)),-DCMAKE_CUDA_COMPILER=$(_cuda_cross_nvcc),) \
		-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$(CURDIR)/$(BUILD_DIR)/$(subst /,-,$(1))-linux$(cuda_build)/out \
		-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$(CURDIR)/$(BIN_DIR)/$(1)/linux$(cuda_subdir) \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(CURDIR)/$(BIN_DIR)/$(1)/linux$(cuda_subdir) \
		-G Ninja -Wno-dev
	@$(CMAKE) --build $(BUILD_DIR)/$(subst /,-,$(1))-linux$(cuda_build)
	@cp $(BUILD_DIR)/$(subst /,-,$(1))-linux$(cuda_build)/out/lora $(BIN_DIR)/$(1)/linux$(cuda_subdir)/lora
	@echo "OK $(1)/linux$(cuda_subdir)"
endef

x86_64/linux: check_tools
	$(call linux_target,x86_64,x86_64,gcc,g++)

i686/linux: check_tools
	$(call linux_target,i686,i686,i686-linux-gnu-gcc,i686-linux-gnu-g++)

aarch64/linux: check_tools
	$(call linux_target,aarch64,aarch64,aarch64-linux-gnu-gcc,aarch64-linux-gnu-g++)

armv7hf/linux: check_tools
	$(call linux_target,armv7hf,armv7,arm-linux-gnueabihf-gcc,arm-linux-gnueabihf-g++)

armv7/linux: check_tools
	$(call linux_target,armv7,armv7,arm-linux-gnueabi-gcc,arm-linux-gnueabi-g++)

riscv64/linux: check_tools
	$(call linux_target,riscv64,riscv64,riscv64-linux-gnu-gcc,riscv64-linux-gnu-g++)

powerpc64le/linux: check_tools
	$(call linux_target,powerpc64le,powerpc64le,powerpc64le-linux-gnu-gcc,powerpc64le-linux-gnu-g++)

mips/linux: check_tools
	$(call linux_target,mips,mips,mips-linux-gnu-gcc,mips-linux-gnu-g++)

mipsel/linux: check_tools
	$(call linux_target,mipsel,mips,mipsel-linux-gnu-gcc,mipsel-linux-gnu-g++)

mips64el/linux: check_tools
	$(call linux_target,mips64el,mips64,mips64el-linux-gnuabi64-gcc,mips64el-linux-gnuabi64-g++)

s390x/linux: check_tools
	$(call linux_target,s390x,s390x,s390x-linux-gnu-gcc,s390x-linux-gnu-g++)

loongarch64/linux: check_tools
	$(call linux_target,loongarch64,loongarch64,loongarch64-linux-gnu-gcc,loongarch64-linux-gnu-g++)

## Windows
## $(1)=target  $(2)=processor  $(3)=CC  $(4)=CXX

define windows_target
	@mkdir -p $(BIN_DIR)/$(1)/windows
	@cmake -S . -B $(BUILD_DIR)/$(1)-windows \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_SYSTEM_NAME=Windows \
		-DCMAKE_SYSTEM_PROCESSOR=$(2) \
		-DCMAKE_C_COMPILER=$(3) \
		-DCMAKE_CXX_COMPILER=$(4) \
		-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$(CURDIR)/$(BUILD_DIR)/$(1)-windows/out \
		-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$(CURDIR)/$(BIN_DIR)/$(1)/windows \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(CURDIR)/$(BIN_DIR)/$(1)/windows \
		-G Ninja -Wno-dev > /dev/null
	@cmake --build $(BUILD_DIR)/$(1)-windows
	@cp $(BUILD_DIR)/$(1)-windows/out/lora.exe $(BIN_DIR)/$(1)/windows/lora.exe
	@cp $(BUILD_DIR)/$(1)-windows/out/liblora.dll $(BIN_DIR)/$(1)/windows/liblora.dll
	@echo "OK $(1)/windows"
endef

x86_64/windows:
	$(call windows_target,x86_64,x86_64,x86_64-w64-mingw32-gcc,x86_64-w64-mingw32-g++)

i686/windows:
	$(call windows_target,i686,i686,i686-w64-mingw32-gcc,i686-w64-mingw32-g++)

## Android

define android_target
	@mkdir -p $(BIN_DIR)/$(1)/android
	@cmake -S . -B $(BUILD_DIR)/$(1)-android \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_TOOLCHAIN_FILE=$(NDK_TOOLCHAIN) \
		-DANDROID_ABI=$(2) \
		-DANDROID_PLATFORM=android-21 \
		-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$(CURDIR)/$(BUILD_DIR)/$(1)-android/out \
		-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$(CURDIR)/$(BIN_DIR)/$(1)/android \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(CURDIR)/$(BIN_DIR)/$(1)/android \
		-G Ninja -Wno-dev > /dev/null
	@cmake --build $(BUILD_DIR)/$(1)-android
	@cp $(BUILD_DIR)/$(1)-android/out/lora $(BIN_DIR)/$(1)/android/lora
	@echo "OK $(1)/android"
endef

aarch64/android:
	$(call android_target,aarch64,arm64-v8a)

armv7/android:
	$(call android_target,armv7,armeabi-v7a)

## Utility

test:
	@sh test.sh

clean:
	@rm -rf $(BUILD_DIR)
	@echo "OK clean"
