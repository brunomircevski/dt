# make        -> ./tree      (CPU + CUDA backends; needs nvcc)
# make cpu    -> ./tree_cpu  (CPU backends only; any C++20 compiler)
# make test   -> tests/check.sh on ./tree
# make clean

CXX       ?= g++
NVCC      ?= /opt/cuda/bin/nvcc
CUDA_LIB  ?= /opt/cuda/lib64
# native = the GPU of this machine. For other GPUs list them, e.g.
#   make CUDA_ARCH="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90"
CUDA_ARCH ?= -arch=native

CXXFLAGS  := -std=c++20 -O3 -pthread -Wall -Wextra -Isrc -MMD -MP
NVCCFLAGS := -std=c++20 -O3 $(CUDA_ARCH) -Isrc -MMD -MP -Xcompiler -Wall

BUILD   := build
SOURCES := $(filter-out src/build/gpu_grower_stub.cpp,$(wildcard src/*/*.cpp))
OBJECTS := $(SOURCES:src/%.cpp=$(BUILD)/%.o)

.PHONY: all cpu test clean
all: tree
cpu: tree_cpu

tree: $(OBJECTS) $(BUILD)/build/gpu_grower.o
	$(CXX) $(CXXFLAGS) $^ -L$(CUDA_LIB) -lcudart -o $@

tree_cpu: $(OBJECTS) $(BUILD)/build/gpu_grower_stub.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

test: tree
	tests/check.sh

clean:
	rm -rf $(BUILD) tree tree_cpu

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)
