# make        -> ./tree      (CPU + CUDA backends; needs nvcc)
# make cpu    -> ./tree_cpu  (CPU backends only; any C++20 compiler)
# make clean

CXX       ?= g++
NVCC      ?= /opt/cuda/bin/nvcc
CUDA_LIB  ?= /opt/cuda/lib64
CUDA_ARCH ?= native

CXXFLAGS  := -std=c++20 -O3 -pthread -Wall -Wextra -MMD -MP
NVCCFLAGS := -std=c++20 -O3 -arch=$(CUDA_ARCH) -MMD -MP -Xcompiler -Wall

CPU_SOURCES := main.cpp options.cpp dataset.cpp tree.cpp thread_pool.cpp \
               split_rules.cpp cpu_builder.cpp pruning.cpp trainer.cpp

BUILD := build
CPU_OBJECTS := $(CPU_SOURCES:%.cpp=$(BUILD)/%.o)

.PHONY: all cpu clean
all: tree
cpu: tree_cpu

tree: $(CPU_OBJECTS) $(BUILD)/gpu_builder.o
	$(CXX) $(CXXFLAGS) $^ -L$(CUDA_LIB) -lcudart -o $@

tree_cpu: $(CPU_OBJECTS) $(BUILD)/gpu_builder_stub.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/%.o: %.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/gpu_builder.o: gpu_builder.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD) tree tree_cpu

-include $(wildcard $(BUILD)/*.d)
