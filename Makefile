# ==============================================================================
# BHARAT AI-SOC CHALLENGE 2026-27 | PROBLEM STATEMENT 5
# NATIVE AHEAD-OF-TIME (AOT) C++ ENGINE MAKEFILE
# Target Platform: ARM64 Linux (Raspberry Pi 5 Cortex-A76)
# ==============================================================================

CXX      := g++
ARCH     := $(shell uname -m)

# Resolve ONNX Runtime directory by host architecture
ifeq ($(ARCH), aarch64)
    ORT_DIR := third_party/onnxruntime-linux-aarch64-1.20.1
    ARCH_FLAGS := -march=armv8.2-a+fp16 -mtune=cortex-a76 -D__ARM_NEON
else ifeq ($(ARCH), arm64)
    ORT_DIR := third_party/onnxruntime-linux-aarch64-1.20.1
    ARCH_FLAGS := -march=armv8.2-a+fp16 -mtune=cortex-a76 -D__ARM_NEON
else
    $(info [Notice] Host architecture is $(ARCH). Falling back to system/third_party ONNX Runtime headers.)
    ORT_DIR := third_party/onnxruntime-linux-aarch64-1.20.1
    ARCH_FLAGS :=
endif

ifneq ($(wildcard $(ORT_DIR)/include/*),)
    ORT_INC := $(ORT_DIR)/include
    ORT_LIB_DIR := $(ORT_DIR)/lib
else
    ORT_INC := /usr/include
    ORT_LIB_DIR := /usr/lib
endif

# OpenCV flags via pkg-config if present, otherwise standard /usr/include/opencv4
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null)
ifeq ($(strip $(OPENCV_CFLAGS)),)
    OPENCV_CFLAGS := -I/usr/include/opencv4
endif
OPENCV_LIBS := -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn -lopencv_videoio

# Base compiler flags with C++17, OpenMP, and includes
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -pthread -fopenmp -MMD -MP $(ARCH_FLAGS) \
            -Iinclude -Ithird_party -I$(ORT_INC) $(OPENCV_CFLAGS)

# Conditional HailoRT support: HAILO=auto (default), HAILO=1, or HAILO=0
HAILO ?= auto
ifeq ($(HAILO), 1)
    CXXFLAGS += -I/usr/include/hailort -DHAVE_HAILORT -DUSE_HAILO
    LDFLAGS  += -lhailort
else ifeq ($(HAILO), auto)
    ifneq ($(wildcard /usr/include/hailort/*),)
        CXXFLAGS += -I/usr/include/hailort -DHAVE_HAILORT -DUSE_HAILO
        LDFLAGS  += -lhailort
    endif
endif

# Link OpenCV, ONNX Runtime, POSIX threads, and RT
LDFLAGS  += $(OPENCV_LIBS) \
            -L$(ORT_LIB_DIR) -lonnxruntime \
            -Wl,-rpath,'$$ORIGIN/../$(ORT_LIB_DIR)' \
            -Wl,-rpath,'$$ORIGIN/$(ORT_LIB_DIR)' \
            -lpthread -lrt

BIN_DIR  := bin
OBJ_DIR  := obj
SRC_DIR  := src

TARGET   := $(BIN_DIR)/bharat_qc_native.bin

SRCS     := $(wildcard $(SRC_DIR)/*.cpp)
OBJS     := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))
DEPS     := $(OBJS:.o=.d)

all: $(TARGET)

-include $(DEPS)

$(TARGET): $(OBJS) | $(BIN_DIR)
	@echo "=================================================================="
	@echo " Linking Ahead-Of-Time Native Binary: $(TARGET)"
	@echo " Architecture: $(ARCH) | ORT: $(ORT_DIR) | HAILO: $(HAILO)"
	@echo "=================================================================="
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)
	@chmod +x $(TARGET)
	@echo ""
	@echo ">>> SUCCESS: Native binary compiled to $(TARGET) <<<"
	@echo "To launch native engine:"
	@echo "    bash scripts/run_native.sh"
	@echo "=================================================================="

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@echo " [CXX] $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

clean:
	@echo "Cleaning native build artifacts..."
	@rm -rf $(OBJ_DIR) $(BIN_DIR)/bharat_qc_native.bin

run: $(TARGET)
	@bash scripts/run_native.sh

install: $(TARGET)
	@cp $(TARGET) /usr/local/bin/bharat_qc_native.bin 2>/dev/null || true
	@echo "Installed to /usr/local/bin/bharat_qc_native.bin"

.PHONY: all clean run install
