#!/usr/bin/env bash
# ==============================================================================
# Bharat AI-SoC PS5: Native Engine Launcher
# Target: Raspberry Pi 5 ARM64 Linux (aarch64)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "bin/bharat_qc_native.bin" ]; then
    echo "Binary bin/bharat_qc_native.bin not found. Building now..."
    make -j"$(nproc)"
fi

mkdir -p models/patchcore_refs models/patchcore_banks uploads snapshots

ARCH="$(uname -m)"
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    ORT_LIB="${SCRIPT_DIR}/third_party/onnxruntime-linux-aarch64-1.20.1/lib"
else
    ORT_LIB="/usr/lib"
fi

EXECUTORCH_LIB="${HOME}/executorch/cmake-out/lib"
EXTRA_LIBS=""
if [ -d "$EXECUTORCH_LIB" ]; then
    EXTRA_LIBS=":${EXECUTORCH_LIB}"
fi

export LD_LIBRARY_PATH="${SCRIPT_DIR}/bin:${ORT_LIB}${EXTRA_LIBS}:/usr/lib:/usr/local/lib:${LD_LIBRARY_PATH}"

if [ -e "/dev/hailo0" ] && [ -f "models/unified_yolo11_640.hef" ]; then
    BACKEND_MSG="Hailo-8L NPU (/dev/hailo0) + ARM64 ONNX Runtime"
else
    BACKEND_MSG="ARM64 ONNX Runtime (CPU Fallback - /dev/hailo0 absent)"
fi

echo "=================================================================="
echo " Launching Bharat AI-SoC PS5 Native Engine (.bin)"
echo " Architecture : ${ARCH}"
echo " Vision       : ${BACKEND_MSG}"
echo " Web SCADA UI : http://localhost:5000"
echo "=================================================================="

# Ensure no duplicate instance shares port 5000 via SO_REUSEPORT
pkill -x qc-main-hud 2>/dev/null || true
pkill -x bharat_qc_nativ 2>/dev/null || true
# Let the old engine finish its advisory and release the singleton lock before
# launching another instance. Killing Llama first makes the old engine respawn it.
for attempt in {1..150}; do
    if ! pgrep -x qc-main-hud >/dev/null && ! pgrep -x bharat_qc_nativ >/dev/null; then
        break
    fi
    sleep 0.1
done
if pgrep -x qc-main-hud >/dev/null || pgrep -x bharat_qc_nativ >/dev/null; then
    echo "Previous engine is still shutting down; retry the launcher shortly." >&2
    exit 1
fi
pkill -x llama_daemon 2>/dev/null || true

exec ./bin/bharat_qc_native.bin "$@"
