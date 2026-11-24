#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "=============================================================="
echo "  Bharat AI-SoC PS5 · Raspberry Pi 5 Turnkey Setup & Launch"
echo "  Workspace: $ROOT_DIR"
echo "=============================================================="

ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
  echo "[warn] Host architecture is '$ARCH' (expected aarch64 on Raspberry Pi 5)."
fi

# 1. System packages (OpenCV, build tools, camera utils)
if command -v apt-get >/dev/null 2>&1; then
  echo "[1/6] Installing build & OpenCV dependencies via apt..."
  sudo apt-get update -y
  sudo apt-get install -y \
    build-essential g++ make pkg-config cmake git curl file \
    libopencv-dev v4l-utils
else
  echo "[1/6] apt-get not found; ensure g++, make, pkg-config, and libopencv-dev are installed."
fi

# 2. Repair ONNX Runtime ARM64 shared library symlinks if copied across FAT/NTFS/ZIP
ORT_ARM_LIB="$ROOT_DIR/third_party/onnxruntime-linux-aarch64-1.20.1/lib"
if [[ -d "$ORT_ARM_LIB" ]]; then
  if [[ -f "$ORT_ARM_LIB/libonnxruntime.so.1.20.1" ]]; then
    ln -sf "libonnxruntime.so.1.20.1" "$ORT_ARM_LIB/libonnxruntime.so.1"
    ln -sf "libonnxruntime.so.1" "$ORT_ARM_LIB/libonnxruntime.so"
  fi
fi

export LD_LIBRARY_PATH="$ROOT_DIR/bin:${ORT_ARM_LIB}:${LD_LIBRARY_PATH:-}"
mkdir -p "$ROOT_DIR/bin" "$ROOT_DIR/obj" "$ROOT_DIR/uploads" "$ROOT_DIR/snapshots"

# 3. Detect Hailo-8L NPU vs ONNX Runtime ARM64 CPU fallback
BUILD_HAILO=0
if [[ -e /dev/hailo0 ]] && [[ -f /usr/include/hailo/hailort.hpp ]] && \
   (ldconfig -p 2>/dev/null | grep -q "libhailort" || [[ -f /usr/lib/libhailort.so ]]); then
  BUILD_HAILO=1
  echo "[2/6] Hailo-8L NPU (/dev/hailo0 + libhailort) detected -> building with HAILO=1."
else
  echo "[2/6] Hailo-8L driver/dev headers not detected -> building ARM64 CPU fallback (ONNX Runtime)."
  if [[ -e /dev/hailo0 ]]; then
    echo "      (Hint: /dev/hailo0 exists; install Hailo headers with: sudo apt-get install -y hailo-all dkms pciutils)"
  fi
fi

# 4. Verify required models & PatchCore banks
echo "[3/6] Verifying local models, PatchCore banks, and conveyor demo assets..."
MISSING=0
for req in \
  "config/edge_qc_config.json" \
  "models/stage0_selector_224.onnx" \
  "models/unified_yolo11_640.onnx" \
  "models/screw_patchcore_feat.onnx" \
  "models/tokenizer.model" \
  "models/Llama-3.2-1B-UnifiedQC-FT-INT4.pte" \
  "sample_videos/conveyor_demo_belt.mp4" \
  "models/patchcore_banks/steel_sheet.bin" \
  "models/patchcore_banks/weld.bin" \
  "models/patchcore_banks/machined_part.bin" \
  "models/patchcore_banks/screw.bin" \
  "models/patchcore_banks/casting.bin" \
  "models/patchcore_banks/aluminium.bin"; do
  if [[ ! -f "$ROOT_DIR/$req" ]]; then
    echo "  [MISSING] $req"
    MISSING=1
  fi
done
if [[ "$BUILD_HAILO" -eq 1 && ! -f "$ROOT_DIR/models/unified_yolo11_640.hef" ]]; then
  echo "  [WARN] models/unified_yolo11_640.hef missing; vision engine will fall back to unified_yolo11_640.onnx."
fi
if [[ "$MISSING" -eq 1 ]]; then
  echo "[WARN] One or more model/asset files are missing. Ensure large ignored files (.pte/.onnx/.hef) were copied to $ROOT_DIR/models/."
fi

# 5. Ensure ARM64 Llama ExecuTorch resident daemon is usable
echo "[4/6] Checking Llama 3.2 1B ExecuTorch daemon (bin/llama_daemon)..."
NEED_LLAMA_BUILD=0
if [[ ! -x "$ROOT_DIR/bin/llama_daemon" ]]; then
  NEED_LLAMA_BUILD=1
elif ! file "$ROOT_DIR/bin/llama_daemon" | grep -qiE "aarch64|arm aarch64"; then
  NEED_LLAMA_BUILD=1
elif ldd "$ROOT_DIR/bin/llama_daemon" 2>/dev/null | grep -q "not found"; then
  NEED_LLAMA_BUILD=1
fi

if [[ "$NEED_LLAMA_BUILD" -eq 1 ]]; then
  if [[ -f "$ROOT_DIR/scripts/build_llama_arm64.sh" ]]; then
    echo "  Rebuilding ExecuTorch llama_daemon on host ARM64..."
    bash "$ROOT_DIR/scripts/build_llama_arm64.sh"
  else
    echo "  [ERR] bin/llama_daemon unusable and scripts/build_llama_arm64.sh not found."
    exit 1
  fi
else
  echo "  bin/llama_daemon verified for ARM64."
fi

# 6. Build native C++ SCADA & Vision + LLM binary
echo "[5/6] Compiling bin/bharat_qc_native.bin ..."
make clean
if [[ "$BUILD_HAILO" -eq 1 ]]; then
  HAILO=1 make -j"$(nproc)"
else
  make -j"$(nproc)"
fi
chmod +x "$ROOT_DIR/bin/bharat_qc_native.bin"
chmod +x "$ROOT_DIR/bin/llama_daemon" || true

# Stop any prior instance cleanly (set_thread_name sets comm=qc-main-hud)
pkill -x qc-main-hud 2>/dev/null || true
pkill -x bharat_qc_nativ 2>/dev/null || true
pkill -x llama_daemon 2>/dev/null || true
sleep 1

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
echo "[6/6] Launching Bharat AI-SoC QC Node on port 5000..."
echo "      Local UI : http://localhost:5000"
if [[ -n "$HOST_IP" ]]; then
  echo "      LAN UI   : http://${HOST_IP}:5000"
fi
echo "      Press Ctrl+C to stop."

exec "$ROOT_DIR/bin/bharat_qc_native.bin"
