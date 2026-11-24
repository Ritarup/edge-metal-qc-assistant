#pragma once
// Decodes the raw YOLOv8/YOLO11 output heads of a Hailo .hef compiled WITHOUT
// on-chip NMS into the same packed tensor the ONNX model produces:
//   [1, 4 + num_classes, N]  rows = cx, cy, w, h (input pixels), class scores (0..1)
// so the existing decode_yolo_tensor_into() can be reused unchanged.
//
// Expected heads (dequantized FLOAT32, NHWC on the host), one pair per stride:
//   box head: H x W x (4 * reg_max)   e.g. 80x80x64, 40x40x64, 20x20x64
//   cls head: H x W x num_classes     e.g. 80x80x8,  40x40x8,  20x20x8
// Anchors are ordered largest grid first (stride 8, 16, 32), row-major,
// matching the Ultralytics ONNX export.
//
// No OpenCV / HailoRT dependency, so it can be unit-tested on any machine.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace qc {
namespace vision {

struct YoloRawHead {
    const float* data = nullptr;
    int height = 0;
    int width = 0;
    int channels = 0;
};

// Returns false if the heads do not form matching box/cls pairs.
// need_sigmoid: -1 = auto-detect (values outside [0,1] => logits), 0 = no, 1 = yes.
inline bool assemble_yolo_raw_heads(
    const std::vector<YoloRawHead>& heads,
    int num_classes,
    int input_size,
    float min_conf,
    std::vector<float>& packed,
    std::vector<int64_t>& shape,
    int need_sigmoid = -1,
    int reg_max = 16
) {
    std::vector<YoloRawHead> box, cls;
    for (const auto& h : heads) {
        if (!h.data || h.height <= 0 || h.width <= 0) continue;
        if (h.channels == 4 * reg_max) box.push_back(h);
        else if (h.channels == num_classes) cls.push_back(h);
    }
    if (box.empty() || box.size() != cls.size()) return false;

    auto by_grid_desc = [](const YoloRawHead& a, const YoloRawHead& b) {
        return a.height * a.width > b.height * b.width;
    };
    std::sort(box.begin(), box.end(), by_grid_desc);
    std::sort(cls.begin(), cls.end(), by_grid_desc);
    for (size_t k = 0; k < box.size(); ++k) {
        if (box[k].height != cls[k].height || box[k].width != cls[k].width) return false;
    }

    // Auto-detect whether class scores are raw logits (need sigmoid) or probabilities.
    bool apply_sigmoid = (need_sigmoid == 1);
    if (need_sigmoid < 0) {
        for (const auto& c : cls) {
            const size_t n = static_cast<size_t>(c.height) * c.width * c.channels;
            const size_t step = std::max<size_t>(1, n / 4096);
            for (size_t i = 0; i < n; i += step) {
                if (c.data[i] < 0.0f || c.data[i] > 1.0f) { apply_sigmoid = true; break; }
            }
            if (apply_sigmoid) break;
        }
    }
    // Compare in logit space when sigmoid is pending, to skip exp() on background anchors.
    const float clamped = std::min(std::max(min_conf, 1e-4f), 0.9999f);
    const float gate = apply_sigmoid ? std::log(clamped / (1.0f - clamped)) : clamped;

    int64_t total = 0;
    for (const auto& b : box) total += static_cast<int64_t>(b.height) * b.width;
    const int dims_c = 4 + num_classes;
    packed.assign(static_cast<size_t>(dims_c) * static_cast<size_t>(total), 0.0f);
    shape = {1, dims_c, total};

    std::vector<float> prob(reg_max);
    int64_t offset = 0;
    for (size_t k = 0; k < box.size(); ++k) {
        const int H = box[k].height, W = box[k].width;
        const float stride = static_cast<float>(input_size) / static_cast<float>(W);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int64_t cell = static_cast<int64_t>(y) * W + x;
                const int64_t a = offset + cell;
                const float* cp = cls[k].data + cell * num_classes;

                float best = -1e30f;
                for (int c = 0; c < num_classes; ++c) best = std::max(best, cp[c]);
                if (best < gate) {
                    // Background: scores stay 0 so the decoder skips it, but write the anchor
                    // centre in pixels so the decoder's "normalised coords?" probe (which
                    // samples every Nth anchor) sees pixel-scale values, not zeros.
                    packed[0 * total + a] = (static_cast<float>(x) + 0.5f) * stride;
                    packed[1 * total + a] = (static_cast<float>(y) + 0.5f) * stride;
                    continue;
                }

                for (int c = 0; c < num_classes; ++c) {
                    const float v = apply_sigmoid ? 1.0f / (1.0f + std::exp(-cp[c])) : cp[c];
                    packed[static_cast<size_t>(4 + c) * total + a] = v;
                }

                // DFL: per side, softmax over reg_max bins, expected value = distance (grid units).
                const float* bp = box[k].data + cell * (4 * reg_max);
                float dist[4];
                for (int s = 0; s < 4; ++s) {
                    const float* bins = bp + s * reg_max;
                    float mx = bins[0];
                    for (int j = 1; j < reg_max; ++j) mx = std::max(mx, bins[j]);
                    float sum = 0.0f;
                    for (int j = 0; j < reg_max; ++j) { prob[j] = std::exp(bins[j] - mx); sum += prob[j]; }
                    float e = 0.0f;
                    for (int j = 0; j < reg_max; ++j) e += prob[j] * static_cast<float>(j);
                    dist[s] = e / sum;
                }
                const float ax = static_cast<float>(x) + 0.5f;
                const float ay = static_cast<float>(y) + 0.5f;
                const float x1 = (ax - dist[0]) * stride, y1 = (ay - dist[1]) * stride;
                const float x2 = (ax + dist[2]) * stride, y2 = (ay + dist[3]) * stride;
                packed[0 * total + a] = 0.5f * (x1 + x2);
                packed[1 * total + a] = 0.5f * (y1 + y2);
                packed[2 * total + a] = x2 - x1;
                packed[3 * total + a] = y2 - y1;
            }
        }
        offset += static_cast<int64_t>(H) * W;
    }
    return true;
}

}  // namespace vision
}  // namespace qc
