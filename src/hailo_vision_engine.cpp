#include "hailo_vision_engine.hpp"
#include "cpu_affinity.hpp"
#include "metallic_gate_neon.hpp"
#include "patchcore_engine.hpp"
#include "vector_math.hpp"
#include "hailo_yolo_heads.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <onnxruntime_cxx_api.h>

#ifdef HAVE_HAILORT
#include "hailo/hailort.hpp"
#endif

namespace fs = std::filesystem;

namespace qc {
namespace vision {

namespace {

using SteadyClock = std::chrono::steady_clock;

float elapsed_ms(SteadyClock::time_point t0, SteadyClock::time_point t1) {
    return std::chrono::duration<float, std::milli>(t1 - t0).count();
}

std::string resolve_existing_file(const std::string& primary, const std::string& fallback_rel) {
    std::vector<std::string> candidates;
    if (!primary.empty()) {
        candidates.push_back(primary);
        candidates.push_back("../" + primary);
    }
    if (!fallback_rel.empty() && fallback_rel != primary) {
        candidates.push_back(fallback_rel);
        candidates.push_back("../" + fallback_rel);
    }
    for (const auto& c : candidates) {
        std::error_code ec;
        if (!c.empty() && fs::exists(c, ec) && fs::is_regular_file(c, ec)) {
            return c;
        }
    }
    return primary.empty() ? fallback_rel : primary;
}

std::string resolve_dir_path(const std::string& primary, const std::string& fallback_rel) {
    std::string chosen = primary.empty() ? fallback_rel : primary;
    std::error_code ec;
    if (fs::exists(chosen, ec) && fs::is_directory(chosen, ec)) {
        return chosen;
    }
    std::string parent_cand = "../" + chosen;
    if (fs::exists(parent_cand, ec) && fs::is_directory(parent_cand, ec)) {
        return parent_cand;
    }
    return chosen;
}

struct OrtSessionHolder {
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_cstrs;
    std::vector<const char*> output_cstrs;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    bool ready = false;

    bool load(Ort::Env& env, const Ort::SessionOptions& opts, const std::string& model_path, const char* tag) {
        ready = false;
        input_names.clear();
        output_names.clear();
        input_cstrs.clear();
        output_cstrs.clear();
        input_shape.clear();
        output_shape.clear();

        std::error_code ec;
        if (model_path.empty() || !fs::exists(model_path, ec)) {
            std::cerr << "[HailoVisionEngine] Model missing for " << tag << ": " << model_path << std::endl;
            return false;
        }
        try {
            session = std::make_unique<Ort::Session>(env, model_path.c_str(), opts);
            Ort::AllocatorWithDefaultOptions allocator;

            size_t n_in = session->GetInputCount();
            input_names.reserve(n_in);
            input_cstrs.reserve(n_in);
            for (size_t i = 0; i < n_in; ++i) {
                auto name_alloc = session->GetInputNameAllocated(i, allocator);
                input_names.emplace_back(name_alloc.get());
            }
            for (const auto& s : input_names) input_cstrs.push_back(s.c_str());
            if (n_in > 0) {
                input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            }

            size_t n_out = session->GetOutputCount();
            output_names.reserve(n_out);
            output_cstrs.reserve(n_out);
            for (size_t i = 0; i < n_out; ++i) {
                auto name_alloc = session->GetOutputNameAllocated(i, allocator);
                output_names.emplace_back(name_alloc.get());
            }
            for (const auto& s : output_names) output_cstrs.push_back(s.c_str());
            if (n_out > 0) {
                output_shape = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            }

            ready = true;
            std::cout << "[HailoVisionEngine] Loaded " << tag << " (ONNX Runtime): " << model_path << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[HailoVisionEngine] Failed to load " << tag << " (" << model_path << "): "
                      << e.what() << std::endl;
            ready = false;
            return false;
        }
    }
};

struct LetterboxMeta {
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
};

/**
 * Resize/letterbox BGR crop to (target_size x target_size), convert to RGB float32 [0,1] NCHW.
 */
LetterboxMeta bgr_to_nchw_letterbox(
    const cv::Mat& src_bgr,
    int target_size,
    std::vector<float>& out,
    bool keep_aspect
) {
    LetterboxMeta meta;
    cv::Mat prepared;
    if (keep_aspect && src_bgr.cols != src_bgr.rows) {
        float scale = std::min(
            static_cast<float>(target_size) / static_cast<float>(src_bgr.cols),
            static_cast<float>(target_size) / static_cast<float>(src_bgr.rows)
        );
        int new_w = std::max(1, static_cast<int>(std::round(src_bgr.cols * scale)));
        int new_h = std::max(1, static_cast<int>(std::round(src_bgr.rows * scale)));
        cv::Mat resized;
        cv::resize(src_bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

        int dw = target_size - new_w;
        int dh = target_size - new_h;
        int top = dh / 2;
        int bottom = dh - top;
        int left = dw / 2;
        int right = dw - left;
        cv::copyMakeBorder(resized, prepared, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        meta.scale = scale;
        meta.pad_x = static_cast<float>(left);
        meta.pad_y = static_cast<float>(top);
    } else {
        if (src_bgr.cols == target_size && src_bgr.rows == target_size) {
            prepared = src_bgr;
        } else {
            cv::resize(src_bgr, prepared, cv::Size(target_size, target_size), 0, 0, cv::INTER_LINEAR);
        }
        meta.scale = static_cast<float>(target_size) / static_cast<float>(std::max(1, src_bgr.cols));
        meta.pad_x = 0.0f;
        meta.pad_y = 0.0f;
    }

    cv::Mat rgb;
    cv::cvtColor(prepared, rgb, cv::COLOR_BGR2RGB);

    const int hw = target_size * target_size;
    out.resize(3 * static_cast<size_t>(hw));
    constexpr float inv255 = 1.0f / 255.0f;
    for (int y = 0; y < target_size; ++y) {
        const uint8_t* row = rgb.ptr<uint8_t>(y);
        int row_off = y * target_size;
        for (int x = 0; x < target_size; ++x) {
            int p = row_off + x;
            out[0 * hw + p] = static_cast<float>(row[x * 3 + 0]) * inv255;
            out[1 * hw + p] = static_cast<float>(row[x * 3 + 1]) * inv255;
            out[2 * hw + p] = static_cast<float>(row[x * 3 + 2]) * inv255;
        }
    }
    return meta;
}

#ifdef HAVE_HAILORT
void bgr_to_hwc_uint8(const cv::Mat& src_bgr, int target_size, std::vector<uint8_t>& out, LetterboxMeta& meta) {
    cv::Mat resized;
    float scale = std::min(
        static_cast<float>(target_size) / static_cast<float>(src_bgr.cols),
        static_cast<float>(target_size) / static_cast<float>(src_bgr.rows)
    );
    int new_w = std::max(1, static_cast<int>(std::round(src_bgr.cols * scale)));
    int new_h = std::max(1, static_cast<int>(std::round(src_bgr.rows * scale)));
    cv::resize(src_bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int dw = target_size - new_w;
    int dh = target_size - new_h;
    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) rgb = rgb.clone();

    out.assign(rgb.data, rgb.data + static_cast<size_t>(target_size) * target_size * 3);
    meta.scale = scale;
    meta.pad_x = static_cast<float>(left);
    meta.pad_y = static_cast<float>(top);
}
#endif

std::vector<float> softmax_if_needed(const float* raw, size_t n) {
    std::vector<float> probs(n, 0.0f);
    if (n == 0 || !raw) return probs;

    float sum = 0.0f;
    bool need_softmax = false;
    for (size_t i = 0; i < n; ++i) {
        if (raw[i] < -1e-4f || raw[i] > 1.0001f) {
            need_softmax = true;
        }
        sum += raw[i];
    }
    if (!need_softmax && std::fabs(sum - 1.0f) <= 0.05f) {
        for (size_t i = 0; i < n; ++i) probs[i] = std::max(0.0f, raw[i]);
        return probs;
    }

    float max_v = raw[0];
    for (size_t i = 1; i < n; ++i) {
        if (raw[i] > max_v) max_v = raw[i];
    }
    double exp_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double e = std::exp(static_cast<double>(raw[i] - max_v));
        probs[i] = static_cast<float>(e);
        exp_sum += e;
    }
    if (exp_sum > 0.0) {
        float inv = static_cast<float>(1.0 / exp_sum);
        for (size_t i = 0; i < n; ++i) probs[i] *= inv;
    }
    return probs;
}

struct TileWindow {
    cv::Rect roi;
    bool is_full_frame = false;
};

std::vector<TileWindow> generate_tile_windows(int w, int h, const qc::rules::TilingConfig& tcfg) {
    std::vector<TileWindow> windows;
    if (w <= 0 || h <= 0) return windows;

    // Run single full-frame 640x640 inference on standard <=720 px images and reserve
    // 2x2 overlapping tiles for high-resolution frames (>720 px) to preserve small-defect recall.
    const int min_tile_span = std::max(720, tcfg.tile_size + 80);
    const bool skip_subtiles = !tcfg.enabled || (w <= min_tile_span && h <= min_tile_span);

    if (tcfg.include_full_frame || skip_subtiles) {
        windows.push_back({cv::Rect(0, 0, w, h), true});
    }
    if (skip_subtiles) {
        return windows;
    }

    int rows = std::max(1, tcfg.grid_rows);
    int cols = std::max(1, tcfg.grid_cols);
    float ov = std::max(0.0f, std::min(0.49f, tcfg.overlap));

    float denom_x = static_cast<float>(cols) - static_cast<float>(cols - 1) * ov;
    float denom_y = static_cast<float>(rows) - static_cast<float>(rows - 1) * ov;
    int tile_w = std::max(32, std::min(w, static_cast<int>(std::ceil(static_cast<float>(w) / std::max(1.0f, denom_x)))));
    int tile_h = std::max(32, std::min(h, static_cast<int>(std::ceil(static_cast<float>(h) / std::max(1.0f, denom_y)))));

    float stride_x = (cols > 1) ? static_cast<float>(w - tile_w) / static_cast<float>(cols - 1) : 0.0f;
    float stride_y = (rows > 1) ? static_cast<float>(h - tile_h) / static_cast<float>(rows - 1) : 0.0f;

    for (int r = 0; r < rows; ++r) {
        int y1 = (r == rows - 1 && rows > 1) ? (h - tile_h) : static_cast<int>(std::round(r * stride_y));
        y1 = std::max(0, std::min(h - tile_h, y1));
        for (int c = 0; c < cols; ++c) {
            int x1 = (c == cols - 1 && cols > 1) ? (w - tile_w) : static_cast<int>(std::round(c * stride_x));
            x1 = std::max(0, std::min(w - tile_w, x1));
            windows.push_back({cv::Rect(x1, y1, tile_w, tile_h), false});
        }
    }
    return windows;
}

struct ProjectedCandidate {
    cv::Rect rect;
    float conf = 0.0f;
    int class_id = 0;
    std::string label;
    std::string group;
};

void decode_yolo_tensor_into(
    const float* data,
    const std::vector<int64_t>& shape,
    int num_classes,
    int input_size,
    float min_conf,
    const TileWindow& tile,
    const LetterboxMeta& meta,
    int orig_w,
    int orig_h,
    const qc::rules::QcRulesEngine& rules,
    std::vector<ProjectedCandidate>& out_candidates
) {
    if (!data || shape.size() < 2) return;

    int dims_c = 4 + num_classes;
    int num_anchors = 0;
    bool channels_first = true;

    if (shape.size() == 3) {
        int d1 = static_cast<int>(shape[1]);
        int d2 = static_cast<int>(shape[2]);
        if (d1 == dims_c) {
            channels_first = true;
            num_anchors = d2;
        } else if (d2 == dims_c) {
            channels_first = false;
            num_anchors = d1;
        } else if (d1 < d2) {
            channels_first = true;
            num_anchors = d2;
            dims_c = d1;
            num_classes = std::max(1, dims_c - 4);
        } else {
            channels_first = false;
            num_anchors = d1;
            dims_c = d2;
            num_classes = std::max(1, dims_c - 4);
        }
    } else if (shape.size() == 2) {
        int d0 = static_cast<int>(shape[0]);
        int d1 = static_cast<int>(shape[1]);
        if (d0 == dims_c) {
            channels_first = true;
            num_anchors = d1;
        } else {
            channels_first = false;
            num_anchors = d0;
            dims_c = d1;
            num_classes = std::max(1, dims_c - 4);
        }
    }

    if (num_anchors <= 0 || num_classes <= 0) return;

    auto read_val = [&](int c, int a) -> float {
        return channels_first ? data[c * num_anchors + a] : data[a * dims_c + c];
    };

    // Detect if coordinates are normalized [0, 1] rather than [0, input_size]
    float max_coord_sample = 0.0f;
    int sample_step = std::max(1, num_anchors / 64);
    for (int a = 0; a < num_anchors; a += sample_step) {
        max_coord_sample = std::max(max_coord_sample, std::fabs(read_val(0, a)));
        max_coord_sample = std::max(max_coord_sample, std::fabs(read_val(2, a)));
    }
    const float coord_scale = (max_coord_sample <= 2.0f) ? static_cast<float>(input_size) : 1.0f;

    // For non-square full-frame letterbox vs square tile crop:
    // scale_x / scale_y account for keep_aspect=true when tile is full-frame or non-square,
    // and direct resize when keep_aspect=false. Here we always use keep_aspect=true!
    const float inv_scale = 1.0f / std::max(1e-5f, meta.scale);

    for (int a = 0; a < num_anchors; ++a) {
        float max_score = -1.0f;
        int best_cls = -1;
        for (int c = 0; c < num_classes; ++c) {
            float sc = read_val(4 + c, a);
            if (sc > max_score) {
                max_score = sc;
                best_cls = c;
            }
        }
        if (best_cls < 0 || max_score < min_conf) continue;

        std::string grp = rules.group_from_class_id(best_cls);
        float cls_thresh = std::min(min_conf, rules.get_group_conf_threshold(grp));
        if (max_score < cls_thresh) continue;

        float xc = read_val(0, a) * coord_scale;
        float yc = read_val(1, a) * coord_scale;
        float bw = read_val(2, a) * coord_scale;
        float bh = read_val(3, a) * coord_scale;

        float lx1 = (xc - bw * 0.5f - meta.pad_x) * inv_scale;
        float ly1 = (yc - bh * 0.5f - meta.pad_y) * inv_scale;
        float lx2 = (xc + bw * 0.5f - meta.pad_x) * inv_scale;
        float ly2 = (yc + bh * 0.5f - meta.pad_y) * inv_scale;

        int gx1 = std::max(0, std::min(orig_w - 1, tile.roi.x + static_cast<int>(std::round(lx1))));
        int gy1 = std::max(0, std::min(orig_h - 1, tile.roi.y + static_cast<int>(std::round(ly1))));
        int gx2 = std::max(0, std::min(orig_w, tile.roi.x + static_cast<int>(std::round(lx2))));
        int gy2 = std::max(0, std::min(orig_h, tile.roi.y + static_cast<int>(std::round(ly2))));

        if (gx2 - gx1 < 4 || gy2 - gy1 < 4) continue;

        ProjectedCandidate cand;
        cand.rect = cv::Rect(gx1, gy1, gx2 - gx1, gy2 - gy1);
        cand.conf = std::min(1.0f, std::max(0.0f, max_score));
        cand.class_id = best_cls;
        cand.group = grp;
        cand.label = grp;
        out_candidates.push_back(std::move(cand));
    }
}

std::string to_upper_label(std::string s) {
    std::replace(s.begin(), s.end(), '_', ' ');
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

/**
 * Deterministic OpenCV geometric, texture, and hotspot sub-classifier for
 * PatchCore-detected anomalies on machined parts (e.g. metal nuts/fasteners)
 * so UI/logs display the specific physical defect class rather than a generic tag.
 */
std::pair<std::string, std::string> classify_machined_part_anomaly(
    const cv::Mat& frame_bgr,
    const cv::Rect& hotspot_rect
) {
    if (frame_bgr.empty() || frame_bgr.cols < 16 || frame_bgr.rows < 16) {
        return {"surface_anomaly", "surface_anomaly"};
    }

    const int w = frame_bgr.cols;
    const int h = frame_bgr.rows;
    cv::Mat gray_u8;
    cv::cvtColor(frame_bgr, gray_u8, cv::COLOR_BGR2GRAY);

    // Segment primary metallic part via largest foreground connected component
    cv::Mat mask = (gray_u8 > 28);
    cv::Mat labels, stats, centroids;
    int num_comp = cv::connectedComponentsWithStats(mask, labels, stats, centroids);

    double cx = w * 0.5;
    double cy = h * 0.5;
    cv::Mat fg(h, w, CV_8UC1, cv::Scalar(0));
    int best_area = 0;
    if (num_comp > 1) {
        int best_idx = 1;
        for (int i = 1; i < num_comp; ++i) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area > best_area) {
                best_area = area;
                best_idx = i;
            }
        }
        cx = centroids.at<double>(best_idx, 0);
        cy = centroids.at<double>(best_idx, 1);
        fg = (labels == best_idx);
    } else {
        fg = mask;
        best_area = cv::countNonZero(fg);
    }

    const double fg_frac = static_cast<double>(best_area) / static_cast<double>(w * h);
    cv::Mat fg_inner;
    cv::erode(fg, fg_inner, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));

    // Analyze centered/circular fastener geometry when part is segmented in frame
    if (fg_frac >= 0.20 && fg_frac <= 0.78 && best_area > 64) {
        std::vector<cv::Point> pts;
        cv::findNonZero(fg, pts);
        std::vector<float> radii;
        radii.reserve(pts.size());
        for (const auto& pt : pts) {
            const float dx = static_cast<float>(pt.x - cx);
            const float dy = static_cast<float>(pt.y - cy);
            radii.push_back(std::sqrt(dx * dx + dy * dy));
        }
        std::vector<float> sorted_r = radii;
        size_t idx98 = std::min(sorted_r.size() - 1, static_cast<size_t>(sorted_r.size() * 0.98));
        std::nth_element(sorted_r.begin(), sorted_r.begin() + idx98, sorted_r.end());
        const float rmax = std::max(10.0f, sorted_r[idx98]);

        // 1. Flipped Orientation Check (rear flat stamped face vs front chamfered bore)
        cv::Mat lap_f;
        cv::Laplacian(gray_u8, lap_f, CV_32F);
        int hole_total = 0, hole_dark = 0;
        double ring_lap_sum = 0.0;
        int ring_cnt = 0;
        double g_mid_sum = 0.0, g_out_sum = 0.0;
        int g_mid_cnt = 0, g_out_cnt = 0;

        for (int y = 0; y < h; ++y) {
            const uint8_t* gp = gray_u8.ptr<uint8_t>(y);
            const uint8_t* fp = fg.ptr<uint8_t>(y);
            const float* lp = lap_f.ptr<float>(y);
            const double dy = static_cast<double>(y) - cy;
            for (int x = 0; x < w; ++x) {
                const double dx = static_cast<double>(x) - cx;
                const float rn = static_cast<float>(std::sqrt(dx * dx + dy * dy)) / rmax;
                if (rn < 0.30f) {
                    hole_total++;
                    if (gp[x] <= 28) hole_dark++;
                }
                if (rn >= 0.42f && rn < 0.78f) {
                    ring_lap_sum += std::fabs(lp[x]);
                    ring_cnt++;
                }
                if (fp[x]) {
                    if (rn >= 0.40f && rn < 0.65f) {
                        g_mid_sum += gp[x];
                        g_mid_cnt++;
                    } else if (rn >= 0.65f && rn < 0.95f) {
                        g_out_sum += gp[x];
                        g_out_cnt++;
                    }
                }
            }
        }
        const double hole_ratio = (hole_total > 0) ? (static_cast<double>(hole_dark) / hole_total) : 0.0;
        const double body_lap = (ring_cnt > 0) ? (ring_lap_sum / ring_cnt) : 20.0;
        const double g_mid = (g_mid_cnt > 0) ? (g_mid_sum / g_mid_cnt) : 0.0;
        const double g_out = (g_out_cnt > 0) ? (g_out_sum / g_out_cnt) : 0.0;

        if (fg_frac >= 0.30 && fg_frac <= 0.72 && hole_ratio >= 0.35 &&
            body_lap < 11.5 && g_mid > g_out + 2.0) {
            return {"flipped_part", "deformation"};
        }

        // 2. Bent / Deformed Lug Check via 4-fold (90-degree) contour symmetry
        std::vector<std::vector<float>> ang_bins(36);
        constexpr float kTwoPi = 6.28318530718f;
        constexpr float kPi = 3.14159265359f;
        for (size_t i = 0; i < pts.size(); ++i) {
            float ang = std::atan2(static_cast<float>(pts[i].y - cy), static_cast<float>(pts[i].x - cx));
            int bin = static_cast<int>(std::floor(((ang + kPi) / kTwoPi) * 36.0f));
            bin = std::max(0, std::min(35, bin));
            ang_bins[static_cast<size_t>(bin)].push_back(radii[i] / rmax);
        }
        std::array<float, 36> r95{};
        for (int b = 0; b < 36; ++b) {
            auto& v = ang_bins[static_cast<size_t>(b)];
            if (v.size() > 5) {
                size_t k95 = std::min(v.size() - 1, static_cast<size_t>(v.size() * 0.95));
                std::nth_element(v.begin(), v.begin() + k95, v.end());
                r95[static_cast<size_t>(b)] = v[k95];
            }
        }
        float rot90_sum = 0.0f, rot90_max = 0.0f;
        for (int b = 0; b < 36; ++b) {
            float d = std::fabs(r95[static_cast<size_t>(b)] - r95[static_cast<size_t>((b + 9) % 36)]);
            rot90_sum += d;
            rot90_max = std::max(rot90_max, d);
        }
        const float rot90_avg = rot90_sum / 36.0f;
        if (rot90_max >= 0.115f || rot90_avg >= 0.024f) {
            return {"bent", "deformation"};
        }
    }

    // 3. Hotspot Color Spot vs Linear Scratch analysis (strictly inside part silhouette, excluding exterior black notches)
    cv::Mat inv_dark = (gray_u8 <= 28);
    cv::Mat ff_bg = inv_dark.clone();
    for (int x = 0; x < w; ++x) {
        if (ff_bg.at<uint8_t>(0, x) == 255) cv::floodFill(ff_bg, cv::Point(x, 0), 128);
        if (ff_bg.at<uint8_t>(h - 1, x) == 255) cv::floodFill(ff_bg, cv::Point(x, h - 1), 128);
    }
    for (int y = 0; y < h; ++y) {
        if (ff_bg.at<uint8_t>(y, 0) == 255) cv::floodFill(ff_bg, cv::Point(0, y), 128);
        if (ff_bg.at<uint8_t>(y, w - 1) == 255) cv::floodFill(ff_bg, cv::Point(w - 1, y), 128);
    }
    cv::Mat part_solid = (ff_bg != 128);
    {
        cv::Mat p_lb, p_st, p_cen;
        int p_nc = cv::connectedComponentsWithStats(part_solid, p_lb, p_st, p_cen);
        if (p_nc > 1) {
            int p_bi = 1, p_ba = 0;
            for (int i = 1; i < p_nc; ++i) {
                int a = p_st.at<int>(i, cv::CC_STAT_AREA);
                if (a > p_ba) { p_ba = a; p_bi = i; }
            }
            part_solid = (p_lb == p_bi);
        }
    }

    const double rot90_cen_x = cx;
    const double rot90_cen_y = cy;
    float bore_rmax = 0.0f;
    if (fg_frac >= 0.20 && fg_frac <= 0.78 && best_area > 64) {
        std::vector<cv::Point> fpts;
        cv::findNonZero(fg, fpts);
        if (!fpts.empty()) {
            std::vector<float> frad;
            frad.reserve(fpts.size());
            for (const auto& pt : fpts) {
                frad.push_back(std::hypot(static_cast<float>(pt.x - rot90_cen_x), static_cast<float>(pt.y - rot90_cen_y)));
            }
            size_t fk = std::min(frad.size() - 1, static_cast<size_t>(frad.size() * 0.98));
            std::nth_element(frad.begin(), frad.begin() + fk, frad.end());
            bore_rmax = std::max(10.0f, frad[fk]);
        }
    }

    std::vector<uint8_t> fg_b, fg_g, fg_r, fg_gray;
    fg_b.reserve(static_cast<size_t>(std::max(1, best_area)));
    fg_g.reserve(static_cast<size_t>(std::max(1, best_area)));
    fg_r.reserve(static_cast<size_t>(std::max(1, best_area)));
    fg_gray.reserve(static_cast<size_t>(std::max(1, best_area)));
    for (int y = 0; y < h; ++y) {
        const cv::Vec3b* bp = frame_bgr.ptr<cv::Vec3b>(y);
        const uint8_t* gp = gray_u8.ptr<uint8_t>(y);
        const uint8_t* fp = fg.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            if (fp[x]) {
                fg_b.push_back(bp[x][0]);
                fg_g.push_back(bp[x][1]);
                fg_r.push_back(bp[x][2]);
                fg_gray.push_back(gp[x]);
            }
        }
    }
    auto median_u8 = [](std::vector<uint8_t>& v) -> float {
        if (v.empty()) return 90.0f;
        size_t m = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + m, v.end());
        return static_cast<float>(v[m]);
    };
    const float mb = median_u8(fg_b);
    const float mg = median_u8(fg_g);
    const float mr = median_u8(fg_r);
    const float med_v = std::max(30.0f, median_u8(fg_gray));

    // Check inner part surface for localized dark/colored stain cluster (pinpoint color_spot vs broader scratch)
    if (bore_rmax > 10.0f) {
        cv::Mat inner_solid = part_solid.clone();
        for (int y = 0; y < h; ++y) {
            uint8_t* sp = inner_solid.ptr<uint8_t>(y);
            for (int x = 0; x < w; ++x) {
                if (sp[x]) {
                    const float rn = std::hypot(static_cast<float>(x - rot90_cen_x), static_cast<float>(y - rot90_cen_y)) / bore_rmax;
                    if (rn < 0.35f) sp[x] = 0;
                }
            }
        }
        const int k_er = std::max(5, std::min(15, std::min(w, h) / 28) | 1);
        cv::erode(inner_solid, inner_solid, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k_er, k_er)));
        cv::Mat full_hsv;
        cv::cvtColor(frame_bgr, full_hsv, cv::COLOR_BGR2HSV);
        cv::Mat dark_stain(h, w, CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < h; ++y) {
            const uint8_t* ip = inner_solid.ptr<uint8_t>(y);
            const uint8_t* gp = gray_u8.ptr<uint8_t>(y);
            const cv::Vec3b* hp = full_hsv.ptr<cv::Vec3b>(y);
            uint8_t* dp = dark_stain.ptr<uint8_t>(y);
            for (int x = 0; x < w; ++x) {
                if (!ip[x]) continue;
                const float pv = static_cast<float>(gp[x]);
                const float sat = static_cast<float>(hp[x][1]);
                if (pv < med_v * 0.42f || (sat > 95.0f && pv < med_v * 0.65f)) {
                    dp[x] = 255;
                }
            }
        }
        cv::morphologyEx(dark_stain, dark_stain, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
        std::vector<std::vector<cv::Point>> dcs;
        cv::findContours(dark_stain, dcs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        double max_dark_area = 0.0;
        for (const auto& c : dcs) max_dark_area = std::max(max_dark_area, cv::contourArea(c));
        const double norm_s = 256.0 / static_cast<double>(std::max(1, std::min(w, h)));
        if (max_dark_area * norm_s * norm_s >= 45.0) {
            return {"color_spot", "stain_oxide"};
        }
    }

    cv::Rect roi(
        std::max(0, hotspot_rect.x - 6),
        std::max(0, hotspot_rect.y - 6),
        std::min(w - std::max(0, hotspot_rect.x - 6), hotspot_rect.width + 12),
        std::min(h - std::max(0, hotspot_rect.y - 6), hotspot_rect.height + 12)
    );
    if (roi.width >= 6 && roi.height >= 6) {
        cv::Mat patch_bgr = frame_bgr(roi);
        cv::Mat patch_gray_u8 = gray_u8(roi);
        cv::Mat patch_solid = part_solid(roi);
        cv::Mat patch_hsv;
        cv::cvtColor(patch_bgr, patch_hsv, cv::COLOR_BGR2HSV);

        std::vector<float> p_grays, p_cdiff, p_sats;
        const int max_px = patch_bgr.rows * patch_bgr.cols;
        p_grays.reserve(static_cast<size_t>(max_px));
        p_cdiff.reserve(static_cast<size_t>(max_px));
        p_sats.reserve(static_cast<size_t>(max_px));
        int dark_cnt = 0, bright_cnt = 0, n_px = 0;

        for (int y = 0; y < patch_bgr.rows; ++y) {
            const cv::Vec3b* bp = patch_bgr.ptr<cv::Vec3b>(y);
            const cv::Vec3b* hp = patch_hsv.ptr<cv::Vec3b>(y);
            const uint8_t* gp = patch_gray_u8.ptr<uint8_t>(y);
            const uint8_t* sp = patch_solid.ptr<uint8_t>(y);
            const int gy_abs = roi.y + y;
            for (int x = 0; x < patch_bgr.cols; ++x) {
                if (!sp[x]) continue;
                if (bore_rmax > 10.0f) {
                    const float rn = std::hypot(static_cast<float>((roi.x + x) - rot90_cen_x), static_cast<float>(gy_abs - rot90_cen_y)) / bore_rmax;
                    if (rn < 0.36f) continue;
                }
                n_px++;
                float pv = static_cast<float>(gp[x]);
                p_grays.push_back(pv);
                if (pv < med_v * 0.48f) dark_cnt++;
                if (pv > med_v * 1.25f) bright_cnt++;
                float b = static_cast<float>(bp[x][0]);
                float g = static_cast<float>(bp[x][1]);
                float r = static_cast<float>(bp[x][2]);
                float cd = std::fabs((r - g) - (mr - mg)) + std::fabs((g - b) - (mg - mb));
                p_cdiff.push_back(cd);
                p_sats.push_back(static_cast<float>(hp[x][1]));
            }
        }

        auto pct_f = [](std::vector<float>& v, float q) -> float {
            if (v.empty()) return 0.0f;
            size_t k = std::min(v.size() - 1, static_cast<size_t>(v.size() * q));
            std::nth_element(v.begin(), v.begin() + k, v.end());
            return v[k];
        };
        const float dark10 = pct_f(p_grays, 0.10f);
        const float cdiff95 = pct_f(p_cdiff, 0.95f);
        const float sat95 = pct_f(p_sats, 0.95f);
        const float blot_frac = static_cast<float>(dark_cnt) / static_cast<float>(std::max(1, n_px));
        const float bright_frac = static_cast<float>(bright_cnt) / static_cast<float>(std::max(1, n_px));

        cv::Mat patch_gray_f, gx, gy;
        patch_gray_u8.convertTo(patch_gray_f, CV_32F);
        cv::Sobel(patch_gray_f, gx, CV_32F, 1, 0, 3);
        cv::Sobel(patch_gray_f, gy, CV_32F, 0, 1, 3);
        cv::Mat mag;
        cv::magnitude(gx, gy, mag);
        const double edge_mean = cv::mean(mag)[0];
        const double jxx = cv::mean(gx.mul(gx))[0];
        const double jyy = cv::mean(gy.mul(gy))[0];
        const double jxy = cv::mean(gx.mul(gy))[0];
        const double coh = std::sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy) / (jxx + jyy + 1e-4);

        const bool is_color_spot = (blot_frac >= 0.12f && dark10 < med_v * 0.38f) ||
                                   (sat95 >= 84.0f && dark10 < med_v * 0.45f) ||
                                   (cdiff95 >= 38.0f && blot_frac >= 0.10f);
        if (is_color_spot) {
            return {"color_spot", "stain_oxide"};
        }
        if (bright_frac >= 0.12f || edge_mean >= 28.0 || coh >= 0.32) {
            return {"scratch", "scratch"};
        }
    }

    return {"surface_anomaly", "surface_anomaly"};
}

/**
 * Extracts foreground metal part silhouette via border flood-fill of dark exterior
 * pixels (excluding concave notch gaps while preserving interior surface defects).
 */
bool extract_part_silhouette(
    const cv::Mat& gray,
    cv::Mat& solid_out,
    double& cx_out,
    double& cy_out,
    float& rmax_out
) {
    if (gray.empty() || gray.cols < 32 || gray.rows < 32) return false;
    const int w = gray.cols;
    const int h = gray.rows;

    cv::Mat ff = (gray <= 28);
    for (int x = 0; x < w; ++x) {
        if (ff.at<uint8_t>(0, x) == 255) cv::floodFill(ff, cv::Point(x, 0), 128);
        if (ff.at<uint8_t>(h - 1, x) == 255) cv::floodFill(ff, cv::Point(x, h - 1), 128);
    }
    for (int y = 0; y < h; ++y) {
        if (ff.at<uint8_t>(y, 0) == 255) cv::floodFill(ff, cv::Point(0, y), 128);
        if (ff.at<uint8_t>(y, w - 1) == 255) cv::floodFill(ff, cv::Point(w - 1, y), 128);
    }
    cv::Mat part = (ff != 128);
    cv::Mat lb, st, cen;
    int nc = cv::connectedComponentsWithStats(part, lb, st, cen);
    if (nc <= 1) return false;

    int bi = 1, ba = 0;
    for (int i = 1; i < nc; ++i) {
        int a = st.at<int>(i, cv::CC_STAT_AREA);
        if (a > ba) { ba = a; bi = i; }
    }
    solid_out = (lb == bi);
    cx_out = cen.at<double>(bi, 0);
    cy_out = cen.at<double>(bi, 1);

    std::vector<cv::Point> pts;
    cv::findNonZero(solid_out, pts);
    if (pts.size() < 64) return false;
    std::vector<float> rad;
    rad.reserve(pts.size());
    for (const auto& pt : pts) {
        rad.push_back(std::hypot(static_cast<float>(pt.x - cx_out), static_cast<float>(pt.y - cy_out)));
    }
    const size_t k98 = std::min(rad.size() - 1, static_cast<size_t>(rad.size() * 0.98));
    std::nth_element(rad.begin(), rad.begin() + k98, rad.end());
    rmax_out = std::max(10.0f, rad[k98]);
    return true;
}

/**
 * Pixel-accurate contour refinement for machined-part color_spot (ink/oil/oxide stain)
 * using border flood-fill part masking so outer lug corner glare is excluded.
 */
std::vector<std::vector<cv::Point>> refine_color_spot_contours(const cv::Mat& frame_bgr) {
    std::vector<std::vector<cv::Point>> out;
    if (frame_bgr.empty() || frame_bgr.cols < 32 || frame_bgr.rows < 32) {
        return out;
    }
    const int w = frame_bgr.cols;
    const int h = frame_bgr.rows;
    cv::Mat gray;
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);

    cv::Mat solid;
    double cx = 0.0, cy = 0.0;
    float rmax = 10.0f;
    if (!extract_part_silhouette(gray, solid, cx, cy, rmax)) {
        return out;
    }

    for (int y = 0; y < h; ++y) {
        uint8_t* sp = solid.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            if (std::hypot(static_cast<float>(x - cx), static_cast<float>(y - cy)) / rmax < 0.35f) {
                sp[x] = 0;
            }
        }
    }
    const int k_er = std::max(5, std::min(15, std::min(w, h) / 28) | 1);
    cv::Mat inner;
    cv::erode(solid, inner, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k_er, k_er)));

    std::vector<uint8_t> pg, pb, pgr, pr;
    pg.reserve(static_cast<size_t>(w * h / 3));
    pb.reserve(static_cast<size_t>(w * h / 3));
    pgr.reserve(static_cast<size_t>(w * h / 3));
    pr.reserve(static_cast<size_t>(w * h / 3));
    for (int y = 0; y < h; ++y) {
        const uint8_t* sp = solid.ptr<uint8_t>(y);
        const uint8_t* gp = gray.ptr<uint8_t>(y);
        const cv::Vec3b* bp = frame_bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < w; ++x) {
            if (sp[x] && gp[x] > 28) {
                pg.push_back(gp[x]);
                pb.push_back(bp[x][0]);
                pgr.push_back(bp[x][1]);
                pr.push_back(bp[x][2]);
            }
        }
    }
    if (pg.empty()) return out;
    auto med = [](std::vector<uint8_t>& v) -> float {
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return static_cast<float>(v[v.size() / 2]);
    };
    const float mv = std::max(30.0f, med(pg));
    const float mb = med(pb), mg = med(pgr), mr = med(pr);

    cv::Mat hsv;
    cv::cvtColor(frame_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat stain(h, w, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < h; ++y) {
        const uint8_t* ip = inner.ptr<uint8_t>(y);
        const uint8_t* gp = gray.ptr<uint8_t>(y);
        const cv::Vec3b* bp = frame_bgr.ptr<cv::Vec3b>(y);
        const cv::Vec3b* hp = hsv.ptr<cv::Vec3b>(y);
        uint8_t* dp = stain.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            if (!ip[x]) continue;
            const float pv = static_cast<float>(gp[x]);
            const float cd = std::fabs((bp[x][2] - bp[x][1]) - (mr - mg)) +
                             std::fabs((bp[x][1] - bp[x][0]) - (mg - mb));
            const float sat = static_cast<float>(hp[x][1]);
            if (pv < mv * 0.40f || cd > 36.0f || (sat > 95.0f && pv < mv * 0.85f)) {
                dp[x] = 255;
            }
        }
    }
    cv::morphologyEx(stain, stain, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    std::vector<std::vector<cv::Point>> raw_c;
    cv::findContours(stain, raw_c, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double max_a = 0.0;
    for (const auto& c : raw_c) {
        max_a = std::max(max_a, cv::contourArea(c));
    }
    const double min_a = std::max(20.0, max_a * 0.55);
    if (max_a < 20.0) return out;
    for (const auto& c : raw_c) {
        if (cv::contourArea(c) < min_a) continue;
        std::vector<cv::Point> poly;
        cv::approxPolyDP(c, poly, 1.2, true);
        if (poly.size() >= 3) {
            out.push_back(std::move(poly));
        }
    }
    return out;
}

/**
 * Pixel-accurate contour refinement for machined-part scratches anchored on the
 * PatchCore hotspot seed: normalizes to 256x256 for scale-invariant behavior
 * across static images and 640x640 conveyor streams, stays outside the central
 * chamfer bevel, and clips morphological bridges to the metal silhouette so
 * neighboring clean prongs and background notches are never crossed.
 */
std::vector<std::vector<cv::Point>> refine_scratch_contours(
    const cv::Mat& frame_bgr,
    const std::vector<std::vector<cv::Point>>& seed_contours
) {
    std::vector<std::vector<cv::Point>> out;
    if (frame_bgr.empty() || seed_contours.empty()) return out;
    const int orig_w = frame_bgr.cols;
    const int orig_h = frame_bgr.rows;

    constexpr int NORM_S = 256;
    cv::Mat gray_orig;
    cv::cvtColor(frame_bgr, gray_orig, cv::COLOR_BGR2GRAY);
    cv::Mat gray;
    if (orig_w != NORM_S || orig_h != NORM_S) {
        cv::resize(gray_orig, gray, cv::Size(NORM_S, NORM_S), 0, 0, cv::INTER_AREA);
    } else {
        gray = gray_orig;
    }

    cv::Mat solid;
    double cx = 0.0, cy = 0.0;
    float rmax = 10.0f;
    if (!extract_part_silhouette(gray, solid, cx, cy, rmax)) {
        return out;
    }

    cv::Mat gblur;
    cv::GaussianBlur(gray, gblur, cv::Size(13, 13), 3.2);

    std::vector<std::vector<uint8_t>> rbins(20);
    for (int y = 0; y < NORM_S; ++y) {
        const uint8_t* sp = solid.ptr<uint8_t>(y);
        const uint8_t* bp = gblur.ptr<uint8_t>(y);
        for (int x = 0; x < NORM_S; ++x) {
            if (!sp[x]) continue;
            float rn = std::hypot(static_cast<float>(x - cx), static_cast<float>(y - cy)) / rmax;
            int bin = std::min(19, std::max(0, static_cast<int>(rn * 20.0f)));
            rbins[static_cast<size_t>(bin)].push_back(bp[x]);
        }
    }
    std::array<float, 20> rmed{};
    for (size_t b = 0; b < 20; ++b) {
        if (!rbins[b].empty()) {
            size_t m = rbins[b].size() / 2;
            std::nth_element(rbins[b].begin(), rbins[b].begin() + m, rbins[b].end());
            rmed[b] = static_cast<float>(rbins[b][m]);
        }
    }

    const float sx_in = static_cast<float>(NORM_S) / static_cast<float>(std::max(1, orig_w));
    const float sy_in = static_cast<float>(NORM_S) / static_cast<float>(std::max(1, orig_h));
    std::vector<std::vector<cv::Point>> scaled_seeds;
    scaled_seeds.reserve(seed_contours.size());
    for (const auto& poly : seed_contours) {
        std::vector<cv::Point> sp;
        sp.reserve(poly.size());
        for (const auto& pt : poly) {
            sp.emplace_back(
                std::max(0, std::min(NORM_S - 1, static_cast<int>(std::round(pt.x * sx_in)))),
                std::max(0, std::min(NORM_S - 1, static_cast<int>(std::round(pt.y * sy_in))))
            );
        }
        if (sp.size() >= 3) scaled_seeds.push_back(std::move(sp));
    }
    if (scaled_seeds.empty()) return out;

    cv::Mat spec(NORM_S, NORM_S, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < NORM_S; ++y) {
        const uint8_t* gp_raw = gray.ptr<uint8_t>(y);
        const uint8_t* bp = gblur.ptr<uint8_t>(y);
        uint8_t* sp_out = spec.ptr<uint8_t>(y);
        for (int x = 0; x < NORM_S; ++x) {
            if (gp_raw[x] > 142 || bp[x] > 136) {
                sp_out[x] = 255;
            }
        }
    }
    cv::dilate(spec, spec, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)));

    cv::Mat seed(NORM_S, NORM_S, CV_8UC1, cv::Scalar(0));
    cv::drawContours(seed, scaled_seeds, -1, cv::Scalar(255), cv::FILLED);
    cv::bitwise_and(seed, solid, seed);
    {
        // Strip specular glare islands and inner bore collar creep from the heatmap seed itself
        cv::Mat seed_clean = seed.clone();
        for (int y = 0; y < NORM_S; ++y) {
            const uint8_t* spec_p = spec.ptr<uint8_t>(y);
            uint8_t* cp = seed_clean.ptr<uint8_t>(y);
            for (int x = 0; x < NORM_S; ++x) {
                const float rn = std::hypot(static_cast<float>(x - cx), static_cast<float>(y - cy)) / rmax;
                if (spec_p[x] || rn < 0.42f) {
                    cp[x] = 0;
                }
            }
        }
        if (cv::countNonZero(seed_clean) >= 20) {
            seed = std::move(seed_clean);
        }
        // Reject detached minor glare hotspots far from the primary scratch cluster
        cv::Mat slb, sst, scen;
        const int snc = cv::connectedComponentsWithStats(seed, slb, sst, scen);
        if (snc > 2) {
            int best_id = 1;
            int best_area = -1;
            for (int i = 1; i < snc; ++i) {
                const int area = sst.at<int>(i, cv::CC_STAT_AREA);
                if (area > best_area) {
                    best_area = area;
                    best_id = i;
                }
            }
            const double bx = scen.at<double>(best_id, 0);
            const double by = scen.at<double>(best_id, 1);
            for (int y = 0; y < NORM_S; ++y) {
                const int* lp = slb.ptr<int>(y);
                uint8_t* sp_seed = seed.ptr<uint8_t>(y);
                for (int x = 0; x < NORM_S; ++x) {
                    const int id = lp[x];
                    if (id > 0 && id != best_id) {
                        const double d = std::hypot(scen.at<double>(id, 0) - bx, scen.at<double>(id, 1) - by);
                        if (d > rmax * 0.40 || sst.at<int>(id, cv::CC_STAT_AREA) < best_area * 0.30) {
                            sp_seed[x] = 0;
                        }
                    }
                }
            }
        }
    }
    cv::Moments sm = cv::moments(seed, true);
    if (sm.m00 < 1.0) return out;
    const float scx = static_cast<float>(sm.m10 / sm.m00);
    const float scy = static_cast<float>(sm.m01 / sm.m00);

    cv::Mat grow = seed.clone();
    for (int y = 0; y < NORM_S; ++y) {
        const uint8_t* sp = solid.ptr<uint8_t>(y);
        const uint8_t* spec_p = spec.ptr<uint8_t>(y);
        const uint8_t* bp = gblur.ptr<uint8_t>(y);
        uint8_t* gp = grow.ptr<uint8_t>(y);
        for (int x = 0; x < NORM_S; ++x) {
            if (!sp[x] || spec_p[x]) continue;
            const float rn = std::hypot(static_cast<float>(x - cx), static_cast<float>(y - cy)) / rmax;
            if (rn < 0.42f) continue;
            if (std::hypot(static_cast<float>(x - scx), static_cast<float>(y - scy)) > rmax * 0.56f) continue;
            const int bin = std::min(19, std::max(0, static_cast<int>(rn * 20.0f)));
            const float diff = static_cast<float>(bp[x]) - rmed[static_cast<size_t>(bin)];
            if (diff > 12.0f) {
                gp[x] = 255;
            }
        }
    }

    const cv::Mat k5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(grow, grow, cv::MORPH_CLOSE, k5);
    cv::bitwise_and(grow, solid, grow);

    cv::Mat glb, gst, gcen;
    int gnc = cv::connectedComponentsWithStats(grow, glb, gst, gcen);
    std::vector<uint8_t> keep(static_cast<size_t>(std::max(1, gnc)), 0);
    for (int y = 0; y < NORM_S; ++y) {
        const uint8_t* sp = seed.ptr<uint8_t>(y);
        const int* lp = glb.ptr<int>(y);
        for (int x = 0; x < NORM_S; ++x) {
            if (sp[x] && lp[x] > 0) {
                keep[static_cast<size_t>(lp[x])] = 1;
            }
        }
    }
    cv::Mat final_m(NORM_S, NORM_S, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < NORM_S; ++y) {
        const int* lp = glb.ptr<int>(y);
        uint8_t* fp = final_m.ptr<uint8_t>(y);
        for (int x = 0; x < NORM_S; ++x) {
            if (lp[x] > 0 && keep[static_cast<size_t>(lp[x])]) {
                fp[x] = 255;
            }
        }
    }
    cv::morphologyEx(final_m, final_m, cv::MORPH_CLOSE, k5);
    cv::bitwise_and(final_m, solid, final_m);

    cv::Mat smooth_m;
    cv::GaussianBlur(final_m, smooth_m, cv::Size(9, 9), 2.2);
    cv::threshold(smooth_m, smooth_m, 120, 255, cv::THRESH_BINARY);
    cv::bitwise_and(smooth_m, solid, smooth_m);

    const float sx_out = static_cast<float>(orig_w) / static_cast<float>(NORM_S);
    const float sy_out = static_cast<float>(orig_h) / static_cast<float>(NORM_S);
    std::vector<std::vector<cv::Point>> raw_c;
    cv::findContours(smooth_m, raw_c, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& c : raw_c) {
        if (cv::contourArea(c) < 20.0) continue;
        std::vector<cv::Point> poly_norm;
        cv::approxPolyDP(c, poly_norm, 2.0, true);
        if (poly_norm.size() < 3) continue;
        std::vector<cv::Point> poly_orig;
        poly_orig.reserve(poly_norm.size());
        for (const auto& pt : poly_norm) {
            poly_orig.emplace_back(
                std::max(0, std::min(orig_w - 1, static_cast<int>(std::round(pt.x * sx_out)))),
                std::max(0, std::min(orig_h - 1, static_cast<int>(std::round(pt.y * sy_out))))
            );
        }
        out.push_back(std::move(poly_orig));
    }
    return out;
}

/**
 * Fast resolution-independent check for empty conveyor belt or half-entered
 * transit frames so empty dark/brown belt gaps do not trigger metal defects.
 */
bool is_empty_or_transit_belt(const cv::Mat& frame_bgr) {
    if (frame_bgr.empty() || frame_bgr.cols < 32 || frame_bgr.rows < 32) {
        return false;
    }
    cv::Mat small_bgr, small_gray;
    cv::resize(frame_bgr, small_bgr, cv::Size(160, 160), 0, 0, cv::INTER_AREA);
    cv::cvtColor(small_bgr, small_gray, cv::COLOR_BGR2GRAY);

    // Center metallic inspection window [12%..88%]
    const int x0 = 19, y0 = 19, cw = 122, ch = 122;
    cv::Mat center = small_gray(cv::Rect(x0, y0, cw, ch));

    int bright_cnt = 0;
    uint8_t p95_hist[256] = {};
    int hist[256] = {};
    for (int y = 0; y < center.rows; ++y) {
        const uint8_t* gp = center.ptr<uint8_t>(y);
        for (int x = 0; x < center.cols; ++x) {
            const uint8_t v = gp[x];
            if (v > 60) bright_cnt++;
            hist[v]++;
        }
    }
    (void)p95_hist;
    const int n_cen = center.rows * center.cols;
    const float frac_gt60 = static_cast<float>(bright_cnt) / static_cast<float>(n_cen);
    int accum = 0, p95 = 0;
    const int target95 = static_cast<int>(n_cen * 0.95f);
    for (int v = 0; v < 256; ++v) {
        accum += hist[v];
        if (accum >= target95) {
            p95 = v;
            break;
        }
    }
    if (frac_gt60 < 0.28f && p95 < 105) {
        return true;
    }

    // Count vertical belt columns across [8%..92%] height that are uniform dark/brown belt
    const int by0 = 13, by1 = 147;
    const int bh = by1 - by0;
    int dark_cols = 0;
    std::array<uint8_t, 160> col_is_dark{};
    for (int x = 0; x < 160; ++x) {
        double sum_c = 0.0;
        double sum_sq = 0.0;
        int col_hist[256] = {};
        for (int y = by0; y < by1; ++y) {
            const uint8_t v = small_gray.at<uint8_t>(y, x);
            sum_c += static_cast<double>(v);
            sum_sq += static_cast<double>(v) * static_cast<double>(v);
            col_hist[v]++;
        }
        const double col_mean = sum_c / static_cast<double>(bh);
        const double col_var = std::max(0.0, (sum_sq / static_cast<double>(bh)) - col_mean * col_mean);
        const double col_std = std::sqrt(col_var);
        int c_acc = 0, col_p90 = 0;
        const int target90 = static_cast<int>(bh * 0.90f);
        for (int v = 0; v < 256; ++v) {
            c_acc += col_hist[v];
            if (c_acc >= target90) {
                col_p90 = v;
                break;
            }
        }
        if (col_mean < 40.0 && col_std < 10.5 && col_p90 < 52) {
            dark_cols++;
            col_is_dark[static_cast<size_t>(x)] = 1;
        }
    }
    int left_dark = 0;
    while (left_dark < 160 && col_is_dark[static_cast<size_t>(left_dark)]) {
        left_dark++;
    }
    int right_dark = 0;
    while (right_dark < 160 && col_is_dark[static_cast<size_t>(159 - right_dark)]) {
        right_dark++;
    }
    const float dark_col_frac = static_cast<float>(dark_cols) / 160.0f;
    // Horizontal middle band [28%..72%] check: only bypass left_dark for centered
    // belt-letterboxed wide steel strips (top & bottom belt bars + bright steel reaching right edge)
    // so sliding transit frames (e.g. metal_nut entering from right) still trigger transit gate.
    const int rail_m = 6; // 24px in 640 -> 6px in 160
    const int band_h = 20;
    auto is_160_band_dark = [&](int y_start) {
        double sum = 0.0, sum_sq = 0.0;
        int bright = 0;
        const int n = band_h * 160;
        for (int y = y_start; y < y_start + band_h; ++y) {
            const uint8_t* gp = small_gray.ptr<uint8_t>(y);
            for (int x = 0; x < 160; ++x) {
                const double v = static_cast<double>(gp[x]);
                sum += v;
                sum_sq += v * v;
                if (gp[x] > 64) bright++;
            }
        }
        const double m = sum / static_cast<double>(n);
        const double s = std::sqrt(std::max(0.0, (sum_sq / static_cast<double>(n)) - m * m));
        return (static_cast<float>(bright) / static_cast<float>(n) < 0.08f && m < 48.0 && s < 14.0);
    };
    const bool demo_letterbox_bars = is_160_band_dark(rail_m) && is_160_band_dark(160 - rail_m - band_h);

    int mid_bright = 0;
    int right_mid_bright = 0;
    int mid_hist[256] = {};
    const int my0 = 44, my1 = 116;
    const int mx0 = 19, mx1 = 141;
    const int n_mid = (my1 - my0) * (mx1 - mx0);
    for (int y = my0; y < my1; ++y) {
        const uint8_t* gp = small_gray.ptr<uint8_t>(y);
        for (int x = mx0; x < mx1; ++x) {
            const uint8_t v = gp[x];
            if (v > 60) mid_bright++;
            mid_hist[v]++;
        }
        for (int x = 128; x < 156; ++x) {
            if (gp[x] > 80) right_mid_bright++;
        }
    }
    const float mid_frac_gt60 = static_cast<float>(mid_bright) / static_cast<float>(n_mid);
    const float right_mid_frac = static_cast<float>(right_mid_bright) / static_cast<float>((my1 - my0) * 28);
    int mid_acc = 0, mid_p95 = 0;
    const int mid_target95 = static_cast<int>(n_mid * 0.95f);
    for (int v = 0; v < 256; ++v) {
        mid_acc += mid_hist[v];
        if (mid_acc >= mid_target95) {
            mid_p95 = v;
            break;
        }
    }
    if (demo_letterbox_bars && mid_frac_gt60 >= 0.42f && mid_p95 >= 120 && right_mid_frac >= 0.65f) {
        return false;
    }

    // Wide hot-rolled steel strip frames (e.g. GC10-DET 2048x1000) can carry an unlit border
    // margin while >45% of the central field is bright metal sheet; never drop as empty belt.
    if (frame_bgr.cols >= static_cast<int>(frame_bgr.rows * 1.45) && frac_gt60 >= 0.45f) {
        return false;
    }
    return (dark_col_frac >= 0.25f || left_dark >= 30 || right_dark >= 30);
}

/**
 * Returns true for studio light-background isolated fasteners (MVTec AD screw)
 * so Stage-0 (trained without screw) and YOLO plate tiles do not misroute them as welds.
 */
bool is_screw_studio_part(const cv::Mat& frame_bgr) {
    if (frame_bgr.empty() || frame_bgr.cols < 64 || frame_bgr.rows < 64) {
        return false;
    }
    cv::Mat small_bgr, gray, hsv;
    cv::resize(frame_bgr, small_bgr, cv::Size(160, 160), 0, 0, cv::INTER_AREA);
    cv::cvtColor(small_bgr, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(small_bgr, hsv, cv::COLOR_BGR2HSV);

    int perim_total = 0;
    int perim_light_neutral = 0;
    double perim_sum = 0.0;
    double perim_sq = 0.0;
    for (int y = 0; y < 160; ++y) {
        const uint8_t* gp = gray.ptr<uint8_t>(y);
        const cv::Vec3b* hp = hsv.ptr<cv::Vec3b>(y);
        for (int x = 0; x < 160; ++x) {
            if (x >= 14 && x < 146 && y >= 14 && y < 146) continue;
            const uint8_t g = gp[x];
            const uint8_t s = hp[x][1];
            perim_total++;
            perim_sum += static_cast<double>(g);
            perim_sq += static_cast<double>(g) * static_cast<double>(g);
            if (g >= 155 && g <= 245 && s < 25) {
                perim_light_neutral++;
            }
        }
    }
    if (perim_total <= 0) return false;
    const double pm = perim_sum / static_cast<double>(perim_total);
    const double ps = std::sqrt(std::max(0.0, (perim_sq / static_cast<double>(perim_total)) - pm * pm));
    const float p_frac = static_cast<float>(perim_light_neutral) / static_cast<float>(perim_total);
    if (p_frac < 0.80f || pm < 155.0 || ps > 35.0) {
        return false;
    }

    int dark_obj = 0;
    int inner_tot = 0;
    for (int y = 18; y < 142; ++y) {
        const uint8_t* gp = gray.ptr<uint8_t>(y);
        for (int x = 18; x < 142; ++x) {
            inner_tot++;
            if (static_cast<double>(gp[x]) < pm - 45.0) {
                dark_obj++;
            }
        }
    }
    const float obj_frac = static_cast<float>(dark_obj) / static_cast<float>(std::max(1, inner_tot));
    return (obj_frac >= 0.08f && obj_frac <= 0.55f);
}

bool align_screw_canonical_288(
    const cv::Mat& frame_bgr,
    cv::Mat& out_aligned_288,
    cv::Mat& out_mask_288,
    cv::Mat& out_M_inv
) {
    if (frame_bgr.empty()) return false;
    cv::Mat gray;
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
    const int bw = std::max(8, frame_bgr.cols / 16);
    const int bh = std::max(8, frame_bgr.rows / 16);
    std::vector<uint8_t> perim;
    perim.reserve(static_cast<size_t>(frame_bgr.cols) * 4);
    for (int y = 0; y < frame_bgr.rows; ++y) {
        const uint8_t* gp = gray.ptr<uint8_t>(y);
        for (int x = 0; x < frame_bgr.cols; ++x) {
            if (x < bw || x >= frame_bgr.cols - bw || y < bh || y >= frame_bgr.rows - bh) {
                perim.push_back(gp[x]);
            }
        }
    }
    if (perim.empty()) return false;
    std::nth_element(perim.begin(), perim.begin() + perim.size() / 2, perim.end());
    const float bg_med = static_cast<float>(perim[perim.size() / 2]);

    cv::Mat mask = (gray < std::max(80.0f, bg_med - 26.0f));
    cv::Mat kern = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kern);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kern);

    cv::Mat labels, stats, centroids;
    const int nc = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    int best_c = -1;
    int best_a = 0;
    for (int c = 1; c < nc; ++c) {
        const int a = stats.at<int>(c, cv::CC_STAT_AREA);
        if (a > best_a) {
            best_a = a;
            best_c = c;
        }
    }
    if (best_c < 0 || best_a < 100) return false;

    cv::Mat obj = (labels == best_c);
    cv::Moments mom = cv::moments(obj, true);
    if (mom.m00 < 10.0) return false;
    const double cx = mom.m10 / mom.m00;
    const double cy = mom.m01 / mom.m00;
    double angle = 0.5 * std::atan2(2.0 * mom.mu11, mom.mu20 - mom.mu02);
    const double ux = std::cos(angle);
    const double uy = std::sin(angle);

    std::vector<cv::Point> pts;
    cv::findNonZero(obj, pts);
    double min_u = 1e9, max_u = -1e9;
    for (const auto& pt : pts) {
        const double u = (pt.x - cx) * ux + (pt.y - cy) * uy;
        min_u = std::min(min_u, u);
        max_u = std::max(max_u, u);
    }
    const double len = max_u - min_u;
    double neg_w = 0.0, pos_w = 0.0;
    for (const auto& pt : pts) {
        const double u = (pt.x - cx) * ux + (pt.y - cy) * uy;
        const double v = std::abs(-(pt.x - cx) * uy + (pt.y - cy) * ux);
        if (u < min_u + 0.25 * len) neg_w = std::max(neg_w, v);
        if (u > max_u - 0.25 * len) pos_w = std::max(pos_w, v);
    }
    if (pos_w > neg_w) {
        angle += CV_PI;
    }

    const double scale = (288.0 * 0.90) / (std::min(frame_bgr.rows, frame_bgr.cols) * 0.96);
    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(static_cast<float>(cx), static_cast<float>(cy)), angle * 180.0 / CV_PI, scale);
    M.at<double>(0, 2) += (144.0 - cx);
    M.at<double>(1, 2) += (144.0 - cy);
    cv::warpAffine(frame_bgr, out_aligned_288, M, cv::Size(288, 288), cv::INTER_AREA, cv::BORDER_REPLICATE);

    cv::Mat obj_u8;
    obj.convertTo(obj_u8, CV_8U, 255.0);
    cv::warpAffine(obj_u8, out_mask_288, M, cv::Size(288, 288), cv::INTER_NEAREST);
    cv::dilate(out_mask_288, out_mask_288, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(17, 17)));
    cv::invertAffineTransform(M, out_M_inv);
    return true;
}

/**
 * Returns true if a candidate ROI sits entirely over featureless dark/brown belt background.
 */
bool is_belt_background_roi(const cv::Mat& frame_bgr, const cv::Rect& box) {
    if (frame_bgr.empty() || box.width <= 0 || box.height <= 0) return false;
    cv::Rect clamped = box & cv::Rect(0, 0, frame_bgr.cols, frame_bgr.rows);
    if (clamped.width < 4 || clamped.height < 4) return false;

    cv::Mat roi_gray;
    cv::cvtColor(frame_bgr(clamped), roi_gray, cv::COLOR_BGR2GRAY);
    cv::Scalar mean_s, std_s;
    cv::meanStdDev(roi_gray, mean_s, std_s);
    int bright = cv::countNonZero(roi_gray > 64);
    float bright_frac = static_cast<float>(bright) / static_cast<float>(roi_gray.total());
    return (bright_frac < 0.08f && mean_s[0] < 48.0 && std_s[0] < 14.0);
}

} // namespace

struct HailoVisionEngine::Impl {
    std::unique_ptr<Ort::Env> ort_env;
    OrtSessionHolder yolo_ort;
    OrtSessionHolder stage0_ort;
    OrtSessionHolder screw_feat_ort;
    PatchCoreEngine patchcore;
    std::vector<float> screw_bank_descs;
    int screw_bank_n = 0;

    bool load_screw_bank(const std::string& bin_path) {
        std::ifstream ifs(bin_path, std::ios::binary);
        if (!ifs.is_open()) return false;
        uint32_t magic = 0, ver = 0, nvec = 0, dim = 0;
        ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        ifs.read(reinterpret_cast<char*>(&ver), sizeof(ver));
        ifs.read(reinterpret_cast<char*>(&nvec), sizeof(nvec));
        ifs.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        if (!ifs || magic != 0x57524353U || ver != 1U || dim != 384U || nvec == 0 || nvec > 16384U) {
            return false;
        }
        screw_bank_descs.resize(static_cast<size_t>(nvec) * dim);
        ifs.read(reinterpret_cast<char*>(screw_bank_descs.data()),
                 static_cast<std::streamsize>(screw_bank_descs.size() * sizeof(float)));
        if (!ifs) {
            screw_bank_descs.clear();
            screw_bank_n = 0;
            return false;
        }
        screw_bank_n = static_cast<int>(nvec);
        std::cout << "[ScrewPatchCore] Loaded screw coreset bank (" << screw_bank_n << "x" << dim
                  << ") from " << bin_path << std::endl;
        return true;
    }

    bool run_screw_patchcore(
        const cv::Mat& frame_bgr,
        const qc::rules::QcRulesEngine& rules,
        VisionInferenceResult& res,
        float& pre_ms,
        float& infer_ms,
        float& post_ms,
        const std::function<void(cv::Mat&, const VisionInferenceResult&)>& draw_overlay_fn
    ) {
        if (!screw_feat_ort.ready || screw_bank_n <= 0 || frame_bgr.empty()) {
            return false;
        }

        auto t_pre0 = SteadyClock::now();
        cv::Mat al288, wm288, m_inv;
        if (!align_screw_canonical_288(frame_bgr, al288, wm288, m_inv)) {
            return false;
        }

        // Convert aligned 288x288 BGR -> ImageNet normalized NCHW float32
        cv::Mat rgb_u8;
        cv::cvtColor(al288, rgb_u8, cv::COLOR_BGR2RGB);
        std::vector<float> tensor_vals(static_cast<size_t>(3 * 288 * 288));
        constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
        constexpr float kInvStd[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};
        const size_t plane = 288U * 288U;
        for (int y = 0; y < 288; ++y) {
            const cv::Vec3b* row = rgb_u8.ptr<cv::Vec3b>(y);
            for (int x = 0; x < 288; ++x) {
                const size_t idx = static_cast<size_t>(y) * 288U + static_cast<size_t>(x);
                tensor_vals[idx]             = (static_cast<float>(row[x][0]) * (1.0f / 255.0f) - kMean[0]) * kInvStd[0];
                tensor_vals[plane + idx]     = (static_cast<float>(row[x][1]) * (1.0f / 255.0f) - kMean[1]) * kInvStd[1];
                tensor_vals[2 * plane + idx] = (static_cast<float>(row[x][2]) * (1.0f / 255.0f) - kMean[2]) * kInvStd[2];
            }
        }
        std::vector<int64_t> in_shape = {1, 3, 288, 288};
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            mem_info, tensor_vals.data(), tensor_vals.size(), in_shape.data(), in_shape.size()
        );
        pre_ms += elapsed_ms(t_pre0, SteadyClock::now());

        auto t_inf0 = SteadyClock::now();
        auto outputs = screw_feat_ort.session->Run(
            Ort::RunOptions{nullptr},
            screw_feat_ort.input_cstrs.data(), &in_tensor, 1,
            screw_feat_ort.output_cstrs.data(), screw_feat_ort.output_cstrs.size()
        );
        if (outputs.empty()) return false;
        const float* fmap = outputs[0].GetTensorData<float>(); // (1, 384, 36, 36)

        cv::Mat mask_small;
        cv::resize(wm288, mask_small, cv::Size(36, 36), 0, 0, cv::INTER_NEAREST);

        constexpr int G = 36;
        constexpr int C = 384;
        constexpr int HW = G * G;
        cv::Mat dist_grid(G, G, CV_32FC1, cv::Scalar(0.0f));
        const int nb = screw_bank_n;
        const float* bank_ptr = screw_bank_descs.data();

        // The vision thread is pinned to one core; extra OpenMP workers inherit
        // that affinity and only add scheduling overhead to this small grid.
        #pragma omp parallel for num_threads(1) schedule(static)
        for (int gy = 1; gy < G - 1; ++gy) {
            float* d_row = dist_grid.ptr<float>(gy);
            const uint8_t* m_row = mask_small.ptr<uint8_t>(gy);
            for (int gx = 1; gx < G - 1; ++gx) {
                if (m_row[gx] == 0) continue;
                float q_vec[C];
                const int sp = gy * G + gx;
                for (int c = 0; c < C; ++c) {
                    q_vec[c] = fmap[static_cast<size_t>(c) * HW + sp];
                }
                float best_sim = -1.0f;
                for (int b = 0; b < nb; ++b) {
                    const float* b_vec = bank_ptr + static_cast<size_t>(b) * C;
                    const float dot = descriptor_dot(q_vec, b_vec, C);
                    if (dot > best_sim) best_sim = dot;
                }
                d_row[gx] = std::sqrt(std::max(0.0f, 2.0f - 2.0f * best_sim));
            }
        }
        infer_ms += elapsed_ms(t_inf0, SteadyClock::now());

        auto t_post0 = SteadyClock::now();
        cv::Mat d_blur;
        cv::GaussianBlur(dist_grid, d_blur, cv::Size(3, 3), 0.7, 0.7, cv::BORDER_REPLICATE);

        double max_sm_d = 0.0;
        cv::Point peak_pt(18, 18);
        cv::minMaxLoc(d_blur, nullptr, &max_sm_d, nullptr, &peak_pt);
        const float max_sm = static_cast<float>(max_sm_d);

        constexpr float kScrewThresh = 0.383f;
        const bool is_anom = (max_sm >= kScrewThresh);

        res.family = "machined_part";
        res.family_conf = 0.98f;
        res.is_not_metal = false;
        res.anomaly.threshold = kScrewThresh;
        res.anomaly.bank_size = screw_bank_n;
        res.anomaly.is_anomalous = is_anom;
        res.anomaly.score = is_anom
            ? std::clamp(0.58f + (max_sm - kScrewThresh) * 2.2f, 0.58f, 0.97f)
            : std::clamp(max_sm * 0.90f, 0.12f, 0.38f);

        std::vector<DetectionBox> accepted;
        std::vector<qc::rules::RawBox> raw_boxes;
        const int w_orig = frame_bgr.cols;
        const int h_orig = frame_bgr.rows;
        float area_override = 0.0f;

        if (is_anom) {
            std::string label = "thread_top";
            std::string group = "machining_tear";
            if (peak_pt.x >= 31) {
                label = "manipulated_front";
                group = "deformation";
            } else if (peak_pt.x <= 9) {
                label = "scratch_head";
                group = "scratch";
            } else if (peak_pt.x <= 18) {
                label = "scratch_neck";
                group = "scratch";
            } else if (peak_pt.x <= 24) {
                label = "thread_side";
                group = "machining_tear";
            }

            // Build tightly localized relative response in canonical 36x36 space for box estimation only
            const float peak_exc = std::max(0.03f, max_sm - 0.34f);
            cv::Mat heat_36(G, G, CV_32FC1, cv::Scalar(0.0f));
            for (int gy = 0; gy < G; ++gy) {
                const float* dr = d_blur.ptr<float>(gy);
                float* hr = heat_36.ptr<float>(gy);
                for (int gx = 0; gx < G; ++gx) {
                    const float dist_g = std::hypot(static_cast<float>(gx - peak_pt.x), static_cast<float>(gy - peak_pt.y));
                    const float radial = std::exp(-(dist_g * dist_g) / (2.0f * 2.20f * 2.20f));
                    const float rel = std::max(0.0f, (dr[gx] - 0.34f) / peak_exc);
                    hr[gx] = rel * radial;
                }
            }
            cv::Mat heat_288, heat_orig;
            cv::resize(heat_36, heat_288, cv::Size(288, 288), 0, 0, cv::INTER_CUBIC);
            cv::warpAffine(heat_288, heat_orig, m_inv, frame_bgr.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0.0f));

            // Screw inspection uses crisp box-only visualization without JET heatmap overlay
            res.anomaly.heatmap.release();
            res.anomaly.peak_ratio = 0.0f;

            // Clip hotspot mask against metallic screw silhouette so studio background/shadow stays outside box
            cv::Mat silhouette_gray;
            cv::cvtColor(frame_bgr, silhouette_gray, cv::COLOR_BGR2GRAY);
            const cv::Scalar border_m = cv::mean(silhouette_gray.rowRange(0, std::max(4, h_orig / 20)));
            cv::Mat body_mask = (silhouette_gray < std::max(85.0, border_m[0] - 78.0));

            cv::Mat hot_mask = (heat_orig >= 0.40f) & body_mask;
            cv::Rect hot_r = cv::boundingRect(hot_mask);
            if (hot_r.empty()) {
                // Fallback centered on mapped peak_pt
                const double ax = peak_pt.x * 8.0 + 4.0;
                const double ay = peak_pt.y * 8.0 + 4.0;
                const int ox = static_cast<int>(std::round(m_inv.at<double>(0, 0) * ax + m_inv.at<double>(0, 1) * ay + m_inv.at<double>(0, 2)));
                const int oy = static_cast<int>(std::round(m_inv.at<double>(1, 0) * ax + m_inv.at<double>(1, 1) * ay + m_inv.at<double>(1, 2)));
                const int rad = std::max(28, std::min(w_orig, h_orig) / 18);
                hot_r = cv::Rect(ox - rad, oy - rad, rad * 2, rad * 2);
            }
            const int pad = std::max(12, std::min(w_orig, h_orig) / 85);
            const int bx1 = std::max(0, hot_r.x - pad);
            const int by1 = std::max(0, hot_r.y - pad);
            const int bx2 = std::min(w_orig, hot_r.x + hot_r.width + pad);
            const int by2 = std::min(h_orig, hot_r.y + hot_r.height + pad);

            res.anomaly.hotspot_x1 = bx1;
            res.anomaly.hotspot_y1 = by1;
            res.anomaly.hotspot_x2 = bx2;
            res.anomaly.hotspot_y2 = by2;

            DetectionBox db;
            db.x1 = bx1; db.y1 = by1; db.x2 = bx2; db.y2 = by2;
            db.confidence = std::min(0.99f, res.anomaly.score);
            db.class_id = rules.get_group_rule(group).id;
            db.label = label;
            db.group = group;
            db.bgr_color = rules.get_group_bgr_color(group);
            accepted.push_back(db);

            qc::rules::RawBox rb;
            rb.x1 = static_cast<float>(bx1);
            rb.y1 = static_cast<float>(by1);
            rb.x2 = static_cast<float>(bx2);
            rb.y2 = static_cast<float>(by2);
            rb.conf = db.confidence;
            rb.class_id = db.class_id;
            rb.label = db.label;
            rb.group = db.group;
            raw_boxes.push_back(rb);

            const int hot_px = cv::countNonZero(hot_mask);
            if (w_orig > 0 && h_orig > 0) {
                area_override = static_cast<float>(std::min(100.0, (hot_px * 100.0) / static_cast<double>(w_orig * h_orig)));
            }
        }

        auto analysis = rules.analyze_detections(
            raw_boxes, w_orig, h_orig, res.family, res.anomaly.score, res.anomaly.is_anomalous
        );
        for (auto& db : accepted) {
            for (const auto& gs : analysis.groups) {
                if (gs.group == db.group) { db.severity = gs.severity; break; }
            }
            if (db.severity.empty() || db.severity == "NONE") db.severity = analysis.overall_severity;
        }
        res.detections = std::move(accepted);
        res.group_summaries = std::move(analysis.groups);
        res.has_defects = analysis.has_defects;
        res.primary_label = analysis.primary_label;
        res.primary_group = analysis.primary_group;
        res.display_title = analysis.display_title;
        res.severity = analysis.overall_severity;
        res.verdict = analysis.verdict;
        res.highest_confidence = analysis.highest_conf;
        res.union_area_pct = (area_override > 0.0f) ? area_override : analysis.total_area_pct;
        res.total_boxes = static_cast<int>(res.detections.size());
        res.spread = analysis.spread;

        res.annotated_frame = frame_bgr.clone();
        draw_overlay_fn(res.annotated_frame, res);
        cv::imencode(".jpg", res.annotated_frame, res.encoded_jpeg, {cv::IMWRITE_JPEG_QUALITY, 85});

        post_ms += elapsed_ms(t_post0, SteadyClock::now());
        res.timings_ms = {pre_ms, infer_ms, post_ms, pre_ms + infer_ms + post_ms};
        res.cv_latency_ms = res.timings_ms.total;
        return true;
    }

#ifdef HAVE_HAILORT
    std::unique_ptr<hailort::VDevice> vdevice;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> network_group;
    std::unique_ptr<hailort::InferVStreams> infer_pipeline;
    std::string input_vstream_name;
    std::string output_vstream_name;
    std::vector<std::vector<float>> hailo_out_bufs;  // one per output vstream, reused
    std::vector<float> hailo_packed;                  // decoded [1, 4+nc, N] tensor, reused
    bool hailo_error_logged = false;
    bool hailo_layout_logged = false;
#endif
    bool hailo_configured = false;

    bool run_yolo_pass_ort(
        const cv::Mat& crop_bgr,
        int input_size,
        int num_classes,
        float min_conf,
        const TileWindow& tile,
        int orig_w,
        int orig_h,
        const qc::rules::QcRulesEngine& rules,
        std::vector<ProjectedCandidate>& out_candidates,
        float& accum_pre_ms,
        float& accum_infer_ms
    ) {
        if (!yolo_ort.ready) return false;

        auto t_pre0 = SteadyClock::now();
        std::vector<float> tensor_vals;
        LetterboxMeta meta = bgr_to_nchw_letterbox(crop_bgr, input_size, tensor_vals, true);
        std::vector<int64_t> in_shape = {1, 3, input_size, input_size};
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            mem_info, tensor_vals.data(), tensor_vals.size(),
            in_shape.data(), in_shape.size()
        );
        auto t_pre1 = SteadyClock::now();
        accum_pre_ms += elapsed_ms(t_pre0, t_pre1);

        auto t_inf0 = SteadyClock::now();
        auto outputs = yolo_ort.session->Run(
            Ort::RunOptions{nullptr},
            yolo_ort.input_cstrs.data(), &in_tensor, 1,
            yolo_ort.output_cstrs.data(), yolo_ort.output_cstrs.size()
        );
        auto t_inf1 = SteadyClock::now();
        accum_infer_ms += elapsed_ms(t_inf0, t_inf1);

        if (outputs.empty()) return false;
        auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const float* out_data = outputs[0].GetTensorData<float>();
        decode_yolo_tensor_into(
            out_data, out_shape, num_classes, input_size, min_conf,
            tile, meta, orig_w, orig_h, rules, out_candidates
        );
        return true;
    }

    bool run_yolo_pass_hailo(
        const cv::Mat& crop_bgr,
        int input_size,
        int num_classes,
        float min_conf,
        const TileWindow& tile,
        int orig_w,
        int orig_h,
        const qc::rules::QcRulesEngine& rules,
        std::vector<ProjectedCandidate>& out_candidates,
        float& accum_pre_ms,
        float& accum_infer_ms
    ) {
#ifdef HAVE_HAILORT
        if (!hailo_configured || !infer_pipeline) return false;
        auto t_pre0 = SteadyClock::now();
        std::vector<uint8_t> hwc_input;
        LetterboxMeta meta;
        bgr_to_hwc_uint8(crop_bgr, input_size, hwc_input, meta);
        auto t_pre1 = SteadyClock::now();
        accum_pre_ms += elapsed_ms(t_pre0, t_pre1);

        auto out_vstreams = infer_pipeline->get_output_vstreams();
        if (out_vstreams.empty()) return false;

        // HailoRT requires a buffer for EVERY output vstream. A .hef compiled without
        // on-chip NMS exposes 6 raw YOLO heads (3x box 64ch + 3x cls), not one tensor.
        hailo_out_bufs.resize(out_vstreams.size());
        std::map<std::string, hailort::MemoryView> in_map;
        std::map<std::string, hailort::MemoryView> out_map;
        in_map.emplace(input_vstream_name, hailort::MemoryView(hwc_input.data(), hwc_input.size()));
        for (size_t i = 0; i < out_vstreams.size(); ++i) {
            auto& vs = out_vstreams[i].get();
            hailo_out_bufs[i].resize(vs.get_frame_size() / sizeof(float));
            out_map.emplace(vs.name(), hailort::MemoryView(
                hailo_out_bufs[i].data(), hailo_out_bufs[i].size() * sizeof(float)));
        }

        auto t_inf0 = SteadyClock::now();
        hailo_status status = infer_pipeline->infer(in_map, out_map, 1);
        auto t_inf1 = SteadyClock::now();
        accum_infer_ms += elapsed_ms(t_inf0, t_inf1);

        if (status != HAILO_SUCCESS) {
            if (!hailo_error_logged) {
                std::cerr << "[HailoVisionEngine] Hailo infer failed (status=" << status
                          << "); falling back to ONNX Runtime for this pass." << std::endl;
                hailo_error_logged = true;
            }
            return false;
        }

        if (out_vstreams.size() > 1) {
            auto t_post0 = SteadyClock::now();
            std::vector<YoloRawHead> heads;
            heads.reserve(out_vstreams.size());
            for (size_t i = 0; i < out_vstreams.size(); ++i) {
                auto info = out_vstreams[i].get().get_info();
                heads.push_back({hailo_out_bufs[i].data(),
                                 static_cast<int>(info.shape.height),
                                 static_cast<int>(info.shape.width),
                                 static_cast<int>(info.shape.features)});
                if (!hailo_layout_logged) {
                    const auto order = out_vstreams[i].get().get_user_buffer_format().order;
                    std::cout << "[HailoVisionEngine] output " << out_vstreams[i].get().name()
                              << " " << info.shape.height << "x" << info.shape.width
                              << "x" << info.shape.features
                              << " host_order=" << static_cast<int>(order)
                              << ((order == HAILO_FORMAT_ORDER_NHWC || order == HAILO_FORMAT_ORDER_FCR) ? " (NHWC-compatible)" : " (unexpected order!)")
                              << std::endl;
                }
            }
            hailo_layout_logged = true;

            std::vector<int64_t> packed_shape;
            if (!assemble_yolo_raw_heads(heads, num_classes, input_size, min_conf,
                                         hailo_packed, packed_shape)) {
                if (!hailo_error_logged) {
                    std::cerr << "[HailoVisionEngine] Unrecognised .hef output heads (expected "
                              << "box 64ch + cls " << num_classes << "ch per stride); "
                              << "falling back to ONNX Runtime." << std::endl;
                    hailo_error_logged = true;
                }
                return false;
            }
            decode_yolo_tensor_into(
                hailo_packed.data(), packed_shape, num_classes, input_size, min_conf,
                tile, meta, orig_w, orig_h, rules, out_candidates
            );
            accum_infer_ms += elapsed_ms(t_post0, SteadyClock::now());
            return true;
        }

        // Single-output .hef (e.g. exported with a fused [1, 4+nc, N] head).
        auto out_info = out_vstreams[0].get().get_info();
        std::vector<float>& out_buf = hailo_out_bufs[0];
        std::vector<int64_t> shape = {
            1,
            static_cast<int64_t>(out_info.shape.features),
            static_cast<int64_t>(out_info.shape.height * out_info.shape.width)
        };
        if (shape[1] <= 0 || shape[2] <= 0) {
            int dims_c = 4 + num_classes;
            shape = {1, dims_c, static_cast<int64_t>(out_buf.size()) / dims_c};
        }
        decode_yolo_tensor_into(
            out_buf.data(), shape, num_classes, input_size, min_conf,
            tile, meta, orig_w, orig_h, rules, out_candidates
        );
        return true;
#else
        (void)crop_bgr; (void)input_size; (void)num_classes; (void)min_conf;
        (void)tile; (void)orig_w; (void)orig_h; (void)rules;
        (void)out_candidates; (void)accum_pre_ms; (void)accum_infer_ms;
        return false;
#endif
    }
};

HailoVisionEngine::HailoVisionEngine() : impl_(std::make_unique<Impl>()) {}
HailoVisionEngine::~HailoVisionEngine() = default;

bool HailoVisionEngine::init(const std::string& config_path) {
    return init("", "", "", config_path);
}

bool HailoVisionEngine::init(
    const std::string& hef_path,
    const std::string& yolo_onnx_path,
    const std::string& stage0_onnx_path,
    const std::string& config_path
) {
    rules_.load_config(config_path);
    const auto& cfg = rules_.config();

    hef_path_ = resolve_existing_file(hef_path, cfg.vision.hef_path);
    yolo_onnx_path_ = resolve_existing_file(yolo_onnx_path, cfg.vision.onnx_path);
    stage0_onnx_path_ = resolve_existing_file(stage0_onnx_path, cfg.stage0.model_path);

    is_hailo_ready_ = false;
    vision_backend_ = "onnxruntime_arm64";
    hardware_name_ = "ARM64 ONNX Runtime (Cortex-A76)";

    // 1. Check Hailo-8L hardware node (/dev/hailo0) + .hef
    std::error_code ec;
    const bool hailo_node_present = fs::exists(cfg.vision.hailo_device_node, ec);
    const bool hef_present = !hef_path_.empty() && fs::exists(hef_path_, ec);

    if (hailo_node_present && hef_present) {
#ifdef HAVE_HAILORT
        try {
            auto vdevice_exp = hailort::VDevice::create();
            if (vdevice_exp) {
                impl_->vdevice = vdevice_exp.release();
                auto hef_exp = hailort::Hef::create(hef_path_);
                if (hef_exp) {
                    hailort::Hef hef = hef_exp.release();
                    auto configure_params = impl_->vdevice->create_configure_params(hef);
                    if (configure_params) {
                        auto network_groups = impl_->vdevice->configure(hef, configure_params.value());
                        if (network_groups && !network_groups.value().empty()) {
                            impl_->network_group = network_groups.value()[0];
                            auto in_params = impl_->network_group->make_input_vstream_params(
                                true, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
                            auto out_params = impl_->network_group->make_output_vstream_params(
                                false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
                            if (in_params && out_params) {
                                auto pipe_exp = hailort::InferVStreams::create(
                                    *impl_->network_group, in_params.value(), out_params.value());
                                if (pipe_exp) {
                                    impl_->infer_pipeline = std::make_unique<hailort::InferVStreams>(pipe_exp.release());
                                    auto ins = impl_->infer_pipeline->get_input_vstreams();
                                    auto outs = impl_->infer_pipeline->get_output_vstreams();
                                    if (!ins.empty() && !outs.empty()) {
                                        impl_->input_vstream_name = ins[0].get().name();
                                        impl_->output_vstream_name = outs[0].get().name();
                                        impl_->hailo_configured = true;
                                        is_hailo_ready_ = true;
                                        vision_backend_ = "hailo8l_hef";
                                        hardware_name_ = "Hailo-8L (13 TOPS AI HAT)";
                                        std::cout << "[HailoVisionEngine] Hailo-8L NPU active (" << hef_path_ << ")" << std::endl;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[HailoVisionEngine] Hailo init notice: " << e.what()
                      << ". Falling back to ARM64 ONNX Runtime." << std::endl;
        }
#else
        std::cout << "[HailoVisionEngine] /dev/hailo0 detected, compiled without HAVE_HAILORT; using ARM64 ONNX Runtime." << std::endl;
#endif
    } else {
        std::cout << "[HailoVisionEngine] Hailo device node (" << cfg.vision.hailo_device_node
                  << ") absent; using ARM64 ONNX Runtime fallback (" << yolo_onnx_path_ << ")." << std::endl;
    }

    // 2. Initialize ONNX Runtime for Stage-0 router + unified YOLO11 fallback
    impl_->ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "BharatUnifiedVision");
    Ort::SessionOptions opts;
    const int host_cores = std::max(1, qc::hardware::get_cpu_core_count());
    const int requested_threads = cfg.vision.ort_intra_op_threads;
    const int automatic_threads = is_hailo_ready_ ? 1 : 2;
    const int threads = std::clamp(requested_threads > 0 ? requested_threads : automatic_threads,
                                   1, host_cores);
    // Pi + Hailo: CPU stages stay on core 1, HTTP on 0, Llama on 2/3.
    // CPU fallback can also use core 0. Explicit 3+ thread overrides share
    // Llama's cores and require end-to-end hardware benchmarking.
    std::vector<int> ort_cores;
    if (threads == 1) {
        ort_cores.push_back(qc::hardware::CORE_VISION);
    } else {
        for (int core = 0; core < threads; ++core) ort_cores.push_back(core);
    }
    qc::hardware::pin_current_thread_mask(ort_cores);
    std::cout << "[Vision CPU] ORT threads=" << threads
              << (requested_threads > 0 ? " (explicit override)" : " (automatic)")
              << ", Hailo=" << (is_hailo_ready_ ? "active" : "inactive")
              << ", " << qc::hardware::get_current_affinity_string() << std::endl;
    if (threads > 2) {
        std::cout << "[Vision CPU] Override shares CPU cores with Llama; benchmark joint latency and throughput."
                  << std::endl;
    }
    opts.SetIntraOpNumThreads(threads);
    // Three resident sessions share the vision budget. Idle pools must sleep
    // while another model, the web server, or Llama is running.
    opts.AddConfigEntry("session.intra_op.allow_spinning", "0");
    opts.AddConfigEntry("session.inter_op.allow_spinning", "0");
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    impl_->stage0_ort.load(*impl_->ort_env, opts, stage0_onnx_path_, "Stage0 Router (224)");
    impl_->yolo_ort.load(*impl_->ort_env, opts, yolo_onnx_path_, "Unified YOLO11 (640)");

    // 3. Initialize PatchCore Anomaly Safety Net
    if (cfg.patchcore.enabled) {
        std::string refs_dir = resolve_dir_path(cfg.patchcore.refs_dir, "models/patchcore_refs");
        std::string banks_dir = resolve_dir_path(cfg.patchcore.banks_dir, "models/patchcore_banks");
        impl_->patchcore.init(
            refs_dir,
            banks_dir,
            cfg.patchcore.anomaly_threshold,
            cfg.patchcore.reference_limit_per_family
        );
        // Subtle machined-part defects (e.g. small color spots on compressed conveyor frames)
        // score 0.56-0.61 while clean machined parts score ~0.33.
        impl_->patchcore.set_family_threshold("machined_part", std::min(cfg.patchcore.anomaly_threshold, 0.50f));

        std::string screw_onnx = resolve_existing_file("models/screw_patchcore_feat.onnx", "models/screw_patchcore_feat.onnx");
        std::string screw_bin = resolve_existing_file(banks_dir + "/screw.bin", "models/patchcore_banks/screw.bin");
        if (!screw_onnx.empty() && !screw_bin.empty()) {
            impl_->screw_feat_ort.load(*impl_->ort_env, opts, screw_onnx, "Screw PatchCore ResNet18");
            impl_->load_screw_bank(screw_bin);
        }
    }

    return is_hailo_ready_ || impl_->yolo_ort.ready;
}

Stage0Result HailoVisionEngine::classify_stage0(const cv::Mat& frame_bgr) {
    Stage0Result res;
    const auto& cfg = rules_.config();
    res.family = cfg.default_family;
    res.family_conf = 0.0f;

    if (frame_bgr.empty()) {
        return res;
    }

    auto t0 = SteadyClock::now();
    std::vector<float> probs;
    const auto& classes = cfg.stage0.classes;

    if (impl_->stage0_ort.ready) {
        auto t_pre0 = SteadyClock::now();
        std::vector<float> input_vals;
        int sz = std::max(64, cfg.stage0.input_size);
        bgr_to_nchw_letterbox(frame_bgr, sz, input_vals, false);

        std::vector<int64_t> in_shape = {1, 3, sz, sz};
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_vals.data(), input_vals.size(),
            in_shape.data(), in_shape.size()
        );
        res.pre_ms = elapsed_ms(t_pre0, SteadyClock::now());

        auto t_inf0 = SteadyClock::now();
        auto outputs = impl_->stage0_ort.session->Run(
            Ort::RunOptions{nullptr},
            impl_->stage0_ort.input_cstrs.data(), &in_tensor, 1,
            impl_->stage0_ort.output_cstrs.data(), impl_->stage0_ort.output_cstrs.size()
        );
        res.infer_ms = elapsed_ms(t_inf0, SteadyClock::now());

        if (!outputs.empty()) {
            const float* raw = outputs[0].GetTensorData<float>();
            size_t out_elems = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
            size_t n_cls = std::min(classes.size(), out_elems);
            probs = softmax_if_needed(raw, n_cls);
        }
    }

    auto t1 = SteadyClock::now();
    res.latency_ms = elapsed_ms(t0, t1);

    if (probs.empty()) {
        res.family = cfg.default_family;
        res.family_conf = 0.50f;
        return res;
    }

    size_t best_idx = 0;
    float best_prob = -1.0f;
    float not_metal_prob = 0.0f;
    size_t best_metal_idx = 0;
    float best_metal_prob = -1.0f;

    for (size_t i = 0; i < probs.size(); ++i) {
        if (probs[i] > best_prob) {
            best_prob = probs[i];
            best_idx = i;
        }
        if (classes[i] == cfg.reject_family) {
            not_metal_prob = probs[i];
        } else if (probs[i] > best_metal_prob) {
            best_metal_prob = probs[i];
            best_metal_idx = i;
        }
    }

    // Visual safeguard via ARM NEON HSV saturation gate
    bool visual_non_metal = false;
    if (cfg.stage0.visual_gate_enabled) {
        auto gate = MetallicSurfaceGate::evaluate_roi_bgr(
            frame_bgr.data, frame_bgr.cols, frame_bgr.rows, static_cast<int>(frame_bgr.step),
            0, 0, frame_bgr.cols, frame_bgr.rows,
            cfg.stage0.frame_max_mean_s,
            cfg.stage0.frame_max_colorful_ratio
        );
        if (!gate.is_metallic &&
            (not_metal_prob >= 0.25f || best_metal_prob < cfg.stage0.confidence_threshold)) {
            visual_non_metal = true;
        }
    }

    if ((classes[best_idx] == cfg.reject_family && best_prob >= cfg.stage0.not_metal_threshold) ||
        visual_non_metal) {
        res.family = cfg.reject_family;
        res.family_conf = std::max(not_metal_prob, 1.0f - std::max(0.0f, best_metal_prob));
        res.is_not_metal = true;
        return res;
    }

    if (classes[best_idx] == cfg.reject_family && best_metal_prob >= 0.25f) {
        res.family = classes[best_metal_idx];
        res.family_conf = best_metal_prob;
        res.is_not_metal = false;
        return res;
    }

    if (best_metal_prob >= cfg.stage0.confidence_threshold) {
        res.family = classes[best_metal_idx];
        res.family_conf = best_metal_prob;
    } else {
        res.family = (best_metal_prob > 0.0f) ? classes[best_metal_idx] : cfg.default_family;
        res.family_conf = std::max(0.0f, best_metal_prob);
    }
    res.is_not_metal = (res.family == cfg.reject_family);
    return res;
}

VisionInferenceResult HailoVisionEngine::infer(
    const cv::Mat& frame_bgr,
    float conf_thresh,
    float iou_thresh,
    int conveyor_phase
) {
    auto t_total_start = SteadyClock::now();
    const auto& cfg = rules_.config();

    VisionInferenceResult res;
    res.frame_id = ++frame_counter_;
    res.vision_backend = vision_backend_;
    res.hardware_device = hardware_name_;
    res.anomaly.threshold = cfg.patchcore.anomaly_threshold;

    if (frame_bgr.empty()) {
        res.display_title = "NO FRAME";
        res.timings_ms.total = elapsed_ms(t_total_start, SteadyClock::now());
        res.cv_latency_ms = res.timings_ms.total;
        return res;
    }

    if (conveyor_phase == 1 || (conveyor_phase == 0 && is_empty_or_transit_belt(frame_bgr))) {
        res.family = "conveyor_belt";
        res.family_conf = 1.0f;
        res.has_defects = false;
        res.primary_label = "no_part";
        res.primary_group = "NONE";
        res.display_title = "CONVEYOR CLEAR · NO PART";
        res.severity = "NONE";
        res.verdict = rules_.severity_to_verdict("NONE");
        res.annotated_frame = frame_bgr.clone();
        draw_unified_overlay(res.annotated_frame, res);
        cv::imencode(".jpg", res.annotated_frame, res.encoded_jpeg, {cv::IMWRITE_JPEG_QUALITY, 88});
        const float post_t = elapsed_ms(t_total_start, SteadyClock::now());
        res.timings_ms = {0.0f, 0.0f, post_t, post_t};
        res.cv_latency_ms = res.timings_ms.total;
        return res;
    }

    {
        std::lock_guard<std::mutex> lock(frame_cache_mutex_);
        last_raw_frame_ = frame_bgr.clone();
    }

    float pre_ms = 0.0f;
    float infer_ms = 0.0f;
    float post_ms = 0.0f;

    // Fast path for isolated studio machined screws using aligned ResNet18 PatchCore coreset
    if (is_screw_studio_part(frame_bgr) && impl_->screw_feat_ort.ready && impl_->screw_bank_n > 0) {
        if (impl_->run_screw_patchcore(
                frame_bgr, rules_, res, pre_ms, infer_ms, post_ms,
                [this](cv::Mat& f, const VisionInferenceResult& r) { draw_unified_overlay(f, r); })) {
            std::lock_guard<std::mutex> lock(frame_cache_mutex_);
            last_family_ = res.family;
            return res;
        }
    }

    // Detect belt-letterboxed wide steel strip (e.g. GC10-DET centered in 640x640 DEMO video),
    // skipping the top/bottom 24px conveyor guide rails so dark belt margins are measured accurately.
    const int rail_margin = std::min(24, frame_bgr.rows / 12);
    const int belt_band_h = std::max(12, frame_bgr.rows / 6 - rail_margin);
    const bool has_top_belt = is_belt_background_roi(frame_bgr, cv::Rect(0, rail_margin, frame_bgr.cols, belt_band_h));
    const bool has_bot_belt = is_belt_background_roi(frame_bgr, cv::Rect(0, frame_bgr.rows - rail_margin - belt_band_h, frame_bgr.cols, belt_band_h));
    const bool is_demo_letterbox_strip = (has_top_belt && has_bot_belt);

    // 1. Stage-0 Material / View Router (crop out dark top/bottom belt bars on letterboxed strips)
    Stage0Result s0 = is_demo_letterbox_strip
        ? classify_stage0(frame_bgr(cv::Rect(0, frame_bgr.rows / 4, frame_bgr.cols, std::max(32, frame_bgr.rows / 2))))
        : classify_stage0(frame_bgr);
    pre_ms += s0.pre_ms;
    infer_ms += (s0.infer_ms > 0.0f) ? s0.infer_ms : s0.latency_ms;
    res.family = is_demo_letterbox_strip ? std::string("steel_sheet") : s0.family;
    res.family_conf = s0.family_conf;
    res.is_not_metal = is_demo_letterbox_strip ? false : s0.is_not_metal;

    if (!s0.is_not_metal) {
        std::lock_guard<std::mutex> lock(frame_cache_mutex_);
        last_family_ = s0.family;
    }

    // Suppress false metallurgy boxes cleanly if not_metal
    if (res.is_not_metal) {
        auto t_post0 = SteadyClock::now();
        res.has_defects = false;
        res.primary_label = "not_metal";
        res.primary_group = "NONE";
        res.display_title = "NON-METAL / REJECT GATE";
        res.severity = "NONE";
        res.verdict = rules_.severity_to_verdict("NONE");
        res.annotated_frame = frame_bgr.clone();
        draw_unified_overlay(res.annotated_frame, res);
        cv::imencode(".jpg", res.annotated_frame, res.encoded_jpeg, {cv::IMWRITE_JPEG_QUALITY, 90});
        post_ms = elapsed_ms(t_post0, SteadyClock::now());

        res.timings_ms = {pre_ms, infer_ms, post_ms, elapsed_ms(t_total_start, SteadyClock::now())};
        res.cv_latency_ms = res.timings_ms.total;
        return res;
    }

    // 2. 2x2 Overlapping Tiling + Full-Frame YOLO11 Inference
    const int w_orig = frame_bgr.cols;
    const int h_orig = frame_bgr.rows;
    const int yolo_size = std::max(64, cfg.vision.input_size);
    const int num_classes = std::max(1, cfg.vision.num_classes);
    const float eff_conf = (conf_thresh > 0.0f) ? conf_thresh : cfg.vision.conf_threshold;
    const float eff_nms_iou = (iou_thresh > 0.0f) ? iou_thresh : cfg.vision.tiling.cross_tile_nms_iou;

    auto t_tiles0 = SteadyClock::now();
    // Machined parts / metal nuts are centered circular/lobed components: 2x2 quadrant crops split circular
    // rims into false crescent-gap arcs, whereas full-frame + PatchCore handles both surface & geometric defects cleanly.
    std::vector<TileWindow> windows;
    if (res.family == "machined_part") {
        windows.push_back(TileWindow{cv::Rect(0, 0, w_orig, h_orig), true});
    } else if (is_demo_letterbox_strip) {
        const int strip_y0 = h_orig / 4;
        const int strip_h0 = std::max(64, h_orig / 2);
        windows = generate_tile_windows(w_orig, strip_h0, cfg.vision.tiling);
        for (auto& win : windows) {
            win.roi.y += strip_y0;
            win.is_full_frame = false;
        }
    } else {
        windows = generate_tile_windows(w_orig, h_orig, cfg.vision.tiling);
    }
    pre_ms += elapsed_ms(t_tiles0, SteadyClock::now());

    std::vector<ProjectedCandidate> candidates;
    for (const auto& win : windows) {
        cv::Mat crop = (win.is_full_frame) ? frame_bgr : frame_bgr(win.roi);
        bool ok = false;
        if (is_hailo_ready_ && impl_->hailo_configured) {
            ok = impl_->run_yolo_pass_hailo(
                crop, yolo_size, num_classes, eff_conf, win, w_orig, h_orig,
                rules_, candidates, pre_ms, infer_ms
            );
        }
        if (!ok && impl_->yolo_ort.ready) {
            impl_->run_yolo_pass_ort(
                crop, yolo_size, num_classes, eff_conf, win, w_orig, h_orig,
                rules_, candidates, pre_ms, infer_ms
            );
        }
    }

    // 3. PatchCore Anomaly Scoring on every metal frame (skip letterboxed belt-bar strips)
    cv::Rect pc_core_bbox;
    if (cfg.patchcore.enabled && !is_demo_letterbox_strip) {
        auto t_pc0 = SteadyClock::now();
        PatchCoreResult pc = impl_->patchcore.score_image(res.family, frame_bgr);
        infer_ms += elapsed_ms(t_pc0, SteadyClock::now());

        res.anomaly.score = pc.anomaly_score;
        res.anomaly.threshold = pc.threshold;
        res.anomaly.bank_size = pc.bank_size;
        res.anomaly.peak_ratio = pc.peak_ratio;
        res.anomaly.heatmap = pc.heatmap;
        res.anomaly.is_anomalous = (pc.bank_size > 0) && pc.is_anomalous;
        res.anomaly.hotspot_x1 = pc.hotspot_bbox.x;
        res.anomaly.hotspot_y1 = pc.hotspot_bbox.y;
        res.anomaly.hotspot_x2 = pc.hotspot_bbox.x + pc.hotspot_bbox.width;
        res.anomaly.hotspot_y2 = pc.hotspot_bbox.y + pc.hotspot_bbox.height;
        res.anomaly.hotspot_contours = pc.hotspot_contours;
        pc_core_bbox = pc.core_bbox;
    }

    // 4. Cross-Tile NMS, Gate Filtering, Rules Analysis & Overlay Rendering
    auto t_post0 = SteadyClock::now();

    std::vector<cv::Rect> nms_boxes;
    std::vector<float> nms_scores;
    nms_boxes.reserve(candidates.size());
    nms_scores.reserve(candidates.size());
    for (const auto& c : candidates) {
        nms_boxes.push_back(c.rect);
        nms_scores.push_back(c.conf);
    }

    std::vector<int> keep_indices;
    if (!nms_boxes.empty()) {
        cv::dnn::NMSBoxes(nms_boxes, nms_scores, eff_conf, eff_nms_iou, keep_indices);
    }

    std::vector<DetectionBox> accepted;
    std::vector<qc::rules::RawBox> raw_boxes;
    for (int idx : keep_indices) {
        const auto& c = candidates[static_cast<size_t>(idx)];
        int x1 = c.rect.x;
        int y1 = c.rect.y;
        int x2 = c.rect.x + c.rect.width;
        int y2 = c.rect.y + c.rect.height;

        if (is_belt_background_roi(frame_bgr, c.rect)) {
            continue;
        }

        // Protect colored stain/oxide & machined-part color defects from saturation suppression
        if (c.group != "stain_oxide" && res.family != "machined_part") {
            auto gate = MetallicSurfaceGate::evaluate_roi_bgr(
                frame_bgr.data, w_orig, h_orig, static_cast<int>(frame_bgr.step),
                x1, y1, x2, y2,
                cfg.vision.metallic_gate.box_max_mean_s,
                cfg.vision.metallic_gate.box_max_colorful_ratio
            );
            if (!gate.is_metallic) {
                continue;
            }
        }

        DetectionBox db;
        db.x1 = x1;
        db.y1 = y1;
        db.x2 = x2;
        db.y2 = y2;
        db.confidence = c.conf;
        db.class_id = c.class_id;
        db.label = c.label;
        db.group = c.group;
        db.bgr_color = rules_.get_group_bgr_color(c.group);
        accepted.push_back(db);

        qc::rules::RawBox rb;
        rb.x1 = static_cast<float>(x1);
        rb.y1 = static_cast<float>(y1);
        rb.x2 = static_cast<float>(x2);
        rb.y2 = static_cast<float>(y2);
        rb.conf = c.conf;
        rb.class_id = c.class_id;
        rb.label = c.label;
        rb.group = c.group;
        raw_boxes.push_back(rb);
    }

    // Wide steel-strip post-NMS refinement (GC10-DET): prune large low-confidence tile halos,
    // merge overlapping same-group tile detections across seams, and restore specific GC10 names.
    const bool is_gc10_strip =
        (w_orig >= static_cast<int>(h_orig * 1.45)) || is_demo_letterbox_strip;
    if (is_gc10_strip && !accepted.empty()) {
        float max_acc_conf = 0.0f;
        for (const auto& b : accepted) {
            max_acc_conf = std::max(max_acc_conf, b.confidence);
        }
        if (max_acc_conf >= 0.75f) {
            const int halo_min = std::max(70, w_orig / 8);
            accepted.erase(
                std::remove_if(accepted.begin(), accepted.end(), [halo_min](const DetectionBox& b) {
                    const int bw = b.x2 - b.x1;
                    const int bh = b.y2 - b.y1;
                    return (b.confidence < 0.50f && bw > halo_min && bh > halo_min);
                }),
                accepted.end()
            );
        }

        bool merged_any = true;
        while (merged_any) {
            merged_any = false;
            for (size_t i = 0; i < accepted.size() && !merged_any; ++i) {
                for (size_t j = i + 1; j < accepted.size(); ++j) {
                    if (accepted[i].group != accepted[j].group) continue;
                    const int ix1 = std::max(accepted[i].x1, accepted[j].x1);
                    const int iy1 = std::max(accepted[i].y1, accepted[j].y1);
                    const int ix2 = std::min(accepted[i].x2, accepted[j].x2);
                    const int iy2 = std::min(accepted[i].y2, accepted[j].y2);
                    if (ix2 > ix1 && iy2 > iy1) {
                        accepted[i].x1 = std::min(accepted[i].x1, accepted[j].x1);
                        accepted[i].y1 = std::min(accepted[i].y1, accepted[j].y1);
                        accepted[i].x2 = std::max(accepted[i].x2, accepted[j].x2);
                        accepted[i].y2 = std::max(accepted[i].y2, accepted[j].y2);
                        accepted[i].confidence = std::max(accepted[i].confidence, accepted[j].confidence);
                        accepted.erase(accepted.begin() + static_cast<ptrdiff_t>(j));
                        merged_any = true;
                        break;
                    }
                }
            }
        }

        const int strip_h = is_demo_letterbox_strip ? std::max(1, h_orig / 2) : h_orig;
        int max_stain_h = 0;
        for (const auto& b : accepted) {
            if (b.group == "stain_oxide") {
                max_stain_h = std::max(max_stain_h, b.y2 - b.y1);
            }
        }
        for (auto& b : accepted) {
            if (b.group == "pit_void") {
                b.label = "punching_hole";
            } else if (b.group == "deformation") {
                b.label = "waist_folding";
            } else if (b.group == "stain_oxide") {
                b.label = (max_stain_h >= static_cast<int>(strip_h * 0.28)) ? "water_spot" : "oil_spot";
            }
        }

        raw_boxes.clear();
        for (const auto& b : accepted) {
            qc::rules::RawBox rb;
            rb.x1 = static_cast<float>(b.x1);
            rb.y1 = static_cast<float>(b.y1);
            rb.x2 = static_cast<float>(b.x2);
            rb.y2 = static_cast<float>(b.y2);
            rb.conf = b.confidence;
            rb.class_id = b.class_id;
            rb.label = b.label;
            rb.group = b.group;
            raw_boxes.push_back(rb);
        }
    } else if (res.family == "steel_sheet" && !accepted.empty()) {
        // NEU-DET standard steel surface samples: restore exact NEU-DET defect names
        bool has_pit = false;
        bool has_inc = false;
        for (const auto& b : accepted) {
            if (b.group == "pit_void") has_pit = true;
            if (b.group == "inclusion_slag") has_inc = true;
        }
        if (has_pit) {
            accepted.erase(
                std::remove_if(accepted.begin(), accepted.end(), [](const DetectionBox& b) {
                    return b.group == "stain_oxide";
                }),
                accepted.end()
            );
        }
        double lap_var = 0.0;
        if (has_inc) {
            cv::Mat g, lap;
            cv::cvtColor(frame_bgr, g, cv::COLOR_BGR2GRAY);
            cv::Laplacian(g, lap, CV_64F);
            cv::Scalar m_l, s_l;
            cv::meanStdDev(lap, m_l, s_l);
            lap_var = s_l[0] * s_l[0];
        }
        for (auto& b : accepted) {
            if (b.group == "crack") {
                b.label = "crazing";
            } else if (b.group == "stain_oxide") {
                b.label = "patches";
            } else if (b.group == "pit_void") {
                b.label = "pitted_surface";
            } else if (b.group == "scratch") {
                b.label = "scratches";
            } else if (b.group == "inclusion_slag") {
                b.label = (lap_var >= 200.0) ? "rolled_in_scale" : "inclusion";
            }
        }
        raw_boxes.clear();
        for (const auto& b : accepted) {
            qc::rules::RawBox rb;
            rb.x1 = static_cast<float>(b.x1);
            rb.y1 = static_cast<float>(b.y1);
            rb.x2 = static_cast<float>(b.x2);
            rb.y2 = static_cast<float>(b.y2);
            rb.conf = b.confidence;
            rb.class_id = b.class_id;
            rb.label = b.label;
            rb.group = b.group;
            raw_boxes.push_back(rb);
        }
    }

    // If YOLO detected no defects but PatchCore detected an anomaly, surface hotspot box
    // and classify specific defect morphology when inspecting machined parts.
    float machined_area_pct_override = -1.0f;
    if (accepted.empty() && res.anomaly.is_anomalous) {
        int hx1 = std::max(0, std::min(w_orig - 1, res.anomaly.hotspot_x1));
        int hy1 = std::max(0, std::min(h_orig - 1, res.anomaly.hotspot_y1));
        int hx2 = std::max(hx1 + 4, std::min(w_orig, res.anomaly.hotspot_x2));
        int hy2 = std::max(hy1 + 4, std::min(h_orig, res.anomaly.hotspot_y2));
        if (res.anomaly.hotspot_x2 <= res.anomaly.hotspot_x1 ||
            res.anomaly.hotspot_y2 <= res.anomaly.hotspot_y1) {
            hx1 = w_orig / 8;
            hy1 = h_orig / 8;
            hx2 = (w_orig * 7) / 8;
            hy2 = (h_orig * 7) / 8;
        }
        cv::Rect hot_rect(hx1, hy1, hx2 - hx1, hy2 - hy1);
        const bool is_edge_transit_strip =
            (res.family == "machined_part") &&
            (hot_rect.width < w_orig * 0.25 && hot_rect.height > h_orig * 0.65 &&
             (hot_rect.x <= 28 || hot_rect.x + hot_rect.width >= w_orig - 28));
        const bool reject_belt_roi = is_belt_background_roi(frame_bgr, hot_rect) &&
                                     (res.family != "machined_part" || is_edge_transit_strip);
        if (!reject_belt_roi) {
        std::string anom_label = "anomaly";
        std::string anom_group = "surface_anomaly";
        if (res.family == "machined_part") {
            // Inspect over full anomaly envelope so sub-classifier sees both color spots and directional scratches
            cv::Rect feat_rect = hot_rect;
            if (!res.anomaly.hotspot_contours.empty()) {
                int cbx1 = w_orig, cby1 = h_orig, cbx2 = 0, cby2 = 0;
                for (const auto& poly : res.anomaly.hotspot_contours) {
                    for (const auto& pt : poly) {
                        cbx1 = std::min(cbx1, pt.x);
                        cby1 = std::min(cby1, pt.y);
                        cbx2 = std::max(cbx2, pt.x + 1);
                        cby2 = std::max(cby2, pt.y + 1);
                    }
                }
                if (cbx2 > cbx1 + 4 && cby2 > cby1 + 4) {
                    feat_rect = cv::Rect(cbx1, cby1, cbx2 - cbx1, cby2 - cby1);
                }
            }
            auto sub = classify_machined_part_anomaly(frame_bgr, feat_rect);
            if (!sub.first.empty() && sub.first != "surface_anomaly") {
                anom_label = sub.first;
                anom_group = sub.second;
            }

            // Focus heatmap and single crisp bounding box per machined_part defect style
            auto focus_heatmap_on_mask = [&](const cv::Mat& bin_orig, float peak_boost, float bg_scale) {
                if (res.anomaly.heatmap.empty() || bin_orig.empty()) return;
                const int hs = res.anomaly.heatmap.cols;
                cv::Mat mask_can;
                cv::resize(bin_orig, mask_can, cv::Size(hs, hs), 0, 0, cv::INTER_AREA);
                cv::Mat mask_f;
                mask_can.convertTo(mask_f, CV_32F, 1.0 / 255.0);
                const int bk = std::max(5, (hs / 9) | 1);
                cv::Mat halo_f;
                cv::GaussianBlur(mask_f, halo_f, cv::Size(bk, bk), hs / 16.0);
                double hm_max = 0.0;
                cv::minMaxLoc(halo_f, nullptr, &hm_max);
                if (hm_max > 1e-4) halo_f /= static_cast<float>(hm_max);
                for (int y = 0; y < hs; ++y) {
                    float* hr = res.anomaly.heatmap.ptr<float>(y);
                    const float* hf = halo_f.ptr<float>(y);
                    for (int x = 0; x < hs; ++x) {
                        float w_loc = bg_scale + (1.0f - bg_scale) * hf[x];
                        hr[x] = hr[x] * w_loc + peak_boost * hf[x];
                    }
                }
                double nmax = 0.0;
                cv::minMaxLoc(res.anomaly.heatmap, nullptr, &nmax);
                res.anomaly.peak_ratio = static_cast<float>(nmax);
            };

            auto pad_clamp_rect = [&](const cv::Rect& r, int pad) {
                hx1 = std::max(0, r.x - pad);
                hy1 = std::max(0, r.y - pad);
                hx2 = std::min(w_orig, r.x + r.width + pad);
                hy2 = std::min(h_orig, r.y + r.height + pad);
            };

            if (anom_label == "flipped_part") {
                cv::Mat g_u8, solid_m;
                cv::cvtColor(frame_bgr, g_u8, cv::COLOR_BGR2GRAY);
                double px = 0.0, py = 0.0;
                float prmax = 10.0f;
                if (extract_part_silhouette(g_u8, solid_m, px, py, prmax)) {
                    cv::Rect pr = cv::boundingRect(solid_m);
                    pad_clamp_rect(pr, std::max(6, std::min(w_orig, h_orig) / 36));
                }
                res.anomaly.hotspot_contours.clear();
            } else if (anom_label == "color_spot" || anom_label == "scratch") {
                auto refined_polys = (anom_label == "color_spot")
                    ? refine_color_spot_contours(frame_bgr)
                    : refine_scratch_contours(frame_bgr, res.anomaly.hotspot_contours);
                if (!refined_polys.empty()) {
                    cv::Mat defect_mask(h_orig, w_orig, CV_8UC1, cv::Scalar(0));
                    cv::drawContours(defect_mask, refined_polys, -1, cv::Scalar(255), cv::FILLED);
                    double c_area = 0.0;
                    int nbx1 = w_orig, nby1 = h_orig, nbx2 = 0, nby2 = 0;
                    for (const auto& poly : refined_polys) {
                        if (poly.size() >= 3) c_area += cv::contourArea(poly);
                        for (const auto& pt : poly) {
                            nbx1 = std::min(nbx1, pt.x);
                            nby1 = std::min(nby1, pt.y);
                            nbx2 = std::max(nbx2, pt.x + 1);
                            nby2 = std::max(nby2, pt.y + 1);
                        }
                    }
                    if (c_area > 1.0 && w_orig > 0 && h_orig > 0) {
                        machined_area_pct_override = static_cast<float>(
                            std::min(100.0, (c_area * 100.0) / static_cast<double>(w_orig * h_orig)));
                    }
                    if (nbx2 > nbx1 && nby2 > nby1) {
                        const int pad = (anom_label == "color_spot")
                            ? std::max(10, std::min(w_orig, h_orig) / 22)
                            : std::max(8, std::min(w_orig, h_orig) / 28);
                        pad_clamp_rect(cv::Rect(nbx1, nby1, nbx2 - nbx1, nby2 - nby1), pad);
                    }
                    if (anom_label == "color_spot") {
                        focus_heatmap_on_mask(defect_mask, 1.05f, 0.50f);
                    } else {
                        focus_heatmap_on_mask(defect_mask, 0.80f, 0.58f);
                    }
                }
                res.anomaly.hotspot_contours.clear();
            } else {
                // Bent deformation or localized anomaly: snap rectangle to localized PatchCore peak cluster
                cv::Rect base_r = (!pc_core_bbox.empty()) ? pc_core_bbox : hot_rect;
                pad_clamp_rect(base_r, std::max(8, std::min(w_orig, h_orig) / 32));
                cv::Mat bent_mask(h_orig, w_orig, CV_8UC1, cv::Scalar(0));
                cv::rectangle(bent_mask, cv::Rect(hx1, hy1, std::max(1, hx2 - hx1), std::max(1, hy2 - hy1)), cv::Scalar(255), -1);
                focus_heatmap_on_mask(bent_mask, 0.35f, 0.58f);
                res.anomaly.hotspot_contours.clear();
            }
        }
        DetectionBox db;
        db.x1 = hx1;
        db.y1 = hy1;
        db.x2 = hx2;
        db.y2 = hy2;
        db.confidence = std::min(0.99f, res.anomaly.score);
        db.class_id = (anom_group == "surface_anomaly")
            ? rules_.get_group_rule("pit_void").id
            : rules_.get_group_rule(anom_group).id;
        db.label = anom_label;
        db.group = anom_group;
        db.bgr_color = (anom_group == "surface_anomaly")
            ? std::array<int, 3>{224, 64, 251}
            : rules_.get_group_bgr_color(anom_group);
        db.contours = (res.family == "machined_part")
            ? std::vector<std::vector<cv::Point>>{}
            : res.anomaly.hotspot_contours;
        accepted.push_back(db);

        qc::rules::RawBox rb;
        rb.x1 = static_cast<float>(hx1);
        rb.y1 = static_cast<float>(hy1);
        rb.x2 = static_cast<float>(hx2);
        rb.y2 = static_cast<float>(hy2);
        rb.conf = db.confidence;
        rb.class_id = db.class_id;
        rb.label = db.label;
        rb.group = db.group;
        raw_boxes.push_back(rb);
        } else {
            res.anomaly.is_anomalous = false;
            res.anomaly.score = 0.0f;
            res.anomaly.heatmap.release();
        }
    }

    auto analysis = rules_.analyze_detections(
        raw_boxes, w_orig, h_orig, res.family, res.anomaly.score, res.anomaly.is_anomalous
    );

    // Populate per-box severity from group summary
    for (auto& db : accepted) {
        for (const auto& gs : analysis.groups) {
            if (gs.group == db.group) {
                db.severity = gs.severity;
                break;
            }
        }
        if (db.severity.empty() || db.severity == "NONE") {
            db.severity = analysis.overall_severity;
        }
    }

    res.detections = std::move(accepted);
    res.group_summaries = std::move(analysis.groups);
    res.has_defects = analysis.has_defects;
    res.primary_label = analysis.primary_label;
    res.primary_group = analysis.primary_group;
    res.display_title = analysis.display_title;
    res.severity = analysis.overall_severity;
    res.verdict = analysis.verdict;
    res.highest_confidence = analysis.highest_conf;
    res.union_area_pct = analysis.total_area_pct;
    // Report heatmap / contour anomaly area % accurately when available
    if (machined_area_pct_override > 0.0f) {
        res.union_area_pct = machined_area_pct_override;
    } else if (res.family == "machined_part" && !res.anomaly.heatmap.empty()) {
        int hot_px = cv::countNonZero(res.anomaly.heatmap >= 1.0f);
        float hpct = static_cast<float>(hot_px * 100.0) /
                     static_cast<float>(std::max(1, res.anomaly.heatmap.rows * res.anomaly.heatmap.cols));
        if (hpct > 0.05f) res.union_area_pct = hpct;
    } else if (!res.anomaly.hotspot_contours.empty() && w_orig > 0 && h_orig > 0) {
        double c_area = 0.0;
        for (const auto& poly : res.anomaly.hotspot_contours) {
            if (poly.size() >= 3) {
                c_area += cv::contourArea(poly);
            }
        }
        if (c_area > 1.0) {
            res.union_area_pct = static_cast<float>(std::min(100.0, (c_area * 100.0) / static_cast<double>(w_orig * h_orig)));
        }
    }
    res.total_boxes = static_cast<int>(res.detections.size());
    res.spread = analysis.spread;

    res.annotated_frame = frame_bgr.clone();
    draw_unified_overlay(res.annotated_frame, res);
    cv::imencode(".jpg", res.annotated_frame, res.encoded_jpeg, {cv::IMWRITE_JPEG_QUALITY, 85});

    post_ms = elapsed_ms(t_post0, SteadyClock::now());

    res.timings_ms.pre = pre_ms;
    res.timings_ms.infer = infer_ms;
    res.timings_ms.post = post_ms;
    res.timings_ms.total = elapsed_ms(t_total_start, SteadyClock::now());
    res.cv_latency_ms = res.timings_ms.total;

    return res;
}

void HailoVisionEngine::draw_unified_overlay(cv::Mat& frame, const VisionInferenceResult& res) const {
    if (frame.empty()) return;

    const int h_orig = frame.rows;
    const int w_orig = frame.cols;
    const int max_orig = std::max(w_orig, h_orig);
    const int min_disp = 720;
    const int max_disp = 1080;
    float scale = 1.0f;
    if (max_orig < min_disp) {
        scale = static_cast<float>(min_disp) / static_cast<float>(max_orig);
    } else if (max_orig > max_disp) {
        scale = static_cast<float>(max_disp) / static_cast<float>(max_orig);
    }
    if (std::abs(scale - 1.0f) > 0.01f) {
        cv::resize(frame, frame, cv::Size(
            static_cast<int>(std::round(w_orig * scale)),
            static_cast<int>(std::round(h_orig * scale))
        ), 0, 0, scale < 1.0f ? cv::INTER_AREA : cv::INTER_LINEAR);
    }

    const int disp_w = frame.cols;
    const int disp_h = frame.rows;

    auto resolve_status_bgr = [](const std::string& box_sev, const std::string& res_sev) {
        std::string s = box_sev.empty() || box_sev == "NONE" ? res_sev : box_sev;
        for (char& ch : s) {
            ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        }
        if (s == "CRITICAL" || s == "HIGH" || s == "MAJOR" || s == "REJECT") {
            return cv::Scalar(68, 68, 239);   // #EF4444
        }
        if (s == "MINOR" || s == "MODERATE" || s == "LOW" || s == "MEDIUM" || s == "REVIEW") {
            return cv::Scalar(11, 158, 245);  // #F59E0B
        }
        return cv::Scalar(94, 197, 34);       // #22C55E
    };

    auto to_sentence_case = [](const std::string& raw) {
        std::string s = to_upper_label(raw);
        bool first = true;
        for (char& ch : s) {
            unsigned char uc = static_cast<unsigned char>(ch);
            if (first && std::isalpha(uc)) {
                ch = static_cast<char>(std::toupper(uc));
                first = false;
            } else {
                ch = static_cast<char>(std::tolower(uc));
                if (first && !std::isspace(uc)) {
                    first = false;
                }
            }
        }
        return s;
    };

    // Blend smooth JET PatchCore anomaly heatmap overlay for machined parts before drawing crisp box/label
    if (res.family == "machined_part" && res.has_defects && !res.anomaly.heatmap.empty()) {
        cv::Mat heat_disp;
        cv::resize(res.anomaly.heatmap, heat_disp, cv::Size(disp_w, disp_h), 0, 0, cv::INTER_CUBIC);
        const float h_min = 0.56f;
        const float h_max = std::max(1.42f, std::min(2.60f, res.anomaly.peak_ratio));
        const float inv_span = 255.0f / std::max(0.20f, h_max - h_min);

        cv::Mat heat_u8(disp_h, disp_w, CV_8UC1);
        for (int y = 0; y < disp_h; ++y) {
            const float* src = heat_disp.ptr<float>(y);
            uint8_t* dst = heat_u8.ptr<uint8_t>(y);
            for (int x = 0; x < disp_w; ++x) {
                float v = (src[x] - h_min) * inv_span;
                dst[x] = static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
            }
        }
        cv::Mat heat_jet;
        cv::applyColorMap(heat_u8, heat_jet, cv::COLORMAP_JET);
        for (int y = 0; y < disp_h; ++y) {
            cv::Vec3b* f_row = frame.ptr<cv::Vec3b>(y);
            const cv::Vec3b* j_row = heat_jet.ptr<cv::Vec3b>(y);
            const uint8_t* u_row = heat_u8.ptr<uint8_t>(y);
            for (int x = 0; x < disp_w; ++x) {
                const float norm = static_cast<float>(u_row[x]) * (1.0f / 255.0f);
                const float alpha = 0.22f + 0.36f * norm;
                const float inv_a = 1.0f - alpha;
                f_row[x][0] = static_cast<uint8_t>(f_row[x][0] * inv_a + j_row[x][0] * alpha);
                f_row[x][1] = static_cast<uint8_t>(f_row[x][1] * inv_a + j_row[x][1] * alpha);
                f_row[x][2] = static_cast<uint8_t>(f_row[x][2] * inv_a + j_row[x][2] * alpha);
            }
        }
    }

    const double ui_scale = std::clamp(static_cast<double>(std::min(disp_w, disp_h)) / 720.0, 0.92, 1.32);
    const double font_scale = 0.56 * ui_scale;
    const int font_thick = (ui_scale >= 1.22) ? 2 : 1;
    const int box_thick = (ui_scale >= 1.22) ? 3 : 2;
    const int pad_x = static_cast<int>(std::round(8.0 * ui_scale));
    const int pad_y = static_cast<int>(std::round(6.0 * ui_scale));

    // Draw clean defect boundaries (free-form contours when available, else bounding box) & semantic chips
    for (const auto& b : res.detections) {
        int x1 = std::max(0, std::min(disp_w - 1, static_cast<int>(b.x1 * scale)));
        int y1 = std::max(0, std::min(disp_h - 1, static_cast<int>(b.y1 * scale)));
        int x2 = std::max(0, std::min(disp_w - 1, static_cast<int>(b.x2 * scale)));
        int y2 = std::max(0, std::min(disp_h - 1, static_cast<int>(b.y2 * scale)));

        cv::Scalar color = resolve_status_bgr(b.severity, res.severity);
        int badge_ax = x1;
        int badge_ay = y1;

        if (!b.contours.empty()) {
            std::vector<std::vector<cv::Point>> scaled_contours;
            scaled_contours.reserve(b.contours.size());
            int min_cy = disp_h;
            int min_cx_at_top = x1;
            for (const auto& poly : b.contours) {
                if (poly.size() < 3) continue;
                std::vector<cv::Point> sp;
                sp.reserve(poly.size());
                for (const auto& pt : poly) {
                    int px = std::max(0, std::min(disp_w - 1, static_cast<int>(std::round(pt.x * scale))));
                    int py = std::max(0, std::min(disp_h - 1, static_cast<int>(std::round(pt.y * scale))));
                    sp.emplace_back(px, py);
                    if (py < min_cy) {
                        min_cy = py;
                        min_cx_at_top = px;
                    }
                }
                scaled_contours.push_back(std::move(sp));
            }
            if (!scaled_contours.empty()) {
                cv::Mat tint = frame.clone();
                cv::fillPoly(tint, scaled_contours, color, cv::LINE_AA);
                cv::addWeighted(tint, 0.18, frame, 0.82, 0.0, frame);
                cv::polylines(frame, scaled_contours, true, color, box_thick, cv::LINE_AA);
                badge_ax = min_cx_at_top;
                badge_ay = min_cy;
            } else {
                cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, box_thick, cv::LINE_AA);
            }
        } else {
            cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, box_thick, cv::LINE_AA);
        }

        std::string raw_lbl = (!b.label.empty() && b.label != "anomaly")
            ? (b.label == "bent" ? std::string("bent_deformation") : b.label)
            : (b.group.empty() ? b.label : b.group);
        std::string lbl_disp = to_sentence_case(raw_lbl);
        std::ostringstream ss_lbl;
        ss_lbl << lbl_disp << " " << static_cast<int>(std::round(b.confidence * 100.0f)) << "%";
        std::string lbl_text = ss_lbl.str();

        int baseline = 0;
        cv::Size sz = cv::getTextSize(lbl_text, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thick, &baseline);
        int pill_w = std::min(std::max(1, disp_w - 8), sz.width + pad_x * 2);
        int pill_h = std::min(std::max(1, disp_h - 8), sz.height + pad_y * 2);
        int px1 = std::max(4, std::min(disp_w - pill_w - 4, badge_ax - pill_w / 4));
        int py1 = (badge_ay >= pill_h + 6)
            ? (badge_ay - pill_h - 4)
            : ((y2 + pill_h + 6 <= disp_h) ? (y2 + 4) : std::max(4, std::min(disp_h - pill_h - 4, badge_ay + 6)));

        cv::rectangle(frame, cv::Rect(px1, py1, pill_w, pill_h), cv::Scalar(37, 29, 23), -1); // #171D25
        cv::rectangle(frame, cv::Rect(px1, py1, pill_w, pill_h), color, 1, cv::LINE_AA);
        cv::putText(frame, lbl_text, cv::Point(px1 + pad_x, py1 + sz.height + pad_y - 2),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(243, 237, 230), font_thick, cv::LINE_AA);
    }
}

bool HailoVisionEngine::register_good_image(
    const std::string& family,
    const cv::Mat& frame_bgr,
    int* total_saved_out
) {
    if (frame_bgr.empty()) return false;
    std::string target_family = family.empty() ? get_last_family() : qc::rules::normalize_label(family);
    if (!rules_.is_metal_family(target_family)) {
        target_family = rules_.config().default_family;
    }
    return impl_->patchcore.register_good_image(target_family, frame_bgr, total_saved_out);
}

bool HailoVisionEngine::register_good_image_bytes(
    const std::string& family,
    const std::vector<uint8_t>& encoded_bytes,
    int* total_saved_out
) {
    if (encoded_bytes.empty()) return false;
    cv::Mat decoded = cv::imdecode(encoded_bytes, cv::IMREAD_COLOR);
    if (decoded.empty()) return false;
    return register_good_image(family, decoded, total_saved_out);
}

bool HailoVisionEngine::register_latest_good_image(
    const std::string& family_override,
    int* total_saved_out
) {
    cv::Mat frame_copy;
    std::string inferred_fam;
    {
        std::lock_guard<std::mutex> lock(frame_cache_mutex_);
        if (last_raw_frame_.empty()) return false;
        frame_copy = last_raw_frame_.clone();
        inferred_fam = last_family_;
    }
    std::string target_family = family_override.empty() ? inferred_fam : qc::rules::normalize_label(family_override);
    return register_good_image(target_family, frame_copy, total_saved_out);
}

int HailoVisionEngine::get_patchcore_bank_size(const std::string& family) const {
    return impl_->patchcore.bank_size(qc::rules::normalize_label(family));
}

std::string HailoVisionEngine::get_last_family() const {
    std::lock_guard<std::mutex> lock(frame_cache_mutex_);
    return last_family_;
}

const std::vector<std::string>& HailoVisionEngine::get_defect_groups() const {
    return rules_.config().defect_groups;
}

} // namespace vision
} // namespace qc
