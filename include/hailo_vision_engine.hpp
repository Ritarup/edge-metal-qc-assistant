#pragma once

#include "qc_rules.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace qc {
namespace vision {

struct DetectionBox {
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
    float confidence = 0.0f;
    int class_id = 0;
    std::string label;    // Fine defect label (e.g. "scratch", "porosity", "anomaly")
    std::string group;    // Unified 8-group name (e.g. "scratch", "weld_porosity", "pit_void")
    std::string severity = "NONE"; // NONE, MINOR, MODERATE, CRITICAL
    std::array<int, 3> bgr_color = {0, 215, 255};
    std::vector<std::vector<cv::Point>> contours; // Optional free-form defect boundary polygon(s)
};

struct AnomalyTelemetry {
    float score = 0.0f;
    float threshold = 0.55f;
    bool is_anomalous = false;
    int bank_size = 0;
    int hotspot_x1 = 0;
    int hotspot_y1 = 0;
    int hotspot_x2 = 0;
    int hotspot_y2 = 0;
    float peak_ratio = 0.0f;
    std::vector<std::vector<cv::Point>> hotspot_contours;
    cv::Mat heatmap; // CANONICAL_SIZE x CANONICAL_SIZE CV_32FC1 normalized anomaly score ratio
};

struct TimingsMs {
    float pre = 0.0f;
    float infer = 0.0f;
    float post = 0.0f;
    float total = 0.0f;
};

struct Stage0Result {
    std::string family = "steel_sheet"; // steel_sheet, aluminium, weld, casting, machined_part, not_metal
    float family_conf = 0.0f;
    bool is_not_metal = false;
    float pre_ms = 0.0f;
    float infer_ms = 0.0f;
    float latency_ms = 0.0f;
};

struct VisionInferenceResult {
    uint64_t frame_id = 0;
    std::string family = "steel_sheet";
    float family_conf = 0.0f;
    std::string vision_backend = "onnxruntime_arm64"; // "hailo8l_hef" or "onnxruntime_arm64"
    std::string hardware_device = "ARM64 ONNX Runtime";

    bool has_defects = false;
    bool is_not_metal = false;
    std::vector<DetectionBox> detections;
    std::vector<qc::rules::GroupSummary> group_summaries;

    std::string primary_label = "NONE";
    std::string primary_group = "NONE";
    std::string display_title = "NO DEFECT DETECTED";
    std::string severity = "NONE"; // NONE, MINOR, MODERATE, CRITICAL
    std::string verdict = "PASS";  // PASS, REVIEW, REJECT

    float highest_confidence = 0.0f;
    float union_area_pct = 0.0f;
    int total_boxes = 0;
    std::string spread = "localized";

    AnomalyTelemetry anomaly;
    TimingsMs timings_ms;
    float cv_latency_ms = 0.0f;

    cv::Mat annotated_frame;
    std::vector<uint8_t> encoded_jpeg;
};

/**
 * Unified Multi-Metal Edge Vision Engine.
 * - Stage 0 Router (224x224 ONNX) + HSV Metallic Visual Safeguard
 * - Dual Vision Backend: Hailo-8L NPU (.hef via /dev/hailo0 when compiled with HAVE_HAILORT)
 *   with automatic seamless fallback to ARM64 ONNX Runtime (.onnx)
 * - 2x2 overlapping 640x640 tiling + full-frame inference + cross-tile NMS
 * - PatchCore anomaly scoring & runtime good-reference registration
 * - Exact steady_clock timings_ms (pre, infer, post, total)
 */
class HailoVisionEngine {
public:
    HailoVisionEngine();
    ~HailoVisionEngine();

    bool init(const std::string& config_path = "config/edge_qc_config.json");

    bool init(
        const std::string& hef_path,
        const std::string& yolo_onnx_path,
        const std::string& stage0_onnx_path,
        const std::string& config_path = "config/edge_qc_config.json"
    );

    Stage0Result classify_stage0(const cv::Mat& frame_bgr);

    VisionInferenceResult infer(
        const cv::Mat& frame_bgr,
        float conf_thresh = -1.0f,
        float iou_thresh = -1.0f,
        int conveyor_phase = 0 // 0: automatic gate; 1: transit; 2: centered demo part
    );

    bool register_good_image(
        const std::string& family,
        const cv::Mat& frame_bgr,
        int* total_saved_out = nullptr
    );

    bool register_good_image_bytes(
        const std::string& family,
        const std::vector<uint8_t>& encoded_bytes,
        int* total_saved_out = nullptr
    );

    bool register_latest_good_image(
        const std::string& family_override = "",
        int* total_saved_out = nullptr
    );

    int get_patchcore_bank_size(const std::string& family) const;
    std::string get_last_family() const;

    bool is_hailo_active() const { return is_hailo_ready_; }
    std::string get_vision_backend() const { return vision_backend_; }
    std::string get_hardware_name() const { return hardware_name_; }
    const qc::rules::QcRulesEngine& rules() const { return rules_; }
    const std::vector<std::string>& get_defect_groups() const;

private:
    bool is_hailo_ready_ = false;
    std::string vision_backend_ = "onnxruntime_arm64";
    std::string hardware_name_ = "ARM64 ONNX Runtime";
    std::string hef_path_;
    std::string yolo_onnx_path_;
    std::string stage0_onnx_path_;

    qc::rules::QcRulesEngine rules_;
    std::atomic<uint64_t> frame_counter_{0};
    mutable std::mutex frame_cache_mutex_;
    cv::Mat last_raw_frame_;
    std::string last_family_ = "steel_sheet";

    struct Impl;
    std::unique_ptr<Impl> impl_;

    void draw_unified_overlay(
        cv::Mat& frame,
        const VisionInferenceResult& res
    ) const;
};

} // namespace vision
} // namespace qc
