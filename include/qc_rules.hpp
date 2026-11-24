#pragma once

#include <array>
#include <map>
#include <string>
#include <vector>

namespace qc {
namespace rules {

struct AreaThresholds {
    float minor_max_pct = 5.0f;
    float moderate_max_pct = 20.0f;
};

struct GroupRule {
    int id = 0;
    std::string group;
    std::string display_name;
    int risk = 2; // 1=low, 2=moderate, 3=critical
    float conf_threshold = 0.28f;
    AreaThresholds area_thresholds;
    int box_escalation_count = 4;
    std::array<int, 3> bgr_color = {0, 215, 255};
};

struct FamilyMetadata {
    std::string name;
    std::string display_name;
    bool patchcore_always = false;
    float default_conf_threshold = 0.28f;
};

struct Stage0Config {
    std::string model_path = "models/stage0_selector_224.onnx";
    std::string input_name = "images";
    std::string output_name = "output0";
    int input_size = 224;
    std::vector<std::string> classes = {
        "steel_sheet", "aluminium", "weld", "casting", "machined_part", "not_metal"
    };
    float confidence_threshold = 0.45f;
    float not_metal_threshold = 0.55f;
    float low_confidence_patchcore_trigger = 0.60f;
    bool visual_gate_enabled = true;
    int saturation_pixel_threshold = 65;
    float frame_max_mean_s = 54.0f;
    float frame_max_colorful_ratio = 0.35f;
};

struct TilingConfig {
    bool enabled = true;
    int grid_rows = 2;
    int grid_cols = 2;
    int tile_size = 640;
    float overlap = 0.20f;
    bool include_full_frame = true;
    float cross_tile_nms_iou = 0.45f;
};

struct MetallicGateConfig {
    int saturation_pixel_threshold = 65;
    float box_max_mean_s = 46.0f;
    float box_max_colorful_ratio = 0.25f;
    float frame_max_mean_s = 54.0f;
    float frame_max_colorful_ratio = 0.35f;
};

struct VisionConfig {
    std::string hef_path = "models/unified_yolo11_640.hef";
    std::string onnx_path = "models/unified_yolo11_640.onnx";
    std::string hailo_device_node = "/dev/hailo0";
    std::string input_name = "images";
    std::string output_name = "output0";
    int input_size = 640;
    int num_classes = 8;
    int num_anchors = 8400;
    float conf_threshold = 0.28f;
    float nms_iou_threshold = 0.45f;
    int lock_in_frames = 4;
    // 0: automatic Pi policy (Hailo: 1 CPU thread, CPU fallback: up to 2).
    // Positive values are explicit whole-system benchmarking overrides.
    int ort_intra_op_threads = 0;
    TilingConfig tiling;
    MetallicGateConfig metallic_gate;
};

struct PatchCoreConfig {
    bool enabled = true;
    int reference_limit_per_family = 20;
    float anomaly_threshold = 0.55f;
    std::string refs_dir = "models/patchcore_refs";
    std::string banks_dir = "models/patchcore_banks";
    std::vector<std::string> families = {
        "steel_sheet", "aluminium", "weld", "casting", "machined_part"
    };
    std::vector<std::string> always_score_families = {"casting", "machined_part"};
    float low_confidence_trigger = 0.60f;
};

struct LlmConfig {
    std::string pte_path = "models/Llama-3.2-1B-UnifiedQC-FT-INT4.pte";
    std::string tokenizer_path = "models/tokenizer.model";
    float temperature = 0.0f;
    int seq_len = 256;
    int max_new_tokens = 64;
    int cpu_threads = 4;
    std::string system_prompt =
        "You are an industrial multi-metal quality-control AI assistant. Given material family, "
        "defect label, unified group, severity, area %, and PatchCore anomaly score, reply in one line: "
        "Possible Cause: <root cause> | Recommended Action: <operator action>";
};

struct SeverityVerdictConfig {
    std::vector<std::string> severity_levels = {"NONE", "MINOR", "MODERATE", "CRITICAL"};
    std::map<std::string, std::string> verdict_by_severity = {
        {"NONE", "PASS"},
        {"MINOR", "REVIEW"},
        {"MODERATE", "REVIEW"},
        {"CRITICAL", "REJECT"}
    };
    float anomaly_only_minor_threshold = 0.55f;
    float anomaly_only_moderate_threshold = 0.72f;
    float anomaly_only_critical_threshold = 0.88f;
    std::array<int, 3> clean_color_bgr = {113, 204, 46};
};

struct EdgeQcConfig {
    std::string config_path;
    std::vector<std::string> defect_groups = {
        "scratch", "pit_void", "crack", "stain_oxide",
        "inclusion_slag", "weld_porosity", "deformation", "machining_tear"
    };
    std::map<std::string, GroupRule> group_rules;
    std::map<std::string, std::string> label_aliases;
    SeverityVerdictConfig severity_verdict;

    std::vector<std::string> family_classes = {
        "steel_sheet", "aluminium", "weld", "casting", "machined_part", "not_metal"
    };
    std::vector<std::string> metal_families = {
        "steel_sheet", "aluminium", "weld", "casting", "machined_part"
    };
    std::string reject_family = "not_metal";
    std::string default_family = "steel_sheet";
    std::map<std::string, FamilyMetadata> family_metadata;

    Stage0Config stage0;
    VisionConfig vision;
    PatchCoreConfig patchcore;
    LlmConfig llm;
};

struct RawBox {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float conf = 0.0f;
    int class_id = -1;
    std::string label;
    std::string group;
};

struct GroupSummary {
    std::string group;
    std::string primary_label;
    std::string display_name;
    float max_conf = 0.0f;
    float conf_pct = 0.0f;
    float area_pct = 0.0f;
    int box_count = 0;
    std::string pattern = "localized";
    std::string severity = "NONE";
    std::string verdict = "PASS";
    float dominance = 0.0f;
    std::array<int, 3> bgr_color = {0, 215, 255};
    std::vector<RawBox> boxes;
};

struct MultiDefectAnalysis {
    bool has_defects = false;
    bool is_not_metal = false;
    std::string family = "steel_sheet";
    std::vector<GroupSummary> groups;
    std::string primary_label = "NONE";
    std::string primary_group = "NONE";
    std::string signature = "NONE";
    std::string display_title = "NO DEFECT DETECTED";
    float highest_conf = 0.0f;
    float total_area_pct = 0.0f;
    int total_boxes = 0;
    std::string spread = "localized";
    std::string overall_severity = "NONE";
    std::string verdict = "PASS";
};

/**
 * Computes exact union area percentage of bounding boxes over frame (w, h).
 */
float compute_union_area_pct(const std::vector<RawBox>& boxes, int w, int h);

/**
 * Normalizes a defect class or alias string to lowercase underscore format.
 */
std::string normalize_label(std::string name);

/**
 * Config-driven Multi-Metal Quality Control Rules Engine.
 * Parses config/edge_qc_config.json and maps detector labels, severities,
 * verdicts, BGR overlay colors, and LLM prompts/fallbacks uniformly.
 */
class QcRulesEngine {
public:
    QcRulesEngine();

    bool load_config(const std::string& config_path = "config/edge_qc_config.json");
    bool is_loaded() const { return loaded_; }
    const EdgeQcConfig& config() const { return cfg_; }

    std::string map_label_to_group(const std::string& raw_label) const;
    std::string group_from_class_id(int class_id) const;
    const GroupRule& get_group_rule(const std::string& group) const;
    std::array<int, 3> get_group_bgr_color(const std::string& group) const;
    float get_group_conf_threshold(const std::string& group) const;

    bool is_metal_family(const std::string& family) const;
    bool should_always_score_patchcore(const std::string& family) const;
    std::string get_family_display_name(const std::string& family) const;

    std::string evaluate_group_severity(
        const std::string& group,
        float area_pct,
        float max_conf,
        int box_count
    ) const;

    std::string evaluate_anomaly_severity(float anomaly_score) const;
    std::string severity_to_verdict(const std::string& severity) const;
    static int severity_rank(const std::string& severity);
    static std::string max_severity(const std::string& a, const std::string& b);

    MultiDefectAnalysis analyze_detections(
        const std::vector<RawBox>& detections,
        int w_orig,
        int h_orig,
        const std::string& family = "steel_sheet",
        float anomaly_score = 0.0f,
        bool is_anomalous = false
    ) const;

    std::string build_llama_advisory_prompt(
        const std::string& family,
        const std::string& label,
        const std::string& group,
        const std::string& severity,
        float confidence,
        float area_pct,
        int box_count,
        float anomaly_score
    ) const;

private:
    EdgeQcConfig cfg_;
    GroupRule default_group_rule_;
    bool loaded_ = false;

    void init_defaults();
};

} // namespace rules
} // namespace qc
