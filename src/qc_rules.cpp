#include "qc_rules.hpp"
#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace qc {
namespace rules {

namespace {

std::string to_upper_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

std::string resolve_config_path(const std::string& requested) {
    std::vector<std::string> candidates;
    if (!requested.empty()) candidates.push_back(requested);
    candidates.push_back("config/edge_qc_config.json");
    candidates.push_back("../config/edge_qc_config.json");

    for (const auto& c : candidates) {
        std::error_code ec;
        if (!c.empty() && fs::exists(c, ec) && fs::is_regular_file(c, ec)) {
            return c;
        }
    }
    return "";
}

template <typename T>
T json_or(const json& obj, const char* key, const T& fallback) {
    if (obj.is_object() && obj.contains(key) && !obj[key].is_null()) {
        try {
            return obj[key].get<T>();
        } catch (...) {
            return fallback;
        }
    }
    return fallback;
}

} // namespace

std::string normalize_label(std::string name) {
    size_t start = name.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = name.find_last_not_of(" \t\r\n");
    name = name.substr(start, end - start + 1);

    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    for (char& c : name) {
        if (c == ' ' || c == '-') c = '_';
    }
    const std::string prefix = "component_anomaly:";
    if (name.rfind(prefix, 0) == 0) {
        name = name.substr(prefix.length());
    }
    return name;
}

float compute_union_area_pct(const std::vector<RawBox>& boxes, int w, int h) {
    if (boxes.empty() || w <= 0 || h <= 0) {
        return 0.0f;
    }

    const int sw = std::max(1, (w + 3) / 4);
    const int sh = std::max(1, (h + 3) / 4);
    std::vector<uint8_t> mask(static_cast<size_t>(sw) * static_cast<size_t>(sh), 0);

    for (const auto& b : boxes) {
        int x1 = std::max(0, std::min(static_cast<int>(std::floor(b.x1 / 4.0f)), sw));
        int y1 = std::max(0, std::min(static_cast<int>(std::floor(b.y1 / 4.0f)), sh));
        int x2 = std::max(0, std::min(static_cast<int>(std::ceil(b.x2 / 4.0f)), sw));
        int y2 = std::max(0, std::min(static_cast<int>(std::ceil(b.y2 / 4.0f)), sh));
        if (x2 <= x1 || y2 <= y1) continue;

        for (int y = y1; y < y2; ++y) {
            std::fill_n(&mask[static_cast<size_t>(y) * sw + x1], x2 - x1, 1);
        }
    }

    int active_pixels = 0;
    for (uint8_t val : mask) {
        if (val) active_pixels++;
    }
    return 100.0f * static_cast<float>(active_pixels) / static_cast<float>(sw * sh);
}

QcRulesEngine::QcRulesEngine() {
    init_defaults();
}

void QcRulesEngine::init_defaults() {
    cfg_ = EdgeQcConfig{};
    default_group_rule_ = GroupRule{
        0, "scratch", "Scratch / Abrasion", 2, 0.28f, {4.0f, 16.0f}, 4, {0, 215, 255}
    };

    struct DefaultSpec {
        int id;
        const char* group;
        const char* disp;
        int risk;
        float conf;
        float minor_max;
        float mod_max;
        int esc_cnt;
        std::array<int, 3> bgr;
    };

    const DefaultSpec specs[] = {
        {0, "scratch",        "Scratch / Abrasion",        2, 0.28f,  4.0f, 16.0f, 4, {  0, 215, 255}},
        {1, "pit_void",       "Pit / Void / Blowhole",     2, 0.28f,  5.0f, 22.0f, 4, {224,  64, 251}},
        {2, "crack",          "Crack / Crazing",           3, 0.25f,  2.5f, 12.0f, 3, { 77,  77, 255}},
        {3, "stain_oxide",    "Stain / Oxide / Spot",      1, 0.30f, 12.0f, 32.0f, 5, {255, 229,   0}},
        {4, "inclusion_slag", "Inclusion / Slag / Scale",  3, 0.28f,  3.5f, 14.0f, 3, {  0, 160, 255}},
        {5, "weld_porosity",  "Weld Porosity / Fusion",    3, 0.26f,  3.0f, 12.0f, 3, {180, 105, 255}},
        {6, "deformation",    "Deformation / Dent / Bend", 3, 0.28f,  3.0f, 10.0f, 3, {  0,  90, 255}},
        {7, "machining_tear", "Machining Tear / Burr",     2, 0.28f,  3.5f, 15.0f, 4, { 80, 220, 120}}
    };

    cfg_.defect_groups.clear();
    cfg_.group_rules.clear();
    cfg_.label_aliases.clear();
    for (const auto& s : specs) {
        cfg_.defect_groups.emplace_back(s.group);
        cfg_.group_rules[s.group] = GroupRule{
            s.id, s.group, s.disp, s.risk, s.conf, {s.minor_max, s.mod_max}, s.esc_cnt, s.bgr
        };
        cfg_.label_aliases[s.group] = s.group;
    }
}

bool QcRulesEngine::load_config(const std::string& config_path) {
    init_defaults();
    std::string resolved = resolve_config_path(config_path);
    if (resolved.empty()) {
        std::cerr << "[QcRulesEngine] Warning: config file not found (" << config_path
                  << "), using built-in defaults." << std::endl;
        loaded_ = false;
        return false;
    }

    std::ifstream ifs(resolved);
    if (!ifs.is_open()) {
        std::cerr << "[QcRulesEngine] Warning: failed to open " << resolved << std::endl;
        loaded_ = false;
        return false;
    }

    json j;
    try {
        ifs >> j;
    } catch (const std::exception& e) {
        std::cerr << "[QcRulesEngine] JSON parse error in " << resolved << ": " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }

    cfg_.config_path = resolved;

    if (j.contains("defect_groups") && j["defect_groups"].is_array() && !j["defect_groups"].empty()) {
        cfg_.defect_groups.clear();
        for (const auto& g : j["defect_groups"]) {
            if (g.is_string()) {
                cfg_.defect_groups.push_back(normalize_label(g.get<std::string>()));
            }
        }
    }

    if (j.contains("group_rules") && j["group_rules"].is_object()) {
        for (auto it = j["group_rules"].begin(); it != j["group_rules"].end(); ++it) {
            std::string gname = normalize_label(it.key());
            const json& rj = it.value();
            GroupRule rule = cfg_.group_rules.count(gname) ? cfg_.group_rules[gname] : default_group_rule_;
            rule.group = gname;
            rule.id = json_or<int>(rj, "id", rule.id);
            rule.display_name = json_or<std::string>(rj, "display_name", gname);
            rule.risk = json_or<int>(rj, "risk", rule.risk);
            rule.conf_threshold = json_or<float>(rj, "conf_threshold", rule.conf_threshold);
            rule.box_escalation_count = json_or<int>(rj, "box_escalation_count", rule.box_escalation_count);
            if (rj.contains("area_thresholds") && rj["area_thresholds"].is_object()) {
                const json& aj = rj["area_thresholds"];
                rule.area_thresholds.minor_max_pct = json_or<float>(aj, "minor_max_pct", rule.area_thresholds.minor_max_pct);
                rule.area_thresholds.moderate_max_pct = json_or<float>(aj, "moderate_max_pct", rule.area_thresholds.moderate_max_pct);
            }
            if (rj.contains("bgr_color") && rj["bgr_color"].is_array() && rj["bgr_color"].size() >= 3) {
                rule.bgr_color = {
                    rj["bgr_color"][0].get<int>(),
                    rj["bgr_color"][1].get<int>(),
                    rj["bgr_color"][2].get<int>()
                };
            }
            cfg_.group_rules[gname] = rule;
            cfg_.label_aliases[gname] = gname;
        }
    }

    if (j.contains("severity_verdict_rules") && j["severity_verdict_rules"].is_object()) {
        const json& sj = j["severity_verdict_rules"];
        if (sj.contains("severity_levels") && sj["severity_levels"].is_array()) {
            cfg_.severity_verdict.severity_levels.clear();
            for (const auto& lvl : sj["severity_levels"]) {
                if (lvl.is_string()) cfg_.severity_verdict.severity_levels.push_back(to_upper_copy(lvl.get<std::string>()));
            }
        }
        if (sj.contains("verdict_by_severity") && sj["verdict_by_severity"].is_object()) {
            cfg_.severity_verdict.verdict_by_severity.clear();
            for (auto it = sj["verdict_by_severity"].begin(); it != sj["verdict_by_severity"].end(); ++it) {
                if (it.value().is_string()) {
                    cfg_.severity_verdict.verdict_by_severity[to_upper_copy(it.key())] = to_upper_copy(it.value().get<std::string>());
                }
            }
        }
        cfg_.severity_verdict.anomaly_only_minor_threshold = json_or<float>(sj, "anomaly_only_minor_threshold", cfg_.severity_verdict.anomaly_only_minor_threshold);
        cfg_.severity_verdict.anomaly_only_moderate_threshold = json_or<float>(sj, "anomaly_only_moderate_threshold", cfg_.severity_verdict.anomaly_only_moderate_threshold);
        cfg_.severity_verdict.anomaly_only_critical_threshold = json_or<float>(sj, "anomaly_only_critical_threshold", cfg_.severity_verdict.anomaly_only_critical_threshold);
        if (sj.contains("clean_color_bgr") && sj["clean_color_bgr"].is_array() && sj["clean_color_bgr"].size() >= 3) {
            cfg_.severity_verdict.clean_color_bgr = {
                sj["clean_color_bgr"][0].get<int>(),
                sj["clean_color_bgr"][1].get<int>(),
                sj["clean_color_bgr"][2].get<int>()
            };
        }
    }

    if (j.contains("label_aliases") && j["label_aliases"].is_object()) {
        for (auto it = j["label_aliases"].begin(); it != j["label_aliases"].end(); ++it) {
            if (it.value().is_string()) {
                cfg_.label_aliases[normalize_label(it.key())] = normalize_label(it.value().get<std::string>());
            }
        }
    }

    if (j.contains("material_families") && j["material_families"].is_object()) {
        const json& mf = j["material_families"];
        if (mf.contains("classes") && mf["classes"].is_array()) {
            cfg_.family_classes.clear();
            for (const auto& c : mf["classes"]) {
                if (c.is_string()) cfg_.family_classes.push_back(normalize_label(c.get<std::string>()));
            }
        }
        if (mf.contains("metal_families") && mf["metal_families"].is_array()) {
            cfg_.metal_families.clear();
            for (const auto& c : mf["metal_families"]) {
                if (c.is_string()) cfg_.metal_families.push_back(normalize_label(c.get<std::string>()));
            }
        }
        cfg_.reject_family = normalize_label(json_or<std::string>(mf, "reject_family", cfg_.reject_family));
        cfg_.default_family = normalize_label(json_or<std::string>(mf, "default_family", cfg_.default_family));
        if (mf.contains("family_metadata") && mf["family_metadata"].is_object()) {
            for (auto it = mf["family_metadata"].begin(); it != mf["family_metadata"].end(); ++it) {
                std::string fname = normalize_label(it.key());
                const json& fj = it.value();
                FamilyMetadata meta;
                meta.name = fname;
                meta.display_name = json_or<std::string>(fj, "display_name", fname);
                meta.patchcore_always = json_or<bool>(fj, "patchcore_always", false);
                meta.default_conf_threshold = json_or<float>(fj, "default_conf_threshold", 0.28f);
                cfg_.family_metadata[fname] = meta;
            }
        }
    }

    if (j.contains("stage0") && j["stage0"].is_object()) {
        const json& s0 = j["stage0"];
        cfg_.stage0.model_path = json_or<std::string>(s0, "model_path", cfg_.stage0.model_path);
        cfg_.stage0.input_name = json_or<std::string>(s0, "input_name", cfg_.stage0.input_name);
        cfg_.stage0.output_name = json_or<std::string>(s0, "output_name", cfg_.stage0.output_name);
        cfg_.stage0.input_size = json_or<int>(s0, "input_size", cfg_.stage0.input_size);
        cfg_.stage0.confidence_threshold = json_or<float>(s0, "confidence_threshold", cfg_.stage0.confidence_threshold);
        cfg_.stage0.not_metal_threshold = json_or<float>(s0, "not_metal_threshold", cfg_.stage0.not_metal_threshold);
        cfg_.stage0.low_confidence_patchcore_trigger = json_or<float>(s0, "low_confidence_patchcore_trigger", cfg_.stage0.low_confidence_patchcore_trigger);
        if (s0.contains("classes") && s0["classes"].is_array() && !s0["classes"].empty()) {
            cfg_.stage0.classes.clear();
            for (const auto& c : s0["classes"]) {
                if (c.is_string()) cfg_.stage0.classes.push_back(normalize_label(c.get<std::string>()));
            }
        }
        if (s0.contains("visual_gate") && s0["visual_gate"].is_object()) {
            const json& vg = s0["visual_gate"];
            cfg_.stage0.visual_gate_enabled = json_or<bool>(vg, "enabled", cfg_.stage0.visual_gate_enabled);
            cfg_.stage0.saturation_pixel_threshold = json_or<int>(vg, "saturation_pixel_threshold", cfg_.stage0.saturation_pixel_threshold);
            cfg_.stage0.frame_max_mean_s = json_or<float>(vg, "frame_max_mean_s", cfg_.stage0.frame_max_mean_s);
            cfg_.stage0.frame_max_colorful_ratio = json_or<float>(vg, "frame_max_colorful_ratio", cfg_.stage0.frame_max_colorful_ratio);
        }
    }

    if (j.contains("vision") && j["vision"].is_object()) {
        const json& vj = j["vision"];
        cfg_.vision.hef_path = json_or<std::string>(vj, "hef_path", cfg_.vision.hef_path);
        cfg_.vision.onnx_path = json_or<std::string>(vj, "onnx_path", cfg_.vision.onnx_path);
        cfg_.vision.hailo_device_node = json_or<std::string>(vj, "hailo_device_node", cfg_.vision.hailo_device_node);
        cfg_.vision.input_name = json_or<std::string>(vj, "input_name", cfg_.vision.input_name);
        cfg_.vision.output_name = json_or<std::string>(vj, "output_name", cfg_.vision.output_name);
        cfg_.vision.input_size = json_or<int>(vj, "input_size", cfg_.vision.input_size);
        cfg_.vision.num_classes = json_or<int>(vj, "num_classes", cfg_.vision.num_classes);
        cfg_.vision.num_anchors = json_or<int>(vj, "num_anchors", cfg_.vision.num_anchors);
        cfg_.vision.conf_threshold = json_or<float>(vj, "conf_threshold", cfg_.vision.conf_threshold);
        cfg_.vision.nms_iou_threshold = json_or<float>(vj, "nms_iou_threshold", cfg_.vision.nms_iou_threshold);
        cfg_.vision.lock_in_frames = json_or<int>(vj, "lock_in_frames", cfg_.vision.lock_in_frames);
        cfg_.vision.ort_intra_op_threads = std::max(0, json_or<int>(vj, "ort_intra_op_threads", 0));
        if (vj.contains("tiling") && vj["tiling"].is_object()) {
            const json& tj = vj["tiling"];
            cfg_.vision.tiling.enabled = json_or<bool>(tj, "enabled", cfg_.vision.tiling.enabled);
            if (tj.contains("grid") && tj["grid"].is_array() && tj["grid"].size() >= 2) {
                cfg_.vision.tiling.grid_rows = std::max(1, tj["grid"][0].get<int>());
                cfg_.vision.tiling.grid_cols = std::max(1, tj["grid"][1].get<int>());
            }
            cfg_.vision.tiling.tile_size = json_or<int>(tj, "tile_size", cfg_.vision.tiling.tile_size);
            cfg_.vision.tiling.overlap = json_or<float>(tj, "overlap", cfg_.vision.tiling.overlap);
            cfg_.vision.tiling.include_full_frame = json_or<bool>(tj, "include_full_frame", cfg_.vision.tiling.include_full_frame);
            cfg_.vision.tiling.cross_tile_nms_iou = json_or<float>(tj, "cross_tile_nms_iou", cfg_.vision.tiling.cross_tile_nms_iou);
        }
        if (vj.contains("metallic_surface_gate") && vj["metallic_surface_gate"].is_object()) {
            const json& mg = vj["metallic_surface_gate"];
            cfg_.vision.metallic_gate.saturation_pixel_threshold = json_or<int>(mg, "saturation_pixel_threshold", cfg_.vision.metallic_gate.saturation_pixel_threshold);
            cfg_.vision.metallic_gate.box_max_mean_s = json_or<float>(mg, "box_max_mean_s", cfg_.vision.metallic_gate.box_max_mean_s);
            cfg_.vision.metallic_gate.box_max_colorful_ratio = json_or<float>(mg, "box_max_colorful_ratio", cfg_.vision.metallic_gate.box_max_colorful_ratio);
            cfg_.vision.metallic_gate.frame_max_mean_s = json_or<float>(mg, "frame_max_mean_s", cfg_.vision.metallic_gate.frame_max_mean_s);
            cfg_.vision.metallic_gate.frame_max_colorful_ratio = json_or<float>(mg, "frame_max_colorful_ratio", cfg_.vision.metallic_gate.frame_max_colorful_ratio);
        }
    }

    if (j.contains("patchcore") && j["patchcore"].is_object()) {
        const json& pj = j["patchcore"];
        cfg_.patchcore.enabled = json_or<bool>(pj, "enabled", cfg_.patchcore.enabled);
        cfg_.patchcore.reference_limit_per_family = json_or<int>(pj, "reference_limit_per_family", cfg_.patchcore.reference_limit_per_family);
        cfg_.patchcore.anomaly_threshold = json_or<float>(pj, "anomaly_threshold", cfg_.patchcore.anomaly_threshold);
        cfg_.patchcore.refs_dir = json_or<std::string>(pj, "refs_dir", cfg_.patchcore.refs_dir);
        cfg_.patchcore.banks_dir = json_or<std::string>(pj, "banks_dir", cfg_.patchcore.banks_dir);
        cfg_.patchcore.low_confidence_trigger = json_or<float>(pj, "low_confidence_trigger", cfg_.patchcore.low_confidence_trigger);
        if (pj.contains("families") && pj["families"].is_array()) {
            cfg_.patchcore.families.clear();
            for (const auto& f : pj["families"]) {
                if (f.is_string()) cfg_.patchcore.families.push_back(normalize_label(f.get<std::string>()));
            }
        }
        if (pj.contains("always_score_families") && pj["always_score_families"].is_array()) {
            cfg_.patchcore.always_score_families.clear();
            for (const auto& f : pj["always_score_families"]) {
                if (f.is_string()) cfg_.patchcore.always_score_families.push_back(normalize_label(f.get<std::string>()));
            }
        }
    }

    if (j.contains("llm") && j["llm"].is_object()) {
        const json& lj = j["llm"];
        cfg_.llm.pte_path = json_or<std::string>(lj, "pte_path", cfg_.llm.pte_path);
        cfg_.llm.tokenizer_path = json_or<std::string>(lj, "tokenizer_path", cfg_.llm.tokenizer_path);
        cfg_.llm.temperature = json_or<float>(lj, "temperature", cfg_.llm.temperature);
        cfg_.llm.seq_len = json_or<int>(lj, "seq_len", cfg_.llm.seq_len);
        cfg_.llm.max_new_tokens = json_or<int>(lj, "max_new_tokens", cfg_.llm.max_new_tokens);
        cfg_.llm.cpu_threads = json_or<int>(lj, "cpu_threads", cfg_.llm.cpu_threads);
        cfg_.llm.system_prompt = json_or<std::string>(lj, "system_prompt", cfg_.llm.system_prompt);
    }

    loaded_ = true;
    std::cout << "[QcRulesEngine] Loaded config (" << resolved << "): "
              << cfg_.defect_groups.size() << " defect groups, "
              << cfg_.label_aliases.size() << " label aliases, "
              << cfg_.family_classes.size() << " material families." << std::endl;
    return true;
}

std::string QcRulesEngine::map_label_to_group(const std::string& raw_label) const {
    std::string norm = normalize_label(raw_label);
    if (norm.empty()) return "scratch";

    auto it_alias = cfg_.label_aliases.find(norm);
    if (it_alias != cfg_.label_aliases.end()) {
        return it_alias->second;
    }
    if (cfg_.group_rules.count(norm)) {
        return norm;
    }
    for (const auto& g : cfg_.defect_groups) {
        if (norm.find(g) != std::string::npos) {
            return g;
        }
    }
    return "pit_void";
}

std::string QcRulesEngine::group_from_class_id(int class_id) const {
    if (class_id >= 0 && static_cast<size_t>(class_id) < cfg_.defect_groups.size()) {
        return cfg_.defect_groups[static_cast<size_t>(class_id)];
    }
    for (const auto& kv : cfg_.group_rules) {
        if (kv.second.id == class_id) {
            return kv.first;
        }
    }
    return "pit_void";
}

const GroupRule& QcRulesEngine::get_group_rule(const std::string& group) const {
    std::string g = map_label_to_group(group);
    auto it = cfg_.group_rules.find(g);
    if (it != cfg_.group_rules.end()) {
        return it->second;
    }
    return default_group_rule_;
}

std::array<int, 3> QcRulesEngine::get_group_bgr_color(const std::string& group) const {
    return get_group_rule(group).bgr_color;
}

float QcRulesEngine::get_group_conf_threshold(const std::string& group) const {
    return get_group_rule(group).conf_threshold;
}

bool QcRulesEngine::is_metal_family(const std::string& family) const {
    std::string f = normalize_label(family);
    if (f == cfg_.reject_family) return false;
    return std::find(cfg_.metal_families.begin(), cfg_.metal_families.end(), f) != cfg_.metal_families.end();
}

bool QcRulesEngine::should_always_score_patchcore(const std::string& family) const {
    std::string f = normalize_label(family);
    auto it = cfg_.family_metadata.find(f);
    if (it != cfg_.family_metadata.end() && it->second.patchcore_always) {
        return true;
    }
    return std::find(cfg_.patchcore.always_score_families.begin(),
                     cfg_.patchcore.always_score_families.end(), f) !=
           cfg_.patchcore.always_score_families.end();
}

std::string QcRulesEngine::get_family_display_name(const std::string& family) const {
    std::string f = normalize_label(family);
    auto it = cfg_.family_metadata.find(f);
    if (it != cfg_.family_metadata.end() && !it->second.display_name.empty()) {
        return it->second.display_name;
    }
    return family;
}

int QcRulesEngine::severity_rank(const std::string& severity) {
    std::string s = to_upper_copy(severity);
    if (s == "CRITICAL" || s == "HIGH") return 3;
    if (s == "MODERATE" || s == "MEDIUM") return 2;
    if (s == "MINOR" || s == "LOW") return 1;
    return 0;
}

std::string QcRulesEngine::max_severity(const std::string& a, const std::string& b) {
    return severity_rank(a) >= severity_rank(b) ? to_upper_copy(a) : to_upper_copy(b);
}

std::string QcRulesEngine::evaluate_group_severity(
    const std::string& group,
    float area_pct,
    float max_conf,
    int box_count
) const {
    if (box_count <= 0 && area_pct <= 0.0f) {
        return "NONE";
    }

    const GroupRule& rule = get_group_rule(group);
    std::string sev = "MINOR";
    if (area_pct < rule.area_thresholds.minor_max_pct) {
        sev = "MINOR";
    } else if (area_pct < rule.area_thresholds.moderate_max_pct) {
        sev = "MODERATE";
    } else {
        sev = "CRITICAL";
    }

    if (box_count >= rule.box_escalation_count) {
        if (rule.risk >= 3) {
            if (sev == "MINOR") sev = "MODERATE";
            else if (sev == "MODERATE") sev = "CRITICAL";
        } else if (rule.risk == 2) {
            if (sev == "MODERATE") sev = "CRITICAL";
            else if (sev == "MINOR" && box_count >= rule.box_escalation_count + 2) sev = "MODERATE";
        }
    }

    if (rule.risk >= 3 && max_conf >= 0.85f && area_pct >= rule.area_thresholds.minor_max_pct && sev == "MINOR") {
        sev = "MODERATE";
    }
    return sev;
}

std::string QcRulesEngine::evaluate_anomaly_severity(float anomaly_score) const {
    if (anomaly_score >= cfg_.severity_verdict.anomaly_only_critical_threshold) return "CRITICAL";
    if (anomaly_score >= cfg_.severity_verdict.anomaly_only_moderate_threshold) return "MODERATE";
    if (anomaly_score >= cfg_.severity_verdict.anomaly_only_minor_threshold) return "MINOR";
    return "NONE";
}

std::string QcRulesEngine::severity_to_verdict(const std::string& severity) const {
    std::string s = to_upper_copy(severity);
    auto it = cfg_.severity_verdict.verdict_by_severity.find(s);
    if (it != cfg_.severity_verdict.verdict_by_severity.end()) {
        return it->second;
    }
    if (s == "CRITICAL") return "REJECT";
    if (s == "MODERATE" || s == "MINOR") return "REVIEW";
    return "PASS";
}

MultiDefectAnalysis QcRulesEngine::analyze_detections(
    const std::vector<RawBox>& detections,
    int w_orig,
    int h_orig,
    const std::string& family,
    float anomaly_score,
    bool is_anomalous
) const {
    MultiDefectAnalysis res;
    res.family = normalize_label(family);
    if (res.family.empty()) res.family = cfg_.default_family;

    if (res.family == cfg_.reject_family) {
        res.has_defects = false;
        res.is_not_metal = true;
        res.primary_label = "not_metal";
        res.primary_group = "NONE";
        res.signature = "NOT_METAL";
        res.display_title = "NON-METAL / REJECT GATE";
        res.overall_severity = "NONE";
        res.verdict = severity_to_verdict("NONE");
        return res;
    }

    if ((detections.empty() && !is_anomalous) || w_orig <= 0 || h_orig <= 0) {
        res.has_defects = false;
        res.primary_label = "NONE";
        res.primary_group = "NONE";
        res.signature = "NONE";
        res.display_title = "NO DEFECT DETECTED";
        res.overall_severity = "NONE";
        res.verdict = severity_to_verdict("NONE");
        return res;
    }

    if ((detections.empty() || (detections.size() == 1 && detections[0].label == "anomaly")) && is_anomalous) {
        res.has_defects = true;
        res.primary_label = "anomaly";
        res.primary_group = "surface_anomaly";
        res.signature = "anomaly";
        res.display_title = "SURFACE ANOMALY";
        res.highest_conf = detections.empty()
            ? std::min(0.99f, std::max(0.0f, anomaly_score))
            : detections[0].conf;
        res.total_area_pct = detections.empty()
            ? 0.0f
            : compute_union_area_pct(detections, w_orig, h_orig);
        res.total_boxes = static_cast<int>(detections.size());
        res.spread = "localized";
        res.overall_severity = evaluate_anomaly_severity(anomaly_score);
        if (res.overall_severity == "NONE") res.overall_severity = "MINOR";
        res.verdict = severity_to_verdict(res.overall_severity);
        if (!detections.empty()) {
            GroupSummary gs;
            gs.group = "surface_anomaly";
            gs.primary_label = "anomaly";
            gs.boxes = detections;
            gs.box_count = 1;
            gs.area_pct = res.total_area_pct;
            gs.max_conf = res.highest_conf;
            gs.conf_pct = res.highest_conf * 100.0f;
            gs.pattern = "localized";
            gs.severity = res.overall_severity;
            gs.verdict = res.verdict;
            res.groups.push_back(std::move(gs));
        }
        return res;
    }

    res.has_defects = true;
    res.total_boxes = static_cast<int>(detections.size());
    res.total_area_pct = compute_union_area_pct(detections, w_orig, h_orig);

    std::map<std::string, std::vector<RawBox>> by_group;
    for (const auto& d : detections) {
        std::string g = d.group.empty() ? map_label_to_group(d.label) : map_label_to_group(d.group);
        RawBox normalized = d;
        normalized.group = g;
        if (normalized.label.empty()) normalized.label = g;
        by_group[g].push_back(normalized);
    }

    std::string max_sev = "NONE";
    for (const auto& pair : by_group) {
        GroupSummary summary;
        summary.group = pair.first;
        summary.boxes = pair.second;
        summary.box_count = static_cast<int>(pair.second.size());
        summary.area_pct = compute_union_area_pct(pair.second, w_orig, h_orig);

        float max_c = -1.0f;
        std::string best_lbl = pair.first;
        for (const auto& b : pair.second) {
            if (b.conf > max_c) {
                max_c = b.conf;
                best_lbl = b.label;
            }
        }
        summary.primary_label = best_lbl;
        summary.max_conf = std::max(0.0f, max_c);
        summary.conf_pct = summary.max_conf * 100.0f;
        summary.pattern = (summary.box_count >= 2 || summary.area_pct >= 20.0f) ? "clustered" : "localized";
        summary.severity = evaluate_group_severity(summary.group, summary.area_pct, summary.max_conf, summary.box_count);
        summary.verdict = severity_to_verdict(summary.severity);
        summary.dominance = summary.conf_pct * (1.0f + summary.area_pct / 50.0f) +
                            static_cast<float>(severity_rank(summary.severity)) * 25.0f;

        const GroupRule& rule = get_group_rule(summary.group);
        summary.display_name = rule.display_name;
        summary.bgr_color = rule.bgr_color;

        max_sev = max_severity(max_sev, summary.severity);
        res.groups.push_back(std::move(summary));
    }

    std::sort(res.groups.begin(), res.groups.end(), [](const GroupSummary& a, const GroupSummary& b) {
        return a.dominance > b.dominance;
    });

    if (!res.groups.empty()) {
        res.primary_group = res.groups[0].group;
        res.primary_label = res.groups[0].primary_label;
        res.highest_conf = res.groups[0].max_conf;
        res.spread = (res.total_boxes >= 2 || res.total_area_pct >= 20.0f) ? "clustered" : "localized";

        std::stringstream ss_sig, ss_title;
        for (size_t i = 0; i < res.groups.size(); ++i) {
            if (i > 0) {
                ss_sig << " + ";
                ss_title << " + ";
            }
            ss_sig << res.groups[i].group;
            std::string t = res.groups[i].group;
            if (res.groups.size() == 1 && !res.groups[i].primary_label.empty() &&
                res.groups[i].primary_label != "anomaly" &&
                res.groups[i].primary_label != res.groups[i].group) {
                if (res.groups[i].primary_label == "bent") {
                    t = "BENT DEFORMATION";
                } else {
                    t = res.groups[i].primary_label;
                }
            }
            std::replace(t.begin(), t.end(), '_', ' ');
            ss_title << to_upper_copy(t);
        }
        res.signature = ss_sig.str();
        res.display_title = ss_title.str();
    }

    if (is_anomalous) {
        max_sev = max_severity(max_sev, evaluate_anomaly_severity(anomaly_score));
    }
    res.overall_severity = max_sev;
    res.verdict = severity_to_verdict(res.overall_severity);
    return res;
}

std::string QcRulesEngine::build_llama_advisory_prompt(
    const std::string& family,
    const std::string& label,
    const std::string& group,
    const std::string& severity,
    float confidence,
    float area_pct,
    int box_count,
    float anomaly_score
) const {
    (void)box_count;
    // Keep prompt <= ~110 tokens so 256-ctx INT4 .pte has ~145 tokens for rich 2-3 sentence 2-part advisories
    std::string sys_prompt = cfg_.llm.system_prompt;
    if (sys_prompt.empty() || sys_prompt.size() > 300) {
        sys_prompt = "Industrial Metal QC AI. Write two full detailed technical sentences each (plain text): Possible Cause: <physics, material stress, process trigger> | Recommended Action: <containment, machine fix, verification>";
    }

    const std::string fam = family.empty() ? cfg_.default_family : family;
    const int conf_pct = static_cast<int>(std::lround(confidence * 100.0f));
    std::ostringstream user_msg;

    if (fam == "not_metal" || label == "not_metal") {
        user_msg << "Family: not_metal | Status: REJECT (non-metallic object in ROI) | Sev: NONE"
                 << " | Conf: " << conf_pct << "%"
                 << " | Area: 0.0%"
                 << " | Anom: " << std::fixed << std::setprecision(2) << anomaly_score;
    } else if (label == "clean" || severity == "PASS") {
        user_msg << "Family: " << fam
                 << " | Status: PASS (clean nominal surface, no defects) | Sev: NONE"
                 << " | Conf: " << conf_pct << "%"
                 << " | Area: 0.0%"
                 << " | Anom: " << std::fixed << std::setprecision(2) << anomaly_score;
    } else {
        std::string sev_short = severity.empty() ? "MODERATE" : severity;
        user_msg << "Family: " << fam
                 << " | Defect: " << (label.empty() ? "anomaly" : label)
                 << " (" << (group.empty() ? "pit_void" : group) << ")"
                 << " | Sev: " << sev_short
                 << " | Conf: " << conf_pct << "%"
                 << " | Area: " << std::fixed << std::setprecision(1) << area_pct << "%"
                 << " | Anom: " << std::fixed << std::setprecision(2) << anomaly_score;
    }

    std::ostringstream prompt;
    prompt << "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
           << sys_prompt << "<|eot_id|>"
           << "<|start_header_id|>user<|end_header_id|>\n\n"
           << user_msg.str() << "<|eot_id|>"
           << "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return prompt.str();
}

} // namespace rules
} // namespace qc
