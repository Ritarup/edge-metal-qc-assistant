#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace qc {
namespace vision {

struct PatchCoreResult {
    float anomaly_score = 0.0f;
    float threshold = 0.55f;
    bool is_anomalous = false;
    int bank_size = 0;
    float peak_ratio = 0.0f;
    cv::Rect hotspot_bbox;
    cv::Rect core_bbox;
    std::vector<std::vector<cv::Point>> hotspot_contours;
    cv::Mat heatmap; // CANONICAL_SIZE x CANONICAL_SIZE CV_32FC1 normalized score ratio (1.0 == threshold)
};

/**
 * Real-time C++17 PatchCore Anomaly Detection Engine for multi-metal Edge QC.
 * Maintains per-family coreset memory banks of multi-scale OpenCV patch feature
 * descriptors (Lab color/chroma deviations, Sobel/Laplacian gradients, local
 * structure tensor trace & anisotropy, multi-scale DoG, and directional LBP)
 * with binary disk persistence and thread-safe runtime good-image registration.
 */
class PatchCoreEngine {
public:
    static constexpr int CANONICAL_SIZE = 192;
    static constexpr int DESC_DIM = 28;
    static constexpr int MAX_CORESET_SIZE = 768;

    PatchCoreEngine() = default;
    ~PatchCoreEngine() = default;

    PatchCoreEngine(const PatchCoreEngine&) = delete;
    PatchCoreEngine& operator=(const PatchCoreEngine&) = delete;

    /**
     * Initializes per-family memory banks by loading persisted `<family>.bin`
     * files from `banks_dir` and/or indexing existing reference images under
     * `refs_dir/<family>/` (up to `max_refs_per_family`).
     */
    bool init(
        const std::string& refs_dir,
        const std::string& banks_dir,
        float default_threshold = 0.55f,
        int max_refs_per_family = 20
    );

    /**
     * Registers a good reference image for `family`:
     * saves a PNG under `refs_dir/<family>/` (rotating out oldest files above
     * `max_refs_per_family`), updates the in-memory coreset patch bank, and
     * persists `banks_dir/<family>.bin`.
     */
    bool register_good_image(
        const std::string& family,
        const cv::Mat& bgr_img,
        int* total_saved_out = nullptr
    );

    /**
     * Scores `bgr_img` against the memory bank for `family`.
     * Returns normalized anomaly_score in [0, 1], active threshold, anomaly flag,
     * coreset bank_size, and localized hotspot_bbox in original pixel coordinates.
     */
    PatchCoreResult score_image(const std::string& family, const cv::Mat& bgr_img) const;

    /**
     * Returns the number of coreset patch vectors currently loaded for `family`.
     */
    int bank_size(const std::string& family) const;

    /**
     * Returns the number of indexed/persisted reference images for `family`.
     */
    int reference_count(const std::string& family) const;

    /**
     * Sets or queries a per-family anomaly decision threshold override.
     */
    void set_family_threshold(const std::string& family, float threshold);
    float get_family_threshold(const std::string& family) const;

    /**
     * Utility accessors for active memory banks.
     */
    bool has_bank(const std::string& family) const;
    std::vector<std::string> list_families() const;

private:
    struct FamilyBank {
        int num_vectors = 0;
        int desc_dim = DESC_DIM;
        int ref_count = 0;
        float calib_scale = 0.44f;
        uint64_t refs_signature = 0;
        float threshold = 0.55f;
        bool has_custom_threshold = false;
        std::vector<float> descriptors; // Row-major [num_vectors * desc_dim]
        std::vector<float> sq_norms;    // Precomputed L2 squared norms [num_vectors]
        std::vector<float> base_descriptors; // Seed coreset loaded from .bin when refs dir has no PNGs
        float base_calib_scale = 0.44f;

        // Cached per-reference descriptors for incremental registration updates
        std::vector<std::string> ref_paths;
        std::vector<std::vector<float>> ref_bank_blocks;
        std::vector<std::vector<float>> ref_query_blocks;
    };

    std::string refs_dir_ = "models/patchcore_refs";
    std::string banks_dir_ = "models/patchcore_banks";
    float default_threshold_ = 0.55f;
    int max_refs_per_family_ = 20;
    uint64_t save_seq_ = 0;

    mutable std::mutex mutex_;
    std::map<std::string, FamilyBank> banks_;

    static std::string sanitize_family(const std::string& family);
    static bool is_supported_image_ext(const std::string& ext);
    static std::vector<std::string> list_sorted_ref_images(const std::string& family_dir);
    static uint64_t compute_refs_signature(const std::vector<std::string>& paths);

    static bool extract_descriptors(
        const cv::Mat& bgr_img,
        int stride,
        std::vector<float>& out_descs,
        int& out_grid_w,
        int& out_grid_h
    );

    static void greedy_coreset_select(
        const float* candidates,
        int num_candidates,
        int desc_dim,
        int target_count,
        std::vector<float>& out_selected
    );

    static void rebuild_bank_from_cache(FamilyBank& bank);
    bool load_bank_bin(const std::string& bin_path, FamilyBank& bank) const;
    bool save_bank_bin(const std::string& bin_path, const FamilyBank& bank) const;
    bool build_family_from_disk(const std::string& family, FamilyBank& bank) const;
};

} // namespace vision
} // namespace qc
