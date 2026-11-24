#include "patchcore_engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace fs = std::filesystem;

namespace qc {
namespace vision {

namespace {

constexpr uint32_t BANK_MAGIC = 0x50434232U; // "PCB2"
constexpr uint32_t BANK_VERSION = 2U;
constexpr int REF_EXTRACT_STRIDE = 6;
constexpr int QUERY_EXTRACT_STRIDE = 12;
constexpr int PER_REF_CACHE_MAX = 256;
constexpr int MARGIN_PX = 8;

const std::array<const char*, 5> DEFAULT_METAL_FAMILIES = {
    "steel_sheet", "aluminium", "weld", "casting", "machined_part"
};

struct FileEntry {
    std::string path;
    std::string filename;
    uint64_t file_size = 0;
    fs::file_time_type mtime{};
};

std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

cv::Mat normalize_to_bgr8(const cv::Mat& img) {
    if (img.empty()) {
        return {};
    }
    cv::Mat tmp;
    if (img.channels() == 1) {
        cv::cvtColor(img, tmp, cv::COLOR_GRAY2BGR);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, tmp, cv::COLOR_BGRA2BGR);
    } else if (img.channels() == 3) {
        tmp = img;
    } else {
        return {};
    }
    if (tmp.depth() != CV_8U) {
        double min_v = 0.0, max_v = 0.0;
        cv::minMaxLoc(tmp.reshape(1), &min_v, &max_v);
        cv::Mat converted;
        if (max_v <= 1.5 && min_v >= -0.5) {
            tmp.convertTo(converted, CV_8U, 255.0);
        } else {
            tmp.convertTo(converted, CV_8U);
        }
        return converted;
    }
    return tmp;
}

uint64_t fnv1a_hash(uint64_t h, const void* data, size_t len) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(bytes[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

std::vector<FileEntry> scan_family_images(const std::string& family_dir, bool sort_by_mtime_asc) {
    std::vector<FileEntry> entries;
    std::error_code ec;
    if (!fs::exists(family_dir, ec) || !fs::is_directory(family_dir, ec)) {
        return entries;
    }
    for (const auto& entry : fs::directory_iterator(family_dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        std::string ext = to_lower_copy(entry.path().extension().string());
        if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp" && ext != ".webp") {
            continue;
        }
        FileEntry fe;
        fe.path = entry.path().string();
        fe.filename = entry.path().filename().string();
        fe.file_size = static_cast<uint64_t>(entry.file_size(ec));
        fe.mtime = entry.last_write_time(ec);
        entries.push_back(std::move(fe));
    }
    if (sort_by_mtime_asc) {
        std::sort(entries.begin(), entries.end(), [](const FileEntry& a, const FileEntry& b) {
            if (a.mtime != b.mtime) {
                return a.mtime < b.mtime;
            }
            return a.filename < b.filename;
        });
    } else {
        std::sort(entries.begin(), entries.end(), [](const FileEntry& a, const FileEntry& b) {
            return a.filename < b.filename;
        });
    }
    return entries;
}

inline float l2_sq_vec(const float* a, const float* b, int dim) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int d = 0;
    for (; d + 8 <= dim; d += 8) {
        const float32x4_t d0 = vsubq_f32(vld1q_f32(a + d), vld1q_f32(b + d));
        const float32x4_t d1 = vsubq_f32(vld1q_f32(a + d + 4), vld1q_f32(b + d + 4));
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
    }
    for (; d + 4 <= dim; d += 4) {
        const float32x4_t d0 = vsubq_f32(vld1q_f32(a + d), vld1q_f32(b + d));
        acc0 = vfmaq_f32(acc0, d0, d0);
    }
    float s = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; d < dim; ++d) {
        const float diff = a[d] - b[d];
        s += diff * diff;
    }
    return s;
#else
    float s = 0.0f;
    for (int d = 0; d < dim; ++d) {
        const float diff = a[d] - b[d];
        s += diff * diff;
    }
    return s;
#endif
}

float compute_top3_mean_nn(
    const float* queries,
    int n_queries,
    const float* bank,
    int n_bank,
    int dim,
    std::vector<float>* out_nn_dists = nullptr
) {
    if (n_queries <= 0 || n_bank <= 0) {
        if (out_nn_dists) {
            out_nn_dists->clear();
        }
        return 0.0f;
    }
    std::vector<float> nn_dists(static_cast<size_t>(n_queries));
    for (int i = 0; i < n_queries; ++i) {
        const float* q = queries + static_cast<size_t>(i) * dim;
        float min_sq = std::numeric_limits<float>::max();
        for (int j = 0; j < n_bank; ++j) {
            const float* m = bank + static_cast<size_t>(j) * dim;
            const float d_sq = l2_sq_vec(q, m, dim);
            if (d_sq < min_sq) {
                min_sq = d_sq;
            }
        }
        nn_dists[static_cast<size_t>(i)] = std::sqrt(std::max(0.0f, min_sq));
    }
    if (out_nn_dists) {
        *out_nn_dists = nn_dists;
    }
    const int k = std::min(3, n_queries);
    std::partial_sort(nn_dists.begin(), nn_dists.begin() + k, nn_dists.end(), std::greater<float>());
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        sum += nn_dists[static_cast<size_t>(i)];
    }
    return sum / static_cast<float>(k);
}

std::vector<float> subsample_query_from_dense(const std::vector<float>& dense, int gw, int gh, int desc_dim) {
    std::vector<float> q_descs;
    const int q_gh = (gh + 1) / 2;
    const int q_gw = (gw + 1) / 2;
    q_descs.reserve(static_cast<size_t>(q_gh * q_gw) * desc_dim);
    for (int r = 0; r < gh; r += 2) {
        for (int c = 0; c < gw; c += 2) {
            const float* src = dense.data() + static_cast<size_t>(r * gw + c) * desc_dim;
            q_descs.insert(q_descs.end(), src, src + desc_dim);
        }
    }
    return q_descs;
}

} // namespace

std::string PatchCoreEngine::sanitize_family(const std::string& family) {
    std::string cleaned;
    cleaned.reserve(family.size());
    for (unsigned char c : family) {
        if (std::isalnum(c) || c == '_' || c == '-') {
            cleaned.push_back(static_cast<char>(std::tolower(c)));
        }
    }
    if (cleaned == "not_metal" || cleaned == "." || cleaned == "..") {
        return {};
    }
    return cleaned;
}

bool PatchCoreEngine::is_supported_image_ext(const std::string& ext) {
    std::string e = to_lower_copy(ext);
    return e == ".png" || e == ".jpg" || e == ".jpeg" || e == ".bmp" || e == ".webp";
}

std::vector<std::string> PatchCoreEngine::list_sorted_ref_images(const std::string& family_dir) {
    auto entries = scan_family_images(family_dir, true);
    std::vector<std::string> out;
    out.reserve(entries.size());
    for (auto& e : entries) {
        out.push_back(std::move(e.path));
    }
    return out;
}

uint64_t PatchCoreEngine::compute_refs_signature(const std::vector<std::string>& paths) {
    uint64_t h = 14695981039346656037ULL;
    std::vector<std::pair<std::string, uint64_t>> items;
    items.reserve(paths.size());
    std::error_code ec;
    for (const auto& p : paths) {
        fs::path fp(p);
        uint64_t sz = static_cast<uint64_t>(fs::file_size(fp, ec));
        items.emplace_back(fp.filename().string(), sz);
    }
    std::sort(items.begin(), items.end());
    for (const auto& item : items) {
        h = fnv1a_hash(h, item.first.data(), item.first.size());
        h = fnv1a_hash(h, &item.second, sizeof(item.second));
    }
    return h;
}

bool PatchCoreEngine::extract_descriptors(
    const cv::Mat& bgr_img,
    int stride,
    std::vector<float>& out_descs,
    int& out_grid_w,
    int& out_grid_h
) {
    cv::Mat work = normalize_to_bgr8(bgr_img);
    if (work.empty()) {
        out_descs.clear();
        out_grid_w = 0;
        out_grid_h = 0;
        return false;
    }

    cv::Mat resized;
    if (work.cols == CANONICAL_SIZE && work.rows == CANONICAL_SIZE) {
        resized = work;
    } else {
        const int interp = (work.cols > CANONICAL_SIZE || work.rows > CANONICAL_SIZE)
                               ? cv::INTER_AREA
                               : cv::INTER_LINEAR;
        cv::resize(work, resized, cv::Size(CANONICAL_SIZE, CANONICAL_SIZE), 0.0, 0.0, interp);
    }

    cv::Mat smoothed;
    cv::GaussianBlur(resized, smoothed, cv::Size(3, 3), 0.85);

    cv::Mat lab_u8;
    cv::cvtColor(smoothed, lab_u8, cv::COLOR_BGR2Lab);
    cv::Mat lab_f32;
    lab_u8.convertTo(lab_f32, CV_32F);

    std::vector<cv::Mat> lab_ch(3);
    cv::split(lab_f32, lab_ch);

    const cv::Mat& L_u8 = lab_ch[0];
    cv::Mat L = L_u8 * (1.0f / 255.0f);
    cv::Mat A = (lab_ch[1] - 128.0f) * (1.0f / 64.0f);
    cv::Mat B = (lab_ch[2] - 128.0f) * (1.0f / 64.0f);

    cv::Mat A_sm, B_sm, A_bg, B_bg;
    cv::GaussianBlur(A, A_sm, cv::Size(5, 5), 1.2);
    cv::GaussianBlur(B, B_sm, cv::Size(5, 5), 1.2);
    cv::boxFilter(A, A_bg, CV_32F, cv::Size(21, 21), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    cv::boxFilter(B, B_bg, CV_32F, cv::Size(21, 21), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    cv::Mat A_spot_raw = A_sm - A_bg;
    cv::Mat B_spot_raw = B_sm - B_bg;

    cv::Mat gx, gy, lap_raw;
    cv::Sobel(L_u8, gx, CV_32F, 1, 0, 3, 1.0 / 255.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(L_u8, gy, CV_32F, 0, 1, 3, 1.0 / 255.0, 0.0, cv::BORDER_REPLICATE);
    cv::Laplacian(L_u8, lap_raw, CV_32F, 3, 1.0 / 255.0, 0.0, cv::BORDER_REPLICATE);
    cv::Mat lap = cv::abs(lap_raw);

    cv::Mat gx2 = gx.mul(gx);
    cv::Mat gy2 = gy.mul(gy);
    cv::Mat gxy = gx.mul(gy);
    cv::Mat mag;
    cv::sqrt(gx2 + gy2, mag);

    cv::Mat agx, agy, bgx, bgy;
    cv::Sobel(A_sm, agx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(A_sm, agy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(B_sm, bgx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(B_sm, bgy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Mat cmag;
    cv::sqrt(agx.mul(agx) + agy.mul(agy) + bgx.mul(bgx) + bgy.mul(bgy), cmag);

    cv::Mat L_b7, L_b17;
    cv::GaussianBlur(L, L_b7, cv::Size(7, 7), 1.4);
    cv::GaussianBlur(L, L_b17, cv::Size(17, 17), 3.5);
    cv::Mat dog5 = cv::abs(L - L_b7);
    cv::Mat dog15 = cv::abs(L - L_b17);

    cv::Mat Jxx, Jyy, Jxy;
    cv::boxFilter(gx2, Jxx, CV_32F, cv::Size(11, 11), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    cv::boxFilter(gy2, Jyy, CV_32F, cv::Size(11, 11), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    cv::boxFilter(gxy, Jxy, CV_32F, cv::Size(11, 11), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    cv::Mat trace = Jxx + Jyy;
    cv::Mat diff_j = Jxx - Jyy;
    cv::Mat eig_diff;
    cv::sqrt(diff_j.mul(diff_j) + 4.0f * Jxy.mul(Jxy), eig_diff);
    cv::Mat tr_pos;
    cv::max(trace, 0.0f, tr_pos);
    cv::Mat tr_sqrt_raw;
    cv::sqrt(tr_pos, tr_sqrt_raw);
    cv::Mat aniso_raw = eig_diff / (trace + 3e-4f);

    cv::Mat L_loc_raw, L2_loc, A_loc_raw, B_loc_raw;
    cv::boxFilter(L, L_loc_raw, CV_32F, cv::Size(11, 11), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    cv::boxFilter(L.mul(L), L2_loc, CV_32F, cv::Size(11, 11), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    cv::Mat L_var;
    cv::max(L2_loc - L_loc_raw.mul(L_loc_raw), 0.0f, L_var);
    cv::Mat L_std_raw;
    cv::sqrt(L_var, L_std_raw);
    cv::boxFilter(A, A_loc_raw, CV_32F, cv::Size(11, 11), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    cv::boxFilter(B, B_loc_raw, CV_32F, cv::Size(11, 11), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);

    // 4-orientation second-difference directional micro-texture (LBP-like energy)
    cv::Mat lbp_max(CANONICAL_SIZE, CANONICAL_SIZE, CV_32F);
    for (int y = 0; y < CANONICAL_SIZE; ++y) {
        const int ym = std::max(0, y - 2);
        const int yp = std::min(CANONICAL_SIZE - 1, y + 2);
        const float* row_m = L.ptr<float>(ym);
        const float* row_c = L.ptr<float>(y);
        const float* row_p = L.ptr<float>(yp);
        float* dst = lbp_max.ptr<float>(y);
        for (int x = 0; x < CANONICAL_SIZE; ++x) {
            const int xm = std::max(0, x - 2);
            const int xp = std::min(CANONICAL_SIZE - 1, x + 2);
            const float c2 = 2.0f * row_c[x];
            const float d1 = std::abs(c2 - row_m[x] - row_p[x]);
            const float d2 = std::abs(c2 - row_c[xm] - row_c[xp]);
            const float d3 = std::abs(c2 - row_m[xm] - row_p[xp]);
            const float d4 = std::abs(c2 - row_m[xp] - row_p[xm]);
            dst[x] = std::max(std::max(d1, d2), std::max(d3, d4));
        }
    }

    std::array<cv::Mat, 14> fine;
    fine[0] = L_loc_raw * 0.7f;
    fine[1] = L_std_raw * 4.0f;
    fine[2] = A_loc_raw * 4.0f;
    fine[3] = B_loc_raw * 4.0f;
    cv::boxFilter(A_spot_raw, fine[4], CV_32F, cv::Size(7, 7), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    fine[4] *= 10.0f;
    cv::boxFilter(B_spot_raw, fine[5], CV_32F, cv::Size(7, 7), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    fine[5] *= 10.0f;
    cv::boxFilter(mag, fine[6], CV_32F, cv::Size(11, 11), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    fine[6] *= 2.8f;
    cv::boxFilter(cmag, fine[7], CV_32F, cv::Size(9, 9), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    fine[7] *= 1.6f;
    cv::boxFilter(dog5, fine[8], CV_32F, cv::Size(9, 9), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    fine[8] *= 10.0f;
    cv::boxFilter(dog15, fine[9], CV_32F, cv::Size(11, 11), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    fine[9] *= 6.0f;
    cv::boxFilter(lap, fine[10], CV_32F, cv::Size(9, 9), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    fine[10] *= 3.0f;
    fine[11] = tr_sqrt_raw * 3.0f;

    cv::Mat gate;
    cv::min(tr_sqrt_raw * 8.0f, 1.0f, gate);
    fine[12] = aniso_raw.mul(gate) * 1.5f;
    cv::boxFilter(lbp_max, fine[13], CV_32F, cv::Size(11, 11), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    fine[13] *= 4.0f;

    std::array<cv::Mat, 14> ctx;
    cv::Mat half_tmp;
    const cv::Size half_sz(CANONICAL_SIZE / 2, CANONICAL_SIZE / 2);
    for (int c = 0; c < 14; ++c) {
        cv::resize(fine[c], half_tmp, half_sz, 0.0, 0.0, cv::INTER_AREA);
        cv::boxFilter(half_tmp, ctx[c], CV_32F, cv::Size(11, 11), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    }

    int gw = 0;
    int gh = 0;
    for (int y = MARGIN_PX; y < CANONICAL_SIZE - MARGIN_PX; y += stride) {
        ++gh;
    }
    for (int x = MARGIN_PX; x < CANONICAL_SIZE - MARGIN_PX; x += stride) {
        ++gw;
    }

    out_grid_w = gw;
    out_grid_h = gh;
    out_descs.resize(static_cast<size_t>(gw * gh) * DESC_DIM);

    int idx = 0;
    for (int y = MARGIN_PX; y < CANONICAL_SIZE - MARGIN_PX; y += stride) {
        const int hy = y >> 1;
        for (int x = MARGIN_PX; x < CANONICAL_SIZE - MARGIN_PX; x += stride) {
            const int hx = x >> 1;
            float* dst = out_descs.data() + static_cast<size_t>(idx) * DESC_DIM;
            for (int c = 0; c < 14; ++c) {
                dst[c] = fine[c].ptr<float>(y)[x];
                dst[14 + c] = ctx[c].ptr<float>(hy)[hx] * 0.65f;
            }
            ++idx;
        }
    }
    return true;
}

void PatchCoreEngine::greedy_coreset_select(
    const float* candidates,
    int num_candidates,
    int desc_dim,
    int target_count,
    std::vector<float>& out_selected
) {
    if (num_candidates <= 0 || target_count <= 0) {
        out_selected.clear();
        return;
    }
    if (num_candidates <= target_count) {
        out_selected.assign(candidates, candidates + static_cast<size_t>(num_candidates) * desc_dim);
        return;
    }

    out_selected.resize(static_cast<size_t>(target_count) * desc_dim);
    std::vector<float> min_dist_sq(static_cast<size_t>(num_candidates), std::numeric_limits<float>::max());

    int chosen = num_candidates / 2;
    for (int step = 0; step < target_count; ++step) {
        const float* center = candidates + static_cast<size_t>(chosen) * desc_dim;
        std::memcpy(
            out_selected.data() + static_cast<size_t>(step) * desc_dim,
            center,
            static_cast<size_t>(desc_dim) * sizeof(float)
        );
        if (step + 1 == target_count) {
            break;
        }
        int next_chosen = 0;
        float max_min_sq = -1.0f;
        for (int i = 0; i < num_candidates; ++i) {
            const float* pt = candidates + static_cast<size_t>(i) * desc_dim;
            const float d_sq = l2_sq_vec(pt, center, desc_dim);
            if (d_sq < min_dist_sq[static_cast<size_t>(i)]) {
                min_dist_sq[static_cast<size_t>(i)] = d_sq;
            }
            if (min_dist_sq[static_cast<size_t>(i)] > max_min_sq) {
                max_min_sq = min_dist_sq[static_cast<size_t>(i)];
                next_chosen = i;
            }
        }
        chosen = next_chosen;
    }
}

void PatchCoreEngine::rebuild_bank_from_cache(FamilyBank& bank) {
    const int k = static_cast<int>(bank.ref_bank_blocks.size());
    const int base_vecs = static_cast<int>(bank.base_descriptors.size()) / DESC_DIM;
    bank.ref_count = k;
    bank.desc_dim = DESC_DIM;

    if (k == 0 && base_vecs == 0) {
        bank.num_vectors = 0;
        bank.descriptors.clear();
        bank.sq_norms.clear();
        bank.calib_scale = 0.44f;
        return;
    }

    std::vector<float> all_candidates;
    int total_cands = base_vecs;
    for (const auto& blk : bank.ref_bank_blocks) {
        total_cands += static_cast<int>(blk.size()) / DESC_DIM;
    }
    all_candidates.reserve(static_cast<size_t>(total_cands) * DESC_DIM);
    if (base_vecs > 0) {
        all_candidates.insert(all_candidates.end(), bank.base_descriptors.begin(), bank.base_descriptors.end());
    }
    for (const auto& blk : bank.ref_bank_blocks) {
        all_candidates.insert(all_candidates.end(), blk.begin(), blk.end());
    }

    greedy_coreset_select(
        all_candidates.data(),
        total_cands,
        DESC_DIM,
        MAX_CORESET_SIZE,
        bank.descriptors
    );
    bank.num_vectors = static_cast<int>(bank.descriptors.size()) / DESC_DIM;

    bank.sq_norms.resize(static_cast<size_t>(bank.num_vectors));
    for (int m = 0; m < bank.num_vectors; ++m) {
        const float* row = bank.descriptors.data() + static_cast<size_t>(m) * DESC_DIM;
        float s = 0.0f;
        for (int d = 0; d < DESC_DIM; ++d) {
            s += row[d] * row[d];
        }
        bank.sq_norms[static_cast<size_t>(m)] = s;
    }

    // Calibrate baseline scale via leave-one-image-out normal variance when K >= 3
    if (k < 3 || bank.ref_query_blocks.size() != static_cast<size_t>(k)) {
        bank.calib_scale = (base_vecs > 0) ? bank.base_calib_scale : 0.44f;
        return;
    }

    const int step_sub = (k > 6) ? 2 : 1;
    std::vector<float> loo_scores;
    loo_scores.reserve(static_cast<size_t>(k));
    std::vector<float> other_bank;

    for (int i = 0; i < k; ++i) {
        other_bank.clear();
        for (int j = 0; j < k; ++j) {
            if (j == i) {
                continue;
            }
            const auto& blk = bank.ref_bank_blocks[static_cast<size_t>(j)];
            const int n_blk = static_cast<int>(blk.size()) / DESC_DIM;
            for (int p = 0; p < n_blk; p += step_sub) {
                const float* src = blk.data() + static_cast<size_t>(p) * DESC_DIM;
                other_bank.insert(other_bank.end(), src, src + DESC_DIM);
            }
        }
        const int n_other = static_cast<int>(other_bank.size()) / DESC_DIM;
        const auto& qb = bank.ref_query_blocks[static_cast<size_t>(i)];
        const int n_q = static_cast<int>(qb.size()) / DESC_DIM;
        if (n_other > 0 && n_q > 0) {
            float top3 = compute_top3_mean_nn(qb.data(), n_q, other_bank.data(), n_other, DESC_DIM);
            loo_scores.push_back(top3);
        }
    }

    if (loo_scores.empty()) {
        bank.calib_scale = 0.44f;
    } else {
        std::sort(loo_scores.begin(), loo_scores.end());
        const float loo_med = loo_scores[loo_scores.size() / 2];
        bank.calib_scale = std::clamp(loo_med * 1.03f, 0.43f, 0.58f);
    }
}

bool PatchCoreEngine::load_bank_bin(const std::string& bin_path, FamilyBank& bank) const {
    std::ifstream ifs(bin_path, std::ios::binary);
    if (!ifs.is_open()) {
        return false;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t num_vec = 0;
    uint32_t dim = 0;
    uint32_t ref_cnt = 0;
    float calib = 0.44f;
    uint64_t sig = 0;

    ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    ifs.read(reinterpret_cast<char*>(&num_vec), sizeof(num_vec));
    ifs.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    ifs.read(reinterpret_cast<char*>(&ref_cnt), sizeof(ref_cnt));
    ifs.read(reinterpret_cast<char*>(&calib), sizeof(calib));
    ifs.read(reinterpret_cast<char*>(&sig), sizeof(sig));

    if (!ifs || magic != BANK_MAGIC || version != BANK_VERSION || dim != static_cast<uint32_t>(DESC_DIM)) {
        return false;
    }
    if (num_vec == 0 || num_vec > 65536U) {
        return false;
    }

    std::vector<float> descs(static_cast<size_t>(num_vec) * dim);
    std::vector<float> norms(static_cast<size_t>(num_vec));
    ifs.read(reinterpret_cast<char*>(descs.data()), static_cast<std::streamsize>(descs.size() * sizeof(float)));
    ifs.read(reinterpret_cast<char*>(norms.data()), static_cast<std::streamsize>(norms.size() * sizeof(float)));
    if (!ifs) {
        return false;
    }

    bank.num_vectors = static_cast<int>(num_vec);
    bank.desc_dim = static_cast<int>(dim);
    bank.ref_count = static_cast<int>(ref_cnt);
    bank.calib_scale = std::clamp(calib, 0.30f, 0.80f);
    bank.refs_signature = sig;
    bank.descriptors = std::move(descs);
    bank.sq_norms = std::move(norms);
    return true;
}

bool PatchCoreEngine::save_bank_bin(const std::string& bin_path, const FamilyBank& bank) const {
    if (bank.num_vectors <= 0 || bank.desc_dim != DESC_DIM) {
        return false;
    }
    std::error_code ec;
    fs::create_directories(fs::path(bin_path).parent_path(), ec);

    const std::string tmp_path = bin_path + ".tmp";
    std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        return false;
    }

    const uint32_t magic = BANK_MAGIC;
    const uint32_t version = BANK_VERSION;
    const uint32_t num_vec = static_cast<uint32_t>(bank.num_vectors);
    const uint32_t dim = static_cast<uint32_t>(bank.desc_dim);
    const uint32_t ref_cnt = static_cast<uint32_t>(bank.ref_count);
    const float calib = bank.calib_scale;
    const uint64_t sig = bank.refs_signature;

    ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
    ofs.write(reinterpret_cast<const char*>(&num_vec), sizeof(num_vec));
    ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    ofs.write(reinterpret_cast<const char*>(&ref_cnt), sizeof(ref_cnt));
    ofs.write(reinterpret_cast<const char*>(&calib), sizeof(calib));
    ofs.write(reinterpret_cast<const char*>(&sig), sizeof(sig));
    ofs.write(
        reinterpret_cast<const char*>(bank.descriptors.data()),
        static_cast<std::streamsize>(bank.descriptors.size() * sizeof(float))
    );
    ofs.write(
        reinterpret_cast<const char*>(bank.sq_norms.data()),
        static_cast<std::streamsize>(bank.sq_norms.size() * sizeof(float))
    );
    ofs.flush();
    const bool ok = static_cast<bool>(ofs);
    ofs.close();

    if (!ok) {
        fs::remove(tmp_path, ec);
        return false;
    }
    fs::rename(tmp_path, bin_path, ec);
    if (ec) {
        fs::remove(tmp_path, ec);
        return false;
    }
    return true;
}

bool PatchCoreEngine::build_family_from_disk(const std::string& family, FamilyBank& bank) const {
    const fs::path fam_dir = fs::path(refs_dir_) / family;
    std::vector<std::string> paths = list_sorted_ref_images(fam_dir.string());
    if (static_cast<int>(paths.size()) > max_refs_per_family_) {
        paths.erase(paths.begin(), paths.begin() + (static_cast<int>(paths.size()) - max_refs_per_family_));
    }

    bank.ref_paths.clear();
    bank.ref_bank_blocks.clear();
    bank.ref_query_blocks.clear();

    for (const auto& p : paths) {
        cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
        if (img.empty()) {
            continue;
        }
        std::vector<float> dense;
        int gw = 0, gh = 0;
        if (!extract_descriptors(img, REF_EXTRACT_STRIDE, dense, gw, gh)) {
            continue;
        }
        std::vector<float> q_descs = subsample_query_from_dense(dense, gw, gh, DESC_DIM);
        const int n_dense = static_cast<int>(dense.size()) / DESC_DIM;
        std::vector<float> reduced;
        greedy_coreset_select(dense.data(), n_dense, DESC_DIM, PER_REF_CACHE_MAX, reduced);

        bank.ref_paths.push_back(p);
        bank.ref_bank_blocks.push_back(std::move(reduced));
        bank.ref_query_blocks.push_back(std::move(q_descs));
    }

    if (bank.ref_paths.empty()) {
        bank.num_vectors = 0;
        bank.ref_count = 0;
        bank.refs_signature = 0;
        return false;
    }

    bank.refs_signature = compute_refs_signature(bank.ref_paths);
    rebuild_bank_from_cache(bank);
    return bank.num_vectors > 0;
}

bool PatchCoreEngine::init(
    const std::string& refs_dir,
    const std::string& banks_dir,
    float default_threshold,
    int max_refs_per_family
) {
    std::lock_guard<std::mutex> lock(mutex_);

    refs_dir_ = refs_dir.empty() ? "models/patchcore_refs" : refs_dir;
    banks_dir_ = banks_dir.empty() ? "models/patchcore_banks" : banks_dir;
    default_threshold_ = std::clamp(default_threshold, 0.05f, 0.99f);
    max_refs_per_family_ = std::max(1, max_refs_per_family);
    banks_.clear();

    std::error_code ec;
    fs::create_directories(refs_dir_, ec);
    fs::create_directories(banks_dir_, ec);

    std::vector<std::string> families;
    for (const char* f : DEFAULT_METAL_FAMILIES) {
        families.emplace_back(f);
    }

    if (fs::exists(refs_dir_, ec) && fs::is_directory(refs_dir_, ec)) {
        for (const auto& entry : fs::directory_iterator(refs_dir_, ec)) {
            if (entry.is_directory(ec)) {
                std::string f = sanitize_family(entry.path().filename().string());
                if (!f.empty() && std::find(families.begin(), families.end(), f) == families.end()) {
                    families.push_back(f);
                }
            }
        }
    }

    if (fs::exists(banks_dir_, ec) && fs::is_directory(banks_dir_, ec)) {
        for (const auto& entry : fs::directory_iterator(banks_dir_, ec)) {
            if (entry.is_regular_file(ec) && to_lower_copy(entry.path().extension().string()) == ".bin") {
                std::string f = sanitize_family(entry.path().stem().string());
                if (!f.empty() && std::find(families.begin(), families.end(), f) == families.end()) {
                    families.push_back(f);
                }
            }
        }
    }

    for (const auto& fam : families) {
        FamilyBank bank;
        bank.threshold = default_threshold_;

        const fs::path fam_refs = fs::path(refs_dir_) / fam;
        const fs::path fam_bin = fs::path(banks_dir_) / (fam + ".bin");

        std::vector<std::string> paths = list_sorted_ref_images(fam_refs.string());
        if (static_cast<int>(paths.size()) > max_refs_per_family_) {
            paths.erase(paths.begin(), paths.begin() + (static_cast<int>(paths.size()) - max_refs_per_family_));
        }

        const uint64_t disk_sig = compute_refs_signature(paths);
        bool loaded_bin = false;
        if (fs::exists(fam_bin, ec) && load_bank_bin(fam_bin.string(), bank)) {
            if (paths.empty() || bank.refs_signature == disk_sig) {
                bank.ref_paths = paths;
                if (paths.empty()) {
                    bank.base_descriptors = bank.descriptors;
                    bank.base_calib_scale = bank.calib_scale;
                } else {
                    bank.ref_count = static_cast<int>(paths.size());
                }
                loaded_bin = true;
            }
        }

        if (!loaded_bin && !paths.empty()) {
            if (build_family_from_disk(fam, bank)) {
                save_bank_bin(fam_bin.string(), bank);
            }
        }

        if (bank.num_vectors > 0) {
            banks_[fam] = std::move(bank);
        }
    }

    std::cout << "[PatchCoreEngine] Initialized (" << banks_.size() << " active family banks, threshold="
              << default_threshold_ << ", max_refs=" << max_refs_per_family_ << ")" << std::endl;
    return true;
}

bool PatchCoreEngine::register_good_image(
    const std::string& family,
    const cv::Mat& bgr_img,
    int* total_saved_out
) {
    const std::string fam = sanitize_family(family);
    if (fam.empty()) {
        if (total_saved_out) {
            *total_saved_out = 0;
        }
        return false;
    }
    cv::Mat clean_bgr = normalize_to_bgr8(bgr_img);
    if (clean_bgr.empty()) {
        if (total_saved_out) {
            *total_saved_out = reference_count(fam);
        }
        return false;
    }

    // Extract descriptors outside the bank mutex
    std::vector<float> dense;
    int gw = 0, gh = 0;
    if (!extract_descriptors(clean_bgr, REF_EXTRACT_STRIDE, dense, gw, gh)) {
        return false;
    }
    std::vector<float> q_descs = subsample_query_from_dense(dense, gw, gh, DESC_DIM);
    const int n_dense = static_cast<int>(dense.size()) / DESC_DIM;
    std::vector<float> reduced_new;
    greedy_coreset_select(dense.data(), n_dense, DESC_DIM, PER_REF_CACHE_MAX, reduced_new);

    std::lock_guard<std::mutex> lock(mutex_);

    const fs::path fam_refs = fs::path(refs_dir_) / fam;
    const fs::path fam_bin = fs::path(banks_dir_) / (fam + ".bin");
    std::error_code ec;
    fs::create_directories(fam_refs, ec);
    fs::create_directories(banks_dir_, ec);

    // Rotate out oldest files if already at or above max_refs_per_family_
    std::vector<std::string> existing_paths = list_sorted_ref_images(fam_refs.string());
    if (static_cast<int>(existing_paths.size()) >= max_refs_per_family_) {
        const int remove_count = static_cast<int>(existing_paths.size()) - max_refs_per_family_ + 1;
        for (int i = 0; i < remove_count && i < static_cast<int>(existing_paths.size()); ++i) {
            fs::remove(existing_paths[static_cast<size_t>(i)], ec);
        }
        existing_paths.erase(
            existing_paths.begin(),
            existing_paths.begin() + std::min(remove_count, static_cast<int>(existing_paths.size()))
        );
    }

    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()
                        ).count();
    std::string dest_path;
    for (int attempt = 0; attempt < 16; ++attempt) {
        ++save_seq_;
        char name_buf[128];
        std::snprintf(
            name_buf,
            sizeof(name_buf),
            "ref_%lld_%06llu.png",
            static_cast<long long>(now_ms),
            static_cast<unsigned long long>(save_seq_)
        );
        fs::path candidate = fam_refs / name_buf;
        if (!fs::exists(candidate, ec)) {
            dest_path = candidate.string();
            break;
        }
    }
    if (dest_path.empty() || !cv::imwrite(dest_path, clean_bgr)) {
        if (total_saved_out) {
            *total_saved_out = static_cast<int>(existing_paths.size());
        }
        return false;
    }

    FamilyBank& bank = banks_[fam];
    if (!bank.has_custom_threshold) {
        bank.threshold = default_threshold_;
    }
    if (bank.base_descriptors.empty() && bank.ref_paths.empty() && bank.num_vectors > 0 && bank.ref_bank_blocks.empty()) {
        bank.base_descriptors = bank.descriptors;
        bank.base_calib_scale = bank.calib_scale;
    }

    // Reconcile cached per-image blocks with remaining existing files on disk
    std::unordered_map<std::string, size_t> old_idx;
    for (size_t i = 0; i < bank.ref_paths.size(); ++i) {
        if (i < bank.ref_bank_blocks.size() && i < bank.ref_query_blocks.size()) {
            old_idx[bank.ref_paths[i]] = i;
        }
    }

    std::vector<std::string> new_paths;
    std::vector<std::vector<float>> new_bank_blocks;
    std::vector<std::vector<float>> new_query_blocks;
    new_paths.reserve(existing_paths.size() + 1);
    new_bank_blocks.reserve(existing_paths.size() + 1);
    new_query_blocks.reserve(existing_paths.size() + 1);

    for (const auto& p : existing_paths) {
        auto it = old_idx.find(p);
        if (it != old_idx.end()) {
            const size_t idx = it->second;
            new_paths.push_back(p);
            new_bank_blocks.push_back(std::move(bank.ref_bank_blocks[idx]));
            new_query_blocks.push_back(std::move(bank.ref_query_blocks[idx]));
        } else {
            cv::Mat prev_img = cv::imread(p, cv::IMREAD_COLOR);
            if (prev_img.empty()) {
                continue;
            }
            std::vector<float> p_dense;
            int pgw = 0, pgh = 0;
            if (!extract_descriptors(prev_img, REF_EXTRACT_STRIDE, p_dense, pgw, pgh)) {
                continue;
            }
            std::vector<float> p_q = subsample_query_from_dense(p_dense, pgw, pgh, DESC_DIM);
            const int np = static_cast<int>(p_dense.size()) / DESC_DIM;
            std::vector<float> p_red;
            greedy_coreset_select(p_dense.data(), np, DESC_DIM, PER_REF_CACHE_MAX, p_red);
            new_paths.push_back(p);
            new_bank_blocks.push_back(std::move(p_red));
            new_query_blocks.push_back(std::move(p_q));
        }
    }

    new_paths.push_back(dest_path);
    new_bank_blocks.push_back(std::move(reduced_new));
    new_query_blocks.push_back(std::move(q_descs));

    bank.ref_paths = std::move(new_paths);
    bank.ref_bank_blocks = std::move(new_bank_blocks);
    bank.ref_query_blocks = std::move(new_query_blocks);
    bank.refs_signature = compute_refs_signature(bank.ref_paths);

    rebuild_bank_from_cache(bank);
    save_bank_bin(fam_bin.string(), bank);

    if (total_saved_out) {
        *total_saved_out = static_cast<int>(bank.ref_paths.size());
    }
    return true;
}

PatchCoreResult PatchCoreEngine::score_image(const std::string& family, const cv::Mat& bgr_img) const {
    PatchCoreResult res;
    res.threshold = default_threshold_;

    const std::string fam = sanitize_family(family);
    if (fam.empty() || bgr_img.empty()) {
        return res;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = banks_.find(fam);
        if (it == banks_.end() || it->second.num_vectors <= 0) {
            if (it != banks_.end() && it->second.has_custom_threshold) {
                res.threshold = it->second.threshold;
            }
            return res;
        }
    }

    std::vector<float> q_descs;
    int gw = 0, gh = 0;
    if (!extract_descriptors(bgr_img, QUERY_EXTRACT_STRIDE, q_descs, gw, gh)) {
        return res;
    }
    const int n_q = static_cast<int>(q_descs.size()) / DESC_DIM;
    if (n_q <= 0 || gw <= 0 || gh <= 0) {
        return res;
    }

    std::vector<float> nn_dists;
    float top3_mean = 0.0f;
    float calib = 0.44f;
    float thresh = default_threshold_;
    int bsize = 0;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = banks_.find(fam);
        if (it == banks_.end() || it->second.num_vectors <= 0) {
            return res;
        }
        const FamilyBank& bank = it->second;
        bsize = bank.num_vectors;
        calib = bank.calib_scale;
        thresh = bank.has_custom_threshold ? bank.threshold : default_threshold_;

        top3_mean = compute_top3_mean_nn(
            q_descs.data(),
            n_q,
            bank.descriptors.data(),
            bank.num_vectors,
            DESC_DIM,
            &nn_dists
        );
    }

    const float t2 = top3_mean * top3_mean;
    const float c2 = calib * calib;
    const float score = std::clamp(t2 / std::max(1e-8f, t2 + c2), 0.0f, 1.0f);

    res.anomaly_score = score;
    res.threshold = thresh;
    res.is_anomalous = (score >= thresh);
    res.bank_size = bsize;

    if (res.is_anomalous && !nn_dists.empty()) {
        const float raw_thresh = std::max(1e-4f, calib * std::sqrt(thresh / std::max(1e-4f, 1.0f - thresh)));

        // Build normalized score grid (1.0f == PatchCore anomaly decision boundary)
        cv::Mat raw_grid(gh, gw, CV_32FC1, cv::Scalar(0.0f));
        for (int r = 0; r < gh; ++r) {
            float* row_p = raw_grid.ptr<float>(r);
            for (int c = 0; c < gw; ++c) {
                row_p[c] = nn_dists[static_cast<size_t>(r * gw + c)] / raw_thresh;
            }
        }

        // Suppress isolated single-patch rim/angle spikes while preserving contiguous defect clusters
        cv::Mat blur_grid;
        cv::GaussianBlur(raw_grid, blur_grid, cv::Size(3, 3), 0.85, 0.85, cv::BORDER_REPLICATE);
        cv::Mat norm_grid = 0.30f * raw_grid + 0.70f * blur_grid;

        // Project onto canonical 192x192 space and interpolate smoothly for pixel-level contours
        cv::Mat heat_can(CANONICAL_SIZE, CANONICAL_SIZE, CV_32FC1, cv::Scalar(0.0f));
        const int span = std::max(8, CANONICAL_SIZE - 2 * MARGIN_PX);
        cv::Mat inner_heat;
        cv::resize(norm_grid, inner_heat, cv::Size(span, span), 0, 0, cv::INTER_CUBIC);
        inner_heat.copyTo(heat_can(cv::Rect(MARGIN_PX, MARGIN_PX, span, span)));
        // Lightly pad immediate border band with clamped edge values * 0.85 so rim defects remain unbroken
        for (int y = 0; y < CANONICAL_SIZE; ++y) {
            const int sy = std::clamp(y, MARGIN_PX, MARGIN_PX + span - 1);
            float* dst_row = heat_can.ptr<float>(y);
            const float* src_row = heat_can.ptr<float>(sy);
            for (int x = 0; x < CANONICAL_SIZE; ++x) {
                if (x >= MARGIN_PX && x < MARGIN_PX + span && y >= MARGIN_PX && y < MARGIN_PX + span) {
                    continue;
                }
                const int sx_idx = std::clamp(x, MARGIN_PX, MARGIN_PX + span - 1);
                const int d_edge = std::max(
                    std::max(MARGIN_PX - x, x - (MARGIN_PX + span - 1)),
                    std::max(MARGIN_PX - y, y - (MARGIN_PX + span - 1))
                );
                const float decay = std::max(0.0f, 1.0f - static_cast<float>(d_edge) * 0.06f);
                dst_row[x] = src_row[sx_idx] * decay;
            }
        }
        cv::GaussianBlur(heat_can, heat_can, cv::Size(7, 7), 1.4, 1.4, cv::BORDER_REPLICATE);

        // Foreground mask on canonical resolution so far-field dark background is excluded
        cv::Mat can_bgr = normalize_to_bgr8(bgr_img);
        cv::Mat can_gray;
        cv::cvtColor(can_bgr, can_gray, cv::COLOR_BGR2GRAY);
        cv::Mat fg_mask = (can_gray > 22);
        cv::dilate(fg_mask, fg_mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));

        // Mask dark empty background if part occupies a bounded centered footprint
        const float fg_cov = static_cast<float>(cv::countNonZero(fg_mask)) / static_cast<float>(CANONICAL_SIZE * CANONICAL_SIZE);
        if (fg_cov > 0.12f && fg_cov < 0.88f) {
            for (int y = 0; y < CANONICAL_SIZE; ++y) {
                float* hr = heat_can.ptr<float>(y);
                const uint8_t* mr = fg_mask.ptr<uint8_t>(y);
                for (int x = 0; x < CANONICAL_SIZE; ++x) {
                    if (mr[x] == 0) hr[x] *= 0.65f;
                }
            }
        }

        double min_v = 0.0, max_v = 0.0;
        cv::Point min_loc, max_loc;
        cv::minMaxLoc(heat_can, &min_v, &max_v, &min_loc, &max_loc);
        const float peak_heat = static_cast<float>(max_v);
        res.peak_ratio = peak_heat;
        res.heatmap = heat_can.clone();

        const float seed_cut = std::min(1.06f, std::max(0.92f, peak_heat * 0.72f));
        const float eff_seed = std::min(seed_cut, peak_heat * 0.88f);
        const float grow_cut = std::max(0.84f, eff_seed * 0.88f);

        cv::Mat grow_mask = (heat_can >= grow_cut);
        cv::Mat seed_mask = (heat_can >= eff_seed);
        cv::Mat cc_labels, cc_stats, cc_centroids;
        int n_cc = cv::connectedComponentsWithStats(grow_mask, cc_labels, cc_stats, cc_centroids);

        std::vector<uint8_t> keep_cc(static_cast<size_t>(std::max(1, n_cc)), 0);
        std::vector<int> seed_areas(static_cast<size_t>(std::max(1, n_cc)), 0);
        int best_cc = -1;
        double best_cc_peak = -1.0;
        int max_kept_area = 0;
        for (int y = 0; y < CANONICAL_SIZE; ++y) {
            const int* lr = cc_labels.ptr<int>(y);
            const uint8_t* sr = seed_mask.ptr<uint8_t>(y);
            const float* hr = heat_can.ptr<float>(y);
            for (int x = 0; x < CANONICAL_SIZE; ++x) {
                int cid = lr[x];
                if (cid <= 0) continue;
                int area = cc_stats.at<int>(cid, cv::CC_STAT_AREA);
                if (sr[x] > 0 && area >= 20) {
                    keep_cc[static_cast<size_t>(cid)] = 1;
                    seed_areas[static_cast<size_t>(cid)] = area;
                    max_kept_area = std::max(max_kept_area, area);
                }
                if (hr[x] > best_cc_peak) {
                    best_cc_peak = hr[x];
                    best_cc = cid;
                }
            }
        }
        // Drop minor secondary rim slivers (< 45% of primary anomaly cluster area), preserving equal-scale multi-lobe anomalies (e.g. flipped parts)
        if (max_kept_area >= 120) {
            for (int cid = 1; cid < n_cc; ++cid) {
                if (keep_cc[static_cast<size_t>(cid)] && cid != best_cc &&
                    seed_areas[static_cast<size_t>(cid)] < static_cast<int>(max_kept_area * 0.45)) {
                    keep_cc[static_cast<size_t>(cid)] = 0;
                }
            }
        }
        if (best_cc > 0 && std::none_of(keep_cc.begin(), keep_cc.end(), [](uint8_t v) { return v != 0; })) {
            keep_cc[static_cast<size_t>(best_cc)] = 1;
        }

        cv::Mat final_mask(CANONICAL_SIZE, CANONICAL_SIZE, CV_8UC1, cv::Scalar(0));
        cv::Mat core_mask(CANONICAL_SIZE, CANONICAL_SIZE, CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < CANONICAL_SIZE; ++y) {
            const int* lr = cc_labels.ptr<int>(y);
            const uint8_t* sr = seed_mask.ptr<uint8_t>(y);
            uint8_t* fr = final_mask.ptr<uint8_t>(y);
            uint8_t* cr = core_mask.ptr<uint8_t>(y);
            for (int x = 0; x < CANONICAL_SIZE; ++x) {
                int cid = lr[x];
                if (cid > 0 && keep_cc[static_cast<size_t>(cid)]) {
                    fr[x] = 255;
                    if (sr[x] > 0) cr[x] = 255;
                }
            }
        }
        cv::morphologyEx(final_mask, final_mask, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

        std::vector<std::vector<cv::Point>> raw_contours;
        cv::findContours(final_mask, raw_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        const float sx = static_cast<float>(bgr_img.cols) / static_cast<float>(CANONICAL_SIZE);
        const float sy = static_cast<float>(bgr_img.rows) / static_cast<float>(CANONICAL_SIZE);
        int bx1 = bgr_img.cols, by1 = bgr_img.rows, bx2 = 0, by2 = 0;

        for (const auto& c : raw_contours) {
            if (cv::contourArea(c) < 16.0) continue;
            std::vector<cv::Point> poly_can;
            cv::approxPolyDP(c, poly_can, 1.35, true);
            if (poly_can.size() < 3) poly_can = c;
            std::vector<cv::Point> poly_orig;
            poly_orig.reserve(poly_can.size());
            for (const auto& pt : poly_can) {
                int ox = std::clamp(static_cast<int>(std::round(pt.x * sx)), 0, std::max(0, bgr_img.cols - 1));
                int oy = std::clamp(static_cast<int>(std::round(pt.y * sy)), 0, std::max(0, bgr_img.rows - 1));
                poly_orig.emplace_back(ox, oy);
                bx1 = std::min(bx1, ox);
                by1 = std::min(by1, oy);
                bx2 = std::max(bx2, ox + 1);
                by2 = std::max(by2, oy + 1);
            }
            if (poly_orig.size() >= 3) {
                res.hotspot_contours.push_back(std::move(poly_orig));
            }
        }

        if (bx2 > bx1 && by2 > by1) {
            res.hotspot_bbox = cv::Rect(bx1, by1, bx2 - bx1, by2 - by1);
        } else {
            res.hotspot_bbox = cv::Rect(
                std::clamp(static_cast<int>(std::round((max_loc.x - 10) * sx)), 0, std::max(0, bgr_img.cols - 1)),
                std::clamp(static_cast<int>(std::round((max_loc.y - 10) * sy)), 0, std::max(0, bgr_img.rows - 1)),
                std::max(8, static_cast<int>(std::round(20 * sx))),
                std::max(8, static_cast<int>(std::round(20 * sy)))
            );
        }

        cv::Mat primary_core = core_mask;
        if (best_cc > 0) {
            cv::Mat best_mask = (cc_labels == best_cc);
            cv::Mat narrowed;
            cv::bitwise_and(core_mask, best_mask, narrowed);
            if (cv::countNonZero(narrowed) >= 4) {
                primary_core = std::move(narrowed);
            }
        }
        cv::Rect core_can = cv::boundingRect(primary_core);
        if (!core_can.empty()) {
            const int cx1 = std::clamp(static_cast<int>(std::floor(core_can.x * sx)), 0, std::max(0, bgr_img.cols - 1));
            const int cy1 = std::clamp(static_cast<int>(std::floor(core_can.y * sy)), 0, std::max(0, bgr_img.rows - 1));
            const int cx2 = std::clamp(static_cast<int>(std::ceil((core_can.x + core_can.width) * sx)), cx1 + 1, bgr_img.cols);
            const int cy2 = std::clamp(static_cast<int>(std::ceil((core_can.y + core_can.height) * sy)), cy1 + 1, bgr_img.rows);
            res.core_bbox = cv::Rect(cx1, cy1, cx2 - cx1, cy2 - cy1);
        } else {
            res.core_bbox = res.hotspot_bbox;
        }
    }

    return res;
}

int PatchCoreEngine::bank_size(const std::string& family) const {
    const std::string fam = sanitize_family(family);
    if (fam.empty()) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = banks_.find(fam);
    if (it == banks_.end()) {
        return 0;
    }
    return it->second.num_vectors;
}

int PatchCoreEngine::reference_count(const std::string& family) const {
    const std::string fam = sanitize_family(family);
    if (fam.empty()) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = banks_.find(fam);
    if (it == banks_.end()) {
        return 0;
    }
    return std::max(it->second.ref_count, static_cast<int>(it->second.ref_paths.size()));
}

void PatchCoreEngine::set_family_threshold(const std::string& family, float threshold) {
    const std::string fam = sanitize_family(family);
    if (fam.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    FamilyBank& bank = banks_[fam];
    bank.threshold = std::clamp(threshold, 0.05f, 0.99f);
    bank.has_custom_threshold = true;
}

float PatchCoreEngine::get_family_threshold(const std::string& family) const {
    const std::string fam = sanitize_family(family);
    if (fam.empty()) {
        return default_threshold_;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = banks_.find(fam);
    if (it != banks_.end() && it->second.has_custom_threshold) {
        return it->second.threshold;
    }
    return default_threshold_;
}

bool PatchCoreEngine::has_bank(const std::string& family) const {
    return bank_size(family) > 0;
}

std::vector<std::string> PatchCoreEngine::list_families() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> out;
    out.reserve(banks_.size());
    for (const auto& kv : banks_) {
        if (kv.second.num_vectors > 0) {
            out.push_back(kv.first);
        }
    }
    return out;
}

} // namespace vision
} // namespace qc
