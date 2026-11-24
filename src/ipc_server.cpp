#include "ipc_server.hpp"
#include "cpu_affinity.hpp"
#include "embedded_ui.hpp"
#include "qc_rules.hpp"

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#undef CPPHTTPLIB_OPENSSL_SUPPORT
#endif
#include "httplib.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {
constexpr const char* CSV_LOG_PATH = "rpi5_qc_inspection_log.csv";
constexpr const char* CSV_HEADER_LINE =
    "Timestamp,Family,DefectClass,Group,ScoreConf,AnomalyScore,AreaPct,Boxes,Spread,Severity,Verdict,PossibleCause,RecommendedAction,AdvisorySource,PreMs,InferMs,PostMs,TotalMs,LlmLatencyMs";

double finite_or_zero(double v) {
    return std::isfinite(v) ? v : 0.0;
}

double round_decimals(double v, int decimals) {
    if (!std::isfinite(v)) return 0.0;
    double scale = std::pow(10.0, decimals);
    return std::round(v * scale) / scale;
}

bool is_video_extension(std::string ext) {
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv" || ext == ".webm";
}

bool is_image_extension(std::string ext) {
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp";
}

std::string escape_csv(const std::string& val) {
    std::string out;
    out.reserve(val.size() + 4);
    for (char c : val) {
        if (c == '"') {
            out += "\"\"";
        } else if (c == '\r' || c == '\n') {
            out += ' ';
        } else {
            out += c;
        }
    }
    return "\"" + out + "\"";
}

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur += '"';
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cur += c;
            }
        } else {
            if (c == '"') {
                in_quotes = true;
            } else if (c == ',') {
                fields.push_back(cur);
                cur.clear();
            } else if (c != '\r') {
                cur += c;
            }
        }
    }
    fields.push_back(cur);
    return fields;
}

std::string format_one_decimal(float v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f", std::isfinite(v) ? v : 0.0f);
    return std::string(buf);
}

std::string trim_string(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

void reset_analyzing_state(qc::ipc::LiveTelemetryState& st) {
    st.is_locked = false;
    st.lock_cnt = 0;
    st.llm_busy = false;
    st.advisory_pending = true;
    st.llm_elapsed = 0.0f;
    st.boxes = 0;
    st.conf = 0.0f;
    st.area = 0.0f;
    st.severity = "NONE";
    st.verdict = "SCANNING";
    st.defects.clear();
    st.defect = "ANALYZING...";
    st.display_title = "ANALYZING...";
    st.cause = "Evaluating image on unified vision pipeline...";
    st.action = "Waiting for Llama 3.2 1B advisory...";
    st.engine_source = "Llama 3.2 1B";
    st.ttft_ms = 0.0f;
    st.tokens_per_s = 0.0f;
    st.llm_ms = 0.0f;
    st.tokens = 0;
    st.narrative.clear();
}
} // namespace

namespace qc {
namespace ipc {

IPCServer::IPCServer(int port) : port_(port), svr_(std::make_unique<httplib::Server>()) {
    init_playlist();
    load_logs_from_csv();
}

IPCServer::~IPCServer() {
    stop();
}

void IPCServer::set_vision_engine(qc::vision::HailoVisionEngine* engine) {
    vision_engine_ = engine;
    if (vision_engine_) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        current_state_.vision_backend = vision_engine_->get_vision_backend();
        current_state_.hardware_device = vision_engine_->get_hardware_name();
        current_state_.anomaly_max_refs = vision_engine_->rules().config().patchcore.reference_limit_per_family;
        current_state_.anomaly.threshold = vision_engine_->rules().config().patchcore.anomaly_threshold;
        current_state_.system_status = current_state_.vision_backend + " · LLAMA 3.2 1B";
        std::lock_guard<std::mutex> block(bench_mutex_);
        last_backend_ = current_state_.vision_backend;
    }
}

void IPCServer::init_playlist() {
    playlist_.clear();
    video_playlist_.clear();

    std::vector<fs::path> dirs = {
        "sample_test_images",
        "../sample_test_images",
        "native/sample_test_images"
    };

    fs::path chosen_dir;
    for (const auto& d : dirs) {
        if (fs::exists(d) && fs::is_directory(d)) {
            chosen_dir = d;
            break;
        }
    }

    if (!chosen_dir.empty()) {
        std::vector<fs::path> root_files;
        std::vector<fs::path> sub_files;
        std::error_code ec;
        for (const auto& entry : fs::recursive_directory_iterator(chosen_dir, ec)) {
            if (!entry.is_regular_file()) continue;
            std::string fname = entry.path().filename().string();
            if (!fname.empty() && fname[0] == '.') continue;
            if (!is_image_extension(entry.path().extension().string())) continue;
            if (entry.path().parent_path() == chosen_dir) {
                root_files.push_back(entry.path());
            } else {
                sub_files.push_back(entry.path());
            }
        }
        std::sort(root_files.begin(), root_files.end());
        std::sort(sub_files.begin(), sub_files.end());
        for (const auto& p : root_files) {
            std::string rel = fs::relative(p, chosen_dir, ec).generic_string();
            playlist_.push_back({rel.empty() ? p.filename().string() : rel, p.generic_string()});
        }
        for (const auto& p : sub_files) {
            std::string rel = fs::relative(p, chosen_dir, ec).generic_string();
            playlist_.push_back({rel.empty() ? p.filename().string() : rel, p.generic_string()});
        }
    }

    if (playlist_.empty()) {
        playlist_.push_back({"test_00_clean_steel.jpg", "sample_test_images/test_00_clean_steel.jpg"});
        playlist_.push_back({"test_crazing.jpg", "sample_test_images/test_crazing.jpg"});
        playlist_.push_back({"test_inclusion.jpg", "sample_test_images/test_inclusion.jpg"});
        playlist_.push_back({"test_patches.jpg", "sample_test_images/test_patches.jpg"});
        playlist_.push_back({"test_pitted_surface.jpg", "sample_test_images/test_pitted_surface.jpg"});
        playlist_.push_back({"test_rolled-in_scale.jpg", "sample_test_images/test_rolled-in_scale.jpg"});
        playlist_.push_back({"test_scratches.jpg", "sample_test_images/test_scratches.jpg"});
    }

    if (fs::exists("uploads") && fs::is_directory("uploads")) {
        std::vector<fs::path> up_imgs, up_vids;
        for (const auto& entry : fs::directory_iterator("uploads")) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            if (is_image_extension(ext)) up_imgs.push_back(entry.path());
            else if (is_video_extension(ext)) up_vids.push_back(entry.path());
        }
        std::sort(up_imgs.begin(), up_imgs.end());
        for (const auto& p : up_imgs) {
            playlist_.push_back({p.filename().string(), p.generic_string()});
        }
        std::sort(up_vids.begin(), up_vids.end());
        for (const auto& p : up_vids) {
            video_playlist_.push_back({p.filename().string(), p.generic_string()});
        }
    }

    demo_video_path_.clear();
    for (const fs::path& dir_cand : {fs::path("."), fs::path(".."), fs::path("sample_videos"), fs::path("../sample_videos")}) {
        if (!fs::exists(dir_cand) || !fs::is_directory(dir_cand)) continue;
        std::vector<fs::path> vids;
        for (const auto& entry : fs::directory_iterator(dir_cand)) {
            if (entry.is_regular_file() && is_video_extension(entry.path().extension().string())) {
                vids.push_back(entry.path());
            }
        }
        std::sort(vids.begin(), vids.end());
        for (const auto& vp : vids) {
            std::string fname = vp.filename().string();
            if (fname == "conveyor_demo_belt.mp4") {
                if (demo_video_path_.empty()) {
                    demo_video_path_ = vp.generic_string();
                }
                continue;
            }
            bool dup = false;
            for (const auto& existing : video_playlist_) {
                if (existing.first == fname) { dup = true; break; }
            }
            if (!dup) {
                video_playlist_.push_back({fname, vp.generic_string()});
            }
        }
    }
    if (demo_video_path_.empty() && !video_playlist_.empty()) {
        demo_video_path_ = video_playlist_[0].second;
    }

    current_state_.active_idx = 0;
    video_idx_ = 0;
    current_state_.active_tag = "DEMO · Conveyor Belt";
}

std::string IPCServer::get_current_image_path() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (playlist_.empty()) return "";
    int sz = static_cast<int>(playlist_.size());
    int idx = (current_state_.active_idx % sz + sz) % sz;
    return playlist_[idx].second;
}

std::string IPCServer::get_current_video_path() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (video_playlist_.empty()) return "";
    int sz = static_cast<int>(video_playlist_.size());
    int idx = (video_idx_ % sz + sz) % sz;
    return video_playlist_[idx].second;
}

std::string IPCServer::get_demo_video_path() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return demo_video_path_;
}

int IPCServer::get_active_index() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (current_state_.mode == "VIDEO") return video_idx_;
    return current_state_.active_idx;
}

std::string IPCServer::get_active_mode() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_state_.mode;
}

bool IPCServer::is_paused() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_state_.is_paused || current_state_.demo_auto_paused;
}

bool IPCServer::is_manual_paused() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_state_.is_paused;
}

void IPCServer::set_auto_paused(bool paused) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_state_.demo_auto_paused = paused;
}

int IPCServer::consume_demo_step_delta() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    int d = demo_step_delta_;
    demo_step_delta_ = 0;
    return d;
}

void IPCServer::update_frame_jpeg(const std::vector<uint8_t>& jpeg_data) {
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        current_jpeg_ = jpeg_data;
    }
    frame_seq_.fetch_add(1, std::memory_order_release);
}

std::vector<uint8_t> IPCServer::get_frame_jpeg() const {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return current_jpeg_;
}

std::vector<uint8_t> IPCServer::get_frame_jpeg_with_seq(uint64_t& out_seq) const {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    out_seq = frame_seq_.load(std::memory_order_acquire);
    return current_jpeg_;
}

void IPCServer::notify_vision() {
    {
        std::lock_guard<std::mutex> lock(wake_mutex_);
        wake_pending_ = true;
    }
    wake_cv_.notify_one();
}

void IPCServer::wait_for_vision_wakeup(int timeout_ms) {
    std::unique_lock<std::mutex> lock(wake_mutex_);
    if (!wake_pending_) {
        wake_cv_.wait_for(lock, std::chrono::milliseconds(std::max(1, timeout_ms)), [this]() {
            return wake_pending_;
        });
    }
    wake_pending_ = false;
}

void IPCServer::next_sample() {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (current_state_.mode == "DEMO") {
            sample_generation_id_++;
            demo_step_delta_ += 1;
            current_state_.demo_auto_paused = false;
            reset_analyzing_state(current_state_);
            current_state_.active_tag = "DEMO · Conveyor Belt";
        } else if (current_state_.mode == "VIDEO") {
            if (video_playlist_.empty()) return;
            sample_generation_id_++;
            video_idx_ = (video_idx_ + 1) % static_cast<int>(video_playlist_.size());
            reset_analyzing_state(current_state_);
            current_state_.active_tag = "VIDEO [" + std::to_string(video_idx_ + 1) + "/" +
                                        std::to_string(video_playlist_.size()) + "]";
        } else {
            if (playlist_.empty()) return;
            sample_generation_id_++;
            current_state_.active_idx = (current_state_.active_idx + 1) % static_cast<int>(playlist_.size());
            reset_analyzing_state(current_state_);
            current_state_.active_tag = current_state_.mode + " " + std::to_string(current_state_.active_idx + 1) + "/" +
                                        std::to_string(playlist_.size()) + " · " + playlist_[current_state_.active_idx].first;
        }
    }
    notify_vision();
}

void IPCServer::prev_sample() {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (current_state_.mode == "DEMO") {
            sample_generation_id_++;
            demo_step_delta_ -= 1;
            current_state_.demo_auto_paused = false;
            reset_analyzing_state(current_state_);
            current_state_.active_tag = "DEMO · Conveyor Belt";
        } else if (current_state_.mode == "VIDEO") {
            int vsz = static_cast<int>(video_playlist_.size());
            if (vsz == 0) return;
            sample_generation_id_++;
            video_idx_ = (video_idx_ - 1 + vsz) % vsz;
            reset_analyzing_state(current_state_);
            current_state_.active_tag = "VIDEO [" + std::to_string(video_idx_ + 1) + "/" +
                                        std::to_string(vsz) + "]";
        } else {
            if (playlist_.empty()) return;
            sample_generation_id_++;
            int sz = static_cast<int>(playlist_.size());
            current_state_.active_idx = (current_state_.active_idx - 1 + sz) % sz;
            reset_analyzing_state(current_state_);
            current_state_.active_tag = current_state_.mode + " " + std::to_string(current_state_.active_idx + 1) + "/" +
                                        std::to_string(sz) + " · " + playlist_[current_state_.active_idx].first;
        }
    }
    notify_vision();
}

void IPCServer::select_sample(int index) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (current_state_.mode == "VIDEO") {
            int vsz = static_cast<int>(video_playlist_.size());
            if (vsz == 0) return;
            sample_generation_id_++;
            video_idx_ = std::max(0, std::min(index, vsz - 1));
            reset_analyzing_state(current_state_);
            current_state_.active_tag = "VIDEO [" + std::to_string(video_idx_ + 1) + "/" +
                                        std::to_string(vsz) + "]";
        } else {
            int sz = static_cast<int>(playlist_.size());
            if (sz == 0) return;
            sample_generation_id_++;
            if (current_state_.mode == "DEMO") {
                current_state_.mode = "IMAGE";
            }
            current_state_.active_idx = std::max(0, std::min(index, sz - 1));
            reset_analyzing_state(current_state_);
            current_state_.active_tag = current_state_.mode + " [" + std::to_string(current_state_.active_idx + 1) + "/" +
                                        std::to_string(sz) + "] · " + playlist_[current_state_.active_idx].first;
        }
    }
    notify_vision();
}

void IPCServer::select_sample_by_path(const std::string& path) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        fs::path target_p(path);
        std::string norm_path = target_p.generic_string();
        std::string fname = target_p.filename().string();

        if (current_state_.mode == "VIDEO") {
            for (size_t i = 0; i < video_playlist_.size(); ++i) {
                if (video_playlist_[i].second == norm_path ||
                    video_playlist_[i].second == path ||
                    video_playlist_[i].first == norm_path ||
                    video_playlist_[i].first == fname) {
                    sample_generation_id_++;
                    video_idx_ = static_cast<int>(i);
                    reset_analyzing_state(current_state_);
                    current_state_.active_tag = "VIDEO [" + std::to_string(i + 1) + "/" +
                                                std::to_string(video_playlist_.size()) + "]";
                    notify_vision();
                    return;
                }
            }
        }

        for (size_t i = 0; i < playlist_.size(); ++i) {
            std::string item_fname = fs::path(playlist_[i].second).filename().string();
            if (playlist_[i].second == norm_path ||
                playlist_[i].second == path ||
                playlist_[i].first == norm_path ||
                playlist_[i].first == path ||
                item_fname == fname) {
                sample_generation_id_++;
                if (current_state_.mode == "DEMO") {
                    current_state_.mode = "IMAGE";
                }
                current_state_.active_idx = static_cast<int>(i);
                reset_analyzing_state(current_state_);
                if (current_state_.mode == "IMAGE") {
                    current_state_.active_tag = "IMAGE [" + std::to_string(i + 1) + "/" +
                                                std::to_string(playlist_.size()) + "] · " + playlist_[i].first;
                } else {
                    current_state_.active_tag = current_state_.mode + " " + std::to_string(i + 1) + "/" +
                                                std::to_string(playlist_.size()) + " · " + playlist_[i].first;
                }
                notify_vision();
                return;
            }
        }

        // If path exists on disk outside the pre-indexed playlist, register it at front cleanly
        if (!path.empty() && fs::exists(path) && fs::is_regular_file(path)) {
            std::string ext = target_p.extension().string();
            if (is_image_extension(ext)) {
                sample_generation_id_++;
                playlist_.insert(playlist_.begin(), {fname, norm_path});
                current_state_.active_idx = 0;
                reset_analyzing_state(current_state_);
                current_state_.active_tag = current_state_.mode + " [1/" + std::to_string(playlist_.size()) + "] · " + fname;
            } else if (is_video_extension(ext)) {
                sample_generation_id_++;
                video_playlist_.insert(video_playlist_.begin(), {fname, norm_path});
                video_idx_ = 0;
                reset_analyzing_state(current_state_);
                current_state_.active_tag = "VIDEO [1/" + std::to_string(video_playlist_.size()) + "]";
            }
        }
    }
    notify_vision();
}

void IPCServer::set_active_index(int idx) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        int sz = static_cast<int>(playlist_.size());
        if (sz == 0) return;
        sample_generation_id_++;
        current_state_.active_idx = (idx % sz + sz) % sz;
        reset_analyzing_state(current_state_);
        if (current_state_.mode == "DEMO") {
            current_state_.active_tag = "DEMO · Conveyor Belt";
        } else if (current_state_.mode == "IMAGE") {
            current_state_.active_tag = "IMAGE [" + std::to_string(current_state_.active_idx + 1) + "/" +
                                        std::to_string(sz) + "] · " + playlist_[current_state_.active_idx].first;
        } else {
            current_state_.active_tag = current_state_.mode + " " + std::to_string(current_state_.active_idx + 1) + "/" +
                                        std::to_string(sz) + " · " + playlist_[current_state_.active_idx].first;
        }
    }
    notify_vision();
}

size_t IPCServer::get_playlist_size() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (current_state_.mode == "VIDEO") return video_playlist_.size();
    return playlist_.size();
}

std::string IPCServer::get_image_path_at(int idx) const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (playlist_.empty()) return "";
    int sz = static_cast<int>(playlist_.size());
    int real_idx = (idx % sz + sz) % sz;
    return playlist_[real_idx].second;
}

void IPCServer::toggle_pause() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (current_state_.mode == "IMAGE" || current_state_.mode == "LOG") {
        current_state_.is_paused = false;
        return;
    }
    current_state_.is_paused = !current_state_.is_paused;
}

void IPCServer::switch_mode(const std::string& mode) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (current_state_.mode == mode) {
            return;
        }
        current_state_.mode = mode;
        sample_generation_id_++;
        if (mode != "LOG") {
            current_state_.is_paused = false;
            current_state_.demo_auto_paused = false;
            demo_step_delta_ = 0;
            reset_analyzing_state(current_state_);
        }
        if (mode == "DEMO") {
            current_state_.active_tag = "DEMO · Conveyor Belt";
        } else if (mode == "IMAGE" && !playlist_.empty()) {
            current_state_.active_tag = "IMAGE [" + std::to_string(current_state_.active_idx + 1) + "/" +
                                        std::to_string(playlist_.size()) + "] · " + playlist_[current_state_.active_idx].first;
        } else if (mode == "VIDEO") {
            current_state_.active_tag = "VIDEO [" + std::to_string(video_playlist_.empty() ? 0 : video_idx_ + 1) + "/" +
                                        std::to_string(video_playlist_.size()) + "]";
        } else if (mode == "CAMERA") {
            current_state_.active_tag = "LIVE CAM · Stream";
        } else if (mode == "LOG") {
            current_state_.active_tag = "CSV LOGBOOK";
        }
    }
    notify_vision();
}

uint64_t IPCServer::get_sample_generation_id() const {
    return sample_generation_id_.load();
}

void IPCServer::update_state_for_generation(uint64_t gen_id, const LiveTelemetryState& state) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (sample_generation_id_.load() != gen_id) {
        return;
    }
    std::string m = current_state_.mode;
    int idx = current_state_.active_idx;
    bool paused = current_state_.is_paused;
    bool auto_paused = current_state_.demo_auto_paused;

    current_state_ = state;
    current_state_.mode = m;
    current_state_.active_idx = idx;
    current_state_.is_paused = paused;
    current_state_.demo_auto_paused = auto_paused;
}

void IPCServer::update_state(const LiveTelemetryState& state) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    std::string m = current_state_.mode;
    int idx = current_state_.active_idx;
    bool paused = current_state_.is_paused;
    bool auto_paused = current_state_.demo_auto_paused;

    current_state_ = state;
    current_state_.mode = m;
    current_state_.active_idx = idx;
    current_state_.is_paused = paused;
    current_state_.demo_auto_paused = auto_paused;
}

LiveTelemetryState IPCServer::get_state() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_state_;
}

void IPCServer::record_bench_sample(const qc::vision::TimingsMs& timings, const std::string& backend) {
    std::lock_guard<std::mutex> lock(bench_mutex_);
    bench_samples_++;
    sum_pre_ms_ += finite_or_zero(timings.pre);
    sum_infer_ms_ += finite_or_zero(timings.infer);
    sum_post_ms_ += finite_or_zero(timings.post);
    sum_total_ms_ += finite_or_zero(timings.total);
    last_timings_ = timings;
    if (!backend.empty()) {
        last_backend_ = backend;
    }
}

void IPCServer::record_bench_llm(float ttft_ms, float tokens_per_s, float total_ms) {
    std::lock_guard<std::mutex> lock(bench_mutex_);
    last_llm_ttft_ms_ = ttft_ms;
    last_llm_tps_ = tokens_per_s;
    last_llm_total_ms_ = total_ms;
}

double IPCServer::read_process_rss_mib() {
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        unsigned long long total_pages = 0;
        unsigned long long resident_pages = 0;
        if (statm >> total_pages >> resident_pages) {
            long page_size = sysconf(_SC_PAGESIZE);
            if (page_size <= 0) page_size = 4096;
            return (static_cast<double>(resident_pages) * static_cast<double>(page_size)) / (1024.0 * 1024.0);
        }
    }
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            double kb = 0.0;
            if (iss >> kb) {
                return kb / 1024.0;
            }
        }
    }
    return 0.0;
}

void IPCServer::load_logs_from_csv() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    inspection_logs_.clear();
    if (!fs::exists(CSV_LOG_PATH)) {
        return;
    }
    std::ifstream in(CSV_LOG_PATH);
    if (!in.is_open()) return;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto cols = parse_csv_line(line);
        if (cols.empty() || cols[0] == "Timestamp" || cols[0] == "timestamp") continue;

        InspectionRecord rec;
        if (cols.size() >= 19) {
            rec.timestamp = cols[0];
            rec.family = cols[1];
            rec.defect = cols[2];
            rec.group = cols[3];
            rec.score_conf = cols[4];
            try { rec.anomaly_score = std::stof(cols[5]); } catch (...) { rec.anomaly_score = 0.0f; }
            try { rec.area_pct = std::stof(cols[6]); } catch (...) { rec.area_pct = 0.0f; }
            try { rec.regions_boxes = std::stoi(cols[7]); } catch (...) { rec.regions_boxes = 0; }
            rec.spread = cols[8];
            rec.severity = cols[9];
            rec.verdict = cols[10];
            rec.cause = cols[11];
            rec.action = cols[12];
            rec.source = cols[13];
            try { rec.pre_ms = std::stof(cols[14]); } catch (...) { rec.pre_ms = 0.0f; }
            try { rec.infer_ms = std::stof(cols[15]); } catch (...) { rec.infer_ms = 0.0f; }
            try { rec.post_ms = std::stof(cols[16]); } catch (...) { rec.post_ms = 0.0f; }
            try { rec.total_ms = std::stof(cols[17]); } catch (...) { rec.total_ms = 0.0f; }
            try { rec.latency_ms = std::stof(cols[18]); } catch (...) { rec.latency_ms = 0.0f; }
            inspection_logs_.push_back(std::move(rec));
        } else if (cols.size() >= 14) {
            rec.timestamp = cols[0];
            rec.family = cols[2];
            rec.defect = cols[3];
            rec.group = "-";
            rec.score_conf = cols[4];
            try { rec.area_pct = std::stof(cols[6]); } catch (...) { rec.area_pct = 0.0f; }
            try { rec.regions_boxes = std::stoi(cols[7]); } catch (...) { rec.regions_boxes = 0; }
            rec.spread = cols[8];
            rec.severity = cols[9];
            rec.verdict = (rec.severity == "High" || rec.severity == "CRITICAL") ? "REJECT" :
                          ((rec.severity == "NONE" || rec.severity == "PASS") ? "PASS" : "REVIEW");
            rec.cause = cols[10];
            rec.action = cols[11];
            rec.source = cols[12];
            try { rec.latency_ms = std::stof(cols[13]); } catch (...) { rec.latency_ms = 0.0f; }
            inspection_logs_.push_back(std::move(rec));
        } else if (cols.size() >= 8) {
            rec.timestamp = cols[0];
            rec.defect = cols[1];
            rec.score_conf = cols[2];
            rec.severity = cols[3];
            try { rec.area_pct = std::stof(cols[4]); } catch (...) { rec.area_pct = 0.0f; }
            try { rec.regions_boxes = std::stoi(cols[5]); } catch (...) { rec.regions_boxes = 1; }
            rec.cause = cols[6];
            rec.action = cols[7];
            inspection_logs_.push_back(std::move(rec));
        }
    }
    if (inspection_logs_.size() > 500) {
        inspection_logs_.erase(inspection_logs_.begin(), inspection_logs_.end() - 500);
    }
}

void IPCServer::append_log(const InspectionRecord& record) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    inspection_logs_.push_back(record);
    if (inspection_logs_.size() > 500) {
        inspection_logs_.erase(inspection_logs_.begin());
    }

    bool need_header = !fs::exists(CSV_LOG_PATH) || fs::file_size(CSV_LOG_PATH) == 0;
    std::ofstream csv(CSV_LOG_PATH, std::ios::app);
    if (csv.is_open()) {
        if (need_header) {
            csv << CSV_HEADER_LINE << "\n";
        }
        csv << escape_csv(record.timestamp) << ","
            << escape_csv(record.family) << ","
            << escape_csv(record.defect) << ","
            << escape_csv(record.group) << ","
            << escape_csv(record.score_conf) << ","
            << std::fixed << std::setprecision(4) << record.anomaly_score << ","
            << std::fixed << std::setprecision(2) << record.area_pct << ","
            << record.regions_boxes << ","
            << escape_csv(record.spread) << ","
            << escape_csv(record.severity) << ","
            << escape_csv(record.verdict) << ","
            << escape_csv(record.cause) << ","
            << escape_csv(record.action) << ","
            << escape_csv(record.source) << ","
            << std::fixed << std::setprecision(2) << record.pre_ms << ","
            << std::fixed << std::setprecision(2) << record.infer_ms << ","
            << std::fixed << std::setprecision(2) << record.post_ms << ","
            << std::fixed << std::setprecision(2) << record.total_ms << ","
            << static_cast<long>(std::lround(record.latency_ms)) << "\n";
    }
}

std::string IPCServer::get_logs_json() const {
    std::lock_guard<std::mutex> lock(log_mutex_);
    json arr = json::array();
    for (auto it = inspection_logs_.rbegin(); it != inspection_logs_.rend(); ++it) {
        std::string sev_val = it->severity.empty() ? "NONE" : it->severity;
        std::string verd_val = it->verdict.empty() ? "PASS" : it->verdict;
        std::string fam_val = it->family.empty() ? "steel_sheet" : it->family;
        std::string grp_val = it->group.empty() ? "NONE" : it->group;
        arr.push_back({
            {"ts", it->timestamp},
            {"timestamp", it->timestamp},
            {"family", fam_val},
            {"part", fam_val},
            {"defect", it->defect},
            {"group", grp_val},
            {"score_conf", it->score_conf},
            {"anomaly_score", round_decimals(it->anomaly_score, 4)},
            {"area_pct", format_one_decimal(it->area_pct)},
            {"regions_boxes", it->regions_boxes},
            {"spread", it->spread.empty() ? "-" : it->spread},
            {"severity", sev_val},
            {"sev", sev_val},
            {"verdict", verd_val},
            {"cause", it->cause},
            {"action", it->action},
            {"src", it->source},
            {"advisory_source", it->source},
            {"timings_ms", {
                {"pre", round_decimals(it->pre_ms, 2)},
                {"infer", round_decimals(it->infer_ms, 2)},
                {"post", round_decimals(it->post_ms, 2)},
                {"total", round_decimals(it->total_ms, 2)}
            }},
            {"ms", static_cast<long>(std::lround(it->latency_ms))}
        });
    }
    return arr.dump();
}

void IPCServer::clear_logs() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    inspection_logs_.clear();
    std::ofstream csv(CSV_LOG_PATH, std::ios::trunc);
    if (csv.is_open()) {
        csv << CSV_HEADER_LINE << "\n";
    }
}

std::string IPCServer::get_csv_logs() const {
    std::lock_guard<std::mutex> lock(log_mutex_);
    std::stringstream ss;
    ss << CSV_HEADER_LINE << "\n";
    for (const auto& r : inspection_logs_) {
        ss << escape_csv(r.timestamp) << ","
           << escape_csv(r.family) << ","
           << escape_csv(r.defect) << ","
           << escape_csv(r.group) << ","
           << escape_csv(r.score_conf) << ","
           << std::fixed << std::setprecision(4) << r.anomaly_score << ","
           << std::fixed << std::setprecision(2) << r.area_pct << ","
           << r.regions_boxes << ","
           << escape_csv(r.spread) << ","
           << escape_csv(r.severity) << ","
           << escape_csv(r.verdict) << ","
           << escape_csv(r.cause) << ","
           << escape_csv(r.action) << ","
           << escape_csv(r.source) << ","
           << std::fixed << std::setprecision(2) << r.pre_ms << ","
           << std::fixed << std::setprecision(2) << r.infer_ms << ","
           << std::fixed << std::setprecision(2) << r.post_ms << ","
           << std::fixed << std::setprecision(2) << r.total_ms << ","
           << static_cast<long>(std::lround(r.latency_ms)) << "\n";
    }
    return ss.str();
}

std::string IPCServer::serialize_json_state() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    json j;
    j["frame_id"] = current_state_.frame_id;
    j["frame_seq"] = frame_seq_.load(std::memory_order_acquire);
    j["sample_gen"] = sample_generation_id_.load(std::memory_order_acquire);
    j["mode"] = current_state_.mode;

    int sz = static_cast<int>(playlist_.size());
    int idx = current_state_.active_idx;
    std::string active_name = (idx >= 0 && idx < sz) ? playlist_[idx].first : "sample";
    if (current_state_.mode == "DEMO") {
        j["active_tag"] = "DEMO · Conveyor Belt";
    } else if (current_state_.mode == "IMAGE") {
        j["active_tag"] = "IMAGE [" + std::to_string(idx + 1) + "/" + std::to_string(sz) + "] · " + active_name;
    } else if (current_state_.mode == "VIDEO") {
        int vsz = static_cast<int>(video_playlist_.size());
        j["active_tag"] = "VIDEO [" + std::to_string(vsz > 0 ? video_idx_ + 1 : 0) + "/" + std::to_string(vsz) + "]";
    } else if (current_state_.mode == "CAMERA") {
        j["active_tag"] = "LIVE CAM · Stream";
    } else {
        j["active_tag"] = "CSV LOGBOOK";
    }

    j["family"] = current_state_.family;
    j["family_conf"] = round_decimals(current_state_.family_conf, 4);
    j["is_not_metal"] = current_state_.is_not_metal;
    j["vision_backend"] = current_state_.vision_backend;
    j["vision_hardware"] = current_state_.hardware_device;

    json defects_arr = json::array();
    for (const auto& d : current_state_.defects) {
        defects_arr.push_back({
            {"label", d.label},
            {"group", d.group},
            {"severity", d.severity},
            {"confidence", round_decimals(d.confidence, 4)},
            {"box", {d.x1, d.y1, d.x2, d.y2}},
            {"class_id", d.class_id}
        });
    }
    j["defects"] = defects_arr;

    j["primary_label"] = current_state_.primary_label;
    j["primary_group"] = current_state_.primary_group;
    j["defect"] = current_state_.defect;
    j["display_title"] = current_state_.display_title;
    j["conf"] = round_decimals(current_state_.conf, 4);
    j["area"] = round_decimals(current_state_.area, 2);
    j["boxes"] = current_state_.boxes;
    j["spread"] = current_state_.spread;
    j["severity"] = current_state_.severity;
    j["verdict"] = current_state_.verdict;

    std::string anom_status = "UNINDEXED";
    if (current_state_.is_not_metal) {
        anom_status = "GATE";
    } else if (current_state_.anomaly.is_anomalous) {
        anom_status = "ANOMALY";
    } else if (current_state_.anomaly.bank_size > 0) {
        anom_status = "NORMAL";
    }

    j["anomaly_score"] = round_decimals(current_state_.anomaly.score, 4);
    j["anomaly"] = {
        {"score", round_decimals(current_state_.anomaly.score, 4)},
        {"threshold", round_decimals(current_state_.anomaly.threshold, 4)},
        {"is_anomalous", current_state_.anomaly.is_anomalous},
        {"status", anom_status},
        {"bank_size", current_state_.anomaly.bank_size},
        {"max_refs", current_state_.anomaly_max_refs},
        {"hotspot_box", {
            current_state_.anomaly.hotspot_x1,
            current_state_.anomaly.hotspot_y1,
            current_state_.anomaly.hotspot_x2,
            current_state_.anomaly.hotspot_y2
        }}
    };

    j["timings_ms"] = {
        {"pre", round_decimals(current_state_.timings_ms.pre, 2)},
        {"infer", round_decimals(current_state_.timings_ms.infer, 2)},
        {"post", round_decimals(current_state_.timings_ms.post, 2)},
        {"total", round_decimals(current_state_.timings_ms.total, 2)}
    };
    j["cv_ms"] = round_decimals(current_state_.cv_ms, 2);

    std::string narrative = current_state_.narrative;
    if (narrative.empty() && !current_state_.cause.empty()) {
        narrative = "Possible Cause: " + current_state_.cause +
                    " | Recommended Action: " + current_state_.action;
    }

    j["llm"] = {
        {"ttft_ms", round_decimals(current_state_.ttft_ms, 1)},
        {"tokens_per_s", round_decimals(current_state_.tokens_per_s, 2)},
        {"total_ms", round_decimals(current_state_.llm_ms, 1)},
        {"narrative", narrative},
        {"summary", narrative},
        {"cause", current_state_.cause},
        {"action", current_state_.action},
        {"tokens", current_state_.tokens},
        {"source", current_state_.engine_source}
    };

    j["cause"] = current_state_.cause;
    j["action"] = current_state_.action;
    j["tokens"] = current_state_.tokens;
    j["tok_speed"] = round_decimals(current_state_.tokens_per_s, 2);
    j["llm_ms"] = round_decimals(current_state_.llm_ms, 1);
    j["llm_busy"] = current_state_.llm_busy;
    j["advisory_pending"] = current_state_.advisory_pending;
    j["llm_elapsed"] = round_decimals(current_state_.llm_elapsed, 2);
    j["engine_source"] = current_state_.engine_source;

    j["is_locked"] = current_state_.is_locked;
    j["lock_cnt"] = current_state_.lock_cnt;
    j["is_paused"] = current_state_.is_paused;
    j["demo_auto_paused"] = current_state_.demo_auto_paused;
    j["system_status"] = current_state_.vision_backend + " · LLAMA 3.2 1B";
    j["active_idx"] = (current_state_.mode == "VIDEO") ? video_idx_ : current_state_.active_idx;

    j["playlist"] = json::array();
    if (current_state_.mode == "VIDEO") {
        for (const auto& item : video_playlist_) {
            j["playlist"].push_back({{"name", item.first}, {"path", item.second}});
        }
    } else if (current_state_.mode == "IMAGE") {
        for (const auto& item : playlist_) {
            j["playlist"].push_back({{"name", item.first}, {"path", item.second}});
        }
    }

    return j.dump();
}

std::string IPCServer::serialize_bench_json() const {
    double rss_mib = read_process_rss_mib();

    std::string active_family = "steel_sheet";
    std::string hardware_dev = "ARM64 ONNX Runtime";
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        active_family = current_state_.family;
        hardware_dev = current_state_.hardware_device;
    }

    uint64_t samples = 0;
    double pre = 0.0, inf = 0.0, post = 0.0, total = 0.0;
    double last_pre = 0.0, last_inf = 0.0, last_post = 0.0, last_tot = 0.0;
    std::string backend = "onnxruntime_arm64";
    float ttft = 0.0f, tps = 0.0f, llm_tot = 0.0f;
    {
        std::lock_guard<std::mutex> lock(bench_mutex_);
        samples = bench_samples_;
        if (samples > 0) {
            pre = sum_pre_ms_ / static_cast<double>(samples);
            inf = sum_infer_ms_ / static_cast<double>(samples);
            post = sum_post_ms_ / static_cast<double>(samples);
            total = sum_total_ms_ / static_cast<double>(samples);
        }
        last_pre = last_timings_.pre;
        last_inf = last_timings_.infer;
        last_post = last_timings_.post;
        last_tot = last_timings_.total;
        backend = last_backend_;
        ttft = last_llm_ttft_ms_;
        tps = last_llm_tps_;
        llm_tot = last_llm_total_ms_;
    }

    double fps = (total > 0.0) ? (1000.0 / total) : ((last_tot > 0.0) ? (1000.0 / last_tot) : 0.0);
    std::error_code ec;
    bool physical_hailo = (vision_engine_ != nullptr &&
                           vision_engine_->is_hailo_active() &&
                           fs::exists("/dev/hailo0", ec));

    json j;
    j["vision_backend"] = backend;
    j["hardware_device"] = hardware_dev;
    j["family"] = active_family;
    j["rss_mib"] = round_decimals(rss_mib, 2);
    j["samples_measured"] = samples;
    j["fps"] = round_decimals(fps, 2);
    j["timings_ms"] = {
        {"pre", round_decimals(samples > 0 ? pre : last_pre, 2)},
        {"infer", round_decimals(samples > 0 ? inf : last_inf, 2)},
        {"post", round_decimals(samples > 0 ? post : last_post, 2)},
        {"total", round_decimals(samples > 0 ? total : last_tot, 2)}
    };
    j["last_timings_ms"] = {
        {"pre", round_decimals(last_pre, 2)},
        {"infer", round_decimals(last_inf, 2)},
        {"post", round_decimals(last_post, 2)},
        {"total", round_decimals(last_tot, 2)}
    };
    j["patchcore_ms"] = 0.0;
    j["llm"] = {
        {"ttft_ms", round_decimals(ttft, 1)},
        {"tokens_per_s", round_decimals(tps, 2)},
        {"total_ms", round_decimals(llm_tot, 1)}
    };
    if (physical_hailo) {
        j["rpi5_npu_fps"] = round_decimals(fps, 2);
    } else {
        j["rpi5_npu_fps"] = "Pending Physical RPi5";
    }
    j["usb_c_power_w"] = "Pending Physical RPi5";
    return j.dump();
}

bool IPCServer::start() {
    is_running_ = true;
    server_thread_ = std::thread(&IPCServer::server_loop, this);
    return true;
}

void IPCServer::stop() {
    if (is_running_) {
        is_running_ = false;
        notify_vision();
        if (svr_) {
            svr_->stop();
        }
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
    }
}

void IPCServer::server_loop() {
    hardware::set_thread_name("scada_http_c0");
    hardware::pin_current_thread(hardware::CORE_IO_IPC);

    std::cout << "[IPCServer] Pure C++ Native SCADA Web Server listening on port " << port_
              << " (" << hardware::get_current_affinity_string() << ")" << std::endl;

    // 1. Embedded Web UI Routes
    svr_->Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
        res.set_content(ui::get_index_html(), "text/html; charset=utf-8");
    });
    svr_->Get("/index.html", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
        res.set_content(ui::get_index_html(), "text/html; charset=utf-8");
    });

    // 2. Unified Shared JSON Telemetry & Benchmark Routes
    svr_->Get("/api/state", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Cache-Control", "no-store, max-age=0");
        res.set_content(serialize_json_state(), "application/json");
    });
    svr_->Get("/api/telemetry", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(serialize_json_state(), "application/json");
    });
    svr_->Get("/api/bench", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(serialize_bench_json(), "application/json");
    });

    // 3. PatchCore Runtime Good-Image Registration
    svr_->Post("/api/register_good", [this](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        if (!vision_engine_) {
            res.status = 503;
            res.set_content("{\"success\":false,\"error\":\"Vision engine not attached\"}", "application/json");
            return;
        }

        std::string req_family;
        if (req.has_param("family")) {
            req_family = trim_string(req.get_param_value("family"));
        }

        std::vector<uint8_t> img_bytes;
        if (req.is_multipart_form_data()) {
            if (req_family.empty() && req.has_file("family")) {
                req_family = trim_string(req.get_file_value("family").content);
            }
            for (const auto& kv : req.files) {
                if (kv.first == "family") continue;
                if (!kv.second.content.empty()) {
                    img_bytes.assign(kv.second.content.begin(), kv.second.content.end());
                    break;
                }
            }
        } else if (!req.body.empty()) {
            std::string trimmed = trim_string(req.body);
            if (!trimmed.empty() && trimmed.front() == '{') {
                try {
                    auto jbody = json::parse(trimmed);
                    if (req_family.empty() && jbody.contains("family") && jbody["family"].is_string()) {
                        req_family = trim_string(jbody["family"].get<std::string>());
                    }
                } catch (...) {
                }
            } else {
                img_bytes.assign(req.body.begin(), req.body.end());
            }
        }

        const auto& rules = vision_engine_->rules();
        std::string target_family;
        int images_saved = 0;
        bool ok = false;

        if (!img_bytes.empty()) {
            cv::Mat decoded = cv::imdecode(img_bytes, cv::IMREAD_COLOR);
            if (decoded.empty()) {
                res.status = 400;
                res.set_content("{\"success\":false,\"error\":\"Invalid image payload\"}", "application/json");
                return;
            }
            if (!req_family.empty()) {
                target_family = qc::rules::normalize_label(req_family);
            } else {
                auto s0 = vision_engine_->classify_stage0(decoded);
                target_family = (!s0.is_not_metal) ? s0.family : vision_engine_->get_last_family();
            }
            if (!rules.is_metal_family(target_family)) {
                target_family = rules.config().default_family;
            }
            ok = vision_engine_->register_good_image(target_family, decoded, &images_saved);
        } else {
            if (!req_family.empty()) {
                target_family = qc::rules::normalize_label(req_family);
            } else {
                target_family = vision_engine_->get_last_family();
            }
            if (!rules.is_metal_family(target_family)) {
                target_family = rules.config().default_family;
            }
            ok = vision_engine_->register_latest_good_image(target_family, &images_saved);
            if (!ok) {
                std::string cur_path = get_current_image_path();
                if (!cur_path.empty() && fs::exists(cur_path)) {
                    cv::Mat disk_img = cv::imread(cur_path);
                    if (!disk_img.empty()) {
                        ok = vision_engine_->register_good_image(target_family, disk_img, &images_saved);
                    }
                }
            }
        }

        if (!ok) {
            res.status = 400;
            res.set_content("{\"success\":false,\"error\":\"No valid frame available to register\"}", "application/json");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (current_state_.family == target_family) {
                current_state_.anomaly.bank_size = vision_engine_->get_patchcore_bank_size(target_family);
            }
        }

        json resp = {
            {"success", true},
            {"family", target_family},
            {"images_saved", images_saved}
        };
        res.set_content(resp.dump(), "application/json");
    });

    // 4. Image / Frame Stream (supports ?since=<frame_seq> for zero-bandwidth unchanged polling)
    auto handle_frame_req = [this](const httplib::Request& req, httplib::Response& res) {
        uint64_t seq = 0;
        std::vector<uint8_t> frame = get_frame_jpeg_with_seq(seq);
        res.set_header("Cache-Control", "no-cache, no-store, must-revalidate");
        res.set_header("X-Frame-Seq", std::to_string(seq));
        if (req.has_param("since") && seq > 0) {
            try {
                uint64_t since_seq = std::stoull(req.get_param_value("since"));
                if (since_seq == seq) {
                    res.status = 204;
                    return;
                }
            } catch (...) {
            }
        }
        if (!frame.empty()) {
            res.set_content(reinterpret_cast<const char*>(frame.data()), frame.size(), "image/jpeg");
        } else {
            res.status = 404;
        }
    };
    svr_->Get("/api/frame.jpg", handle_frame_req);
    svr_->Get("/api/stream", handle_frame_req);

    // 5. Playlist & Controls (Single-menu mode navigation only)
    svr_->Get("/api/next", [this](const httplib::Request&, httplib::Response& res) {
        next_sample();
        res.set_header("Cache-Control", "no-store, max-age=0");
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    svr_->Get("/api/prev", [this](const httplib::Request&, httplib::Response& res) {
        prev_sample();
        res.set_header("Cache-Control", "no-store, max-age=0");
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    svr_->Get("/api/select_item", [this](const httplib::Request& req, httplib::Response& res) {
        if (req.has_param("mode")) {
            std::string req_mode = trim_string(req.get_param_value("mode"));
            if (!req_mode.empty()) {
                std::lock_guard<std::mutex> lock(state_mutex_);
                current_state_.mode = req_mode;
            }
        }
        if (req.has_param("path")) {
            select_sample_by_path(req.get_param_value("path"));
        } else if (req.has_param("idx")) {
            try { select_sample(std::stoi(req.get_param_value("idx"))); } catch (...) {}
        } else if (req.has_param("index")) {
            try { select_sample(std::stoi(req.get_param_value("index"))); } catch (...) {}
        }
        res.set_header("Cache-Control", "no-store, max-age=0");
        res.set_content("{\"status\":\"ok\",\"success\":true}", "application/json");
    });
    svr_->Get("/api/switch_mode", [this](const httplib::Request& req, httplib::Response& res) {
        std::string mode = req.has_param("mode") ? req.get_param_value("mode") : "DEMO";
        switch_mode(mode);
        json r = {{"mode", mode}, {"success", true}};
        res.set_header("Cache-Control", "no-store, max-age=0");
        res.set_content(r.dump(), "application/json");
    });
    svr_->Get("/api/toggle_pause", [this](const httplib::Request&, httplib::Response& res) {
        toggle_pause();
        bool paused = is_paused();
        res.set_header("Cache-Control", "no-store, max-age=0");
        res.set_content("{\"is_paused\":" + std::string(paused ? "true" : "false") + ",\"success\":true}", "application/json");
    });
    svr_->Get("/api/control", [this](const httplib::Request& req, httplib::Response& res) {
        std::string cmd = req.has_param("cmd") ? req.get_param_value("cmd") : "";
        if (cmd == "next") {
            next_sample();
            res.set_content("{\"status\":\"ok\",\"cmd\":\"next\"}", "application/json");
        } else if (cmd == "prev") {
            prev_sample();
            res.set_content("{\"status\":\"ok\",\"cmd\":\"prev\"}", "application/json");
        } else if (cmd == "pause" || cmd == "toggle_pause") {
            toggle_pause();
            bool paused = is_paused();
            res.set_content("{\"is_paused\":" + std::string(paused ? "true" : "false") + ",\"success\":true}", "application/json");
        } else if (cmd == "mode" && req.has_param("mode")) {
            std::string mode = req.get_param_value("mode");
            switch_mode(mode);
            json r = {{"mode", mode}, {"success", true}};
            res.set_content(r.dump(), "application/json");
        } else if (cmd == "select") {
            if (req.has_param("path")) {
                select_sample_by_path(req.get_param_value("path"));
            } else if (req.has_param("idx")) {
                try { select_sample(std::stoi(req.get_param_value("idx"))); } catch (...) {}
            }
            res.set_content("{\"status\":\"ok\",\"success\":true}", "application/json");
        } else {
            res.status = 400;
            res.set_content("{\"success\":false,\"error\":\"Unsupported control cmd\"}", "application/json");
        }
    });

    // 6. File Uploads (Multipart Form Data)
    svr_->Post("/api/upload", [this](const httplib::Request& req, httplib::Response& res) {
        if (!req.is_multipart_form_data() || req.files.empty()) {
            res.set_content("{\"success\":false,\"status\":\"error\",\"error\":\"No file uploaded\"}", "application/json");
            return;
        }
        fs::create_directories("uploads");
        std::string last_media_type = "IMAGE";
        std::string last_name;
        bool saved_any = false;

        for (const auto& file : req.files) {
            std::string clean_name = fs::path(file.second.filename).filename().string();
            if (clean_name.empty()) continue;

            std::string ext = fs::path(clean_name).extension().string();
            bool is_vid = is_video_extension(ext);
            bool is_img = is_image_extension(ext);
            if (!is_vid && !is_img) continue;

            std::string save_path = "uploads/" + clean_name;
            std::ofstream out(save_path, std::ios::binary);
            if (!out.is_open()) continue;
            out.write(file.second.content.data(), file.second.content.size());
            out.close();

            std::lock_guard<std::mutex> lock(state_mutex_);
            sample_generation_id_++;
            current_state_.is_paused = false;
            reset_analyzing_state(current_state_);
            if (is_vid) {
                auto it = std::find_if(video_playlist_.begin(), video_playlist_.end(),
                                       [&](const auto& p) { return p.second == save_path || p.first == clean_name; });
                if (it != video_playlist_.end()) video_playlist_.erase(it);
                video_playlist_.insert(video_playlist_.begin(), {clean_name, save_path});
                video_idx_ = 0;
                current_state_.mode = "VIDEO";
                current_state_.active_tag = "VIDEO [1/" + std::to_string(video_playlist_.size()) + "]";
                last_media_type = "VIDEO";
            } else {
                auto it = std::find_if(playlist_.begin(), playlist_.end(),
                                       [&](const auto& p) { return p.second == save_path || p.first == clean_name; });
                if (it != playlist_.end()) playlist_.erase(it);
                playlist_.insert(playlist_.begin(), {clean_name, save_path});
                current_state_.active_idx = 0;
                current_state_.mode = "IMAGE";
                current_state_.active_tag = "IMAGE [1/" + std::to_string(playlist_.size()) + "] · " + clean_name;
                last_media_type = "IMAGE";
            }
            last_name = clean_name;
            saved_any = true;
        }

        if (!saved_any) {
            res.set_content("{\"success\":false,\"status\":\"error\",\"error\":\"Unsupported or empty upload file\"}", "application/json");
            return;
        }

        notify_vision();
        json ok_resp = {
            {"success", true},
            {"status", "ok"},
            {"media_type", last_media_type},
            {"name", last_name},
            {"message", "Uploaded successfully"}
        };
        res.set_content(ok_resp.dump(), "application/json");
    });

    // 7. Logs & Reports
    svr_->Get("/api/logs", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(get_logs_json(), "application/json");
    });
    svr_->Post("/api/clear_logs", [this](const httplib::Request&, httplib::Response& res) {
        clear_logs();
        res.set_content("{\"success\":true,\"status\":\"ok\",\"message\":\"Logs cleared\"}", "application/json");
    });
    svr_->Get("/api/download_csv", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Content-Disposition", "attachment; filename=rpi5_qc_inspection_log.csv");
        res.set_content(get_csv_logs(), "text/csv");
    });
    svr_->Post("/api/save_frame", [this](const httplib::Request&, httplib::Response& res) {
        std::vector<uint8_t> frame = get_frame_jpeg();
        if (!frame.empty()) {
            fs::create_directories("snapshots");
            auto now = std::chrono::system_clock::now();
            auto t = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << "snapshots/snap_" << t << ".jpg";
            std::string snap_path = ss.str();
            std::ofstream out(snap_path, std::ios::binary);
            if (out.is_open()) {
                out.write(reinterpret_cast<const char*>(frame.data()), frame.size());
            }
            res.set_content("{\"success\":true,\"filename\":\"" + snap_path + "\"}", "application/json");
            return;
        }
        res.set_content("{\"success\":false}", "application/json");
    });

    svr_->listen("0.0.0.0", port_);
}

} // namespace ipc
} // namespace qc
