#pragma once

#include "hailo_vision_engine.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace httplib {
    class Server;
}

namespace qc {
namespace ipc {

struct LiveTelemetryState {
    uint64_t frame_id = 0;
    std::string mode = "DEMO";
    std::string active_tag = "DEMO 1/1 · Industrial Conveyor";

    // Stage-0 material family + vision backend
    std::string family = "steel_sheet";
    float family_conf = 1.0f;
    bool is_not_metal = false;
    std::string vision_backend = "onnxruntime_arm64";
    std::string hardware_device = "ARM64 ONNX Runtime (Cortex-A76)";

    // Unified detection telemetry
    std::vector<qc::vision::DetectionBox> defects;
    std::string primary_label = "NONE";
    std::string primary_group = "NONE";
    std::string defect = "NO DEFECT DETECTED";
    std::string display_title = "NO DEFECT DETECTED";
    float conf = 0.0f;
    float area = 0.0f;
    int boxes = 0;
    std::string spread = "localized";
    std::string severity = "NONE"; // NONE, MINOR, MODERATE, CRITICAL
    std::string verdict = "PASS";  // PASS, REVIEW, REJECT

    // PatchCore anomaly telemetry
    qc::vision::AnomalyTelemetry anomaly;
    int anomaly_max_refs = 20;

    // Steady-clock pipeline breakdown (ms)
    qc::vision::TimingsMs timings_ms;
    float cv_ms = 0.0f;

    // LLM Advisory & TTFT telemetry
    std::string cause = "Waiting for Llama 3.2 1B advisory...";
    std::string action = "Waiting for Llama 3.2 1B advisory...";
    std::string narrative = "";
    std::string engine_source = "Llama 3.2 1B (Initializing...)";
    float ttft_ms = 0.0f;
    float tokens_per_s = 0.0f;
    float llm_ms = 0.0f;
    int tokens = 0;
    bool llm_busy = false;
    bool advisory_pending = true;
    float llm_elapsed = 0.0f;

    // Lock & UI controls
    bool is_locked = true;
    int lock_cnt = 4;
    bool is_paused = false;
    bool demo_auto_paused = false;
    std::string system_status = "ONNXRUNTIME_ARM64 · LLAMA 3.2 1B";
    int active_idx = 0;
};

struct InspectionRecord {
    std::string timestamp;
    std::string family = "steel_sheet";
    std::string defect;
    std::string group = "NONE";
    std::string score_conf = "0.0%";
    float anomaly_score = 0.0f;
    float area_pct = 0.0f;
    int regions_boxes = 0;
    std::string spread = "-";
    std::string severity = "NONE";
    std::string verdict = "PASS";
    std::string cause;
    std::string action;
    std::string source = "System";
    float pre_ms = 0.0f;
    float infer_ms = 0.0f;
    float post_ms = 0.0f;
    float total_ms = 0.0f;
    float latency_ms = 0.0f;
};

/**
 * Production C++ HTTP SCADA Server.
 * Powered by cpp-httplib and nlohmann::json.
 * Pinned to Core 0 of Raspberry Pi 5 / ARM64 host.
 * Serves the single-menu dark SCADA interface and unified REST API on port 5000.
 */
class IPCServer {
public:
    explicit IPCServer(int port = 5000);
    ~IPCServer();

    void set_vision_engine(qc::vision::HailoVisionEngine* engine);

    bool start();
    void stop();

    void update_state(const LiveTelemetryState& state);
    void update_state_for_generation(uint64_t gen_id, const LiveTelemetryState& state);
    LiveTelemetryState get_state() const;
    uint64_t get_sample_generation_id() const;

    void record_bench_sample(const qc::vision::TimingsMs& timings, const std::string& backend);
    void record_bench_llm(float ttft_ms, float tokens_per_s, float total_ms);

    void update_frame_jpeg(const std::vector<uint8_t>& jpeg_data);
    std::vector<uint8_t> get_frame_jpeg() const;
    std::vector<uint8_t> get_frame_jpeg_with_seq(uint64_t& out_seq) const;
    uint64_t get_frame_seq() const { return frame_seq_.load(); }
    void notify_vision();
    void wait_for_vision_wakeup(int timeout_ms);

    std::string get_current_image_path() const;
    std::string get_current_video_path() const;
    std::string get_demo_video_path() const;
    int get_active_index() const;
    std::string get_active_mode() const;
    bool is_paused() const;
    bool is_manual_paused() const;
    void set_auto_paused(bool paused);
    int consume_demo_step_delta();

    void next_sample();
    void prev_sample();
    void select_sample(int index);
    void select_sample_by_path(const std::string& path);
    void set_active_index(int idx);
    size_t get_playlist_size() const;
    std::string get_image_path_at(int idx) const;
    void toggle_pause();
    void switch_mode(const std::string& mode);

    void append_log(const InspectionRecord& record);
    std::string get_logs_json() const;
    void clear_logs();
    std::string get_csv_logs() const;

private:
    int port_;
    qc::vision::HailoVisionEngine* vision_engine_ = nullptr;
    std::atomic<bool> is_running_{false};
    std::atomic<uint64_t> sample_generation_id_{1};
    std::atomic<uint64_t> frame_seq_{0};
    std::thread server_thread_;

    mutable std::mutex state_mutex_;
    mutable std::mutex frame_mutex_;
    mutable std::mutex log_mutex_;
    mutable std::mutex bench_mutex_;
    std::mutex wake_mutex_;
    std::condition_variable wake_cv_;
    bool wake_pending_{false};

    LiveTelemetryState current_state_;
    std::vector<uint8_t> current_jpeg_;
    std::vector<InspectionRecord> inspection_logs_;
    std::unique_ptr<httplib::Server> svr_;

    std::vector<std::pair<std::string, std::string>> playlist_;       // display_name, full_path (images)
    std::vector<std::pair<std::string, std::string>> video_playlist_; // display_name, full_path (videos)
    std::string demo_video_path_;
    int video_idx_ = 0;
    int demo_step_delta_ = 0;

    // Rolling benchmark telemetry
    uint64_t bench_samples_ = 0;
    double sum_pre_ms_ = 0.0;
    double sum_infer_ms_ = 0.0;
    double sum_post_ms_ = 0.0;
    double sum_total_ms_ = 0.0;
    qc::vision::TimingsMs last_timings_;
    std::string last_backend_ = "onnxruntime_arm64";
    float last_llm_ttft_ms_ = 0.0f;
    float last_llm_tps_ = 0.0f;
    float last_llm_total_ms_ = 0.0f;

    void init_playlist();
    void load_logs_from_csv();
    void server_loop();
    std::string serialize_json_state() const;
    std::string serialize_bench_json() const;
    static double read_process_rss_mib();
};

} // namespace ipc
} // namespace qc
