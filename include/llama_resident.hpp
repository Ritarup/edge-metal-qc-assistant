#pragma once

#include "qc_rules.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace qc {
namespace reasoning {

struct AdvisoryResult {
    bool success = false;
    std::string possible_cause;
    std::string recommended_action;
    std::string summary; // "Possible Cause: ... | Recommended Action: ..."
    std::string advisory_source;
    float ttft_ms = 0.0f;
    float tokens_per_s = 0.0f;
    float total_ms = 0.0f;
    int tokens_generated = 0;
};

struct AdvisoryRequest {
    std::string family = "steel_sheet";
    std::string label = "scratch";
    std::string group = "scratch";
    std::string severity = "MODERATE";
    float confidence = 0.0f;
    float area_pct = 0.0f;
    int box_count = 1;
    float anomaly_score = 0.0f;
    bool is_anomalous = false;
    std::promise<AdvisoryResult> promise;
};

/**
 * Warm Resident In-Memory ExecuTorch Llama 3.2 1B Engine.
 * Loads INT4 .pte flatbuffer and tokenizer ONCE at boot into RAM,
 * formats unified multi-metal QC telemetry prompts, and records
 * TTFT (ttft_ms), decode throughput (tokens_per_s), and total_ms.
 */
class LlamaResidentEngine {
public:
    LlamaResidentEngine();
    ~LlamaResidentEngine();

    bool init(const std::string& config_path = "config/edge_qc_config.json");

    bool init(
        const std::string& pte_path,
        const std::string& tokenizer_path,
        const std::string& runner_path = "",
        const std::string& config_path = "config/edge_qc_config.json"
    );

    std::future<AdvisoryResult> request_advisory_async(
        const std::string& family,
        const std::string& label,
        const std::string& group,
        const std::string& severity,
        float confidence,
        float area_pct,
        int box_count,
        float anomaly_score = 0.0f,
        bool is_anomalous = false
    );

    AdvisoryResult request_advisory_sync(
        const std::string& family,
        const std::string& label,
        const std::string& group,
        const std::string& severity,
        float confidence,
        float area_pct,
        int box_count,
        float anomaly_score = 0.0f,
        bool is_anomalous = false
    );

    void shutdown();
    bool is_ready() const { return is_initialized_.load(); }

private:
    qc::rules::QcRulesEngine rules_;
    std::string pte_path_;
    std::string tokenizer_path_;
    std::string runner_bin_;
    std::atomic<bool> is_initialized_{false};
    std::atomic<bool> is_running_{false};

    int daemon_pid_ = -1;
    int daemon_in_fd_ = -1;
    int daemon_out_fd_ = -1;
    FILE* daemon_stream_ = nullptr;

    std::thread worker_thread_;
    std::mutex queue_mutex_;
    std::mutex daemon_io_mutex_;
    std::condition_variable cv_;
    std::queue<std::unique_ptr<AdvisoryRequest>> request_queue_;

    void worker_loop();
    bool start_resident_daemon();
    bool restart_resident_daemon_locked();
    AdvisoryResult execute_inference(const AdvisoryRequest& req);
    std::string build_prompt(const AdvisoryRequest& req) const;
    void parse_output(
        const std::string& raw_output,
        std::string& cause,
        std::string& action
    ) const;
};

} // namespace reasoning
} // namespace qc
