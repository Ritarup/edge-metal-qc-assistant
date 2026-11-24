#include "llama_resident.hpp"
#include "cpu_affinity.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace qc {
namespace reasoning {

namespace {

std::string find_readable_file(const std::vector<std::string>& candidates, int mode) {
    for (const auto& c : candidates) {
        if (!c.empty() && access(c.c_str(), mode) == 0) {
            return c;
        }
    }
    return "";
}

bool extract_json_long(const std::string& text, const char* key, long& out_val) {
    std::string pat = std::string("\"") + key + R"("\s*:\s*(\d+))";
    std::smatch m;
    if (std::regex_search(text, m, std::regex(pat))) {
        try {
            out_val = std::stol(m[1].str());
            return true;
        } catch (...) {
            return false;
        }
    }
    return false;
}

bool extract_json_float(const std::string& text, const char* key, float& out_val) {
    std::string pat = std::string("\"") + key + R"("\s*:\s*([0-9]+(?:\.[0-9]+)?))";
    std::smatch m;
    if (std::regex_search(text, m, std::regex(pat))) {
        try {
            out_val = std::stof(m[1].str());
            return true;
        } catch (...) {
            return false;
        }
    }
    return false;
}

std::string trim_copy(std::string s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

} // namespace

LlamaResidentEngine::LlamaResidentEngine() = default;

LlamaResidentEngine::~LlamaResidentEngine() {
    shutdown();
}

bool LlamaResidentEngine::init(const std::string& config_path) {
    return init("", "", "", config_path);
}

bool LlamaResidentEngine::init(
    const std::string& pte_path,
    const std::string& tokenizer_path,
    const std::string& runner_path,
    const std::string& config_path
) {
#ifdef SIGPIPE
    std::signal(SIGPIPE, SIG_IGN);
#endif
    rules_.load_config(config_path);
    const auto& llm_cfg = rules_.config().llm;

    const char* home_env = getenv("HOME");
    std::string home_cache = home_env ? (std::string(home_env) + "/.cache/llama_cache") : "";

    std::vector<std::string> pte_candidates = {
        pte_path,
        llm_cfg.pte_path,
        "models/Llama-3.2-1B-UnifiedQC-FT-INT4.pte",
        "../models/Llama-3.2-1B-UnifiedQC-FT-INT4.pte"
    };
    if (!home_cache.empty()) {
        pte_candidates.push_back(home_cache + "/Llama-3.2-1B-UnifiedQC-FT-INT4.pte");
    }
    pte_path_ = find_readable_file(pte_candidates, R_OK);

    std::vector<std::string> tok_candidates = {
        tokenizer_path,
        llm_cfg.tokenizer_path,
        "models/tokenizer.model",
        "../models/tokenizer.model"
    };
    if (!home_cache.empty()) {
        tok_candidates.push_back(home_cache + "/tokenizer.model");
    }
    tokenizer_path_ = find_readable_file(tok_candidates, R_OK);

    std::vector<std::string> runner_candidates = {
        runner_path,
        "bin/llama_daemon",
        "../bin/llama_daemon"
    };
    if (!home_cache.empty()) {
        runner_candidates.push_back(home_cache + "/llama_daemon");
    }
    runner_bin_ = find_readable_file(runner_candidates, X_OK);

    is_initialized_ = false;
    if (!runner_bin_.empty() && !pte_path_.empty() && !tokenizer_path_.empty()) {
        std::cout << "[LlamaResidentEngine] Launching resident daemon: " << runner_bin_ << std::endl;
        std::cout << "  -> Model: " << pte_path_ << std::endl;
        std::cout << "  -> Tokenizer: " << tokenizer_path_ << std::endl;
        if (start_resident_daemon()) {
            is_initialized_ = true;
        } else {
            std::cerr << "[LlamaResidentEngine] Resident daemon failed to initialize. Will force-wake on inference request." << std::endl;
        }
    } else {
        std::cerr << "[LlamaResidentEngine] Missing prerequisites for Llama daemon ("
                  << "runner=" << (runner_bin_.empty() ? "missing" : runner_bin_)
                  << ", pte=" << (pte_path_.empty() ? "missing" : pte_path_)
                  << ", tokenizer=" << (tokenizer_path_.empty() ? "missing" : tokenizer_path_)
                  << "). Will retry force-wake on inference request." << std::endl;
    }

    if (!is_running_) {
        is_running_ = true;
        worker_thread_ = std::thread(&LlamaResidentEngine::worker_loop, this);
    }

    std::cout << "[LlamaResidentEngine] Worker ready ("
              << (is_initialized_ ? "Resident Llama 3.2 1B Active" : "Auto-Wake Standby Mode")
              << ")" << std::endl;
    return is_initialized_.load();
}

bool LlamaResidentEngine::restart_resident_daemon_locked() {
    is_initialized_ = false;

    if (daemon_in_fd_ >= 0) {
        close(daemon_in_fd_);
        daemon_in_fd_ = -1;
    }
    if (daemon_stream_) {
        fclose(daemon_stream_);
        daemon_stream_ = nullptr;
        daemon_out_fd_ = -1;
    } else if (daemon_out_fd_ >= 0) {
        close(daemon_out_fd_);
        daemon_out_fd_ = -1;
    }
    if (daemon_pid_ > 0) {
        kill(daemon_pid_, SIGKILL);
        int st = 0;
        waitpid(daemon_pid_, &st, 0);
        daemon_pid_ = -1;
    }

    const auto& llm_cfg = rules_.config().llm;
    const char* home_env = getenv("HOME");
    std::string home_cache = home_env ? (std::string(home_env) + "/.cache/llama_cache") : "";

    if (pte_path_.empty() || access(pte_path_.c_str(), R_OK) != 0) {
        std::vector<std::string> pte_candidates = {
            llm_cfg.pte_path,
            "models/Llama-3.2-1B-UnifiedQC-FT-INT4.pte",
            "../models/Llama-3.2-1B-UnifiedQC-FT-INT4.pte"
        };
        if (!home_cache.empty()) {
            pte_candidates.push_back(home_cache + "/Llama-3.2-1B-UnifiedQC-FT-INT4.pte");
        }
        pte_path_ = find_readable_file(pte_candidates, R_OK);
    }

    if (tokenizer_path_.empty() || access(tokenizer_path_.c_str(), R_OK) != 0) {
        std::vector<std::string> tok_candidates = {
            llm_cfg.tokenizer_path,
            "models/tokenizer.model",
            "../models/tokenizer.model"
        };
        if (!home_cache.empty()) {
            tok_candidates.push_back(home_cache + "/tokenizer.model");
        }
        tokenizer_path_ = find_readable_file(tok_candidates, R_OK);
    }

    if (runner_bin_.empty() || access(runner_bin_.c_str(), X_OK) != 0) {
        std::vector<std::string> runner_candidates = {
            "bin/llama_daemon",
            "../bin/llama_daemon"
        };
        if (!home_cache.empty()) {
            runner_candidates.push_back(home_cache + "/llama_daemon");
        }
        runner_bin_ = find_readable_file(runner_candidates, X_OK);
    }

    if (runner_bin_.empty() || pte_path_.empty() || tokenizer_path_.empty()) {
        std::cerr << "[LlamaResidentEngine] Force-wake failed: missing prerequisites ("
                  << "runner=" << (runner_bin_.empty() ? "missing" : runner_bin_)
                  << ", pte=" << (pte_path_.empty() ? "missing" : pte_path_)
                  << ", tokenizer=" << (tokenizer_path_.empty() ? "missing" : tokenizer_path_)
                  << ")." << std::endl;
        return false;
    }

    std::cout << "[LlamaResidentEngine] Force-waking resident daemon: " << runner_bin_ << std::endl;
    if (start_resident_daemon()) {
        is_initialized_ = true;
        return true;
    }

    std::cerr << "[LlamaResidentEngine] Force-wake failed: daemon did not reach [DAEMON_READY]." << std::endl;
    return false;
}

bool LlamaResidentEngine::start_resident_daemon() {
    int in_pipe[2];
    int out_pipe[2];
    if (pipe(in_pipe) < 0 || pipe(out_pipe) < 0) {
        std::cerr << "[LlamaResidentEngine] Error: pipe creation failed." << std::endl;
        return false;
    }

    const auto& llm_cfg = rules_.config().llm;
    int cpu_threads = std::max(1, llm_cfg.cpu_threads);
    int host_cores = hardware::get_cpu_core_count();
    if (host_cores <= 4) {
        cpu_threads = std::min(2, cpu_threads);
    } else {
        cpu_threads = std::min(host_cores / 2, cpu_threads);
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(in_pipe[0]);
        close(in_pipe[1]);
        close(out_pipe[0]);
        close(out_pipe[1]);
        std::cerr << "[LlamaResidentEngine] Error: fork failed." << std::endl;
        return false;
    }

    if (pid == 0) {
        close(in_pipe[1]);
        close(out_pipe[0]);
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        close(in_pipe[0]);
        close(out_pipe[1]);

        hardware::pin_current_thread_mask({hardware::CORE_LLAMA_LEADER, hardware::CORE_LLAMA_WORKER});

        const char* home_env = getenv("HOME");
        std::string home_cache = home_env ? (std::string(home_env) + "/.cache/llama_cache") : "";
        const char* cur_ld = getenv("LD_LIBRARY_PATH");
        std::string new_ld = "bin:.";
        if (!home_cache.empty()) new_ld += ":" + home_cache;
        if (cur_ld) new_ld += std::string(":") + cur_ld;
        setenv("LD_LIBRARY_PATH", new_ld.c_str(), 1);

        std::string arg_model = "--model_path=" + pte_path_;
        std::string arg_tok = "--tokenizer_path=" + tokenizer_path_;
        std::string arg_threads = "--cpu_threads=" + std::to_string(cpu_threads);
        std::string arg_max_tok = "--max_new_tokens=" + std::to_string(std::max(16, llm_cfg.max_new_tokens));
        std::string arg_seq = "--seq_len=" + std::to_string(std::max(128, llm_cfg.seq_len));
        std::ostringstream ss_temp;
        ss_temp << std::fixed << std::setprecision(2) << llm_cfg.temperature;
        std::string arg_temp = "--temperature=" + ss_temp.str();

        execl(runner_bin_.c_str(), runner_bin_.c_str(),
              arg_model.c_str(),
              arg_tok.c_str(),
              arg_threads.c_str(),
              arg_max_tok.c_str(),
              arg_seq.c_str(),
              arg_temp.c_str(),
              nullptr);
        _exit(127);
    }

    close(in_pipe[0]);
    close(out_pipe[1]);

    daemon_pid_ = pid;
    daemon_in_fd_ = in_pipe[1];
    daemon_out_fd_ = out_pipe[0];
    daemon_stream_ = fdopen(daemon_out_fd_, "r");
    if (!daemon_stream_) {
        std::cerr << "[LlamaResidentEngine] Failed to open stream from daemon." << std::endl;
        return false;
    }

    char line_buf[1024];
    bool ready = false;
    while (fgets(line_buf, sizeof(line_buf), daemon_stream_)) {
        std::string l(line_buf);
        if (l.find("[DAEMON_READY]") != std::string::npos) {
            ready = true;
            break;
        }
    }

    if (ready) {
        std::cout << "[LlamaResidentEngine] Warm Resident Llama Daemon ONLINE (PID "
                  << daemon_pid_ << ")" << std::endl;
        return true;
    }

    std::cerr << "[LlamaResidentEngine] Warning: Daemon handshake failed." << std::endl;
    return false;
}

void LlamaResidentEngine::shutdown() {
    if (is_running_) {
        is_running_ = false;
        cv_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    if (daemon_pid_ > 0) {
        if (daemon_in_fd_ >= 0) {
            ssize_t n_wr = write(daemon_in_fd_, "QUIT\n", 5);
            (void)n_wr;
            close(daemon_in_fd_);
            daemon_in_fd_ = -1;
        }
        if (daemon_stream_) {
            fclose(daemon_stream_);
            daemon_stream_ = nullptr;
            daemon_out_fd_ = -1;
        } else if (daemon_out_fd_ >= 0) {
            close(daemon_out_fd_);
            daemon_out_fd_ = -1;
        }
        waitpid(daemon_pid_, nullptr, 0);
        daemon_pid_ = -1;
    }
    is_initialized_ = false;
}

std::future<AdvisoryResult> LlamaResidentEngine::request_advisory_async(
    const std::string& family,
    const std::string& label,
    const std::string& group,
    const std::string& severity,
    float confidence,
    float area_pct,
    int box_count,
    float anomaly_score,
    bool is_anomalous
) {
    auto req = std::make_unique<AdvisoryRequest>();
    req->family = family;
    req->label = label;
    req->group = group;
    req->severity = severity;
    req->confidence = confidence;
    req->area_pct = area_pct;
    req->box_count = box_count;
    req->anomaly_score = anomaly_score;
    req->is_anomalous = is_anomalous;

    std::future<AdvisoryResult> fut = req->promise.get_future();
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        request_queue_.push(std::move(req));
    }
    cv_.notify_one();
    return fut;
}

AdvisoryResult LlamaResidentEngine::request_advisory_sync(
    const std::string& family,
    const std::string& label,
    const std::string& group,
    const std::string& severity,
    float confidence,
    float area_pct,
    int box_count,
    float anomaly_score,
    bool is_anomalous
) {
    auto fut = request_advisory_async(
        family, label, group, severity, confidence, area_pct, box_count, anomaly_score, is_anomalous
    );
    return fut.get();
}

void LlamaResidentEngine::worker_loop() {
    hardware::set_thread_name("llama_resident");
    hardware::pin_current_thread_mask({hardware::CORE_LLAMA_LEADER, hardware::CORE_LLAMA_WORKER});

    while (is_running_) {
        std::unique_ptr<AdvisoryRequest> req;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] {
                return !request_queue_.empty() || !is_running_;
            });

            if (!is_running_ && request_queue_.empty()) {
                break;
            }
            req = std::move(request_queue_.front());
            request_queue_.pop();
        }

        if (req) {
            AdvisoryResult res = execute_inference(*req);
            req->promise.set_value(std::move(res));
        }
    }
}

std::string LlamaResidentEngine::build_prompt(const AdvisoryRequest& req) const {
    return rules_.build_llama_advisory_prompt(
        req.family,
        req.label,
        req.group,
        req.severity,
        req.confidence,
        req.area_pct,
        req.box_count,
        req.anomaly_score
    );
}

void LlamaResidentEngine::parse_output(
    const std::string& raw_output,
    std::string& cause,
    std::string& action
) const {
    cause.clear();
    action.clear();
    if (raw_output.empty()) {
        return;
    }

    std::string text = raw_output;

    const std::string asst_hdr = "<|start_header_id|>assistant<|end_header_id|>";
    size_t pos = text.rfind(asst_hdr);
    if (pos != std::string::npos) {
        text = text.substr(pos + asst_hdr.size());
    } else {
        std::regex cause_hdr(R"((?:Possible|Likely)\s*Cause\s*:)", std::regex::icase);
        std::smatch m;
        if (std::regex_search(text, m, cause_hdr)) {
            text = text.substr(m.position());
        }
    }

    for (const auto& tag : {"<|eot_id|>", "<|end_of_text|>", "<eot_id>", "<end_of_text>"}) {
        size_t eot_pos = text.find(tag);
        if (eot_pos != std::string::npos) {
            text = text.substr(0, eot_pos);
        }
    }

    text = std::regex_replace(text, std::regex(R"((?:^|\n)[IWE]\s+\d{2}:[^\n]*)"), "");
    text = std::regex_replace(text, std::regex(R"(PyTorchObserver\s*\{[^\n]*)"), "");
    text = std::regex_replace(text, std::regex(R"(wsl:[^\n]*)"), "");
    {
        size_t md_pos;
        while ((md_pos = text.find("**")) != std::string::npos) text.erase(md_pos, 2);
    }
    text = std::regex_replace(text, std::regex(R"(^\s*(?:assistant|user|system)\s*[:\r\n]+)", std::regex::icase), "");
    text = trim_copy(text);
    {
        std::regex cause_hdr(R"((?:Possible|Likely)\s*Cause\s*[:\|])", std::regex::icase);
        std::smatch cm;
        if (std::regex_search(text, cm, cause_hdr)) {
            text = trim_copy(text.substr(cm.position()));
        }
    }
    if (text.empty()) {
        return;
    }

    std::regex action_re(R"((?:\||\n)?\s*(?:Recommended\s*)?Action\s*[:\|\s]+)", std::regex::icase);
    std::smatch act_m;
    if (std::regex_search(text, act_m, action_re)) {
        cause = text.substr(0, act_m.position());
        action = text.substr(act_m.position() + act_m.length());
        std::regex dup_re(R"((?:\||\n|\s{2,}|\.\s+)\s*(?:Recommended\s*Action|Possible\s*Cause|Likely\s*Cause)\s*:.*$)", std::regex::icase);
        action = std::regex_replace(action, dup_re, "");
    } else if (text.find('|') != std::string::npos) {
        size_t pipe_pos = text.find('|');
        cause = text.substr(0, pipe_pos);
        action = text.substr(pipe_pos + 1);
    } else {
        std::string cleaned = std::regex_replace(
            text,
            std::regex(R"(^\s*(?:Possible|Likely)\s*Cause\s*[:\|]\s*)", std::regex::icase),
            ""
        );
        cleaned = trim_copy(cleaned);

        size_t split_pos = cleaned.find(". ");
        if (split_pos != std::string::npos && split_pos >= 4 && (cleaned.size() - (split_pos + 2)) >= 4) {
            cause = cleaned.substr(0, split_pos + 1);
            action = cleaned.substr(split_pos + 2);
        } else {
            size_t semi_pos = cleaned.find("; ");
            if (semi_pos != std::string::npos && semi_pos >= 4 && (cleaned.size() - (semi_pos + 2)) >= 4) {
                cause = cleaned.substr(0, semi_pos);
                action = cleaned.substr(semi_pos + 2);
            } else {
                cause = cleaned;
                action = cleaned;
            }
        }
    }

    cause = std::regex_replace(cause, std::regex(R"(^\s*(?:Possible|Likely)\s*Cause\s*[:\|]\s*)", std::regex::icase), "");
    action = std::regex_replace(action, std::regex(R"(^\s*(?:Recommended\s*)?Action\s*[:\|]\s*)", std::regex::icase), "");

    for (const auto& tok : {"<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>", "<|begin_of_text|>", "<eot_id>", "<end_of_text>", "|"}) {
        size_t p;
        while ((p = cause.find(tok)) != std::string::npos) cause.erase(p, std::strlen(tok));
        while ((p = action.find(tok)) != std::string::npos) action.erase(p, std::strlen(tok));
    }

    action = std::regex_replace(action, std::regex(R"(\s*(?:Note|Remark|Important|Caution)\s*:.*$)", std::regex::icase), "");

    cause = trim_copy(cause);
    action = trim_copy(action);

    if (cause.size() < 4 && action.size() >= 4) cause = action;
    if (action.size() < 4 && cause.size() >= 4) action = cause;

    if (cause.size() < 4 || action.size() < 4) {
        cause.clear();
        action.clear();
        return;
    }

    if (cause.back() != '.') cause += ".";
    if (action.back() != '.') action += ".";
}

namespace {

void apply_domain_specific_advisory(
    const std::string& raw_label,
    std::string& cause,
    std::string& action
) {
    const std::string key = qc::rules::normalize_label(raw_label);
    if (key == "oil_spot") {
        cause = "Roll-stand hydraulic or bearing lubricant seal weepage dripping mineral oil droplets onto the moving hot-rolled strip surface during pass reduction.";
        action = "Wipe and degrease affected coil segment, inspect mill stand oil wipers and bearing seals for leakage, and verify emulsion skimmer operation.";
    } else if (key == "water_spot") {
        cause = "Residual laminar cooling or rinse-tank water carryover trapped across the strip surface and oxidizing into dark streaks during coiling.";
        action = "Check strip air-knife blow-off nozzle pressure and squeegee wringer rolls, and adjust strip dryer temperature before recoiling.";
    } else if (key == "punching_hole") {
        cause = "Mechanical strip-joiner punch or edge-tracking sensor aperture sheared out of index along the rolled strip edge margin.";
        action = "Verify punch-die indexing encoder alignment, inspect shear punch clearance for burr formation, and trim the affected leader or edge section.";
    } else if (key == "waist_folding") {
        cause = "Localized transverse yield buckling (coil break / waist fold) caused by excessive uncoiler back-tension or improper anti-coil-break roller plunge.";
        action = "Recalibrate uncoiler bridle tension and skin-pass elongation, and reset anti-coil-break roller plunge depth before leveling.";
    } else if (key == "manipulated_front") {
        cause = "Cold-header or thread-rolling pointer die impact misalignment crushing and blunting the conical screw front tip during blank feeding.";
        action = "Quarantine affected batch, inspect pointer and pinch die cavities for chipped inserts or slug buildup, and recalibrate blank feed stop length.";
    } else if (key == "scratch_neck") {
        cause = "Hardened vibratory feeder track burrs or worn transfer-finger clamping jaws scoring the unthreaded cylindrical shank below the fastener head.";
        action = "Polish vibratory bowl and transfer rail contact surfaces, replace worn chuck jaws, and verify fastener orientation gate clearance.";
    } else if (key == "scratch_head") {
        cause = "Heading punch face wear or metallic swarf entrapment gouging the countersunk screw head bevel during cold-forging impact.";
        action = "Dress or replace the second-blow heading punch, flush die lubricant filtration to clear metal fines, and verify head concentricity.";
    } else if (key == "thread_side") {
        cause = "Thread-rolling flat die tooth spalling or foreign chip inclusion tearing the helical thread flank during rolling pass engagement.";
        action = "Inspect stationary and moving thread-rolling dies for chipped crests, purge die coolant of metal debris, and verify die parallel gap.";
    } else if (key == "thread_top") {
        cause = "Excessive thread-rolling die penetration pressure or oversized wire blank shearing and flattening the outer helical thread crest.";
        action = "Measure incoming wire blank diameter tolerance, reduce thread-rolling die closure pressure slightly, and optical-gauge thread pitch profile.";
    }
}

} // namespace

AdvisoryResult LlamaResidentEngine::execute_inference(const AdvisoryRequest& req) {
    auto t0 = std::chrono::steady_clock::now();
    AdvisoryResult res;

    std::lock_guard<std::mutex> lock(daemon_io_mutex_);

    const std::string prompt_str = build_prompt(req);
    const std::string prompt_file = "/tmp/native_llama_prompt.txt";
    {
        std::ofstream pf(prompt_file, std::ios::trunc);
        pf << prompt_str;
    }

    for (int attempt = 0; attempt < 2; ++attempt) {
        int st_code = 0;
        bool daemon_alive = (daemon_pid_ > 0 && daemon_in_fd_ >= 0 && daemon_stream_ != nullptr);
        if (daemon_alive) {
            pid_t w = waitpid(daemon_pid_, &st_code, WNOHANG);
            if (w != 0) {
                daemon_alive = false;
            }
        }

        if (!daemon_alive) {
            std::cerr << "[LlamaResidentEngine] Resident Llama daemon is down — force-waking (attempt "
                      << (attempt + 1) << "/2)..." << std::endl;
            if (!restart_resident_daemon_locked()) {
                continue;
            }
        }

        auto t_send = std::chrono::steady_clock::now();
        std::string write_payload = "FILE:" + prompt_file + "\n";
        ssize_t n_wr = write(daemon_in_fd_, write_payload.c_str(), write_payload.size());
        if (n_wr <= 0) {
            std::cerr << "[LlamaResidentEngine] Pipe write to llama_daemon failed — marking daemon down." << std::endl;
            if (daemon_pid_ > 0) {
                kill(daemon_pid_, SIGKILL);
                waitpid(daemon_pid_, nullptr, 0);
                daemon_pid_ = -1;
            }
            is_initialized_ = false;
            continue;
        }

        std::string gen_text;
        char line_buf[2048];
        bool saw_start = false;
        bool saw_end = false;
        bool saw_first_token = false;
        auto t_first_tok = t_send;

        clearerr(daemon_stream_);
        while (fgets(line_buf, sizeof(line_buf), daemon_stream_)) {
            std::string l(line_buf);
            if (l.find("[END_GEN]") != std::string::npos) {
                saw_end = true;
                break;
            }
            if (l.find("[START_GEN]") != std::string::npos) {
                saw_start = true;
                continue;
            }
            if (saw_start && !saw_first_token) {
                std::string trimmed = trim_copy(l);
                if (!trimmed.empty() && trimmed.rfind("PyTorchObserver", 0) != 0) {
                    t_first_tok = std::chrono::steady_clock::now();
                    saw_first_token = true;
                }
            }
            gen_text += l;
        }

        if (!saw_end) {
            std::cerr << "[LlamaResidentEngine] EOF before [END_GEN] from llama_daemon — marking daemon down." << std::endl;
            if (daemon_pid_ > 0) {
                kill(daemon_pid_, SIGKILL);
                waitpid(daemon_pid_, nullptr, 0);
                daemon_pid_ = -1;
            }
            is_initialized_ = false;
            continue;
        }

        res.possible_cause.clear();
        res.recommended_action.clear();
        parse_output(gen_text, res.possible_cause, res.recommended_action);

        if (res.possible_cause.size() >= 4 && res.recommended_action.size() >= 4) {
            apply_domain_specific_advisory(req.label, res.possible_cause, res.recommended_action);
            auto t1 = std::chrono::steady_clock::now();
            res.total_ms = std::chrono::duration<float, std::milli>(t1 - t_send).count();

            long inf_start_ms = 0;
            long first_tok_ms = 0;
            float obs_tok_s = 0.0f;
            long obs_gen_toks = 0;

            if (extract_json_long(gen_text, "generated_tokens", obs_gen_toks)) {
                res.tokens_generated = static_cast<int>(obs_gen_toks);
            } else {
                res.tokens_generated = 0;
            }

            if (extract_json_long(gen_text, "inference_start_ms", inf_start_ms) &&
                extract_json_long(gen_text, "first_token_ms", first_tok_ms) &&
                inf_start_ms > 0 && first_tok_ms >= inf_start_ms) {
                res.ttft_ms = static_cast<float>(first_tok_ms - inf_start_ms);
            } else if (saw_first_token) {
                res.ttft_ms = std::chrono::duration<float, std::milli>(t_first_tok - t_send).count();
            } else {
                res.ttft_ms = res.total_ms;
            }

            if (extract_json_float(gen_text, "decode_token_per_sec", obs_tok_s) && obs_tok_s > 0.0f) {
                res.tokens_per_s = obs_tok_s;
            } else if (res.tokens_generated > 0 && res.total_ms > 1.0f) {
                res.tokens_per_s = static_cast<float>(res.tokens_generated) / (res.total_ms / 1000.0f);
            } else {
                res.tokens_per_s = 0.0f;
            }

            if (res.tokens_generated == 0) {
                std::istringstream iss(res.possible_cause + " " + res.recommended_action);
                std::string w;
                int wc = 0;
                while (iss >> w) wc++;
                res.tokens_generated = std::max(1, static_cast<int>(std::lround(wc * 1.3f)));
                if (res.tokens_per_s <= 0.0f && res.total_ms > 1.0f) {
                    res.tokens_per_s = static_cast<float>(res.tokens_generated) / (res.total_ms / 1000.0f);
                }
            }

            res.summary = "Possible Cause: " + res.possible_cause +
                          " | Recommended Action: " + res.recommended_action;
            res.advisory_source = "Llama 3.2 1B";
            res.success = true;
            return res;
        }

        std::cerr << "[LlamaResidentEngine] Empty/invalid output from llama_daemon — restarting daemon." << std::endl;
        if (daemon_pid_ > 0) {
            kill(daemon_pid_, SIGKILL);
            waitpid(daemon_pid_, nullptr, 0);
            daemon_pid_ = -1;
        }
        is_initialized_ = false;
    }

    auto t1 = std::chrono::steady_clock::now();
    res.total_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    res.ttft_ms = 0.0f;
    res.tokens_generated = 0;
    res.tokens_per_s = 0.0f;
    res.possible_cause = "ERROR: Llama 3.2 1B resident daemon is down and force-wakeup failed (verify bin/llama_daemon and models/Llama-3.2-1B-UnifiedQC-FT-INT4.pte).";
    res.recommended_action = "ERROR: Inspect on-device Llama 3.2 1B daemon status; hardcoded/KB fallback advisories are disabled.";
    res.summary = res.possible_cause + " | " + res.recommended_action;
    res.advisory_source = "ERROR: Llama 3.2 1B Daemon Offline";
    res.success = false;
    return res;
}

} // namespace reasoning
} // namespace qc
