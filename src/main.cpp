#include "cpu_affinity.hpp"
#include "demo_timeline.hpp"
#include "hailo_vision_engine.hpp"
#include "ipc_server.hpp"
#include "llama_resident.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

std::atomic<bool> g_shutdown{false};

void handle_signal(int /*sig*/) {
    g_shutdown.store(true);
}

std::string get_utc_timestamp() {
    std::time_t now = std::time(nullptr);
    std::tm tm_buf{};
    localtime_r(&now, &tm_buf);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_buf);
    return std::string(buf);
}

std::string resolve_config_path() {
    for (const auto& candidate : {
             std::string("config/edge_qc_config.json"),
             std::string("../config/edge_qc_config.json"),
             std::string("./config/edge_qc_config.json")}) {
        std::error_code ec;
        if (fs::exists(candidate, ec) && fs::is_regular_file(candidate, ec)) {
            return candidate;
        }
    }
    return "config/edge_qc_config.json";
}

std::string to_upper_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

std::string format_system_status(const qc::vision::HailoVisionEngine& engine, bool llm_ready) {
    std::string backend = to_upper_ascii(engine.get_vision_backend());
    std::string llm_tag = llm_ready ? "LLAMA 3.2 1B RESIDENT" : "LLAMA 3.2 1B OFFLINE";
    return backend + " · " + llm_tag;
}

std::vector<uint8_t> render_offline_banner(const std::string& headline, const std::string& subtitle) {
    cv::Mat frame(720, 1280, CV_8UC3, cv::Scalar(16, 20, 28));
    cv::rectangle(frame, cv::Rect(32, 32, 1216, 656), cv::Scalar(48, 62, 82), 2);
    cv::line(frame, cv::Point(80, 360), cv::Point(1200, 360), cv::Scalar(42, 54, 72), 1);

    int baseline = 0;
    cv::Size h_sz = cv::getTextSize(headline, cv::FONT_HERSHEY_SIMPLEX, 1.35, 3, &baseline);
    cv::Point h_org((frame.cols - h_sz.width) / 2, 330);
    cv::putText(frame, headline, h_org, cv::FONT_HERSHEY_SIMPLEX, 1.35, cv::Scalar(80, 185, 255), 3, cv::LINE_AA);

    cv::Size s_sz = cv::getTextSize(subtitle, cv::FONT_HERSHEY_SIMPLEX, 0.72, 2, &baseline);
    cv::Point s_org((frame.cols - s_sz.width) / 2, 415);
    cv::putText(frame, subtitle, s_org, cv::FONT_HERSHEY_SIMPLEX, 0.72, cv::Scalar(170, 185, 205), 2, cv::LINE_AA);

    std::vector<uint8_t> jpg;
    cv::imencode(".jpg", frame, jpg, {cv::IMWRITE_JPEG_QUALITY, 85});
    return jpg;
}

void populate_telemetry_from_vision(
    const qc::vision::VisionInferenceResult& v_res,
    const qc::vision::HailoVisionEngine& engine,
    qc::ipc::LiveTelemetryState& st,
    const std::string& active_mode,
    bool llm_ready
) {
    st.frame_id = v_res.frame_id;
    st.mode = active_mode;
    st.family = v_res.family;
    st.family_conf = v_res.family_conf;
    st.is_not_metal = v_res.is_not_metal;
    st.vision_backend = v_res.vision_backend;
    st.hardware_device = v_res.hardware_device;

    st.defects = v_res.detections;
    st.primary_label = v_res.primary_label;
    st.primary_group = v_res.primary_group;
    st.defect = v_res.display_title;
    st.display_title = v_res.display_title;
    st.conf = v_res.highest_confidence;
    st.area = v_res.union_area_pct;
    st.boxes = v_res.total_boxes;
    st.spread = v_res.spread;
    st.severity = v_res.severity;
    st.verdict = v_res.verdict;

    st.anomaly = v_res.anomaly;
    st.anomaly_max_refs = engine.rules().config().patchcore.reference_limit_per_family;
    st.timings_ms = v_res.timings_ms;
    st.cv_ms = v_res.cv_latency_ms;
    st.system_status = format_system_status(engine, llm_ready);
}

qc::ipc::InspectionRecord build_inspection_record(
    const qc::vision::VisionInferenceResult& v_res,
    const qc::ipc::LiveTelemetryState& st
) {
    qc::ipc::InspectionRecord rec;
    rec.timestamp = get_utc_timestamp();
    rec.family = v_res.family;
    rec.defect = v_res.primary_label;
    rec.group = v_res.primary_group;

    char conf_buf[32];
    if (v_res.total_boxes > 0 || !v_res.anomaly.is_anomalous) {
        std::snprintf(conf_buf, sizeof(conf_buf), "%.1f%%", v_res.highest_confidence * 100.0f);
    } else {
        std::snprintf(conf_buf, sizeof(conf_buf), "ANOM %.2f", v_res.anomaly.score);
    }
    rec.score_conf = conf_buf;

    rec.anomaly_score = v_res.anomaly.score;
    rec.area_pct = v_res.union_area_pct;
    rec.regions_boxes = v_res.total_boxes;
    rec.spread = v_res.spread;
    rec.severity = v_res.severity;
    rec.verdict = v_res.verdict;
    rec.cause = st.cause;
    rec.action = st.action;
    rec.source = st.engine_source;
    rec.pre_ms = v_res.timings_ms.pre;
    rec.infer_ms = v_res.timings_ms.infer;
    rec.post_ms = v_res.timings_ms.post;
    rec.total_ms = v_res.timings_ms.total;
    rec.latency_ms = st.llm_ms;
    return rec;
}

bool open_live_camera(cv::VideoCapture& cap) {
    cap.release();
    if (cap.open(0, cv::CAP_V4L2)) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(cv::CAP_PROP_FPS, 30);
        return true;
    }
    if (cap.open(0, cv::CAP_ANY)) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        return true;
    }
    return false;
}

} // namespace

int main() {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
#ifdef SIGPIPE
    std::signal(SIGPIPE, SIG_IGN);
#endif

    const int lock_fd = ::open("/tmp/bharat_qc_native.lock", O_CREAT | O_RDWR, 0644);
    if (lock_fd >= 0 && ::flock(lock_fd, LOCK_EX | LOCK_NB) != 0) {
        std::cerr << "[Main] Another bharat_qc_native.bin instance is already running; refusing duplicate port-5000 bind." << std::endl;
        ::close(lock_fd);
        return 1;
    }

    qc::hardware::set_thread_name("qc-main-hud");
    qc::hardware::pin_current_thread(qc::hardware::CORE_IO_IPC);

    const std::string config_path = resolve_config_path();
    std::cout << "==============================================================\n"
              << "  BHARAT AI-SoC · UNIFIED MULTI-METAL SCADA VISION NODE\n"
              << "  Config: " << config_path
              << " · Host Cores: " << qc::hardware::get_cpu_core_count() << "\n"
              << "==============================================================" << std::endl;

    // Vision initialization selects affinity before constructing ORT workers.
    // Restore the main/HTTP core immediately afterwards.
    cv::setNumThreads(1);
    qc::vision::HailoVisionEngine vision_engine;
    const bool vision_ok = vision_engine.init(config_path);
    qc::hardware::pin_current_thread(qc::hardware::CORE_IO_IPC);
    if (!vision_ok) {
        std::cerr << "[Main] Warning: Vision engine did not load HEF or YOLO ONNX ("
                  << vision_engine.get_vision_backend() << "); check models/ directory." << std::endl;
    }

    qc::reasoning::LlamaResidentEngine llama_engine;
    const bool llm_ready = llama_engine.init(config_path);
    if (!llm_ready) {
        std::cerr << "[Main] Warning: LlamaResidentEngine daemon not ready at boot; will force-wake on inference." << std::endl;
    }

    qc::ipc::IPCServer ipc_server(5000);
    ipc_server.set_vision_engine(&vision_engine);
    if (!ipc_server.start()) {
        std::cerr << "[Main] CRITICAL: Failed to bind HTTP SCADA server on port 5000." << std::endl;
        llama_engine.shutdown();
        return 1;
    }

    const int required_lock_frames = std::max(1, vision_engine.rules().config().vision.lock_in_frames);
    std::atomic<bool> vision_running{true};

    std::thread vision_thread([&]() {
        qc::hardware::set_thread_name("qc-vision");
        qc::hardware::pin_current_thread(qc::hardware::CORE_VISION);

        cv::VideoCapture vid_cap;
        cv::VideoCapture cam_cap;
        std::string active_video_path;
        std::string prev_mode;
        bool cam_open = false;

        std::string last_image_path;
        uint64_t last_image_gen_id = 0;
        int last_image_idx = -1;
        bool vision_done_for_sample = false;
        bool current_image_analyzed = false;
        qc::vision::VisionInferenceResult cached_static_vres;

        int lock_cnt = 0;
        std::string lock_candidate_sig;
        std::string llm_dispatched_sample;

        std::future<qc::reasoning::AdvisoryResult> active_llm_future;
        bool llm_busy = false;
        std::chrono::steady_clock::time_point llm_start_time;
        uint64_t llm_target_gen_id = 0;
        bool llm_target_is_static = false;
        qc::vision::VisionInferenceResult llm_target_vres;

        auto demo_last_switch = std::chrono::steady_clock::now();
        bool demo_dwell_active = false;
        bool demo_wait_for_belt_clear = false;
        auto demo_dwell_until = std::chrono::steady_clock::now();
        int demo_cooldown_frames = 0;

        while (vision_running.load() && !g_shutdown.load()) {
            if (llm_busy && active_llm_future.valid()) {
                if (active_llm_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    const auto adv = active_llm_future.get();
                    llm_busy = false;

                    if (ipc_server.get_sample_generation_id() == llm_target_gen_id) {
                        qc::ipc::LiveTelemetryState cur = ipc_server.get_state();
                        cur.llm_busy = false;
                        cur.advisory_pending = false;
                        cur.llm_elapsed = adv.total_ms / 1000.0f;
                        cur.cause = adv.possible_cause;
                        cur.action = adv.recommended_action;
                        cur.narrative = adv.summary;
                        cur.engine_source = adv.advisory_source;
                        cur.ttft_ms = adv.ttft_ms;
                        cur.tokens_per_s = adv.tokens_per_s;
                        cur.llm_ms = adv.total_ms;
                        cur.tokens = adv.tokens_generated;

                        ipc_server.update_state_for_generation(llm_target_gen_id, cur);
                        ipc_server.record_bench_llm(adv.ttft_ms, adv.tokens_per_s, adv.total_ms);
                        ipc_server.append_log(build_inspection_record(llm_target_vres, cur));

                        if (llm_target_is_static) {
                            current_image_analyzed = true;
                            demo_last_switch = std::chrono::steady_clock::now();
                        } else if (cur.mode == "DEMO") {
                            demo_dwell_active = true;
                            demo_dwell_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(4000);
                        }
                    }
                } else if (ipc_server.get_sample_generation_id() == llm_target_gen_id) {
                    qc::ipc::LiveTelemetryState cur = ipc_server.get_state();
                    cur.llm_busy = true;
                    cur.advisory_pending = true;
                    cur.llm_elapsed = std::chrono::duration<float>(
                        std::chrono::steady_clock::now() - llm_start_time).count();
                    ipc_server.update_state_for_generation(llm_target_gen_id, cur);
                }
            }

            const std::string mode = ipc_server.get_active_mode();
            if (mode == "DEMO" && demo_dwell_active && !llm_busy) {
                if (ipc_server.is_manual_paused()) {
                    demo_dwell_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(400);
                } else if (std::chrono::steady_clock::now() >= demo_dwell_until) {
                    demo_dwell_active = false;
                    ipc_server.set_auto_paused(false);
                    lock_cnt = 0;
                    lock_candidate_sig.clear();
                    demo_wait_for_belt_clear = true;
                    demo_cooldown_frames = 12;
                }
            }

            const bool paused = ipc_server.is_paused();
            const uint64_t cur_gen_id = ipc_server.get_sample_generation_id();
            const int cur_idx = ipc_server.get_active_index();

            if (mode != prev_mode) {
                if ((prev_mode == "VIDEO" || prev_mode == "DEMO") && vid_cap.isOpened()) {
                    vid_cap.release();
                    active_video_path.clear();
                }
                if (prev_mode == "CAMERA" && cam_open) {
                    cam_cap.release();
                    cam_open = false;
                }
                lock_cnt = 0;
                lock_candidate_sig.clear();
                llm_dispatched_sample.clear();
                vision_done_for_sample = false;
                current_image_analyzed = false;
                demo_dwell_active = false;
                demo_wait_for_belt_clear = false;
                demo_cooldown_frames = 0;
                ipc_server.set_auto_paused(false);
                demo_last_switch = std::chrono::steady_clock::now();
                prev_mode = mode;
            }

            if (mode == "LOG") {
                ipc_server.wait_for_vision_wakeup(100);
                continue;
            }

            demo_last_switch = std::chrono::steady_clock::now();
            const auto loop_frame_start = std::chrono::steady_clock::now();
            const bool is_static_mode = (mode == "IMAGE");
            qc::vision::VisionInferenceResult v_res;
            std::vector<uint8_t> pending_frame_jpeg;
            bool centered_demo_frame = false;

            if (mode == "CAMERA") {
                if (paused) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(80));
                    continue;
                }
                cv::Mat frame;
                if (!cam_open) {
                    cam_open = open_live_camera(cam_cap);
                }
                if (!cam_open || !cam_cap.read(frame) || frame.empty()) {
                    if (cam_open) {
                        cam_cap.release();
                        cam_open = false;
                    }
                    ipc_server.update_frame_jpeg(render_offline_banner(
                        "NO LIVE CAMERA DETECTED",
                        "Connect USB or CSI camera (/dev/video0) or switch to DEMO / IMAGE / VIDEO"));
                    auto st = ipc_server.get_state();
                    st.mode = "CAMERA";
                    st.defect = "CAMERA OFFLINE";
                    st.display_title = "CAMERA OFFLINE";
                    st.severity = "NONE";
                    st.verdict = "PASS";
                    st.llm_busy = false;
                    st.advisory_pending = false;
                    st.system_status = format_system_status(vision_engine, llm_ready);
                    ipc_server.update_state_for_generation(cur_gen_id, st);
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    continue;
                }
                if (ipc_server.get_sample_generation_id() != cur_gen_id) {
                    continue;
                }
                v_res = vision_engine.infer(frame);
                if (ipc_server.get_sample_generation_id() != cur_gen_id) {
                    continue;
                }
                if (!v_res.encoded_jpeg.empty()) {
                    pending_frame_jpeg = v_res.encoded_jpeg;
                }
                ipc_server.record_bench_sample(v_res.timings_ms, v_res.vision_backend);
            } else if (mode == "VIDEO" || mode == "DEMO") {
                const bool is_demo = (mode == "DEMO");
                const std::string vpath = is_demo ? ipc_server.get_demo_video_path() : ipc_server.get_current_video_path();
                if (vpath.empty()) {
                    ipc_server.update_frame_jpeg(render_offline_banner(
                        is_demo ? "NO CONVEYOR DEMO VIDEO" : "NO VIDEO SAMPLE AVAILABLE",
                        is_demo ? "Generate sample_videos/conveyor_demo_belt.mp4 to run conveyor demo"
                                : "Upload an MP4/AVI/MOV stream in the VIDEO tab to inspect"));
                    auto st = ipc_server.get_state();
                    st.mode = mode;
                    st.defect = "AWAITING VIDEO";
                    st.display_title = "AWAITING VIDEO";
                    st.llm_busy = false;
                    st.advisory_pending = false;
                    ipc_server.update_state_for_generation(cur_gen_id, st);
                    ipc_server.wait_for_vision_wakeup(250);
                    continue;
                }
                if (vpath != active_video_path || !vid_cap.isOpened()) {
                    vid_cap.open(vpath);
                    active_video_path = vpath;
                    lock_cnt = 0;
                    lock_candidate_sig.clear();
                    llm_dispatched_sample.clear();
                    demo_cooldown_frames = 0;
                    demo_dwell_active = false;
                }
                if (!vid_cap.isOpened()) {
                    ipc_server.update_frame_jpeg(render_offline_banner(
                        "VIDEO DECODE ERROR",
                        "Unable to open video stream: " + vpath));
                    auto st = ipc_server.get_state();
                    st.llm_busy = false;
                    st.advisory_pending = false;
                    ipc_server.update_state_for_generation(cur_gen_id, st);
                    ipc_server.wait_for_vision_wakeup(300);
                    continue;
                }
                if (is_demo) {
                    const int step_delta = ipc_server.consume_demo_step_delta();
                    if (step_delta != 0) {
                        const int total_f = static_cast<int>(vid_cap.get(cv::CAP_PROP_FRAME_COUNT));
                        const int cur_f = static_cast<int>(vid_cap.get(cv::CAP_PROP_POS_FRAMES));
                        if (total_f > 1) {
                            int target_f = (cur_f + step_delta * 35) % total_f;
                            if (target_f < 0) target_f += total_f;
                            vid_cap.set(cv::CAP_PROP_POS_FRAMES, target_f);
                        }
                        lock_cnt = 0;
                        lock_candidate_sig.clear();
                        llm_dispatched_sample.clear();
                        demo_cooldown_frames = 0;
                        demo_dwell_active = false;
                        demo_wait_for_belt_clear = false;
                        ipc_server.set_auto_paused(false);
                    }
                }
                if (paused) {
                    ipc_server.wait_for_vision_wakeup(60);
                    continue;
                }
                cv::Mat frame;
                if (!vid_cap.read(frame) || frame.empty()) {
                    vid_cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                    lock_cnt = 0;
                    lock_candidate_sig.clear();
                    llm_dispatched_sample.clear();
                    demo_wait_for_belt_clear = false;
                    demo_cooldown_frames = 0;
                    if (!vid_cap.read(frame) || frame.empty()) {
                        ipc_server.wait_for_vision_wakeup(100);
                        continue;
                    }
                }
                if (ipc_server.get_sample_generation_id() != cur_gen_id) {
                    continue;
                }
                // The bundled generated clip has explicit transit and centered-part
                // windows. Never run PatchCore on a partial part plus conveyor.
                const bool scheduled_demo = is_demo &&
                    static_cast<int>(vid_cap.get(cv::CAP_PROP_FRAME_COUNT)) ==
                        qc::demo::station_frames * static_cast<int>(qc::demo::part_rois.size()) &&
                    std::abs(vid_cap.get(cv::CAP_PROP_FPS) - qc::demo::source_fps) < 0.1;
                if (scheduled_demo) {
                    const int frame_index = std::max(0, static_cast<int>(vid_cap.get(cv::CAP_PROP_POS_FRAMES)) - 1);
                    const int phase = frame_index % qc::demo::station_frames;
                    const bool centered = phase >= qc::demo::inspect_begin && phase < qc::demo::inspect_end;
                    if (phase == qc::demo::inspect_begin) {
                        lock_cnt = 0;
                        lock_candidate_sig.clear();
                        llm_dispatched_sample.clear();
                        demo_wait_for_belt_clear = false;
                        demo_cooldown_frames = 0;
                    }
                    if (centered) {
                        centered_demo_frame = true;
                        const auto& r = qc::demo::part_rois.at(frame_index / qc::demo::station_frames);
                        const cv::Rect roi = cv::Rect(r[0], r[1], r[2], r[3]) & cv::Rect(0, 0, frame.cols, frame.rows);
                        v_res = vision_engine.infer(frame(roi), -1.0f, -1.0f, 2);
                    } else {
                        v_res = vision_engine.infer(frame, -1.0f, -1.0f, 1);
                    }
                } else {
                    v_res = vision_engine.infer(frame);
                }
                if (ipc_server.get_sample_generation_id() != cur_gen_id) {
                    continue;
                }
                if (!v_res.encoded_jpeg.empty()) {
                    pending_frame_jpeg = v_res.encoded_jpeg;
                }
                ipc_server.record_bench_sample(v_res.timings_ms, v_res.vision_backend);
            } else {
                const std::string img_path = ipc_server.get_current_image_path();
                if (img_path.empty()) {
                    ipc_server.update_frame_jpeg(render_offline_banner(
                        "NO SAMPLE IMAGE LOADED",
                        "Add images to sample_test_images/ or upload one via IMAGE tab"));
                    auto st = ipc_server.get_state();
                    st.llm_busy = false;
                    st.advisory_pending = false;
                    ipc_server.update_state_for_generation(cur_gen_id, st);
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                    continue;
                }
                if (img_path != last_image_path || cur_gen_id != last_image_gen_id || cur_idx != last_image_idx) {
                    last_image_path = img_path;
                    last_image_gen_id = cur_gen_id;
                    last_image_idx = cur_idx;
                    vision_done_for_sample = false;
                    current_image_analyzed = false;
                    lock_cnt = 0;
                    lock_candidate_sig.clear();
                }
                if (current_image_analyzed) {
                    ipc_server.wait_for_vision_wakeup(60);
                    continue;
                }
                if (!vision_done_for_sample) {
                    cv::Mat frame = cv::imread(img_path, cv::IMREAD_COLOR);
                    if (frame.empty()) {
                        ipc_server.update_frame_jpeg(render_offline_banner(
                            "IMAGE LOAD FAILED",
                            "Could not decode image: " + img_path));
                        auto st = ipc_server.get_state();
                        st.llm_busy = false;
                        st.advisory_pending = false;
                        ipc_server.update_state_for_generation(cur_gen_id, st);
                        current_image_analyzed = true;
                        ipc_server.wait_for_vision_wakeup(200);
                        continue;
                    }
                    if (ipc_server.get_sample_generation_id() != cur_gen_id) {
                        continue;
                    }
                    // Immediately push a fast preview JPEG so UI image switches feel instant (<30ms)
                    // while full Vision + PatchCore inference completes right behind it.
                    {
                        cv::Mat preview = frame;
                        const int max_side = std::max(frame.cols, frame.rows);
                        if (max_side > 960) {
                            const double s = 960.0 / static_cast<double>(max_side);
                            cv::resize(frame, preview, cv::Size(
                                std::max(1, static_cast<int>(std::round(frame.cols * s))),
                                std::max(1, static_cast<int>(std::round(frame.rows * s)))
                            ), 0, 0, cv::INTER_AREA);
                        } else if (max_side < 720) {
                            const double s = 720.0 / static_cast<double>(std::max(1, max_side));
                            cv::resize(frame, preview, cv::Size(
                                std::max(1, static_cast<int>(std::round(frame.cols * s))),
                                std::max(1, static_cast<int>(std::round(frame.rows * s)))
                            ), 0, 0, cv::INTER_LINEAR);
                        }
                        std::vector<uint8_t> preview_jpg;
                        if (cv::imencode(".jpg", preview, preview_jpg, {cv::IMWRITE_JPEG_QUALITY, 80}) &&
                            ipc_server.get_sample_generation_id() == cur_gen_id) {
                            ipc_server.update_frame_jpeg(preview_jpg);
                        }
                    }
                    if (ipc_server.get_sample_generation_id() != cur_gen_id) {
                        continue;
                    }
                    cached_static_vres = vision_engine.infer(frame);
                    if (ipc_server.get_sample_generation_id() != cur_gen_id) {
                        continue;
                    }
                    if (!cached_static_vres.encoded_jpeg.empty()) {
                        ipc_server.update_frame_jpeg(cached_static_vres.encoded_jpeg);
                    }
                    ipc_server.record_bench_sample(cached_static_vres.timings_ms, cached_static_vres.vision_backend);
                    vision_done_for_sample = true;
                }
                v_res = cached_static_vres;
            }

            qc::ipc::LiveTelemetryState cur = ipc_server.get_state();
            populate_telemetry_from_vision(v_res, vision_engine, cur, mode, llm_ready);

            const bool is_conveyor_clear = (v_res.family == "conveyor_belt");
            const bool defect_or_anomaly = (!is_conveyor_clear && !v_res.is_not_metal) &&
                                           (v_res.has_defects || v_res.anomaly.is_anomalous);
            const std::string advisory_sig =
                v_res.family + "|" + v_res.primary_label + "|" + v_res.severity + "|" +
                (v_res.anomaly.is_anomalous ? "anom" : "norm") + "|" +
                (v_res.is_not_metal ? "nm" : "m");

            if (is_conveyor_clear) {
                // Empty/transit conveyor gap must roll freely without locking, pausing, or invoking Llama.
                demo_wait_for_belt_clear = false;
                demo_cooldown_frames = 0;
                lock_cnt = 0;
                lock_candidate_sig.clear();
                if (!llm_busy) {
                    cur.advisory_pending = false;
                    cur.llm_busy = false;
                    cur.engine_source = "Belt Gate";
                    cur.cause = "Empty conveyor belt gap between parts; no metallic specimen in inspection zone.";
                    cur.action = "Conveyor advancing normally; awaiting next metal part.";
                }
            } else if (is_static_mode) {
                lock_cnt = required_lock_frames;
            } else if (centered_demo_frame && !demo_wait_for_belt_clear && demo_cooldown_frames == 0) {
                // Generated stations contain one stationary part. Compression can
                // change the winning label between frames; don't skip the part
                // while waiting for four identical labels from the same image.
                lock_cnt = std::min(required_lock_frames, lock_cnt + 1);
            } else if (mode == "DEMO" && (demo_wait_for_belt_clear || demo_cooldown_frames > 0)) {
                if (is_conveyor_clear) {
                    // As soon as the empty belt gap appears, clear cooldown so the next part locks immediately
                    demo_cooldown_frames = 0;
                    demo_wait_for_belt_clear = false;
                } else if (demo_cooldown_frames > 0) {
                    demo_cooldown_frames--;
                } else {
                    demo_wait_for_belt_clear = false;
                }
                lock_cnt = 0;
                lock_candidate_sig.clear();
            } else {
                if (advisory_sig == lock_candidate_sig) {
                    lock_cnt = std::min(required_lock_frames, lock_cnt + 1);
                } else {
                    lock_candidate_sig = advisory_sig;
                    lock_cnt = 1;
                }
            }
            const bool is_locked = (!is_conveyor_clear) && (lock_cnt >= required_lock_frames);
            cur.is_locked = is_locked;
            cur.lock_cnt = lock_cnt;

            const std::string current_sample = is_static_mode
                ? (std::to_string(cur_gen_id) + "|" + std::to_string(cur_idx) + "|" + advisory_sig)
                : (mode + "|" + std::to_string(cur_gen_id) + "|" + advisory_sig);

            if (cur.is_locked && current_sample != llm_dispatched_sample) {
                cur.advisory_pending = true;
                if (!llm_busy) {
                    std::string fam;
                    std::string lbl;
                    std::string grp;
                    std::string sev;
                    float conf = 0.0f;
                    float area_pct = 0.0f;
                    int box_count = 0;
                    const float anom_score = v_res.anomaly.score;
                    bool is_anom = false;

                    if (v_res.is_not_metal) {
                        fam = "not_metal";
                        lbl = "not_metal";
                        grp = "none";
                        sev = "NONE";
                        conf = v_res.family_conf;
                    } else if (!defect_or_anomaly) {
                        fam = v_res.family.empty() ? "steel_sheet" : v_res.family;
                        lbl = "clean";
                        grp = "none";
                        sev = "PASS";
                        conf = v_res.family_conf;
                    } else {
                        fam = v_res.family.empty() ? "steel_sheet" : v_res.family;
                        lbl = (v_res.primary_label.empty() || v_res.primary_label == "NONE")
                            ? (v_res.anomaly.is_anomalous ? "anomaly" : "defect")
                            : v_res.primary_label;
                        grp = (v_res.primary_group.empty() || v_res.primary_group == "NONE")
                            ? (v_res.anomaly.is_anomalous ? "anomaly" : "defect")
                            : v_res.primary_group;
                        sev = v_res.severity;
                        conf = v_res.highest_confidence;
                        area_pct = v_res.union_area_pct;
                        box_count = v_res.total_boxes;
                        is_anom = v_res.anomaly.is_anomalous;
                    }

                    cur.llm_busy = true;
                    cur.llm_elapsed = 0.0f;
                    cur.engine_source = "Llama 3.2 1B";
                    cur.cause = "Analyzing inspection telemetry with Llama 3.2 1B...";
                    cur.action = "Waiting for on-device Llama 3.2 1B advisory...";
                    cur.narrative.clear();

                    active_llm_future = llama_engine.request_advisory_async(
                        fam, lbl, grp, sev, conf, area_pct, box_count, anom_score, is_anom);
                    llm_busy = true;
                    llm_start_time = std::chrono::steady_clock::now();
                    llm_target_gen_id = cur_gen_id;
                    llm_target_is_static = is_static_mode;
                    llm_target_vres = v_res;
                    llm_dispatched_sample = current_sample;
                    if (mode == "DEMO") {
                        ipc_server.set_auto_paused(true);
                    }
                } else {
                    cur.llm_busy = true;
                    cur.engine_source = "Llama 3.2 1B";
                }
                ipc_server.update_state_for_generation(cur_gen_id, cur);
            } else {
                if (!llm_busy && cur.is_locked && current_sample == llm_dispatched_sample) {
                    cur.advisory_pending = false;
                    cur.llm_busy = false;
                }
                ipc_server.update_state_for_generation(cur_gen_id, cur);
            }
            if (!pending_frame_jpeg.empty() && ipc_server.get_sample_generation_id() == cur_gen_id) {
                ipc_server.update_frame_jpeg(pending_frame_jpeg);
            }

            if (is_static_mode) {
                ipc_server.wait_for_vision_wakeup(50);
            } else if (mode == "DEMO") {
                const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - loop_frame_start).count();
                const double source_fps = vid_cap.get(cv::CAP_PROP_FPS);
                const int period_ms = (source_fps >= 1.0 && source_fps <= 60.0)
                    ? static_cast<int>(1000.0 / source_fps) : 80;
                const int pace_wait = std::max(1, period_ms - static_cast<int>(elapsed_ms));
                ipc_server.wait_for_vision_wakeup(pace_wait);
            } else if (mode == "VIDEO") {
                ipc_server.wait_for_vision_wakeup(1);
            }
        }
    });

    while (!g_shutdown.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (g_shutdown.load()) break;

        const auto st = ipc_server.get_state();
        const float fps = (st.timings_ms.total > 0.01f) ? (1000.0f / st.timings_ms.total) : 0.0f;
        std::cout << "[HUD] " << st.mode
                  << " | Backend: " << st.vision_backend
                  << " | Family: " << st.family << " (" << std::fixed << std::setprecision(1)
                  << (st.family_conf * 100.0f) << "%)"
                  << " | Defect: " << st.primary_label << " [" << st.primary_group << "]"
                  << " | Severity: " << st.severity << "/" << st.verdict
                  << " | Total: " << std::setprecision(1) << st.timings_ms.total << "ms ("
                  << std::setprecision(1) << fps << " FPS)"
                  << " | Anomaly: " << std::setprecision(2) << st.anomaly.score
                  << " (bank=" << st.anomaly.bank_size << ")" << std::endl;
    }

    std::cout << "\n[Main] Shutdown signal received; stopping threads..." << std::endl;
    vision_running.store(false);
    if (vision_thread.joinable()) {
        vision_thread.join();
    }
    llama_engine.shutdown();
    ipc_server.stop();
    std::cout << "[Main] Clean shutdown complete." << std::endl;
    return 0;
}
