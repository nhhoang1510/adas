#pragma once

#include "road_geometry.hpp"
#include "bev_transformer.hpp"
#include "lane_detector.hpp"
#include "vehicle_tracker.hpp"
#include "detection_filter.hpp"
#include "calibration_manager.hpp"
#include "dual_engine_pipeline.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <thread>
#include <vector>

struct AdasVehicleOut {
    int   id = 0, cls = 0;
    float conf = 0.f;
    cv::Rect2f box;
    float z = 0.f, vz = 0.f, x = 0.f;
    bool  in_path = false, cut_in = false, is_lead = false;
    char  lane_pos[12] = "IN_LANE";
};

struct AdasOutput {
    static constexpr int MAX_VEH = 16;
    double t = 0.0;
    bool   has_lane = false, left_synth = false, right_synth = false;
    double offset_m = 0.0, width_m = 3.5, lat_vel_mps = 0.0, tlc_s = -1.0;
    double margin_left_m = 9.9, margin_right_m = 9.9;
    int    ldw_level = 0;
    char   ldw_status[20] = "NO_LANE";
    char   ldw_text[40] = "KHONG THAY VACH LAN";
    std::array<cv::Point, 12> left_img{}, right_img{};
    int    n_lane_pts = 0;
    int    n_veh = 0;
    std::array<AdasVehicleOut, MAX_VEH> veh{};
    bool   has_lead = false;
    int    lead_idx = -1;
    int    fcw_level = 0;
    char   fcw_text[48] = "AN TOAN";
    double horizon_y = 0.0, vanish_x = 0.0;
    int    hood_y = 0;
    bool   night = false;
    bool   ai_alive = false;
    int    ai_age_frames = 999;
    double ai_ms = 0.0, fast_ms = 0.0;
    char   lane_src[8] = "-";
    double calib_progress = 0.0;
    bool   calibrated = false;
    char   calib_text[48] = "CALIB 0%";
};

class AdasPipeline {
public:
    struct Config {
        DualEnginePipeline::Config engine;
        LaneDetector::Params lane;
        VehicleTracker::Params tracker;
        CalibrationManager::Params calib;
        std::vector<int> fast_cores = {0, 1};
        std::vector<int> ai_cores = {2, 3};
        int  ai_stale_frames = 45;
        bool debug_view = true;
    };

    explicit AdasPipeline(const Config& cfg)
        : m_cfg(cfg), m_lanes(cfg.lane), m_tracker(cfg.tracker), m_calMgr(cfg.calib), m_calib(cfg.engine.calib) {
        m_lastDets.reserve(64); m_filtDets.reserve(64); m_exclude.reserve(64);
        m_calMgr.begin(m_calib, m_calib.auto_calibrated == 1);
    }
    ~AdasPipeline() { stop(); }

    void start() {
        if (m_running.exchange(true)) return;
        m_aiThread = std::thread([this] { aiLoop(); });
    }
    void stop() {
        if (!m_running.exchange(false)) return;
        { std::lock_guard<std::mutex> lk(m_frameMtx); m_frameReady = true; }
        m_frameCv.notify_all();
        if (m_aiThread.joinable()) m_aiThread.join();
    }

    void pinFastThread() const { pinCurrent(m_cfg.fast_cores); }

    void setEgoSpeedKph(double kph) { m_egoMps.store(kph < 0 ? -1.0 : kph / 3.6, std::memory_order_relaxed); }
    void setTurnSignals(bool left, bool right) { m_turnL.store(left); m_turnR.store(right); }

    const AdasOutput& step(const cv::Mat& frame, double t) {
        const auto t0 = std::chrono::steady_clock::now();
        const int W = frame.cols, H = frame.rows;
        AdasOutput& o = m_out;
        o.t = t;

        if (!m_aiBusy.load(std::memory_order_acquire)) {
            const int slot = m_frameSlot.load(std::memory_order_relaxed) ^ 1;
            frame.copyTo(m_frameBuf[slot]);
            m_frameT[slot] = t;
            m_frameSlot.store(slot, std::memory_order_release);
            m_aiBusy.store(true, std::memory_order_release);
            { std::lock_guard<std::mutex> lk(m_frameMtx); m_frameReady = true; }
            m_frameCv.notify_one();
        }

        bool detFresh = false;
        const unsigned seq = m_resSeq.load(std::memory_order_acquire);
        if (seq != m_resSeen && (seq & 1u) == 0) {
            const AiResult& r = m_res[(seq / 2) % 3];
            m_resSeen = seq;
            if (r.det_fresh)  { m_lastDets = r.dets; detFresh = true; }
            o.night = r.night; o.ai_ms = r.cycle_ms;
            m_aiAge = 0;
        } else {
            ++m_aiAge;
        }
        o.ai_age_frames = m_aiAge;
        o.ai_alive = m_aiAge < m_cfg.ai_stale_frames;

        const RoadGeometry g = RoadGeometry::make(m_calib, W, H, m_calib.horizon_y_ratio * H, int(m_calib.hood_y_ratio * H));
        o.horizon_y = g.vh; o.vanish_x = g.cx; o.hood_y = g.hood;

        m_lanes.predict(t);
        cv::Mat calMask;
        {
            m_exclude.clear();
            for (const auto& d : m_lastDets) m_exclude.emplace_back(cv::Point(int(d.box.x1) - 4, int(d.box.y1) - 4), cv::Point(int(d.box.x2) + 4, int(d.box.y2) + 4));
            for (int i = 0; i < o.n_veh; ++i) m_exclude.emplace_back(o.veh[i].box);
            laneMarkingMask(frame, g, m_markMask, m_markGray, m_exclude);
            cv::Mat fused = m_markMask;
            m_static.apply(fused, m_staticMask, g, t,
                           m_calMgr.status().state == CalibrationManager::State::Calibrated);
            fused = m_staticMask;
            m_lanes.setGeometryCalibrated(m_calMgr.status().state == CalibrationManager::State::Calibrated);
            m_lanes.update(fused, true, g);
            calMask = fused;
            std::snprintf(o.lane_src, sizeof o.lane_src, "CV");
            if (m_cfg.debug_view) m_dbgMask = fused;
        }
        const LaneOutput lo = m_lanes.output(g, t, m_turnL.load(), m_turnR.load());
        fillLanes(lo, o);

        const std::vector<VehicleDetection>* in = nullptr;
        if (detFresh) {
            m_filtDets = filterDetections(m_lastDets, g, &m_detStats, false);
            m_lastDets.swap(m_filtDets);
            in = &m_lastDets;
            for (const auto& d : m_lastDets)
                m_calMgr.addVehicle(cv::Rect2f(d.box.x1, d.box.y1, d.box.x2 - d.box.x1, d.box.y2 - d.box.y1), d.box.cls, H, g.hood);
        }
        const double ego = m_egoMps.load(std::memory_order_relaxed);
        const auto& tracks = m_tracker.step(t, in, g, lo, ego);
        fillVehicles(tracks, m_tracker.lead(), o);

        {
            CameraCalibration c = m_calib;
            const auto& lp = m_lanes.params();
            if (m_calMgr.step(t, frame, g, lo, calMask, calMask.empty() ? cv::Mat() : m_lanes.bev(), lp.res_z, lp.z_max,
                              ego, m_turnL.load() || m_turnR.load(), c)) {
                m_calib = c;
                m_lanes.reset();
                m_tracker.reset();
                std::lock_guard<std::mutex> lk(m_calibMtx);
                m_pendingCalib = c; m_calibPending = true;
            }
            const auto& cs = m_calMgr.status();
            o.calib_progress = cs.progress;
            o.calibrated = cs.state == CalibrationManager::State::Calibrated;
            std::snprintf(o.calib_text, sizeof o.calib_text, "%s", cs.text.c_str());
            if (m_cfg.calib.geometry_only && !o.calibrated && o.ldw_level == 2) o.ldw_level = 1;
        }
        o.fast_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return o;
    }

    void drawHud(cv::Mat& img, const AdasOutput& o) const {
        const cv::Scalar colLane = o.ldw_level == 2 ? cv::Scalar(0, 0, 255) : o.ldw_level == 1 ? cv::Scalar(0, 200, 255) : cv::Scalar(60, 255, 120);
        if (o.has_lane && o.n_lane_pts >= 2) {
            for (int k = 1; k < o.n_lane_pts; ++k) {
                cv::line(img, o.left_img[k - 1], o.left_img[k], o.left_synth ? cv::Scalar(120, 120, 120) : colLane, 2, cv::LINE_AA);
                cv::line(img, o.right_img[k - 1], o.right_img[k], o.right_synth ? cv::Scalar(120, 120, 120) : colLane, 2, cv::LINE_AA);
            }
        }
        for (int i = 0; i < o.n_veh; ++i) {
            const auto& v = o.veh[i];
            const cv::Scalar col = v.is_lead ? (o.fcw_level == 2 ? cv::Scalar(0, 0, 255) : o.fcw_level == 1 ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0))
                                 : v.cut_in ? cv::Scalar(0, 140, 255) : v.in_path ? cv::Scalar(0, 220, 255) : cv::Scalar(255, 185, 40);
            cv::rectangle(img, v.box, col, v.is_lead ? 2 : 1);
            char s[48];
            if (v.is_lead) std::snprintf(s, sizeof s, "LEAD #%d %.1fm", v.id, v.z);
            else                        std::snprintf(s, sizeof s, "#%d %s %.0fm", v.id, v.lane_pos, v.z);
            cv::putText(img, s, cv::Point(int(v.box.x), std::max(10, int(v.box.y) - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv::LINE_AA);
        }
        if (m_cfg.debug_view) {
            if (!m_dbgMask.empty() && m_dbgMask.size() == img.size()) img.setTo(cv::Scalar(0, 255, 0), m_dbgMask);
            m_lanes.drawDebug(img, img.rows / 2);
            cv::line(img, cv::Point(0, int(o.horizon_y)), cv::Point(img.cols, int(o.horizon_y)), cv::Scalar(0, 255, 255), 1);
            cv::line(img, cv::Point(0, o.hood_y), cv::Point(img.cols, o.hood_y), cv::Scalar(0, 128, 255), 1);
        }
        char line[192];
        std::snprintf(line, sizeof line, "%s | off %+.2fm | %s", o.ldw_text, o.offset_m, o.fcw_text);
        cv::putText(img, line, cv::Point(6, 16), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        std::snprintf(line, sizeof line, "fast %.1fms ai %.0fms age %d %s | %s", o.fast_ms, o.ai_ms, o.ai_age_frames, o.lane_src, o.calib_text);
        cv::putText(img, line, cv::Point(6, img.rows - 8), cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    }

    const CameraCalibration& calibration() const { return m_calib; }
    const AdasOutput& output() const { return m_out; }
    const LaneDetector& laneDetector() const { return m_lanes; }

private:
    struct AiResult {
        std::vector<VehicleDetection> dets;
        bool det_fresh = false, night = false;
        double cycle_ms = 0.0;
    };

    Config m_cfg;
    LaneDetector m_lanes;
    VehicleTracker m_tracker;
    CalibrationManager m_calMgr;
    CameraCalibration m_calib;
    AdasOutput m_out;

    cv::Mat m_frameBuf[2];
    double m_frameT[2] = {0, 0};
    std::atomic<int> m_frameSlot{0};
    std::atomic<bool> m_aiBusy{false};
    std::mutex m_frameMtx;
    std::condition_variable m_frameCv;
    bool m_frameReady = false;
    AiResult m_res[3];
    std::atomic<unsigned> m_resSeq{0};
    unsigned m_resSeen = 0;
    unsigned m_resWrite = 0;
    std::mutex m_calibMtx;
    CameraCalibration m_pendingCalib;
    bool m_calibPending = false;

    std::atomic<bool> m_running{false};
    std::thread m_aiThread;
    std::atomic<double> m_egoMps{-1.0};
    std::atomic<bool> m_turnL{false}, m_turnR{false};

    StaticMaskFilter m_static;
    cv::Mat m_staticMask;
    cv::Mat m_markMask, m_markGray, m_dbgMask;
    std::vector<VehicleDetection> m_lastDets, m_filtDets;
    std::vector<cv::Rect> m_exclude;
    DetFilterStats m_detStats;
    int m_aiAge = 999;

    static void pinCurrent(const std::vector<int>& cores) {
        if (cores.empty()) return;
        cpu_set_t set; CPU_ZERO(&set);
        for (int c : cores) CPU_SET(c, &set);
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    }

    void aiLoop() {
        pinCurrent(m_cfg.ai_cores);
        DualEnginePipeline engine(m_cfg.engine);
        std::printf("[AdasPipeline] AI thread on cores");
        for (int c : m_cfg.ai_cores) std::printf(" %d", c);
        std::printf(" | det %d\n", engine.detReady());
        while (m_running.load()) {
            {
                std::unique_lock<std::mutex> lk(m_frameMtx);
                m_frameCv.wait_for(lk, std::chrono::milliseconds(200), [this] { return m_frameReady || !m_running.load(); });
                if (!m_running.load()) break;
                if (!m_frameReady) continue;
                m_frameReady = false;
            }
            {
                std::lock_guard<std::mutex> lk(m_calibMtx);
                if (m_calibPending) { engine.updateCalibration(m_pendingCalib); m_calibPending = false; }
            }
            const int slot = m_frameSlot.load(std::memory_order_acquire);
            const cv::Mat& f = m_frameBuf[slot];
            if (!f.empty() && engine.ready()) {
                DualEnginePipeline::Result r = engine.step(f);
                AiResult& dst = m_res[(m_resWrite + 1) % 3];
                m_resSeq.store(2 * (m_resWrite + 1) - 1, std::memory_order_release);
                dst.dets = r.vehicles;
                dst.det_fresh = r.det_fresh;
                dst.night = r.night; dst.cycle_ms = r.cycle_ms;
                ++m_resWrite;
                m_resSeq.store(2 * m_resWrite, std::memory_order_release);
            }
            m_aiBusy.store(false, std::memory_order_release);
        }
    }

    static void fillLanes(const LaneOutput& lo, AdasOutput& o) {
        o.has_lane = lo.has_lane; o.left_synth = lo.left_synth; o.right_synth = lo.right_synth;
        o.offset_m = lo.offset_m; o.width_m = lo.width_m; o.lat_vel_mps = lo.lat_vel_mps; o.tlc_s = lo.tlc_s;
        o.margin_left_m = lo.margin_left_m; o.margin_right_m = lo.margin_right_m;
        o.ldw_level = lo.alert_level;
        std::snprintf(o.ldw_status, sizeof o.ldw_status, "%s", lo.status.c_str());
        std::snprintf(o.ldw_text, sizeof o.ldw_text, "%s", lo.text.c_str());
        o.n_lane_pts = int(std::min(lo.left_img.size(), size_t(12)));
        for (int i = 0; i < o.n_lane_pts; ++i) { o.left_img[i] = lo.left_img[i]; o.right_img[i] = lo.right_img[i]; }
    }
    static void fillVehicles(const std::vector<TrackedVehicle>& tracks, const LeadOutput& ld, AdasOutput& o) {
        o.n_veh = 0; o.has_lead = ld.has_lead; o.lead_idx = -1;
        for (const auto& v : tracks) {
            if (o.n_veh >= AdasOutput::MAX_VEH) break;
            AdasVehicleOut& d = o.veh[o.n_veh];
            d.id = v.id; d.cls = v.cls; d.conf = v.conf; d.box = v.box;
            d.z = float(v.z); d.vz = float(v.vz); d.x = float(v.x);
            d.in_path = v.in_path; d.cut_in = v.cut_in;
            d.is_lead = ld.has_lead && v.id == ld.lead.id;
            std::snprintf(d.lane_pos, sizeof d.lane_pos, "%s", v.lane_pos.c_str());
            if (d.is_lead) o.lead_idx = o.n_veh;
            ++o.n_veh;
        }
        o.fcw_level = ld.fcw_level;
        std::snprintf(o.fcw_text, sizeof o.fcw_text, "%s", ld.fcw_text.c_str());
    }
};
