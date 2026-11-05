#ifndef CAMERAWORKER_H
#define CAMERAWORKER_H

#include <QThread>
#include <QString>
#include <QMetaObject>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>
#include <deque>
#include <mutex>
#include <atomic>
#include <future>
#include <memory>
#include <thread>
#include <queue>
#include <condition_variable>
#include <unistd.h>
#include <fcntl.h>
#include <sched.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include "AdasController.h"
#include "FrameProvider.h"
#include "MjpegServer.hpp"
#include "dual_engine_pipeline.hpp"
#include "fast_vehicle_detector.hpp"
#include "lane_detector.hpp"
#include "detection_filter.hpp"
#include "vehicle_tracker.hpp"
#include "auto_calibrator.hpp"
#include "calibration_manager.hpp"
#include "traffic_sign_engine.hpp"
#include "box_flow.hpp"
#include "ego_motion.hpp"
#include "obd_reader.hpp"
#include <sys/stat.h>

inline void adas_pin_to_cores(const std::vector<int>& cores) {
#if defined(__arm__) || defined(__aarch64__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int c : cores) CPU_SET(c, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
    (void)cores;
#endif
}

inline std::string adas_model_file(const std::string& dir, const std::string& base) {
    const char* e = getenv("ADAS_INT8");
    const std::string q = dir + "/" + base + "_int8.onnx", f = dir + "/" + base + ".onnx";
    if (!(e && e[0] == '0') && access(q.c_str(), R_OK) == 0) return q;
    return access(f.c_str(), R_OK) == 0 ? f : std::string();
}
inline bool adas_is_int8(const std::string& path) {
    return path.size() > 10 && path.compare(path.size() - 10, 10, "_int8.onnx") == 0;
}

inline void adas_pin_to_core(int core) {
    adas_pin_to_cores({core});
}

inline std::vector<int> adas_env_cores(const char* name, std::vector<int> def) {
    const char* e = getenv(name);
    if (!e || !*e) return def;
    std::vector<int> v;
    for (const char* p = e; *p;) {
        char* end = nullptr;
        const long c = strtol(p, &end, 10);
        if (end == p) break;
        if (c >= 0 && c <= 3) v.push_back(int(c));
        p = *end ? end + 1 : end;
    }
    return v.empty() ? def : v;
}

static inline int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct DetectedVehicle {
    cv::Rect box;
    float conf = 0.0f;
    int cls = 0;
    double dist = -1.0;
    bool isLead = false;
    int trackId = 0;
    std::string lanePos = "IN_LANE";
    double vRel = 0.0;
    bool isCutIn = false;
    bool inPath = false;
    double x = 0.0;
    double widthM = 1.8;
    std::vector<cv::Point> trail;
};

struct PerceptionOutput {
    double fps = 0.0;
    int speed = 0;
    std::string laneStatus = "NORMAL";
    std::string laneOffset = "+0.00m";
    std::string laneText = "GIU LAN TOT";
    int threatLevel = 0;
    std::string warningText = "AN TOAN";
    bool hasLead = false;
    double leadDist = -1.0;
    double leadConf = 0.0;
    int inLaneVehicles = 0;
    int speedLimit = 60;
    int leadTrackId = 0;

    std::vector<cv::Point> drivablePoly;
    std::vector<cv::Point> leftLaneLine;
    std::vector<cv::Point> rightLaneLine;
    int alertLevel = 0;
    int roadShape = 0;
    double curveRadius = 0.0;
    bool egoValid = false;
    double yawRate = 0.0;
    int egoTurn = 0;
    double pathCurv = 0.0;
    double roadBend = 0.0;
    std::vector<cv::Point> pathLeft, pathRight;
    double latVel = 0.0;
    double offsetM = 0.0;
    bool hasLane = false;
    bool hasLeadBox = false;
    cv::Rect leadBox;
    std::vector<DetectedVehicle> allVehicles;

    std::vector<SignTracker::Track> signs;
    int activeSign = -1;
    int speedSign = -1, auxSign = -1;
    int speedValue = 0;

    double horizonY = 240.0;
    double vanishingPointX = 320.0;
    double pitchDeg = 0.0;
    bool hasHorizon = false;
    std::string horizonSource = "CALIBRATED";
    std::string calibText = "CALIB 0%";
    double calibProgress = 0.0;
    bool calibrated = false;

    std::string envLighting = "Ban ngay (Chuan)";
    int envBrightness = 135;

    bool onnxActive = false;
    int onnxAge = 999;
};

template<typename T, int CAP = 2>
class OnnxFrameQueue {
public:
    void push(T&& item) {
        std::lock_guard<std::mutex> lk(mtx_);
        if ((int)q_.size() >= CAP) q_.pop();
        q_.push(std::move(item));
        cv_.notify_one();
    }
    bool pop(T& out, int timeout_ms = 300) {
        std::unique_lock<std::mutex> lk(mtx_);
        if (!cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                          [this]{ return !q_.empty() || done_; }))
            return false;
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop();
        return true;
    }
    void set_done() {
        std::lock_guard<std::mutex> lk(mtx_);
        done_ = true;
        cv_.notify_all();
    }
    void reset() {
        std::lock_guard<std::mutex> lk(mtx_);
        done_ = false;
        while (!q_.empty()) q_.pop();
    }
private:
    std::queue<T> q_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool done_ = false;
};

struct HudState {
    double fps; int speed; QString laneStatus, laneOffset, laneText;
    int threat; QString warning; bool hasLead; double leadDist, leadConf;
    int inLane; double horizonY, pitch;
    QString sourceState; bool aiHealthy;
    int sign;
    int speedSign, auxSign;
    int speedValue;
    QString obdState;
    int calibProgress;
    double roadBend;
};

class CameraWorker : public QThread {
    Q_OBJECT

signals:
    void frame_ready(int frameId);

public:
    CameraWorker(AdasController* ctrl, FrameProvider* provider,
                 const std::string& source = "0",
                 const std::string& modelPath = "",
                 int onnxCadence = 5,
                 QObject* parent = nullptr)
        : QThread(parent)
        , m_ctrl(ctrl)
        , m_provider(provider)
        , m_source(source)
        , m_modelPath(modelPath)
        , m_onnxCadence(onnxCadence)
        , m_running(true)
        , m_frameCounter(0)
        , m_sourceChanged(false)
        , m_isTestVideo(false)
        , m_onnxReady(false)
        , m_onnxBusy(false)
    {}

    void requestStop() {
        m_running = false;
        m_onnxQueue.set_done();
    }

    void stop() {
        requestStop();
        wait();
    }

    bool isHealthy() const {
        const int64_t t = nowNs();
        const bool loopAlive = t - m_lastLoopNs.load(std::memory_order_relaxed) < 2000000000LL;
        const int64_t lastAi = m_onnxLastResultNs.load(std::memory_order_relaxed);
        const bool aiAlive = !m_onnxReady.load() || lastAi == 0 ||
                             t - lastAi < 10000000000LL;
        return loopAlive && aiAlive;
    }
    bool isAiHealthy() const { return m_aiHealthy; }

    void setEgoSpeedKph(double kph) { m_egoSpeedMps.store(kph < 0 ? -1.0 : kph / 3.6); }
    void setTurnSignals(bool left, bool right) { m_turnLeft.store(left); m_turnRight.store(right); }
    double egoMps() const { return m_isTestVideo ? -1.0 : m_egoSpeedMps.load(std::memory_order_relaxed); }

    bool switchSource(const std::string& newSource) {
        auto prom = std::make_shared<std::promise<bool>>();
        auto fut = prom->get_future();
        {
            std::lock_guard<std::mutex> lock(m_srcMtx);
            m_pendingSource = newSource;
            m_pendingPromise = prom;
            m_sourceChanged = true;
        }
        if (fut.wait_for(std::chrono::milliseconds(8000)) == std::future_status::ready) {
            return fut.get();
        }
        qWarning("[CameraWorker] switchSource timeout sau 8s!");
        return false;
    }

protected:
    void run() override {
        adas_pin_to_core(1);
        pthread_setname_np(pthread_self(), "adas-fast");
        qInfo("[CameraWorker] Fast Path pinned to Core 1");
        if (const char* e = getenv("ADAS_CV_THREADS")) cv::setNumThreads(std::max(0, atoi(e)));

        m_obd.start([this](double kph) { setEgoSpeedKph(kph); });

        loadCalibrationFor(!(m_source == "0" || m_source == "webcam"), m_source);

        if (m_cvOnly) qInfo("[CameraWorker] ADAS_CV_ONLY: khong dung deep learning (chi computer vision co dien)");
        else startOnnxThread();
        startTrafficSigns();

        m_mjpegServer.setSwitchSourceCallback([this](const std::string& src) {
            return this->switchSource(src);
        });

        m_mjpegServer.start(8080);

        cv::VideoCapture cap;
        std::string active_pipeline;
        bool has_cam = false;

        if (m_source == "0" || m_source == "webcam") {
            has_cam = autoFindAndOpenGStreamer(cap, active_pipeline);
            m_mjpegServer.setCurrentSource("camera");
        } else {
            has_cam = cap.open(m_source, cv::CAP_GSTREAMER);
            if (!has_cam) has_cam = cap.open(m_source);
            if (has_cam) {
                active_pipeline = m_source;
                m_isTestVideo = true;
                m_currentVideoPath = m_source;
                m_mjpegServer.setCurrentSource("video");
            }
        }

        if (!has_cam) {
            qWarning("[CameraWorker] Khong tim thay camera qua GStreamer! Chuyen sang che do Live Simulation & Auto-Reconnect.");
        }

        cv::Mat frame;
        cv::Mat prev_gray;
        int64 t_prev = cv::getTickCount();
        int fps_frames = 0;
        double current_fps = 30.0;
        double estimated_speed = 0.0;
        int fail_count = 0;
        int sim_tick = 0;
        int loopErrors = 0;
        auto next_video_frame_time = std::chrono::steady_clock::now();

        while (m_running.load()) {
            m_lastLoopNs.store(nowNs(), std::memory_order_relaxed);
            try {
                if (m_sourceChanged.exchange(false)) {
                    std::string targetSrc;
                    std::shared_ptr<std::promise<bool>> targetProm;
                    {
                        std::lock_guard<std::mutex> lock(m_srcMtx);
                        targetSrc = m_pendingSource;
                        targetProm = m_pendingPromise;
                        m_pendingPromise.reset();
                    }

                    bool switch_ok = false;
                    if (targetSrc == "camera" || targetSrc.empty()) {
                        cap.release();
                        m_isTestVideo = false;
                        has_cam = autoFindAndOpenGStreamer(cap, active_pipeline);
                        m_mjpegServer.setCurrentSource("camera");
                        prev_gray.release();
                        estimated_speed = 0.0;
                        resetSmoothState();
                        loadCalibrationFor(false, "");
                        switch_ok = has_cam;
                        qInfo("[CameraWorker] Da chuyen sang Webcam GStreamer (status: %d)", switch_ok);
                    } else {
                        cap.release();
                        std::string vpath = targetSrc;
                        if (vpath.rfind("video:", 0) == 0) vpath = vpath.substr(6);
                        if (tryOpenVideoFile(cap, vpath)) {
                            m_isTestVideo = true;
                            has_cam = true;
                            active_pipeline = "Test Video (" + vpath + ")";
                            m_mjpegServer.setCurrentSource("video");
                            prev_gray.release();
                            estimated_speed = 0.0;
                            resetSmoothState();
                            loadCalibrationFor(true, vpath);
                            switch_ok = true;
                            qInfo("[CameraWorker] Da mo video test AI thanh cong: %s", vpath.c_str());
                        } else {
                            qWarning("[CameraWorker] Khong the mo video: %s -> Quay ve camera/simulation", vpath.c_str());
                            m_isTestVideo = false;
                            has_cam = autoFindAndOpenGStreamer(cap, active_pipeline);
                            m_mjpegServer.setCurrentSource("camera");
                            resetSmoothState();
                            loadCalibrationFor(false, "");
                            switch_ok = false;
                        }
                    }

                    if (targetProm) {
                        try { targetProm->set_value(switch_ok); } catch (...) {}
                    }
                }

                if (m_mjpegServer.takeCalibReset()) resetCalibration();

                int64 t_frame_start = cv::getTickCount();
                const auto tp0 = std::chrono::steady_clock::now();

                if (has_cam) {
                    cv::Mat raw_frame;
                    if (!cap.read(raw_frame) || raw_frame.empty()) {
                        if (m_isTestVideo) {
                            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                            if (!cap.read(raw_frame) || raw_frame.empty()) {
                                cap.release();
                                if (!m_currentVideoPath.empty() && tryOpenVideoFile(cap, m_currentVideoPath)) {
                                    cap.read(raw_frame);
                                }
                            }
                            next_video_frame_time = std::chrono::steady_clock::now();
                            if (raw_frame.empty()) {
                                raw_frame = generateVideoNoticeFrame();
                                msleep(33);
                            }
                        } else {
                            fail_count++;
                            if (fail_count > 15) {
                                qWarning("[CameraWorker] Mat tin hieu camera GStreamer, do lai camera...");
                                cap.release();
                                has_cam = false;
                                fail_count = 0;
                                sim_tick = 0;
                            }
                            msleep(10);
                            continue;
                        }
                    }
                    fail_count = 0;
                    if (!m_isTestVideo) m_hadCamera = true;

                    if (!raw_frame.empty() && raw_frame.cols > 640) {
                        int target_w = 640;
                        int target_h = static_cast<int>(std::round(raw_frame.rows * (640.0 / raw_frame.cols)));
                        if (target_h % 2 != 0) target_h--;
                        cv::Mat resized;
                        cv::resize(raw_frame, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);
                        frame = resized;
                    } else {
                        frame = raw_frame;
                    }

                } else {
                    frame = m_simulateNoCam ? generateSimulationFrame(sim_tick, 52.0, 4.0)
                                            : generateNoCameraFrame(m_hadCamera);
                    ++sim_tick;

                    auto now_sim = std::chrono::steady_clock::now();
                    if (now_sim < next_video_frame_time) {
                        std::this_thread::sleep_until(next_video_frame_time);
                    } else if (now_sim - next_video_frame_time > std::chrono::milliseconds(100)) {
                        next_video_frame_time = now_sim;
                    }
                    next_video_frame_time += std::chrono::milliseconds(33);

                    const int probe_every = sim_tick < 900 ? 60 : 600;
                    if (sim_tick % probe_every == 0 && !m_probingCamera.load()) {
                        m_probingCamera.store(true);
                        std::thread([this]() {
                            cv::VideoCapture probeCap;
                            std::string probePipe;
                            bool found = false;
                            try {
                                found = autoFindAndOpenGStreamer(probeCap, probePipe);
                            } catch (...) {}
                            probeCap.release();
                            if (found && m_running.load()) {
                                std::lock_guard<std::mutex> lk(m_srcMtx);
                                m_pendingSource = "camera";
                                m_sourceChanged.store(true);
                            }
                            m_probingCamera.store(false);
                        }).detach();
                    }
                }

                const auto tp1 = std::chrono::steady_clock::now();
                m_frameCounter++;
                fps_frames++;

                int64 t_now = cv::getTickCount();
                double elapsed_sec = (t_now - t_prev) / cv::getTickFrequency();
                if (elapsed_sec >= 0.5) {
                    current_fps = fps_frames / elapsed_sec;
                    fps_frames = 0;
                    t_prev = t_now;
                }

                PerceptionOutput perc;
                if (has_cam || m_simulateNoCam) {
                    std::vector<VehicleDetection> fast_dets;
                    perc = processFrameFastPath(frame, current_fps,
                                                prev_gray, estimated_speed,
                                                fast_dets);

                    if (m_onnxReady.load(std::memory_order_relaxed) && !m_onnxBusy.load(std::memory_order_relaxed)) {
                        m_onnxBusy.store(true, std::memory_order_relaxed);
                        m_onnxFrame.store(m_frameCounter, std::memory_order_relaxed);
                        m_onnxQueue.push(frame.clone());
                    }

                    applyTracking(perc, frame, fast_dets);
                    stepTrafficSigns(frame, perc);
                } else {
                    perc = noCameraOutput(current_fps);
                }
                const auto tp2 = std::chrono::steady_clock::now();

                cv::Mat crop_sample;
                if (perc.hasLeadBox && perc.leadBox.area() > 100) {
                    cv::Rect safeRect = perc.leadBox & cv::Rect(0, 0, frame.cols, frame.rows);
                    if (safeRect.width > 10 && safeRect.height > 10) {
                        crop_sample = frame(safeRect).clone();
                    }
                }

                if (m_provider) {
                    m_provider->updateFrame(frame, crop_sample);
                }

                if (m_mjpegServer.hasClients()) {
                    static auto lastWebTime = std::chrono::steady_clock::now();
                    auto nowWeb = std::chrono::steady_clock::now();
                    auto webElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(nowWeb - lastWebTime).count();
                    if (webElapsedMs >= m_webPeriodMs) {
                        lastWebTime = nowWeb;
                        m_camState = has_cam ? (m_isTestVideo ? "video" : "live") : "none";
                        std::string telemJson = formatTelemetryJson(perc, frame.cols, frame.rows);
                        if (m_debugView && m_mjpegServer.debugMask() && !m_lastLaneMaskDbg.empty() &&
                            m_lastLaneMaskDbg.size() == frame.size()) {
                            cv::Mat dbg = frame.clone();
                            dbg.setTo(cv::Scalar(0, 255, 0), m_lastLaneMaskDbg);
                            m_mjpegServer.setFrame(dbg, telemJson, perc.fps, perc.leadDist, perc.speed, perc.speedLimit);
                        } else {
                            m_mjpegServer.setFrame(frame, telemJson, perc.fps, perc.leadDist, perc.speed, perc.speedLimit);
                        }
                    } else {
                        m_mjpegServer.updateStatusOnly(perc.fps, perc.leadDist, perc.speed, perc.speedLimit);
                    }
                } else {
                    m_mjpegServer.updateStatusOnly(perc.fps, perc.leadDist, perc.speed, perc.speedLimit);
                }

                HudState s{
                    perc.fps, perc.speed,
                    QString::fromStdString(perc.laneStatus),
                    QString::fromStdString(perc.laneOffset),
                    QString::fromStdString(perc.laneText),
                    perc.threatLevel, QString::fromStdString(perc.warningText),
                    perc.hasLead, perc.leadDist, perc.leadConf,
                    perc.inLaneVehicles,
                    perc.horizonY, perc.pitchDeg,
                    has_cam ? (m_isTestVideo ? QStringLiteral("VIDEO_TEST") : QStringLiteral("LIVE"))
                            : m_hadCamera ? QStringLiteral("CAMERA_LOST") : QStringLiteral("NO_CAMERA"),
                    m_cvOnly || m_aiHealthy,
                    perc.activeSign, perc.speedSign, perc.auxSign, perc.speedValue,
                    QString::fromLatin1(m_obd.stateName()),
                    perc.calibrated ? -1 : static_cast<int>(std::round(perc.calibProgress * 100.0)),
                    perc.roadBend
                };
                QMetaObject::invokeMethod(m_ctrl, [c = m_ctrl, s = std::move(s)] {
                    c->setFps(s.fps);                 c->setSpeed(s.speed);
                    c->setLaneStatus(s.laneStatus);   c->setLaneOffset(s.laneOffset);
                    c->setLaneText(s.laneText);       c->setThreatLevel(s.threat);
                    c->setWarningText(s.warning);     c->setHasLead(s.hasLead);
                    c->setLeadDist(s.leadDist);
                    c->setLeadConf(s.leadConf);       c->setInLaneVehicles(s.inLane);
                    c->setHorizonY(s.horizonY);       c->setPitchDeg(s.pitch);
                    c->setSourceState(s.sourceState); c->setAiHealthy(s.aiHealthy);
                    c->setSpeedSign(s.speedSign);     c->setAuxSign(s.auxSign);
                    c->setSpeedValue(s.speedValue);
                    c->setObdState(s.obdState);     c->setCalibProgress(s.calibProgress);
                    c->setRoadBend(s.roadBend);
                    c->setHasSign(s.sign >= 0);
                    if (s.sign >= 0) {
                        const SignInfo& si = signInfo(s.sign);
                        c->setSignCode(QString::number(s.sign));
                        c->setSignName(QString::fromUtf8(si.display));
                        c->setSignType(QString::fromLatin1(signTypeName(si.type)));
                    }
                }, Qt::QueuedConnection);

                emit frame_ready(m_frameCounter);
                {
                    const auto tp3 = std::chrono::steady_clock::now();
                    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
                    m_prof[0] += ms(tp0, tp1); m_prof[1] += ms(tp1, tp2); m_prof[2] += ms(tp2, tp3); ++m_profN;
                    if (ms(m_profT, tp3) >= 2000.0) {
                        std::printf("[Loop] %.1f FPS | read %.1f ms  perception %.1f ms (flow %.1f)  output(web+HUD) %.1f ms | AI %.0f ms/step, idle %.0f ms\n",
                                    current_fps, m_prof[0] / m_profN, m_prof[1] / m_profN, m_flowMs / m_profN, m_prof[2] / m_profN, m_cachedOnnxMs, m_aiIdleMs.load());
                        m_flowMs = 0;
                        std::fflush(stdout);
                        m_prof[0] = m_prof[1] = m_prof[2] = 0; m_profN = 0; m_profT = tp3;
                    }
                }

                if (m_isTestVideo) {
                    auto now_t = std::chrono::steady_clock::now();
                    if (next_video_frame_time == std::chrono::steady_clock::time_point{}) {
                        next_video_frame_time = now_t;
                    }
                    next_video_frame_time += std::chrono::microseconds(33333);
                    if (now_t < next_video_frame_time) {
                        std::this_thread::sleep_until(next_video_frame_time);
                    } else if (now_t - next_video_frame_time > std::chrono::milliseconds(100)) {
                        next_video_frame_time = now_t;
                    }
                }
            loopErrors = 0;
            } catch (const std::exception& ex) {
                if (onLoopError(ex.what(), loopErrors, has_cam, cap)) sim_tick = 0;
            } catch (...) {
                if (onLoopError("non-std exception", loopErrors, has_cam, cap)) sim_tick = 0;
            }
        }

        m_onnxQueue.set_done();
        if (m_onnxThread.joinable()) m_onnxThread.join();
        m_tsr.stop();
        m_mjpegServer.stop();
        if (cap.isOpened()) cap.release();
    }

private:
    bool tryGStreamerPipeline(cv::VideoCapture& cap, const std::string& pipeline, std::string& out_name, const std::string& desc) {
        if (cap.open(pipeline, cv::CAP_GSTREAMER)) {
            cv::Mat test;
            for (int i = 0; i < 6; ++i) {
                if (cap.read(test) && !test.empty()) {
                    out_name = desc;
                    return true;
                }
                msleep(20);
            }
            cap.release();
        }
        return false;
    }

    bool isRealCaptureCamera(const std::string& dev) {
        int fd = ::open(dev.c_str(), O_RDWR | O_NONBLOCK);
        if (fd < 0) return false;
        struct v4l2_capability cap_info;
        memset(&cap_info, 0, sizeof(cap_info));
        bool ok = false;
        if (::ioctl(fd, VIDIOC_QUERYCAP, &cap_info) == 0) {
            uint32_t caps = cap_info.capabilities;
            if (caps & V4L2_CAP_DEVICE_CAPS) {
                caps = cap_info.device_caps;
            }
            if ((caps & V4L2_CAP_VIDEO_CAPTURE) && !(caps & V4L2_CAP_VIDEO_M2M)) {
                ok = true;
            }
        }
        ::close(fd);
        return ok;
    }

    bool autoFindAndOpenGStreamer(cv::VideoCapture& cap, std::string& activePipeline) {
        struct AffinityScope {
            AffinityScope() { adas_pin_to_cores({0, 1}); }
            ~AffinityScope() { adas_pin_to_core(1); }
        } affinityScope;
        std::vector<std::string> devices;
        const char* env_cam = getenv("ADAS_CAMERA_DEVICE");
        if (env_cam && strlen(env_cam) > 0) devices.push_back(env_cam);
        devices.push_back("/dev/adas-cam");

        for (int i = 0; i < 4; ++i) {
            devices.push_back("/dev/video" + std::to_string(i));
        }

        for (const auto& dev : devices) {
            if (access(dev.c_str(), R_OK) != 0) continue;
            if (!isRealCaptureCamera(dev)) continue;

            std::string pipe1 = "v4l2src device=" + dev + " do-timestamp=true ! "
                                "image/jpeg,width=640,height=480,framerate=30/1 ! "
                                "jpegdec ! videoconvert ! "
                                "video/x-raw,format=BGR ! "
                                "appsink drop=true max-buffers=1 sync=false";
            if (tryGStreamerPipeline(cap, pipe1, activePipeline, dev + " [GStreamer MJPEG]")) {
                qInfo("[CameraWorker] GStreamer MJPEG pipeline opened on %s", dev.c_str());
                return true;
            }

            std::string pipe2 = "v4l2src device=" + dev + " do-timestamp=true ! "
                                "video/x-raw,width=640,height=480,framerate=30/1 ! "
                                "videoconvert ! video/x-raw,format=BGR ! "
                                "appsink drop=true max-buffers=1 sync=false";
            if (tryGStreamerPipeline(cap, pipe2, activePipeline, dev + " [GStreamer YUYV]")) {
                qInfo("[CameraWorker] GStreamer YUYV pipeline opened on %s", dev.c_str());
                return true;
            }

            std::string pipe3 = "v4l2src device=" + dev + " ! videoconvert ! video/x-raw,format=BGR ! "
                                "appsink drop=true max-buffers=1 sync=false";
            if (tryGStreamerPipeline(cap, pipe3, activePipeline, dev + " [GStreamer Auto]")) {
                qInfo("[CameraWorker] GStreamer Auto pipeline opened on %s", dev.c_str());
                return true;
            }

            if (cap.open(dev, cv::CAP_V4L2)) {
                cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                cv::Mat test;
                if (cap.read(test) && !test.empty()) {
                    activePipeline = dev + " [V4L2]";
                    return true;
                }
                cap.release();
            }
        }
        return false;
    }

    bool tryGstVideoPipeline(cv::VideoCapture& cap, const std::string& pipe, const std::string& desc) {
        if (cap.open(pipe, cv::CAP_GSTREAMER)) {
            cv::Mat test;
            for (int retry = 0; retry < 10; ++retry) {
                if (cap.read(test) && !test.empty() && test.cols > 0 && test.rows > 0) {
                    qInfo("[CameraWorker] Da mo video thanh cong qua %s: %dx%d", desc.c_str(), test.cols, test.rows);
                    return true;
                }
                msleep(20);
            }
            cap.release();
        }
        return false;
    }

    bool tryOpenVideoFile(cv::VideoCapture& cap, const std::string& vpath) {
        if (access(vpath.c_str(), R_OK) != 0) {
            qWarning("[CameraWorker] File video khong ton tai hoac khong the doc: %s", vpath.c_str());
            return false;
        }

        m_currentVideoPath = vpath;
        struct AffinityScope {
            AffinityScope() { adas_pin_to_cores({0, 1}); }
            ~AffinityScope() { adas_pin_to_core(1); }
        } affinityScope;

        const std::string src = "filesrc location=\"" + vpath + "\" ! qtdemux name=d d.video_0 ! h264parse ! ";
        const std::string sink = " ! appsink drop=false max-buffers=2 sync=false";

        if (tryGstVideoPipeline(cap, src + "v4l2h264dec ! v4l2convert ! video/x-raw,format=BGR,width=640,height=360" + sink,
                                "GStreamer HW decode + ISP (v4l2h264dec/v4l2convert BGR)")) return true;
        if (tryGstVideoPipeline(cap, src + "v4l2h264dec ! v4l2convert ! video/x-raw,format=BGRx,width=640,height=360 ! videoconvert ! video/x-raw,format=BGR" + sink,
                                "GStreamer HW decode + ISP (BGRx)")) return true;
        if (tryGstVideoPipeline(cap, src + "v4l2h264dec ! videoscale method=0 ! video/x-raw,width=640,height=360 ! videoconvert ! video/x-raw,format=BGR" + sink,
                                "GStreamer HW decode + SW scale")) return true;

        std::string p_sw = src + "avdec_h264 max-threads=2 ! videoscale method=0 ! video/x-raw,width=640,height=360 ! videoconvert n-threads=2 ! video/x-raw,format=BGR" + sink;
        if (tryGstVideoPipeline(cap, p_sw, "GStreamer SW (avdec_h264) - transcode test videos to 640x360 for speed")) return true;

        std::string p_uri = "uridecodebin uri=\"file://" + vpath + "\" ! videoscale ! video/x-raw,width=640,height=360 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=false max-buffers=2 sync=false";
        if (tryGstVideoPipeline(cap, p_uri, "GStreamer uridecodebin")) return true;

        if (cap.open(vpath)) {
            cv::Mat test;
            for (int retry = 0; retry < 5; ++retry) {
                if (cap.read(test) && !test.empty() && test.cols > 0 && test.rows > 0) {
                    qInfo("[CameraWorker] Da mo video qua OpenCV native backend: %s (%dx%d)", vpath.c_str(), test.cols, test.rows);
                    return true;
                }
                msleep(20);
            }
            cap.release();
        }

        qWarning("[CameraWorker] Khong the decode duoc video qua bat ky backend nao: %s", vpath.c_str());
        return false;
    }

    void loadCalibrationFor(bool isVideo, const std::string& videoPath) {
        CameraCalibration c;
        std::vector<std::string> candidates;
        std::string savePath;
        if (isVideo) {
            candidates.push_back(CameraCalibration::sidecarFor(videoPath));
            candidates.push_back("/opt/adas/calib/video_default.json");
            savePath = CameraCalibration::sidecarFor(videoPath);
        } else {
            const char* env = getenv("ADAS_CALIB");
            if (env && *env) candidates.push_back(env);
            const char* sp = getenv("ADAS_CALIB_SAVE");
            savePath = (sp && *sp) ? sp : "/var/lib/adas/calib/picam_auto.json";
            candidates.push_back(savePath);
            candidates.push_back("/opt/adas/calib/picam.json");
        }
        bool loaded = false;
        for (const auto& p : candidates) {
            if (CameraCalibration::load(p, c)) { loaded = true; break; }
        }
        if (!loaded) qWarning(isVideo ? "[Calib] Video chua co file hieu chuan -> mac dinh; tu hieu chuan nhanh (~10-20 s)"
                                      : "[Calib] Khong tim thay file hieu chuan -> mac dinh; tu hieu chuan khi chay >30 km/h");
        if (!loaded) c.horizon_y_ratio = 0.45;
        c.print();
        m_calib = c;

        auto envd = [](const char* k, double d) { const char* e = getenv(k); return (e && *e) ? atof(e) : d; };
        const bool quick = isVideo && envd("ADAS_CALIB_QUICK_VIDEO", 1.0) > 0;
        CalibrationManager::Params cp = quick ? CalibrationManager::Params::quickVideo() : CalibrationManager::Params();
        if (!quick) {
            cp.min_speed_kph = envd("ADAS_CALIB_MIN_KPH", 30.0);
            cp.required_s = envd("ADAS_CALIB_SECONDS", 180.0);
            cp.allow_no_speed = envd("ADAS_CALIB_NO_SPEED", 0.0) > 0;
            cp.estimate_lateral = envd("ADAS_CALIB_LATERAL", 1.0) > 0;
        }
        cp.lane_width_ref_m = envd("ADAS_LANE_WIDTH_REF", cp.lane_width_ref_m);
        cp.save_path = savePath;
        m_calibSavePath = savePath;
        if (!isVideo) {
            const auto slash = savePath.find_last_of('/');
            if (slash != std::string::npos) {
                std::string dir;
                for (size_t i = 1; i <= slash; ++i)
                    if (savePath[i] == '/' || i == slash) { dir = savePath.substr(0, i); ::mkdir(dir.c_str(), 0755); }
            }
        }
        m_assumeSpeedMps = envd("ADAS_CALIB_ASSUME_KPH", -1.0) / 3.6;
        m_calMgr.setParams(cp);
        const bool forceRecal = envd("ADAS_CALIB_RESET", 0.0) > 0 || m_forceRecal;
        m_calMgr.begin(c, c.auto_calibrated == 1 && !forceRecal);
        qInfo("[CalibMgr] %s, save -> %s", (c.auto_calibrated == 1 && !forceRecal) ? "da hieu chuan (giam sat lech camera)"
                                                                                 : "bat dau tu hieu chuan", savePath.c_str());

        std::lock_guard<std::mutex> lk(m_calibMtx);
        m_pendingCalib = c;
        m_calibPending = true;
        m_calibSoft = false;
    }

    void resetCalibration() {
        if (!m_calibSavePath.empty() && ::rename(m_calibSavePath.c_str(), (m_calibSavePath + ".bak").c_str()) == 0)
            qInfo("[Calib] Hieu chuan lai: %s -> .bak", m_calibSavePath.c_str());
        m_forceRecal = true;
        loadCalibrationFor(m_isTestVideo, m_currentVideoPath);
        m_forceRecal = false;
        m_lanes.reset();
        m_vehicles.reset();
        m_dispDist = -1.0;
    }

    void applyAutoCalibration(const CameraCalibration& c) {
        m_calib = c;
        m_lanes.reset();
        m_vehicles.reset();
        m_dispDist = -1.0;
        std::lock_guard<std::mutex> lk(m_calibMtx);
        m_pendingCalib = c;
        m_calibPending = true;
        m_calibSoft = true;
    }

    void startOnnxThread() {
        if (m_modelPath.empty()) {
            const char* env_model = getenv("ADAS_MODEL_PATH");
            if (env_model) m_modelPath = env_model;
        }
        struct stat st {};
        if (!m_modelPath.empty() && (stat(m_modelPath.c_str(), &st) != 0 || !S_ISDIR(st.st_mode))) {
            if (m_modelPath != "option_b" && m_modelPath != "dual")
                qWarning("[AI Engine] Model dir '%s' khong ton tai -> dung thu muc mac dinh", m_modelPath.c_str());
            m_modelPath.clear();
        }
        qInfo("[AI Engine] NanoDet-Plus, model dir: %s",
              m_modelPath.empty() ? "(mac dinh)" : m_modelPath.c_str());

        m_onnxQueue.reset();
        m_onnxThread = std::thread([this] { onnxInferLoop(); });
    }

    void onnxInferLoop() {
        adas_pin_to_cores(adas_env_cores("ADAS_AI_CORES", {2, 3}));
        (void)!nice(5);
        pthread_setname_np(pthread_self(), "adas-ai");
        qInfo("[AI Engine] Deep Path thread started on Core 2+3");

        std::unique_ptr<DualEnginePipeline> dualEngine;
        {
            try {
                DualEnginePipeline::Config cfg;
                std::vector<std::string> dirs = {"/opt/adas/models", "../models", "models"};
                if (!m_modelPath.empty()) dirs.insert(dirs.begin(), m_modelPath);
                for (const auto& d : dirs)
                    if (cfg.det.onnx_path.empty()) cfg.det.onnx_path = adas_model_file(d, "nanodet-plus-m_320");
                cfg.det.allow_simulation = false;
                auto parseSize = [](const char* env, int& w, int& h) {
                    const char* v = getenv(env);
                    int a = 0, b = 0;
                    if (v && std::sscanf(v, "%dx%d", &a, &b) == 2 && a >= 64 && b >= 64 && a % 32 == 0 && b % 32 == 0) { w = a; h = b; }
                };
                if (const char* e = getenv("ADAS_DET_HZ")) cfg.det_hz = std::max(0.5, atof(e));
                if (const char* e = getenv("ADAS_DET_THREADS")) cfg.det.num_threads = std::clamp(atoi(e), 1, 4);
                parseSize("ADAS_DET_SIZE", cfg.det.input_w, cfg.det.input_h);
                {
                    std::lock_guard<std::mutex> lk(m_calibMtx);
                    cfg.calib = m_pendingCalib;
                    m_calibPending = false;
                }
                dualEngine = std::make_unique<DualEnginePipeline>(cfg);
                m_onnxReady.store(dualEngine->ready());
                if (dualEngine->ready()) qInfo("[AI Engine] DualEngine ready (det %d)", dualEngine->detReady());
                else qWarning("[AI Engine] DualEngine: KHONG co model nao duoc nap -> chi chay fallback OpenCV");
            } catch (const std::exception& e) {
                qWarning("[AI Engine] DualEngine init error: %s", e.what());
                return;
            }
        }

        while (m_running.load()) {
            cv::Mat frame;
            const auto tw = std::chrono::steady_clock::now();
            if (!m_onnxQueue.pop(frame, 500)) continue;
            m_aiIdleMs.store(0.8 * m_aiIdleMs.load() + 0.2 * std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count());
            const int64_t frameNo = m_onnxFrame.load(std::memory_order_relaxed);
            try {
                if (dualEngine) {
                    {
                        std::lock_guard<std::mutex> lk(m_calibMtx);
                        if (m_calibPending) {
                            if (m_calibSoft) dualEngine->updateCalibration(m_pendingCalib);
                            else dualEngine->setCalibration(m_pendingCalib);
                            m_calibPending = false;
                        }
                    }
                    auto r = dualEngine->step(frame);
                    std::lock_guard<std::mutex> lk(m_fusionMtx);
                    if (r.det_fresh) {
                        m_cachedDetMs = r.det_ms;
                        m_cachedVehicles = std::move(r.vehicles);
                        m_cachedDetFrame = frameNo;
                        ++m_cachedDetSeq;
                    }
                    m_cachedHorizon = r.horizon_y;
                    m_cachedHood = r.hood_y;
                    m_cachedNight = r.night;
                    m_haveGeom = true;
                    m_cachedOnnxAge = 0;
                    m_cachedOnnxMs = r.cycle_ms;
                }
                m_onnxLastResultNs.store(nowNs(), std::memory_order_relaxed);
            } catch (const std::exception& e) {
                qWarning("[DeepPath] Infer error: %s", e.what());
            } catch (...) {
                qWarning("[DeepPath] Infer error (non-std exception)");
            }
            m_onnxBusy.store(false, std::memory_order_relaxed);
        }
        qInfo("[DeepPath] Deep Path thread stopped.");
    }

    PerceptionOutput processFrameFastPath(const cv::Mat& frame, double fps,
                                          cv::Mat& prev_gray, double& est_speed,
                                          std::vector<VehicleDetection>& out_fast_dets) {
        (void)prev_gray; (void)est_speed;
        PerceptionOutput out;
        out.fps = fps;
        const int w = frame.cols, h = frame.rows;

        cv::Mat smallL, grayL;
        cv::resize(frame, smallL, cv::Size(64, 36), 0, 0, cv::INTER_AREA);
        cv::cvtColor(smallL, grayL, cv::COLOR_BGR2GRAY);
        const int lightMean = static_cast<int>(cv::mean(grayL)[0]);
        if (lightMean < 65) out.envLighting = "Ban dem / Thieu sang";
        else if (lightMean > 185) out.envLighting = "Nang gat / Choi sang";
        else out.envLighting = "Ban ngay (Chuan)";
        out.envBrightness = lightMean;

        const double ego = egoMps();
        out.speed = ego >= 0 ? static_cast<int>(std::round(ego * 3.6)) : 0;

        if (!m_aiHealthy) {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            int left_x = static_cast<int>(w * 0.20), right_x = static_cast<int>(w * 0.80);
            if (m_lastLane.has_lane && !m_lastLane.left_img.empty() && !m_lastLane.right_img.empty()) {
                left_x = std::clamp(m_lastLane.left_img.front().x, 0, w - 1);
                right_x = std::clamp(m_lastLane.right_img.front().x, 0, w - 1);
            }
            const double hz = m_lastGeom.valid() ? m_lastGeom.vh : m_calib.horizon_y_ratio * h;
            out_fast_dets = m_fastVehicleDetector.detect(gray, hz, left_x, right_x);
            for (auto& d : out_fast_dets) d.box.conf = std::min(d.box.conf, 0.5f);
        }
        return out;
    }

    void carryLateDetections(const cv::Mat& frame, std::vector<VehicleDetection>* lateDets, int64_t detFrame) {
        if (!m_flowOn || frame.empty()) return;
        const auto t0 = std::chrono::steady_clock::now();
        if (lateDets && detFrame >= 0) m_frameHist.carry(*lateDets, detFrame);
        m_flowMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    void applyTracking(PerceptionOutput& perc, const cv::Mat& frame, const std::vector<VehicleDetection>& fast_dets) {
        const int W = frame.cols, H = frame.rows;
        bool detFresh = false, haveGeom = false, night = false;
        std::vector<VehicleDetection> dets;
        int64_t detFrame = -1;
        double hz = 0.0;
        int hood = 0, age = 999;
        {
            std::lock_guard<std::mutex> lk(m_fusionMtx);
            age = ++m_cachedOnnxAge;
            m_aiHealthy = m_onnxReady.load() && age < kOnnxStaleFrames;
            if (m_cachedDetSeq != m_lastDetSeqUsed) {
                m_lastDetSeqUsed = m_cachedDetSeq;
                detFresh = true;
                dets = m_cachedVehicles;
                detFrame = m_cachedDetFrame;
            }
            haveGeom = m_haveGeom;
            hz = m_cachedHorizon;
            hood = m_cachedHood;
            night = m_cachedNight;
        }
        perc.onnxAge = age;
        perc.onnxActive = m_aiHealthy;
        {
            const double tn = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
            m_detEvents += detFresh;
            if (m_rateT0 < 0) m_rateT0 = tn;
            if (tn - m_rateT0 >= 1.0) {
                m_detHz = m_detEvents / (tn - m_rateT0);
                m_detEvents = 0; m_rateT0 = tn;
                std::lock_guard<std::mutex> lk(m_fusionMtx);
                m_detMs = m_cachedDetMs;
            }
        }
        if (!haveGeom || hz <= 0) {
            hz = m_calib.horizon_y_ratio * H;
            hood = static_cast<int>(m_calib.hood_y_ratio * H);
        }
        const RoadGeometry g = RoadGeometry::make(m_calib, W, H, hz, hood);
        m_lastGeom = g;
        const double t = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();

        m_lanes.predict(t);
        cv::Mat calMask;
        {
            laneMarkingMask(frame, g, m_markMask, m_markGray, m_vehBoxes);
            cv::Mat fused = m_markMask;
            m_staticFilter.apply(fused, m_staticMask, g, t,
                                 m_calMgr.status().state == CalibrationManager::State::Calibrated);
            fused = m_staticMask;
            m_lanes.setGeometryCalibrated(m_calMgr.status().state == CalibrationManager::State::Calibrated);
            m_lanes.update(fused, true, g);
            calMask = fused;
            if (m_debugView) m_lastLaneMaskDbg = fused;
        }
        const LaneOutput lo = m_lanes.output(g, t, m_turnLeft.load(), m_turnRight.load());
        m_lastLane = lo;

        {
            const auto t0 = std::chrono::steady_clock::now();
            m_frameHist.push(m_frameCounter, frame);
            const EgoOutput& e = m_ego.update(t, m_frameHist.newest(), m_frameHist.scale(), g, m_vehBoxes, egoMps());
            m_flowMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            m_lastEgo = e;
            perc.egoValid = e.valid;
            perc.yawRate = e.yawDegS();
            perc.egoTurn = e.turn;
            perc.pathCurv = e.curvature;
            const bool lanesSeen = lo.has_lane && !lo.held;
            perc.roadBend = (e.turn != 0 || !lanesSeen) ? e.curvature
                          : lo.road_shape != 0 && lo.curve_radius_m > 0 ? lo.road_shape / lo.curve_radius_m : 0.0;
            perc.pathLeft.clear(); perc.pathRight.clear();
            if (e.valid && (e.turn != 0 || !lanesSeen)) {
                const double len = e.speed_mps > 0 ? std::clamp(e.speed_mps * 3.0, 10.0, 40.0) : 25.0;
                const auto c = e.path(len, 0.5);
                for (size_t i = 1; i < c.size(); ++i) {
                    const double th = std::atan2(c[i].x - c[i - 1].x, c[i].y - c[i - 1].y);
                    const double hw = m_lanes.params().car_half_width_m;
                    for (int side = -1; side <= 1; side += 2) {
                        const double x = g.car_x + c[i].x + side * hw * std::cos(th), z = c[i].y - side * hw * std::sin(th);
                        if (z < g.nearZ()) continue;
                        const cv::Point2f q = g.toImage(x, z);
                        (side < 0 ? perc.pathLeft : perc.pathRight).emplace_back(cvRound(q.x), cvRound(q.y));
                    }
                }
            }
            m_vehicles.setPathCurvature(e.valid ? e.curvature : 0.0);
        }

        perc.laneStatus = lo.status;
        perc.laneText = lo.text;
        perc.alertLevel = lo.alert_level;
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << (lo.offset_m >= 0 ? "+" : "") << (lo.has_lane ? lo.offset_m : 0.0) << "m";
            perc.laneOffset = oss.str();
        }
        perc.leftLaneLine = lo.left_img;
        perc.rightLaneLine = lo.right_img;
        perc.drivablePoly = lo.drivable_img;
        perc.roadShape = lo.road_shape;
        perc.curveRadius = lo.curve_radius_m;
        perc.latVel = lo.lat_vel_mps;
        perc.offsetM = lo.offset_m;
        perc.hasLane = lo.has_lane;

        perc.horizonY = g.vh;
        perc.vanishingPointX = g.cx;
        perc.pitchDeg = std::atan2(g.vh - m_calib.horizon_y_ratio * H, g.f) * 180.0 / M_PI;
        perc.hasHorizon = true;
        perc.horizonSource = haveGeom ? "GATED" : "CALIBRATED";
        if (night) perc.envLighting = "Ban dem / Thieu sang";

        const std::vector<VehicleDetection>* in = nullptr;
        if (detFresh) {
            const bool logNow = t - m_lastDetLogT >= 2.0 && getenv("ADAS_DET_LOG");
            if (logNow) m_lastDetLogT = t;
            dets = filterDetections(dets, g, &m_detStats, logNow);
            in = &dets;
            m_lastRawDets = dets;
            for (const auto& d : dets)
                m_calMgr.addVehicle(cv::Rect2f(d.box.x1, d.box.y1, d.box.x2 - d.box.x1, d.box.y2 - d.box.y1),
                                     d.box.cls, H, g.hood);
        }
        else if (!m_aiHealthy && !fast_dets.empty()) {
            m_cvDets = filterDetections(fast_dets, g, &m_detStats, false);
            if (lo.has_lane) {
                const double centre = g.car_x - lo.offset_m, reach = 1.5 * lo.width_m;
                m_cvDets.erase(std::remove_if(m_cvDets.begin(), m_cvDets.end(), [&](const VehicleDetection& d) {
                    double x, z;
                    return !g.toGround(0.5 * (d.box.x1 + d.box.x2), d.box.y2, x, z) || std::abs(x - centre) > reach;
                }), m_cvDets.end());
            }
            in = &m_cvDets; m_lastRawDets = m_cvDets;
        }
        const double ego = egoMps();
        carryLateDetections(frame, in == &dets ? &dets : nullptr, detFrame);
        const auto& tracks = m_vehicles.step(t, in, g, lo, ego);
        const LeadOutput& ld = m_vehicles.lead();

        perc.allVehicles.clear();
        perc.inLaneVehicles = 0;
        m_vehBoxes.clear();
        for (const auto& d : m_lastRawDets)
            m_vehBoxes.emplace_back(cv::Point(cvRound(d.box.x1) - 4, cvRound(d.box.y1) - 4),
                                    cv::Point(cvRound(d.box.x2) + 4, cvRound(d.box.y2) + 4));
        for (const auto& v : tracks) {
            m_vehBoxes.emplace_back(cvRound(v.box.x), cvRound(v.box.y), cvRound(v.box.width), cvRound(v.box.height));
            DetectedVehicle dv;
            dv.box = cv::Rect(cvRound(v.box.x), cvRound(v.box.y), cvRound(v.box.width), cvRound(v.box.height));
            dv.conf = v.conf * 100.0f;
            dv.cls = v.cls;
            dv.dist = v.z;
            dv.trackId = v.id;
            dv.lanePos = v.lane_pos;
            dv.vRel = v.vz * 3.6;
            dv.isCutIn = v.cut_in;
            dv.isLead = ld.has_lead && v.id == ld.lead.id;
            dv.trail = v.trail;
            dv.inPath = v.in_path;
            dv.x = v.x;
            dv.widthM = v.width_m;
            if (v.in_path) ++perc.inLaneVehicles;
            perc.allVehicles.push_back(std::move(dv));
        }

        if (ld.has_lead) {
            const auto& L = ld.lead;
            perc.hasLead = true;
            perc.hasLeadBox = true;
            perc.leadBox = cv::Rect(cvRound(L.box.x), cvRound(L.box.y), cvRound(L.box.width), cvRound(L.box.height));
            perc.leadTrackId = L.id;
            perc.leadConf = L.conf * 100.0;
            if (m_dispDist < 0 || std::abs(L.z - m_dispDist) >= 0.2) m_dispDist = std::round(L.z * 10.0) / 10.0;
            perc.leadDist = m_dispDist;
        } else {
            m_dispDist = -1.0;
            perc.hasLead = false;
            perc.hasLeadBox = false;
            perc.leadDist = -1.0;
            perc.leadConf = 0.0;
            perc.leadTrackId = 0;
        }

        {
            double spd = egoMps();
            if (spd < 0 && m_assumeSpeedMps > 0) spd = m_assumeSpeedMps;
            CameraCalibration c = m_calib;
            const auto& lp = m_lanes.params();
            if (m_calMgr.step(t, frame, g, lo, calMask, calMask.empty() ? cv::Mat() : m_lanes.bev(), lp.res_z, lp.z_max,
                              spd, m_turnLeft.load() || m_turnRight.load() || m_lastEgo.turn != 0 ||
                                       std::abs(m_lastEgo.yawDegS()) > 3.0, c))
                applyAutoCalibration(c);
            const auto& cs = m_calMgr.status();
            perc.calibText = cs.text;
            perc.calibProgress = cs.progress;
            perc.calibrated = cs.state == CalibrationManager::State::Calibrated;
            perc.horizonSource = perc.calibrated ? "AUTO-CALIB" : cs.text;
        }

        perc.threatLevel = ld.fcw_level;
        perc.warningText = ld.fcw_level > 0 ? ld.fcw_text : (lo.alert_level > 0 ? lo.text : std::string("AN TOAN"));
    }

    void startTrafficSigns() {
        const char* on = getenv("ADAS_TSR");
        if (on && on[0] == '0') { qInfo("[TSR] tat (ADAS_TSR=0)"); return; }
        TrafficSignEngine::Config cfg;
        const char* mp = getenv("ADAS_TSR_MODEL");
        if (mp && *mp) cfg.onnx_path = mp;
        std::vector<std::string> dirs = {"/opt/adas/models", "../models", "models"};
        if (!m_modelPath.empty()) dirs.insert(dirs.begin(), m_modelPath);
        for (const auto& d : dirs)
            if (cfg.onnx_path.empty()) cfg.onnx_path = adas_model_file(d, "tsr_640x224");
        if (cfg.onnx_path.empty()) { qInfo("[TSR] khong co models/tsr_640x224.onnx -> tat nhan dien bien bao"); return; }
        const char* th = getenv("ADAS_TSR_THREADS");
        cfg.num_threads = th ? std::clamp(atoi(th), 1, 4) : 2;
        const char* cf = getenv("ADAS_TSR_CONF");
        if (cf) cfg.conf_thresh = float(atof(cf));
        std::string valuePath;
        for (const auto& d : dirs)
            if (valuePath.empty() && access((d + "/speed_value.onnx").c_str(), R_OK) == 0) valuePath = d + "/speed_value.onnx";
        if (!m_tsr.start(cfg, adas_env_cores("ADAS_TSR_CORES", {2, 3}), m_cvOnly ? 5 : 10, valuePath)) qWarning("[TSR] khong nap duoc model %s", cfg.onnx_path.c_str());
        if (m_followHz > 0 && !valuePath.empty()) m_followReader = std::make_unique<SpeedValueReader>(valuePath);
        m_signs.requireSpeedValue(!valuePath.empty());
    }

    void stepTrafficSigns(const cv::Mat& frame, PerceptionOutput& perc) {
        if (!m_tsr.ready() || frame.empty()) return;
        const double t = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        bool fresh = false;
        if (m_followReader && m_followReader->ready() && t - m_followLastT >= 1.0 / m_followHz - 1e-3) {
            m_followFrames.emplace_back(t, frame.clone());
            while (t - m_followFrames.front().first > 1.5) m_followFrames.pop_front();
            m_followLastT = t; fresh = true;
        }
        TrafficSignRunner::Result r;
        if (m_tsr.poll(r, m_tsrSeq)) {
            m_tsrSeq = r.seq; m_tsrMs = r.ms;
            m_signs.update(r.dets, r.t);
            if (m_followReader)
                for (const auto& [ft, f] : m_followFrames) if (ft > r.t) m_signs.follow(f, ft, *m_followReader);
            fresh = false;
            if (m_tsrRateT0 < 0) m_tsrRateT0 = t;
            if (++m_tsrRuns, t - m_tsrRateT0 >= 3.0) { m_tsrHz = m_tsrRuns / (t - m_tsrRateT0); m_tsrRuns = 0; m_tsrRateT0 = t; }
        }
        if (fresh) m_signs.follow(m_followFrames.back().second, t, *m_followReader);
        const int period = m_signs.speedPending(t) ? m_tsrFastMs : m_tsrPeriodMs;
        if (!m_tsr.busy() && (t - m_tsrLastSubmit) * 1000.0 >= period) {
            const float sc = float(frame.cols) / m_tsr.inputW();
            const int bh = std::min(frame.rows, int(std::lround(m_tsr.inputH() * sc)));
            const double hz = perc.horizonY > 0 ? perc.horizonY : frame.rows * m_calib.horizon_y_ratio;
            const int y0 = std::clamp(int(std::lround(hz - 0.65 * bh)), 0, frame.rows - bh);
            cv::Mat band = frame(cv::Rect(0, y0, frame.cols, bh)).clone();
            m_tsrBand[0] = y0; m_tsrBand[1] = y0 + bh;
            m_tsr.submit(std::move(band), 0, y0, 1.f, t);
            m_tsrLastSubmit = t;
        }
        perc.signs = m_signs.visible(t);
        perc.activeSign = m_signs.active(t);
        perc.speedSign = m_signs.speedSign();
        perc.speedValue = m_signs.speedValue();
        if (perc.speedValue > 0) perc.speedLimit = perc.speedValue;
        perc.auxSign = m_signs.auxSign(t);
    }

    void resetSmoothState() {
        m_signs.reset();
        m_followFrames.clear();
        m_lanes.reset();
        m_vehBoxes.clear();
        m_lastLaneMaskDbg.release();
        m_vehicles.reset();
        m_lastLane = LaneOutput();
        m_ego.reset(); m_lastEgo = EgoOutput(); m_frameHist.clear();
        m_lastRawDets.clear();
        m_dispDist = -1.0;
        std::lock_guard<std::mutex> lk(m_fusionMtx);
        m_cachedVehicles.clear();
        m_cachedOnnxAge = 999;
        m_lastDetSeqUsed = m_cachedDetSeq;
        m_haveGeom = false;
    }

    std::string formatTelemetryJson(const PerceptionOutput& p, int imgW = 640, int imgH = 480) {
        std::ostringstream ss;
        ss << "{"
           << "\"imgW\":" << imgW << ","
           << "\"imgH\":" << imgH << ","
           << "\"fps\":" << p.fps << ","
           << "\"speed\":" << p.speed << ","
           << "\"limit\":" << p.speedLimit << ","
           << "\"dist\":" << p.leadDist << ","
           << "\"threat\":" << p.threatLevel << ","
           << "\"alert\":" << p.alertLevel << ","
           << "\"lane\":\"" << p.laneStatus << "\","
           << "\"offset\":\"" << p.laneOffset << "\","
           << "\"hasLead\":" << (p.hasLead ? "true" : "false") << ","
           << "\"leadTrackId\":" << p.leadTrackId << ","
           << "\"warn\":\"" << p.warningText << "\","
           << "\"laneText\":\"" << p.laneText << "\","
           << "\"inLane\":" << p.inLaneVehicles << ","
           << "\"hoodY\":" << m_lastGeom.hood << ",";

        if (p.hasLeadBox) {
            ss << "\"leadBox\":[" << p.leadBox.x << "," << p.leadBox.y << "," << p.leadBox.width << "," << p.leadBox.height << "],";
        } else {
            ss << "\"leadBox\":null,";
        }

        ss << "\"hzY\":" << p.horizonY << ","
           << "\"vpX\":" << p.vanishingPointX << ","
           << "\"pitch\":" << p.pitchDeg << ",";

        ss << "\"poly\":[";
        for (size_t i = 0; i < p.drivablePoly.size(); ++i) {
            if (i > 0) ss << ",";
            ss << "[" << p.drivablePoly[i].x << "," << p.drivablePoly[i].y << "]";
        }
        ss << "],";

        ss << "\"leftLane\":[";
        for (size_t i = 0; i < p.leftLaneLine.size(); ++i) {
            if (i > 0) ss << ",";
            ss << "[" << p.leftLaneLine[i].x << "," << p.leftLaneLine[i].y << "]";
        }
        ss << "],";

        ss << "\"rightLane\":[";
        for (size_t i = 0; i < p.rightLaneLine.size(); ++i) {
            if (i > 0) ss << ",";
            ss << "[" << p.rightLaneLine[i].x << "," << p.rightLaneLine[i].y << "]";
        }
        ss << "],";
        for (int side = 0; side < 2; ++side) {
            const auto& pl = side ? p.pathRight : p.pathLeft;
            ss << (side ? "\"pathR\":[" : "\"pathL\":[");
            for (size_t i = 0; i < pl.size(); ++i) ss << (i ? "," : "") << "[" << pl[i].x << "," << pl[i].y << "]";
            ss << "],";
        }

        ss << "\"vehs\":[";
        bool firstV = true;
        for (const auto& v : p.allVehicles) {
            if (v.isLead) continue;
            if (!firstV) ss << ",";
            firstV = false;
            ss << "{"
               << "\"id\":" << v.trackId << ","
               << "\"box\":[" << v.box.x << "," << v.box.y << "," << v.box.width << "," << v.box.height << "],"
               << "\"dist\":" << v.dist << ","
               << "\"cutIn\":" << (v.isCutIn ? "true" : "false") << ","
               << "\"pos\":\"" << v.lanePos << "\""
               << "}";
        }
        ss << "],";

        ss << "\"rawVehs\":[";
        for (size_t i = 0; i < m_lastRawDets.size(); ++i) {
            const auto& b = m_lastRawDets[i].box;
            if (i > 0) ss << ",";
            ss << "{\"box\":[" << b.x1 << "," << b.y1 << "," << (b.x2 - b.x1) << "," << (b.y2 - b.y1)
               << "],\"conf\":" << b.conf << ",\"cls\":\"" << vehicleClassName(b.cls) << "\"}";
        }
        ss << "],"
           << "\"laneConfL\":" << m_lastLane.conf_left << ","
           << "\"laneConfR\":" << m_lastLane.conf_right << ","
           << "\"laneWidth\":" << m_lastLane.width_m << ","
           << "\"lanePts\":[" << m_lastLane.pts_left << "," << m_lastLane.pts_right << "],"

           << "\"roadShape\":" << m_lastLane.road_shape << ","
           << "\"latVel\":" << m_lastLane.lat_vel_mps << ","
           << "\"curveR\":" << m_lastLane.curve_radius_m << ","
           << "\"ego\":{\"valid\":" << (m_lastEgo.valid ? "true" : "false") << ",\"yaw\":" << m_lastEgo.yawDegS()
           << ",\"v\":" << m_lastEgo.speed_mps << ",\"vSrc\":" << m_lastEgo.speed_src << ",\"curv\":" << m_lastEgo.curvature
           << ",\"turn\":" << m_lastEgo.turn << ",\"pts\":" << m_lastEgo.inliers << "},"
           << "\"cvOnly\":" << (m_cvOnly ? "true" : "false") << ","
           << "\"detFilter\":[" << m_detStats.in << "," << m_detStats.kept << "],"
           << "\"aiAge\":" << p.onnxAge << ","
           << "\"hzSource\":\"" << p.horizonSource << "\","
           << "\"calib\":{\"text\":\"" << m_calMgr.status().text << "\",\"progress\":" << m_calMgr.status().progress
           << ",\"phase\":" << m_calMgr.status().phase << ",\"validS\":" << m_calMgr.status().valid_s
           << ",\"vp\":" << m_calMgr.status().vp << ",\"flow\":" << m_calMgr.status().flow
           << ",\"hood\":" << m_calMgr.status().hood << ",\"gate\":" << (m_calMgr.status().gate_ok ? "true" : "false") << "},";
        ss << "\"obd\":" << m_obd.json() << ",\"video\":" << (m_isTestVideo ? "true" : "false")
           << ",\"cam\":\"" << m_camState << "\",";
        ss << "\"calibCfg\":{\"horizon\":" << m_calib.horizon_y_ratio << ",\"vpX\":" << m_calib.vanish_x_ratio
           << ",\"hood\":" << m_calib.hood_y_ratio << ",\"camH\":" << m_calib.camera_height_m
           << ",\"latOff\":" << m_calib.camera_lateral_offset_m << ",\"hfov\":" << m_calib.hfov_deg << "},";
        ss << "\"signs\":[";
        for (size_t i = 0; i < p.signs.size(); ++i) {
            const auto& g = p.signs[i];
            ss << (i ? "," : "") << "[" << int(g.box.x) << "," << int(g.box.y) << "," << int(g.box.width) << ","
               << int(g.box.height) << "," << g.cls << "," << std::fixed << std::setprecision(2) << g.conf << "," << g.value << "]";
        }
        ss.unsetf(std::ios::floatfield); ss << std::setprecision(6);
        ss << "],\"sign\":" << p.activeSign << ",\"speedSign\":" << p.speedSign << ",\"speedValue\":" << p.speedValue
           << ",\"tsr\":{\"on\":" << (m_tsr.ready() ? "true" : "false") << ",\"hz\":" << m_tsrHz << ",\"ms\":" << m_tsrMs
           << ",\"band\":[" << m_tsrBand[0] << "," << m_tsrBand[1] << "]},";
        ss << "\"ai\":{\"detHz\":" << m_detHz << ",\"detMs\":" << m_detMs << "},";
        auto model = [&](const LaneModel& m, bool valid, bool synth) {
            ss << "[" << m.c0 << "," << m.c1 << "," << m.c2 << "," << (valid ? 1 : 0) << "," << (synth ? 1 : 0) << "]";
        };
        const LaneOutput& lo = m_lastLane;
        ss << "\"bev\":{\"L\":"; model(lo.left, lo.has_lane && lo.left_valid, lo.left_synth);
        ss << ",\"R\":"; model(lo.right, lo.has_lane && lo.right_valid, lo.right_synth);
        ss << ",\"zNear\":" << (m_lastGeom.valid() ? m_lastGeom.nearZ() : 3.0) << ",\"carX\":" << m_lastGeom.car_x << ",\"vehs\":[";
        bool firstB = true;
        for (const auto& v : p.allVehicles) {
            if (v.dist <= 0) continue;
            if (!firstB) ss << ",";
            firstB = false;
            ss << "[" << v.x << "," << v.dist << "," << v.widthM << "," << v.trackId << "," << (v.isLead ? 1 : 0) << "," << (v.inPath ? 1 : 0) << "]";
        }
        ss << "]}}";
        return ss.str();
    }

    void draw_real_stream_overlay(cv::Mat& img, const PerceptionOutput& p, bool hasCam, const std::string& pipelineDesc, bool isVideoTest) {
        int w = img.cols;
        int h = img.rows;

        if (p.drivablePoly.size() >= 4) {
            cv::Scalar carpetColor;
            cv::Scalar lineColor;
            if (p.alertLevel == 2) {
                carpetColor = cv::Scalar(0, 0, 220);
                lineColor = cv::Scalar(0, 0, 255);
            } else if (p.alertLevel == 1) {
                carpetColor = cv::Scalar(0, 200, 255);
                lineColor = cv::Scalar(0, 215, 255);
            } else {
                carpetColor = cv::Scalar(0, 190, 80);
                lineColor = cv::Scalar(50, 255, 120);
            }

            cv::Rect roi = cv::boundingRect(p.drivablePoly) & cv::Rect(0, 0, w, h);
            if (roi.width > 0 && roi.height > 0) {
                cv::Mat imgRoi = img(roi);
                cv::Mat carpetRoi = cv::Mat::zeros(roi.size(), img.type());
                std::vector<cv::Point> shiftedPoly;
                shiftedPoly.reserve(p.drivablePoly.size());
                for (const auto& pt : p.drivablePoly) {
                    shiftedPoly.push_back(cv::Point(pt.x - roi.x, pt.y - roi.y));
                }
                std::vector<std::vector<cv::Point>> polys = { shiftedPoly };
                cv::fillPoly(carpetRoi, polys, carpetColor);
                cv::addWeighted(imgRoi, 1.0, carpetRoi, 0.22, 0, imgRoi);
            }

            if (p.leftLaneLine.size() >= 2 && p.rightLaneLine.size() >= 2) {
                cv::polylines(img, std::vector<std::vector<cv::Point>>{p.leftLaneLine}, false, lineColor, 3, cv::LINE_AA);
                cv::polylines(img, std::vector<std::vector<cv::Point>>{p.rightLaneLine}, false, lineColor, 3, cv::LINE_AA);
            }
        }

        if (p.hasLane) {
            const int cx = w / 2, cy = h - 62;
            const int dir = p.alertLevel > 0 ? (p.laneStatus.find("LEFT") != std::string::npos ? -1 : 1)
                          : std::abs(p.latVel) > 0.25 ? (p.latVel < 0 ? -1 : 1) : 0;
            if (dir != 0) {
                const cv::Scalar ac = p.alertLevel == 2 ? cv::Scalar(0, 0, 255) : p.alertLevel == 1 ? cv::Scalar(0, 215, 255) : cv::Scalar(50, 255, 120);
                cv::arrowedLine(img, {cx - dir * 30, cy}, {cx + dir * 40, cy}, ac, 6, cv::LINE_AA, 0, 0.45);
                const char* dt = dir < 0 ? "DRIFTING LEFT" : "DRIFTING RIGHT";
                int bl = 0; const cv::Size ts = cv::getTextSize(dt, cv::FONT_HERSHEY_SIMPLEX, 0.55, 2, &bl);
                cv::putText(img, dt, {cx - ts.width / 2, cy - 16}, cv::FONT_HERSHEY_SIMPLEX, 0.55, ac, 2, cv::LINE_AA);
            }
            char ahead[64];
            if (p.roadShape == 0) snprintf(ahead, sizeof ahead, "AHEAD: STRAIGHT ROAD");
            else snprintf(ahead, sizeof ahead, "AHEAD: CURVE %s (R~%.0f m)", p.roadShape < 0 ? "LEFT" : "RIGHT", p.curveRadius);
            int bl = 0; const cv::Size ts = cv::getTextSize(ahead, cv::FONT_HERSHEY_SIMPLEX, 0.48, 1, &bl);
            const cv::Rect box(cx - ts.width / 2 - 6, cy + 10, ts.width + 12, ts.height + 10);
            if ((box & cv::Rect(0, 0, w, h)) == box) {
                cv::rectangle(img, box, cv::Scalar(15, 20, 30), cv::FILLED);
                cv::putText(img, ahead, {box.x + 6, box.y + ts.height + 4}, cv::FONT_HERSHEY_SIMPLEX, 0.48,
                            p.roadShape == 0 ? cv::Scalar(120, 255, 120) : cv::Scalar(0, 215, 255), 1, cv::LINE_AA);
            }
        }

        if (p.hasHorizon || p.horizonY > 0) {
            int yh = static_cast<int>(p.horizonY);
            int vpx = (p.vanishingPointX > 0) ? static_cast<int>(p.vanishingPointX) : (w / 2);
            if (yh >= static_cast<int>(h * 0.20) && yh <= static_cast<int>(h * 0.75)) {
                int dash = 14, gap = 8;
                for (int x = 0; x < w; x += dash + gap) {
                    if (std::abs(x - vpx) < 28) continue;
                    int x2 = std::min(w - 1, x + dash);
                    cv::line(img, cv::Point(x, yh), cv::Point(x2, yh), cv::Scalar(0, 215, 255), 1, cv::LINE_AA);
                }

                int r_sz = 9;
                cv::line(img, cv::Point(vpx - r_sz, yh), cv::Point(vpx + r_sz, yh), cv::Scalar(0, 240, 255), 1, cv::LINE_AA);
                cv::line(img, cv::Point(vpx, yh - r_sz), cv::Point(vpx, yh + r_sz), cv::Scalar(0, 240, 255), 1, cv::LINE_AA);
                cv::circle(img, cv::Point(vpx, yh), 2, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);

                char hz_txt[48];
                snprintf(hz_txt, sizeof(hz_txt), "HORIZON: Y=%d | PITCH: %+.1f*", yh, p.pitchDeg);
                int baseline = 0;
                cv::Size hz_sz = cv::getTextSize(hz_txt, cv::FONT_HERSHEY_SIMPLEX, 0.35, 1, &baseline);
                int tag_x = std::min(w - hz_sz.width - 12, std::max(10, vpx + 14));
                cv::putText(img, hz_txt, cv::Point(tag_x, yh - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 220, 255), 1, cv::LINE_AA);
            }
        }

        for (const auto& veh : p.allVehicles) {
            if (veh.trail.size() >= 2) {
                cv::Scalar tCol = veh.isLead ?
                    ((p.threatLevel == 2) ? cv::Scalar(0, 0, 255) : (p.threatLevel == 1 ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 120))) :
                    (veh.isCutIn ? cv::Scalar(0, 140, 255) : (veh.lanePos == "IN_LANE" ? cv::Scalar(0, 220, 255) : cv::Scalar(255, 185, 40)));

                for (size_t k = 1; k < veh.trail.size(); ++k) {
                    cv::line(img, veh.trail[k - 1], veh.trail[k], tCol, (k > veh.trail.size() / 2) ? 2 : 1, cv::LINE_AA);
                }
            }
        }

        for (const auto& veh : p.allVehicles) {
            if (veh.isLead) continue;

            cv::Scalar boxCol = (veh.isCutIn) ? cv::Scalar(0, 140, 255) :
                               ((veh.lanePos == "IN_LANE") ? cv::Scalar(0, 220, 255) : cv::Scalar(255, 185, 40));
            cv::rectangle(img, veh.box, boxCol, (veh.isCutIn || veh.lanePos == "IN_LANE") ? 2 : 1);

            cv::circle(img, cv::Point(veh.box.x + veh.box.width / 2, veh.box.y + veh.box.height), 3, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);

            char vtxt[64];
            if (veh.isCutIn) {
                snprintf(vtxt, sizeof(vtxt), "TẠT ĐẦU #%d: %.1fm", veh.trackId, veh.dist);
            } else if (veh.lanePos == "IN_LANE") {
                snprintf(vtxt, sizeof(vtxt), "CÙNG LÀN #%d: %.1fm", veh.trackId, veh.dist);
            } else {
                const char* pos = (veh.lanePos == "LEFT_LANE") ? "TRÁI" : ((veh.lanePos == "RIGHT_LANE") ? "PHẢI" : "XE");
                snprintf(vtxt, sizeof(vtxt), "#%d %s: %.1fm", veh.trackId, pos, veh.dist);
            }

            int baseline = 0;
            cv::Size tsz = cv::getTextSize(vtxt, cv::FONT_HERSHEY_SIMPLEX, 0.40, 1, &baseline);
            int label_y = std::max(tsz.height + 6, veh.box.y - 4);
            cv::rectangle(img, cv::Rect(veh.box.x, label_y - tsz.height - 4, tsz.width + 6, tsz.height + 4), cv::Scalar(15, 23, 42), cv::FILLED);
            cv::rectangle(img, cv::Rect(veh.box.x, label_y - tsz.height - 4, tsz.width + 6, tsz.height + 4), boxCol, 1);
            cv::putText(img, vtxt, cv::Point(veh.box.x + 3, label_y - 2), cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(230, 240, 255), 1, cv::LINE_AA);
        }

        if (p.hasLeadBox) {
            cv::Scalar boxCol = (p.threatLevel == 2) ? cv::Scalar(0, 0, 255) :
                               (p.threatLevel == 1) ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0);
            cv::Scalar tagBg = (p.threatLevel == 2) ? cv::Scalar(0, 0, 220) :
                              (p.threatLevel == 1) ? cv::Scalar(0, 140, 230) : cv::Scalar(0, 180, 50);

            cv::rectangle(img, p.leadBox, boxCol, 3);
            int corner_len = std::max(8, std::min(20, p.leadBox.width / 4));
            int bx = p.leadBox.x, by = p.leadBox.y, bw = p.leadBox.width, bh = p.leadBox.height;
            cv::line(img, {bx, by}, {bx + corner_len, by}, boxCol, 4);
            cv::line(img, {bx, by}, {bx, by + corner_len}, boxCol, 4);
            cv::line(img, {bx + bw, by}, {bx + bw - corner_len, by}, boxCol, 4);
            cv::line(img, {bx + bw, by}, {bx + bw, by + corner_len}, boxCol, 4);
            cv::line(img, {bx, by + bh}, {bx + corner_len, by + bh}, boxCol, 4);
            cv::line(img, {bx, by + bh}, {bx, by + bh - corner_len}, boxCol, 4);
            cv::line(img, {bx + bw, by + bh}, {bx + bw - corner_len, by + bh}, boxCol, 4);
            cv::line(img, {bx + bw, by + bh}, {bx + bw, by + bh - corner_len}, boxCol, 4);

            cv::circle(img, cv::Point(bx + bw / 2, by + bh), 4, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);

            int ego_x = w / 2, ego_y = static_cast<int>(h * 0.85);
            cv::line(img, {ego_x, ego_y}, {bx + bw / 2, by + bh}, boxCol, 2, cv::LINE_AA);

            char lead_txt[96];
            snprintf(lead_txt, sizeof(lead_txt), "LEAD #%d: %.1fm", p.leadTrackId, p.leadDist);

            int baseline = 0;
            cv::Size tsz = cv::getTextSize(lead_txt, cv::FONT_HERSHEY_SIMPLEX, 0.48, 2, &baseline);
            int label_y = std::max(tsz.height + 8, p.leadBox.y - 8);
            cv::rectangle(img, cv::Rect(p.leadBox.x, label_y - tsz.height - 6, tsz.width + 10, tsz.height + 6), tagBg, cv::FILLED);
            cv::putText(img, lead_txt, cv::Point(p.leadBox.x + 5, label_y - 2), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }

        int gw = 250, gh = 96;
        int gx = w - gw - 16, gy = 36;
        if (gx > 0 && gy + gh < h) {
            cv::Rect gRect(gx, gy, gw, gh);
            cv::Mat gRoi = img(gRect);
            cv::Mat gOverlay(gRect.size(), img.type(), cv::Scalar(15, 23, 42));
            cv::addWeighted(gRoi, 0.25, gOverlay, 0.75, 0, gRoi);
            cv::rectangle(img, gRect, cv::Scalar(248, 189, 56), 1);

            cv::putText(img, "ADAS TELEMETRY RADAR", cv::Point(gx + 10, gy + 17),
                        cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(248, 189, 56), 1, cv::LINE_AA);

            cv::Scalar ldwCol = (p.alertLevel == 0) ? cv::Scalar(120, 255, 0) : ((p.alertLevel == 1) ? cv::Scalar(0, 215, 255) : cv::Scalar(0, 0, 255));
            std::string off_text = "Lech tam: " + p.laneOffset;
            cv::putText(img, off_text, cv::Point(gx + 10, gy + 38),
                        cv::FONT_HERSHEY_SIMPLEX, 0.42, ldwCol, 1, cv::LINE_AA);

            cv::Scalar fcwCol = (p.threatLevel == 0) ? cv::Scalar(120, 255, 0) : ((p.threatLevel == 1) ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 0, 255));
            char fcw_text[64];
            if (p.hasLead) {
                snprintf(fcw_text, sizeof(fcw_text), "Xe #%d: %.1fm", p.leadTrackId, p.leadDist);
            } else {
                fcwCol = cv::Scalar(184, 163, 148);
                snprintf(fcw_text, sizeof(fcw_text), "Xe truoc: -- (Trong lan)");
            }
            cv::putText(img, fcw_text, cv::Point(gx + 10, gy + 58),
                        cv::FONT_HERSHEY_SIMPLEX, 0.42, fcwCol, 1, cv::LINE_AA);

            char hz_info[64];
            snprintf(hz_info, sizeof(hz_info), "Pitch: %+.1f* | Chan troi: %d", p.pitchDeg, static_cast<int>(p.horizonY));
            cv::putText(img, hz_info, cv::Point(gx + 10, gy + 78),
                        cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(0, 220, 255), 1, cv::LINE_AA);
        }

        cv::rectangle(img, cv::Rect(0, 0, w, 28), cv::Scalar(15, 20, 30), cv::FILLED);
        char osd[256];
        const char* onnx_tag = p.onnxActive ? " | AI:ON" : " | AI:OFF";
        cv::Scalar onnx_dot_col = p.onnxActive ? cv::Scalar(0, 230, 118) : cv::Scalar(80, 80, 80);
        cv::circle(img, cv::Point(w - 20, 14), 5, onnx_dot_col, -1);

        if (isVideoTest) {
            snprintf(osd, sizeof(osd), "[VIDEO TEST] | %.1f FPS | %d km/h | %s | %s | %s%s",
                     p.fps, p.speed, p.laneOffset.c_str(), p.warningText.c_str(), p.envLighting.c_str(), onnx_tag);
            cv::putText(img, osd, cv::Point(16, 19), cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(0, 229, 255), 1, cv::LINE_AA);
        } else if (hasCam) {
            snprintf(osd, sizeof(osd), "[LIVE CAM] | %.1f FPS | %d km/h | %s | %s | %s%s",
                     p.fps, p.speed, p.laneOffset.c_str(), p.warningText.c_str(), p.envLighting.c_str(), onnx_tag);
            cv::putText(img, osd, cv::Point(16, 19), cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(0, 230, 118), 1, cv::LINE_AA);
        } else {
            snprintf(osd, sizeof(osd), "[SIMULATION] | %.1f FPS | SPI 3.5\": HUD%s", p.fps, onnx_tag);
            cv::putText(img, osd, cv::Point(16, 19), cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(50, 180, 255), 1, cv::LINE_AA);
        }

        int bar_h = 24;
        cv::Rect bottomBar(0, h - bar_h, w, bar_h);
        cv::Mat bRoi = img(bottomBar);
        cv::Mat bOverlay(bottomBar.size(), img.type(), cv::Scalar(11, 15, 25));
        cv::addWeighted(bRoi, 0.20, bOverlay, 0.80, 0, bRoi);

        char bottomInfo[128];
        snprintf(bottomInfo, sizeof(bottomInfo), "FRAME: #%d | %s | AS: %s (%d/255)",
                 m_frameCounter, m_cvOnly ? "CLASSIC CV (NO DEEP LEARNING)" : "NANODET + CLASSIC LANES ADAS",
                 p.envLighting.c_str(), p.envBrightness);
        cv::putText(img, bottomInfo, cv::Point(10, h - 7),
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(241, 245, 249), 1, cv::LINE_AA);

        const char* legend = "LUC: Lan xe | TRACK ID: Da theo vet";
        int b_base = 0;
        cv::Size legSz = cv::getTextSize(legend, cv::FONT_HERSHEY_SIMPLEX, 0.36, 1, &b_base);
        cv::putText(img, legend, cv::Point(w - legSz.width - 12, h - 7),
                    cv::FONT_HERSHEY_SIMPLEX, 0.36, cv::Scalar(148, 163, 184), 1, cv::LINE_AA);
    }

    cv::Mat generateSimulationFrame(int tick, double speed, double dist) {
        cv::Mat img(480, 640, CV_8UC3, cv::Scalar(30, 32, 35));
        cv::rectangle(img, cv::Rect(0, 0, 640, 220), cv::Scalar(65, 55, 45), cv::FILLED);

        std::vector<cv::Point> road_pts = {
            cv::Point(280, 220), cv::Point(360, 220), cv::Point(600, 480), cv::Point(40, 480)
        };
        std::vector<std::vector<cv::Point>> r_poly = { road_pts };
        cv::fillPoly(img, r_poly, cv::Scalar(42, 45, 48));

        int offset = (tick * 6) % 80;
        for (int y = 230 + offset; y < 480; y += 80) {
            int dash_h = std::min(40, 480 - y);
            cv::rectangle(img, cv::Rect(318, y, 4, dash_h), cv::Scalar(240, 240, 240), cv::FILLED);
        }
        return img;
    }

    bool onLoopError(const char* what, int& errors, bool& has_cam, cv::VideoCapture& cap) {
        ++errors;
        if (errors <= 3 || errors % 300 == 0) qWarning("[CameraWorker] loop error #%d: %s", errors, what);
        if (errors == 30) resetSmoothState();
        if (errors == 90 && has_cam && !m_isTestVideo) {
            qWarning("[CameraWorker] loi lap lai, mo lai camera");
            cap.release();
            has_cam = false;
            msleep(10);
            return true;
        }
        msleep(10);
        return false;
    }

    PerceptionOutput noCameraOutput(double fps) const {
        PerceptionOutput out;
        out.fps = fps;
        const double ego = egoMps();
        out.speed = ego >= 0 ? static_cast<int>(std::round(ego * 3.6)) : 0;
        out.laneStatus = "NO_LANE";
        out.laneOffset = "--";
        out.laneText = m_hadCamera ? "MAT CAMERA" : "CHO CAMERA";
        out.warningText = out.laneText;
        out.horizonSource = "NO_CAMERA";
        out.calibText = "";
        out.calibrated = true;
        return out;
    }

    cv::Mat generateNoCameraFrame(bool lost) {
        cv::Mat img(480, 640, CV_8UC3, cv::Scalar(15, 20, 28));
        cv::rectangle(img, cv::Rect(20, 20, 600, 440), cv::Scalar(30, 41, 59), 2);
        cv::putText(img, lost ? "[ MAT CAMERA ]" : "[ CHO CAMERA ]", cv::Point(lost ? 220 : 215, 220),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 191, 255), 2, cv::LINE_AA);
        cv::putText(img, "Canh bao lan / va cham tam tat. Dang do lai camera...", cv::Point(110, 270),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
        return img;
    }

    cv::Mat generateVideoNoticeFrame() {
        cv::Mat img(480, 640, CV_8UC3, cv::Scalar(15, 20, 28));
        cv::rectangle(img, cv::Rect(20, 20, 600, 440), cv::Scalar(30, 41, 59), 2);

        cv::putText(img, "[ CHE DO TEST VIDEO AI ]", cv::Point(170, 180), cv::FONT_HERSHEY_SIMPLEX, 0.70, cv::Scalar(0, 229, 255), 2, cv::LINE_AA);
        cv::putText(img, "CHUA CO VIDEO TEST HOAC DANG TAI LEN...", cv::Point(110, 230), cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
        cv::putText(img, "-> Chon hoac keo tha file video (.mp4) vao o ben phai", cv::Point(90, 275), cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(0, 230, 118), 1, cv::LINE_AA);
        cv::putText(img, "-> Hoac bam nut [Webcam] de quay lai camera truc tiep", cv::Point(90, 310), cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(255, 179, 0), 1, cv::LINE_AA);
        return img;
    }

    AdasController* m_ctrl;
    FrameProvider*  m_provider;
    std::string     m_source;
    std::atomic<bool> m_running;
    int             m_frameCounter;
    MjpegServer     m_mjpegServer;

    std::mutex      m_srcMtx;
    std::string     m_pendingSource;
    std::shared_ptr<std::promise<bool>> m_pendingPromise;
    std::atomic<bool> m_sourceChanged;
    bool            m_isTestVideo;
    std::string     m_currentVideoPath;
    std::atomic<bool> m_probingCamera{false};

    std::string     m_modelPath;
    int             m_onnxCadence;
    std::thread     m_onnxThread;
    std::atomic<bool> m_onnxReady;
    std::atomic<bool> m_onnxBusy;
    OnnxFrameQueue<cv::Mat> m_onnxQueue;

    std::mutex      m_fusionMtx;
    std::vector<VehicleDetection> m_cachedVehicles;
    int             m_cachedOnnxAge = 999;
    double          m_cachedOnnxMs = 0.0;
    double          m_cachedDetMs = 0.0;
    int             m_detEvents = 0;
    double          m_rateT0 = -1.0, m_detHz = 0.0, m_detMs = 0.0;
    uint64_t        m_cachedDetSeq = 0;
    int64_t         m_cachedDetFrame = -1;
    std::atomic<int64_t> m_onnxFrame{-1};
    FrameHistory    m_frameHist;
    double          m_flowMs = 0.0;
    EgoMotion       m_ego;
    EgoOutput       m_lastEgo;
    std::atomic<double> m_aiIdleMs{0.0};
    const bool      m_flowOn = [] { const char* e = getenv("ADAS_FLOW"); return !(e && e[0] == '0'); }();
    uint64_t        m_lastDetSeqUsed = 0;
    double          m_cachedHorizon = 0.0;
    int             m_cachedHood = 0;
    bool            m_cachedNight = false;
    bool            m_haveGeom = false;

    static constexpr int kOnnxStaleFrames = 45;
    static constexpr int kLaneStaleFrames = 60;
    bool            m_aiHealthy = false;
    std::vector<VehicleDetection> m_cvDets;
    std::atomic<int64_t> m_onnxLastResultNs{0};
    std::atomic<int64_t> m_lastLoopNs{0};

    CameraCalibration m_calib;
    std::mutex      m_calibMtx;
    CameraCalibration m_pendingCalib;
    bool            m_calibPending = false;
    bool            m_calibSoft = false;
    CalibrationManager m_calMgr;
    double          m_assumeSpeedMps = -1.0;
    std::string     m_calibSavePath;
    bool            m_forceRecal = false;

    ObdReader       m_obd;
    const char*     m_camState = "none";
    std::atomic<double> m_egoSpeedMps{-1.0};
    std::atomic<bool> m_turnLeft{false};
    std::atomic<bool> m_turnRight{false};

    LaneDetector    m_lanes;
    cv::Mat         m_markMask, m_markGray, m_lastLaneMaskDbg;
    StaticMaskFilter m_staticFilter;
    cv::Mat         m_staticMask;
    std::vector<cv::Rect> m_vehBoxes;
    DetFilterStats  m_detStats;
    double          m_lastDetLogT = 0.0;
    const bool      m_debugView = [] { const char* e = getenv("ADAS_DEBUG_VIEW"); return !(e && e[0] == '0'); }();
    const bool      m_cvOnly = [] { const char* e = getenv("ADAS_CV_ONLY"); return e && e[0] == '1'; }();
    const bool      m_simulateNoCam = [] { const char* e = getenv("ADAS_SIMULATION"); return e && e[0] == '1'; }();
    bool            m_hadCamera = false;
    VehicleTracker  m_vehicles;
    LaneOutput      m_lastLane;
    RoadGeometry    m_lastGeom;
    std::vector<VehicleDetection> m_lastRawDets;
    double          m_dispDist = -1.0;
    double          m_prof[3] = {0, 0, 0};
    int             m_profN = 0;
    std::chrono::steady_clock::time_point m_profT = std::chrono::steady_clock::now();
    FastVehicleDetector m_fastVehicleDetector;

    TrafficSignRunner m_tsr;
    SignTracker     m_signs{[] {
        SignTracker::Params p;
        if (const char* e = getenv("ADAS_TSR_CONF1")) p.confirm_conf = float(atof(e));
        return p;
    }()};
    uint64_t        m_tsrSeq = 0;
    double          m_tsrLastSubmit = -1e9, m_tsrMs = 0.0, m_tsrHz = 0.0, m_tsrRateT0 = -1.0;
    int             m_tsrRuns = 0;
    int             m_tsrBand[2] = {0, 0};
    const int       m_tsrPeriodMs = [] { const char* e = getenv("ADAS_TSR_PERIOD_MS"); return e ? std::max(100, atoi(e)) : 600; }();
    const int       m_tsrFastMs = [] { const char* e = getenv("ADAS_TSR_FAST_MS"); return e ? std::max(50, atoi(e)) : 200; }();
    const double    m_followHz = [] { const char* e = getenv("ADAS_TSR_FOLLOW_HZ"); return e ? std::clamp(atof(e), 0.0, 60.0) : 30.0; }();
    std::unique_ptr<SpeedValueReader> m_followReader;
    std::deque<std::pair<double, cv::Mat>> m_followFrames;
    double          m_followLastT = -1e9;
    const int       m_webPeriodMs = [] { const char* e = getenv("ADAS_WEB_FPS"); int f = e ? std::clamp(atoi(e), 1, 30) : 30; return 1000 / f - 8; }();
};

#endif
