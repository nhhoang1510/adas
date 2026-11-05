#pragma once

#include "nanodet_engine.hpp"
#include "road_geometry.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

class DualEnginePipeline {
public:
  struct Config {
    NanoDetEngine::Config det;
    CameraCalibration calib;
    double det_hz = 15.0;
    double det_horizon_pos = 0.08;
    bool det_far = [] { const char* e = getenv("ADAS_DET_FAR"); return !(e && e[0] == '0'); }();
    int crop_quant_px = 8;
    int crop_hysteresis_px = 16;
    double night_luma_on = 55.0;
    double night_luma_off = 65.0;
    double log_every_s = 1.0;
    float det_conf_day = 0.30f;
    float det_conf_night = 0.20f;
  };

  struct Result {
    std::vector<VehicleDetection>
        vehicles;
    bool det_fresh = false;
    uint64_t det_seq = 0;
    double det_ms = 0.0;

    double horizon_y = 0.0;
    bool horizon_trusted = false;
    int hood_y = 0;
    bool night = false;
    double luma = 0.0;
    double cycle_ms = 0.0;
    bool simulated = false;
  };

  DualEnginePipeline() : DualEnginePipeline(Config{}) {}
  explicit DualEnginePipeline(const Config &cfg)
      : m_cfg(cfg), m_det(cfg.det) {
    std::printf("[DualEngine] det %s @%.0f Hz\n",
                m_det.usesRealModel() ? "MODEL" : (m_det.ready() ? "SIM" : "OFF"),
                cfg.det_hz);
    m_cfg.calib.print();
  }
  DualEnginePipeline(const DualEnginePipeline &) = delete;
  DualEnginePipeline &operator=(const DualEnginePipeline &) = delete;

  bool detReady() const { return m_det.ready(); }
  bool ready() const { return detReady(); }
  const CameraCalibration &calibration() const { return m_cfg.calib; }

  void setCalibration(const CameraCalibration &c) {
    m_cfg.calib = c;
    m_detCropY = -1;
    m_last = Result();
    m_cfg.calib.print();
  }
  void updateCalibration(const CameraCalibration &c) { m_cfg.calib = c; }

  bool setCalibrationFile(const std::string &path) {
    CameraCalibration c = m_cfg.calib;
    if (!CameraCalibration::load(path, c)) {
      std::printf("[Calib] %s not found, keeping %s\n", path.c_str(),
                  m_cfg.calib.source.c_str());
      return false;
    }
    setCalibration(c);
    return true;
  }

  Result step(const cv::Mat &frame, double raw_horizon = -1.0,
              int raw_inliers = 0) {
    Result r = m_last;
    r.det_fresh = false;
    if (frame.empty())
      return r;

    const auto t0 = clock::now();
    const int H = frame.rows, W = frame.cols;
    m_frameW = W;
    m_frameH = H;
    const int hood = std::clamp(int(m_cfg.calib.hood_y_ratio * H), 1, H);

    r.luma = roadLuma(frame, hood);
    m_night = m_night ? (r.luma < m_cfg.night_luma_off)
                      : (r.luma < m_cfg.night_luma_on);
    r.night = m_night;
    if (m_night != m_detNightMode) {
      m_det.setConfThreshold(m_night ? m_cfg.det_conf_night : m_cfg.det_conf_day);
      m_detNightMode = m_night;
    }
    (void)raw_horizon; (void)raw_inliers;
    r.horizon_y = m_cfg.calib.horizon_y_ratio * H;
    r.horizon_trusted = true;
    r.hood_y = hood;

    if (!ready()) {
      m_last = r;
      logStatus(r, t0);
      return r;
    }

    if (detReady() && secondsSince(m_lastDetT, t0) * m_cfg.det_hz >= 1.0) {
      const cv::Rect crop =
          stableCrop(W, H, m_det.config().input_w, m_det.config().input_h,
                     r.horizon_y, m_cfg.det_horizon_pos, m_detCropY);
      cv::Rect far;
      if (m_cfg.det_far) {
        const int fh = std::min(m_det.freeRows(crop.width, crop.height), crop.height);
        const int fw = std::min(m_det.config().input_w, crop.width);
        if (fh >= 64 && fw < crop.width) {
          const int fx = std::clamp(int(std::lround(m_cfg.calib.vanish_x_ratio * W)) - fw / 2, 0, crop.width - fw);
          const int fy = std::clamp(int(std::lround(r.horizon_y)) - crop.y - int(0.6 * fh), 0, crop.height - fh);
          far = cv::Rect(fx, fy, fw, fh);
        }
      }
      double ms = 0.0;
      bool ok = false;
      auto dets = m_det.infer(frame(crop), &ms, &ok, far);
      m_lastDetT = t0;
      if (ok) {
        std::vector<VehicleDetection> keep;
        keep.reserve(dets.size());
        for (auto &d : dets) {
          d.box.y1 += crop.y;
          d.box.y2 += crop.y;
          if (d.box.y1 >= hood)
            continue;
          if (d.box.y2 <= r.horizon_y - 0.05 * H)
            continue;
          keep.push_back(d);
        }
        r.vehicles = std::move(keep);
        r.det_fresh = true;
        r.det_seq = ++m_detSeq;
        r.det_ms = ms;
        r.simulated = !m_det.usesRealModel();
        ++m_detRuns;
      }
    }

    r.cycle_ms = secondsSince(t0, clock::now()) * 1000.0;
    m_last = r;
    logStatus(r, t0);
    return r;
  }

  RoadGeometry geometry(const Result &r) const {
    return RoadGeometry::make(m_cfg.calib, m_frameW, m_frameH, r.horizon_y,
                              r.hood_y);
  }

  double estimateDistance(const cv::Rect2f &box, int vehicle_class,
                          const Result &r) const {
    return geometry(r).estimateRange(box, vehicle_class);
  }

  double correctLaneOffset(double offset_m) const {
    return offset_m - m_cfg.calib.camera_lateral_offset_m;
  }

private:
  using clock = std::chrono::steady_clock;
  Config m_cfg;
  NanoDetEngine m_det;

  Result m_last;
  bool m_night = false;
  bool m_detNightMode = false;
  uint64_t m_detSeq = 0;
  clock::time_point m_lastDetT{}, m_lastLogT{};
  int m_detCropY = -1;
  int m_detRuns = 0;
  int m_frameW = 640, m_frameH = 480;

  static double secondsSince(clock::time_point a, clock::time_point b) {
    if (a == clock::time_point{})
      return 1e9;
    return std::chrono::duration<double>(b - a).count();
  }

  static double roadLuma(const cv::Mat &bgr, int hood) {
    const int top = bgr.rows / 2;
    if (hood - top < 8)
      return 128.0;
    cv::Mat small;
    cv::resize(bgr(cv::Rect(0, top, bgr.cols, hood - top)), small,
               cv::Size(32, 16), 0, 0, cv::INTER_AREA);
    const cv::Scalar m = cv::mean(small);
    return 0.114 * m[0] + 0.587 * m[1] + 0.299 * m[2];
  }

  cv::Rect stableCrop(int W, int H, int in_w, int in_h, double horizon_y,
                      double hpos, int &cropY) const {
    const int ch =
        std::min(H, static_cast<int>(std::lround(double(W) * in_h / in_w)));
    const int q = std::max(1, m_cfg.crop_quant_px);
    int want = static_cast<int>(horizon_y - hpos * ch);
    want = std::clamp((want / q) * q, 0, H - ch);
    if (cropY < 0 || std::abs(want - cropY) > m_cfg.crop_hysteresis_px)
      cropY = want;
    cropY = std::clamp(cropY, 0, H - ch);
    return cv::Rect(0, cropY, W, ch);
  }

  void logStatus(const Result &r, clock::time_point now) {
    if (m_cfg.log_every_s <= 0)
      return;
    if (m_lastLogT == clock::time_point{}) {
      m_lastLogT = now;
      return;
    }
    const double dt = secondsSince(m_lastLogT, now);
    if (dt < m_cfg.log_every_s)
      return;
    std::printf(
        "[DualEngine] det %4.1f Hz %5.1f ms | "
        "horizon %5.1f (cal %5.1f, %s) hood %d | luma %5.1f %s | veh %zu%s\n",
        m_detRuns / dt, r.det_ms, r.horizon_y,
        m_cfg.calib.horizon_y_ratio * m_frameH,
        "auto-cal", r.hood_y, r.luma,
        r.night ? "NIGHT" : "day", r.vehicles.size(),
        r.simulated ? " [SIM]" : "");
    for (const auto &v : r.vehicles) {
      const cv::Rect2f b(v.box.x1, v.box.y1, v.box.x2 - v.box.x1,
                         v.box.y2 - v.box.y1);
      std::printf(
          "    %-9s conf %.2f  box [%4.0f %4.0f %4.0f %4.0f]  dist %5.1f m\n",
          vehicleClassName(v.box.cls), v.box.conf, b.x, b.y, b.width, b.height,
          estimateDistance(b, v.box.cls, r));
    }
    std::fflush(stdout);
    m_detRuns = 0;
    m_lastLogT = now;
  }
};
