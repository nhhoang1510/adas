#pragma once

#include "auto_calibrator.hpp"
#include "lane_detector.hpp"
#include "road_geometry.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>
#include <vector>

class CalibrationManager {
public:
    enum class State { Collecting, Calibrated, Recalibrating };

    struct Params {
        double min_speed_kph = 30.0;
        double required_s = 180.0;
        int    min_vp_samples = 300;
        int    min_hood_frames = 300;
        int    min_flow_samples = 150;
        int    min_lat_samples = 300;
        double straight_c2 = 0.0015;
        double straight_c1 = 0.08;
        double max_lat_vel = 0.5;
        double max_vp_mad = 0.015;
        double flow_min_resp = 0.06;
        double flow_z_near = 0.0, flow_z_far = 22.0;
        bool   allow_no_speed = false;
        bool   estimate_lateral = true;
        bool   estimate_scale = true;
        bool   estimate_lateral_quick = true;
        double max_lat_quick_m = 0.3;
        double lane_width_ref_m = 0.0;
        double drift_ratio = 0.025, drift_hold_s = 60.0;
        int    min_vehicle_samples = 80;
        int    min_yaw_samples = 40;
        int    min_width_samples = 30;
        int    min_pair_samples = 8;
        bool   geometry_only = false;
        std::string save_path;

        static Params quickVideo() {
            Params p;
            p.allow_no_speed = true;
            p.required_s = 8.0;
            p.min_vp_samples = 25;
            p.min_hood_frames = 90;
            p.min_vehicle_samples = 80;
            p.max_vp_mad = 0.02;
            p.estimate_lateral = false;
            p.estimate_scale = false;
            p.geometry_only = true;
            p.lane_width_ref_m = 3.5;
            p.estimate_lateral_quick = false;
            return p;
        }
    };

    struct Status {
        State  state = State::Collecting;
        int    phase = 0;
        double progress = 0.0;
        double valid_s = 0.0;
        int    vp = 0, hood = 0, flow = 0, lat = 0;
        bool   speed_known = false, gate_ok = false;
        std::string text = "CALIB 0%";
    };

    CalibrationManager() = default;
    explicit CalibrationManager(const Params& p) : m_p(p) {}
    void setParams(const Params& p) { m_p = p; }
    const Params& params() const { return m_p; }

    void begin(const CameraCalibration& c, bool calibrated) {
        m_ref = c;
        resetSamples();
        m_st = Status();
        m_st.state = calibrated ? State::Calibrated : State::Collecting;
        m_st.phase = calibrated ? 2 : 0;
        m_st.progress = calibrated ? 1.0 : 0.0;
        m_st.text = calibrated ? "CALIB OK" : "CALIB 0%";
        m_driftS = 0.0;
    }

    void addVehicle(const cv::Rect2f& box, int cls, int H, int hood) { m_auto.addVehicle(box, cls, H, hood); }

    bool step(double t, const cv::Mat& frame, const RoadGeometry& g, const LaneOutput& lo,
              const cv::Mat& laneMask, const cv::Mat& bev, double bev_res_z, double bev_z_max,
              double speed_mps, bool turning, CameraCalibration& c) {
        const double dt = m_lastT < 0 ? 0.0 : std::clamp(t - m_lastT, 0.0, 0.2);
        m_lastT = t;
        const bool speedKnown = speed_mps >= 0.0;
        const bool speedOk = speedKnown ? speed_mps * 3.6 >= m_p.min_speed_kph : m_p.allow_no_speed;
        const bool lanesOk = lo.has_lane && lo.left_valid && lo.right_valid && !lo.left_synth && !lo.right_synth &&
                             std::abs(lo.left_trk.c2) < m_p.straight_c2 && std::abs(lo.right_trk.c2) < m_p.straight_c2 &&
                             std::abs(0.5 * (lo.left_trk.c1 + lo.right_trk.c1)) < m_p.straight_c1 &&
                             std::abs(lo.lat_vel_mps) < m_p.max_lat_vel;
        const bool gateA = speedOk && !turning && lo.has_lane && !lo.held;
        const bool oneLineOk = lo.has_lane && ((lo.left_valid && !lo.left_synth) || (lo.right_valid && !lo.right_synth)) &&
                               std::abs(lo.lat_vel_mps) < m_p.max_lat_vel;
        const bool gate = (m_st.phase == 0 && m_st.state != State::Calibrated) ? gateA
                        : (m_p.geometry_only && m_st.state != State::Calibrated) ? (speedOk && !turning && oneLineOk)
                        : (speedOk && !turning && lanesOk);
        m_st.speed_known = speedKnown;
        m_st.gate_ok = gate;

        if (speedOk && !turning && (m_st.state != State::Calibrated || m_p.geometry_only)) hoodSample(frame);
        else m_prevThumb.release();

        bool changed = false;
        if (m_st.state == State::Calibrated) {
            changed = monitor(t, dt, gateA, laneMask, g);
            if (m_p.geometry_only && speedOk && !turning && oneLineOk && !laneMask.empty() && m_p.lane_width_ref_m > 0) {
                widthSample(laneMask, g, lo);
                if (refineVanishing(c)) changed = true;
                else if (refineScale(c)) changed = true;
            }
            if (m_p.geometry_only && hoodRecheck(c, 0.04)) {
                std::printf("[CalibMgr] hood moved after calibration -> RECALIBRATING geometry\n");
                m_auto.reset();
                m_flow.clear(); m_lat.clear(); m_width.clear(); m_yaw.clear(); m_paintW.clear(); m_wN.clear(); m_cN.clear(); m_laneC.clear(); m_nextRefine = 60;
                m_pairVy.clear(); m_pairVx.clear(); m_otherPx.clear(); m_prevBev.release();
                m_vpRefined = false; m_vpAnchorY = m_vpAnchorX = -1.0;
                m_st = Status();
                m_st.state = State::Recalibrating;
                changed = true;
            }
        } else {
            if (gate) {
                m_st.valid_s += dt;
                if (!laneMask.empty() && m_auto.addLaneMask(laneMask, g.hood)) ++m_st.vp;
                if (m_st.phase == 1 && m_p.geometry_only) {
                    yawSample(lo);
                    if (m_p.lane_width_ref_m > 0 && !laneMask.empty()) widthSample(laneMask, g, lo);
                }
                if (m_st.phase == 1 && !m_p.geometry_only) {
                    if (m_p.estimate_lateral) { m_lat.push_back(0.5 * (lo.left_trk.c0 + lo.right_trk.c0)); ++m_st.lat; }
                    m_width.push_back(lo.right_trk.c0 - lo.left_trk.c0);
                    (void)bev; (void)bev_res_z; (void)bev_z_max;
                    if (m_p.estimate_scale && speedKnown) flowSample(t, frame, g, speed_mps);
                }
            } else {
                m_prevBev.release();
            }
            changed = advance(c, g);
        }
        updateText();
        return changed;
    }

    const Status& status() const { return m_st; }
    double flowRatio() const { return m_flow.size() ? med(m_flow) : 1.0; }

private:
    Params m_p;
    Status m_st;
    CameraCalibration m_ref;
    AutoCalibrator m_auto;
    double m_lastT = -1.0, m_driftS = 0.0;
    bool m_hoodApplied = false, m_vpRefined = false;
    const bool m_debug = std::getenv("ADAS_CALIB_DEBUG") != nullptr;
    int m_dbgTick = 0;
    cv::Mat m_prevThumb, m_thumb, m_diff;
    std::vector<double> m_rowAct, m_rowLum;
    static constexpr int ROW_HIST = 600;
    std::vector<std::vector<float>> m_rowHist;
    int m_rowHistI = 0;
    int m_rowN = 0, m_hoodNextCheck = 0;
    double m_hoodCand = -1.0;
    int m_nextRefine = 60;
    cv::Mat m_prevBev, m_curBev, m_win, m_gray, m_bevGray, m_lowBev, m_colMean;
    double m_prevBevT = -1.0;
    std::vector<double> m_flow, m_lat, m_width, m_yaw, m_whist, m_paintW, m_wN, m_cN, m_pairVy, m_pairVx, m_laneC;
    std::vector<cv::Point2f> m_otherPx;
    int m_pairNew = 0;
    double m_vpAnchorY = -1.0, m_vpAnchorX = -1.0;

    bool hoodRecheck(CameraCalibration& c, double moveTol = 0.03) {
        if (m_rowN < m_hoodNextCheck) return false;
        m_hoodNextCheck = m_rowN + 60;
        bool weak = false;
        const double h = hoodEstimate(c.horizon_y_ratio, &weak);
        if (h <= 0) return false;
        const bool first = !m_hoodApplied;
        if (!first && weak) { m_hoodCand = -1; return false; }
        m_hoodApplied = true;
        if (std::abs(h - c.hood_y_ratio) <= (first ? 0.01 : moveTol)) { m_hoodCand = -1; return false; }
        if (!first && c.hood_y_ratio < 0.98 && h > c.hood_y_ratio) { m_hoodCand = -1; return false; }
        if (!first && std::abs(h - m_hoodCand) > 0.02) { m_hoodCand = h; return false; }
        m_hoodCand = -1;
        std::printf("[CalibMgr] hood %.3fH -> %.3fH (%d frames)\n", c.hood_y_ratio, h, m_rowN);
        c.hood_y_ratio = h;
        return true;
    }

    bool refineScale(CameraCalibration& c) {
        if (int(m_wN.size()) < m_nextRefine) return false;
        m_nextRefine = m_nextRefine < 150 ? 150 : 2 * m_nextRefine;
        const double h = std::clamp(m_p.lane_width_ref_m / med(m_wN), 0.8, 2.5);
        double lat = m_p.estimate_lateral_quick
                   ? std::clamp(-med(m_cN) * h, -m_p.max_lat_quick_m, m_p.max_lat_quick_m) : c.camera_lateral_offset_m;
        lat = std::clamp(lat, c.camera_lateral_offset_m - 0.3, c.camera_lateral_offset_m + 0.3);
        if (std::abs(h - c.camera_height_m) < 0.04 * c.camera_height_m && std::abs(lat - c.camera_lateral_offset_m) < 0.1)
            return false;
        std::printf("[CalibMgr] refine (%zu widths): camera height %.2f -> %.2f m, offset %+.2f -> %+.2f m\n",
                    m_wN.size(), c.camera_height_m, h, c.camera_lateral_offset_m, lat);
        c.camera_height_m = h;
        c.camera_lateral_offset_m = lat;
        if (!m_p.save_path.empty()) c.save(m_p.save_path);
        return true;
    }

    void resetSamples() {
        m_auto.reset();
        m_prevThumb.release(); m_prevBev.release();
        m_rowAct.clear(); m_rowLum.clear(); m_rowN = 0; m_rowHist.clear(); m_rowHistI = 0;
        m_flow.clear(); m_lat.clear(); m_width.clear(); m_yaw.clear(); m_paintW.clear(); m_wN.clear(); m_cN.clear(); m_laneC.clear(); m_nextRefine = 60;
        m_lastT = -1.0; m_prevBevT = -1.0;
        m_hoodApplied = false; m_vpRefined = false; m_hoodNextCheck = 0; m_hoodCand = -1.0;
        m_pairVy.clear(); m_pairVx.clear(); m_pairNew = 0; m_vpAnchorY = m_vpAnchorX = -1.0;
    }

    static double med(std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    }

    void hoodSample(const cv::Mat& frame) {
        if (frame.empty()) return;
        cv::Mat small;
        cv::resize(frame, small, cv::Size(96, 54), 0, 0, cv::INTER_AREA);
        if (small.channels() == 3) cv::cvtColor(small, m_thumb, cv::COLOR_BGR2GRAY); else m_thumb = small.clone();
        if (!m_prevThumb.empty()) {
            cv::absdiff(m_thumb, m_prevThumb, m_diff);
            m_thumb.copyTo(m_prevThumb);
            if (cv::mean(m_diff)[0] > 25.0) return;
            if (cv::mean(m_diff.rowRange(m_diff.rows / 2, m_diff.rows * 3 / 4))[0] < 0.4) return;
            if (m_rowAct.size() != size_t(m_diff.rows)) {
                m_rowAct.assign(m_diff.rows, 0.0); m_rowLum.assign(m_diff.rows, 0.0); m_rowN = 0;
                m_rowHist.assign(m_diff.rows, {}); m_rowHistI = 0;
            }
            for (int r = 0; r < m_diff.rows; ++r) {
                const float v = float(cv::mean(m_diff.row(r))[0]);
                m_rowAct[r] += v;
                m_rowLum[r] += cv::mean(m_thumb.row(r))[0];
                if (int(m_rowHist[r].size()) < ROW_HIST) m_rowHist[r].push_back(v); else m_rowHist[r][m_rowHistI] = v;
            }
            m_rowHistI = (m_rowHistI + 1) % ROW_HIST;
            ++m_rowN;
            if (m_st.phase == 0) m_st.hood = m_rowN;
            return;
        }
        m_thumb.copyTo(m_prevThumb);
    }

    double hoodEstimate(double horizon_ratio, bool* weak = nullptr) const {
        if (weak) *weak = false;
        if (m_rowN < m_p.min_hood_frames || m_rowAct.empty()) return -1.0;
        const int R = int(m_rowAct.size());
        std::vector<double> a(R);
        for (int r = 0; r < R; ++r) {
            std::vector<double> h(m_rowHist[r].begin(), m_rowHist[r].end());
            a[r] = h.empty() ? m_rowAct[r] / m_rowN : med(h);
        }
        const int r0 = std::clamp(int((horizon_ratio + 0.08) * R), 0, R - 1), r1 = std::clamp(int(0.70 * R), r0 + 1, R);
        std::vector<double> road(a.begin() + r0, a.begin() + r1);
        const double level = med(road);
        if (level < 0.3) return -1.0;
        const double thr = 0.35 * level;
        int r = R - 1;
        while (r > r1 && a[r] < thr) --r;
        if (r == R - 1)
            return m_rowN >= 3 * m_p.min_hood_frames ? 1.0 : -1.0;
        int edge = -1;
        double best = 12.0;
        for (int k = r; k + 1 < std::min(R, r + 1 + R / 5); ++k) {
            const double step = std::abs(m_rowLum[k + 1] - m_rowLum[k]) / m_rowN;
            if (step > best) { best = step; edge = k + 1; }
        }
        if (edge > 0) return std::clamp(double(edge) / R - 0.01, horizon_ratio + 0.12, 1.0);
        if (weak) *weak = true;
        if (m_rowN < 3 * m_p.min_hood_frames) return -1.0;
        std::vector<double> lq(R, 0.0);
        auto p75 = [&](int k) {
            std::vector<double> h(m_rowHist[k].begin(), m_rowHist[k].end());
            if (h.empty()) return 0.0;
            std::nth_element(h.begin(), h.begin() + h.size() * 3 / 4, h.end());
            return h[h.size() * 3 / 4];
        };
        std::vector<double> roadQ;
        for (int k = std::max(r0, r - 4); k <= r; ++k) roadQ.push_back(p75(k));
        const double roadLevel = med(roadQ);
        for (int k = r + 1; k < R; ++k) lq[k] = std::log(std::max(0.05, p75(k)));
        int split = -1;
        double bestC = std::log(1.4);
        for (int k = r + 3; k < R - 1; ++k) {
            double up = 0, dn = 0;
            for (int i = r + 1; i < k; ++i) up += lq[i];
            for (int i = k; i < R; ++i) dn += lq[i];
            const double c = up / (k - r - 1) - dn / (R - k);
            if (c > bestC && std::exp(up / (k - r - 1)) >= 0.3 * roadLevel) { bestC = c; split = k; }
        }
        if (split > 0) return std::clamp(double(split) / R - 0.01, horizon_ratio + 0.12, 1.0);
        return std::clamp((r + 1.0) / R - 0.01, horizon_ratio + 0.12, 1.0);
    }

    void yawSample(const LaneOutput& lo) {
        auto take = [&](bool ok, bool synth, const LaneModel& m) {
            if (ok && !synth && std::abs(m.c2) < m_p.straight_c2 && std::abs(m.c1) < 0.2) m_yaw.push_back(m.c1);
        };
        take(lo.left_valid, lo.left_synth, lo.left_trk);
        take(lo.right_valid, lo.right_synth, lo.right_trk);
    }

    void widthSample(const cv::Mat& mask, const RoadGeometry& g, const LaneOutput& lo) {
        const bool useR = lo.right_valid && !lo.right_synth, useL = lo.left_valid && !lo.left_synth;
        if (!useR && !useL) return;
        const LaneModel& ref = useR ? lo.right_trk : lo.left_trk;
        const double res = 0.1, dMin = 2.4, dMax = 6.5;
        const int nb = int(dMax / res) + 1;
        m_whist.assign(nb, 0.0);
        const int vTop = std::max(0, int(g.rowAtZ(25.0))), vBot = std::min(mask.rows, g.hood) - 2;
        for (int v = vTop; v <= vBot; ++v) {
            const uchar* p = mask.ptr<uchar>(v);
            for (int u = 0; u < mask.cols; ++u) {
                if (!p[u]) continue;
                double x, z;
                if (!g.toGround(u, v, x, z)) continue;
                const double d = useR ? ref.x(z) - x : x - ref.x(z);
                if (d >= dMin && d < dMax) m_whist[int(d / res)] += 1.0;
            }
        }
        int best = -1;
        double bestN = 10.0;
        for (int b = int(dMin / res) + 1; b + 1 < nb; ++b) {
            const double n = m_whist[b - 1] + m_whist[b] + m_whist[b + 1];
            if (n > bestN) { bestN = n; best = b; }
        }
        if (best <= 0) return;
        for (int b = int(dMin / res) + 1; b < best; ++b) {
            const double n = m_whist[b - 1] + m_whist[b] + m_whist[b + 1];
            const double nl = m_whist[b - 2 >= 0 ? b - 2 : 0] + m_whist[b - 1] + m_whist[b];
            const double nr = m_whist[b] + m_whist[b + 1] + m_whist[b + 2 < nb ? b + 2 : nb - 1];
            if (n >= 0.35 * bestN && n > 10.0 && n >= nl && n >= nr) { best = b; break; }
        }
        const double wd = (best + 0.5) * res;
        m_paintW.push_back(wd);
        const bool pair = useL && useR && lo.has_lane;
        m_laneC.push_back(pair ? 0.5 * (lo.left_trk.c0 + lo.right_trk.c0) : useR ? ref.c0 - 0.5 * wd : ref.c0 + 0.5 * wd);
        if (m_wN.size() < 2000) {
            m_wN.push_back(wd / g.cam_h);
            m_cN.push_back(m_laneC.back() / g.cam_h);
        }
        m_st.lat = int(m_paintW.size());
        auto imageLine = [&](const LaneModel& M, double dOff, double& a, double& b) {
            m_otherPx.clear();
            const double sgn = useR ? -1.0 : 1.0;
            for (int v = vTop; v <= vBot; ++v) {
                double x0, z;
                if (!g.toGround(0.5 * mask.cols, v, x0, z)) continue;
                const double up = g.toImage(M.x(z) + sgn * dOff, z).x;
                const int lo = std::max(0, int(up) - 20), hi = std::min(mask.cols - 1, int(up) + 20);
                if (lo >= hi) continue;
                const uchar* p = mask.ptr<uchar>(v);
                double best = 1e9, bu = 0.0;
                for (int u = lo; u <= hi; ++u) {
                    if (!p[u]) continue;
                    int e = u;
                    while (e + 1 <= hi && p[e + 1]) ++e;
                    const double c = 0.5 * (u + e);
                    if (e - u < 25 && std::abs(c - up) < best) { best = std::abs(c - up); bu = c; }
                    u = e;
                }
                if (best < 1e8) m_otherPx.emplace_back(float(v), float(bu));
            }
            float vmin = 1e9f, vmax = -1e9f;
            for (const auto& q : m_otherPx) { vmin = std::min(vmin, q.x); vmax = std::max(vmax, q.x); }
            if (m_otherPx.size() < 15 || vmax - vmin < 15) return false;
            const size_t n0 = m_otherPx.size();
            for (int it = 0; it < 2; ++it) {
                cv::Vec4f ln;
                cv::fitLine(m_otherPx, ln, cv::DIST_HUBER, 0, 0.01, 0.01);
                if (std::abs(ln[0]) < 1e-3) return false;
                a = ln[1] / ln[0]; b = ln[3] - a * ln[2];
                if (it == 1) break;
                std::vector<cv::Point2f> in;
                for (const auto& q : m_otherPx) if (std::abs(q.y - (a * q.x + b)) < 3.0) in.push_back(q);
                if (in.size() < 15 || in.size() < 0.5 * n0) return false;
                m_otherPx.swap(in);
            }
            double ss = 0.0;
            for (const auto& q : m_otherPx) { const double e = q.y - (a * q.x + b); ss += e * e; }
            return std::sqrt(ss / m_otherPx.size()) < 1.5;
        };
        double la[3], lb[3]; bool ok[2] = {false, false};
        if (useL) ok[0] = imageLine(lo.left_trk, 0.0, la[0], lb[0]);
        if (useR) ok[1] = imageLine(lo.right_trk, 0.0, la[1], lb[1]);
        const int iRef = useR ? 1 : 0;
        int i1 = 0, i2 = 1;
        if (!(ok[0] && ok[1])) {
            if (!ok[iRef] || !imageLine(ref, wd, la[2], lb[2])) return;
            i1 = iRef; i2 = 2;
        }
        const double a1 = la[i1], b1 = lb[i1], a2 = la[i2], b2 = lb[i2];
        if (std::abs(a1 - a2) < 0.25) return;
        const double vy = (b2 - b1) / (a1 - a2), vx = a1 * vy + b1;
        if (vy < 0.25 * g.H || vy > 0.75 * g.H || vx < 0.2 * g.W || vx > 0.8 * g.W) return;
        m_pairVy.push_back(vy / g.H);
        m_pairVx.push_back(vx / g.W);
        ++m_pairNew;
        if (m_pairVy.size() > 400) { m_pairVy.erase(m_pairVy.begin()); m_pairVx.erase(m_pairVx.begin()); }
    }

    bool refineVanishing(CameraCalibration& c) {
        if (!m_vpRefined || m_pairNew < 30 || m_pairVy.size() < 60) return false;
        m_pairNew = 0;
        const double vy = med(m_pairVy), vx = med(m_pairVx);
        std::vector<double> dy, dx;
        for (size_t i = 0; i < m_pairVy.size(); ++i) { dy.push_back(std::abs(m_pairVy[i] - vy)); dx.push_back(std::abs(m_pairVx[i] - vx)); }
        if (med(dy) > 0.01 || med(dx) > 0.01) return false;
        if (m_vpAnchorY < 0) { m_vpAnchorY = c.horizon_y_ratio; m_vpAnchorX = c.vanish_x_ratio; }
        if (std::abs(vy - m_vpAnchorY) > 0.04 || std::abs(vx - m_vpAnchorX) > 0.04) return false;
        if (std::abs(vy - c.horizon_y_ratio) < 0.006 && std::abs(vx - c.vanish_x_ratio) < 0.006) return false;
        std::printf("[CalibMgr] vanishing point refined from %zu lane pairs: %.3fH %.3fW -> %.3fH %.3fW\n",
                    m_pairVy.size(), c.horizon_y_ratio, c.vanish_x_ratio, vy, vx);
        c.horizon_y_ratio = std::clamp(vy, 0.3, 0.7);
        c.vanish_x_ratio = std::clamp(vx, 0.3, 0.7);
        m_wN.clear(); m_cN.clear(); m_nextRefine = 150;
        if (!m_p.save_path.empty()) c.save(m_p.save_path);
        return true;
    }

    void flowSample(double t, const cv::Mat& frame, const RoadGeometry& g, double speed) {
        if (frame.empty()) return;
        const double res = 0.05, xr = 3.0;
        const double zn = std::max(m_p.flow_z_near, std::clamp(g.nearZ(), 1.5, 15.0) + 1.0), zf = m_p.flow_z_far;
        if (zf - zn < 6.0) return;
        const int bw = int(2 * xr / res), bh = int((zf - zn) / res);
        const cv::Point2f src[4] = {g.toImage(-xr, zn), g.toImage(xr, zn), g.toImage(-xr, zf), g.toImage(xr, zf)};
        const cv::Point2f dst[4] = {cv::Point2f(0.f, float(bh - 1)), cv::Point2f(float(bw - 1), float(bh - 1)),
                                    cv::Point2f(0.f, 0.f), cv::Point2f(float(bw - 1), 0.f)};
        const cv::Mat Hm = cv::getPerspectiveTransform(src, dst);
        if (frame.channels() == 3) cv::cvtColor(frame, m_gray, cv::COLOR_BGR2GRAY); else m_gray = frame;
        cv::warpPerspective(m_gray, m_bevGray, Hm, cv::Size(bw, bh), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        m_bevGray.convertTo(m_curBev, CV_32F, 1.0 / 255.0);
        cv::GaussianBlur(m_curBev, m_lowBev, cv::Size(0, 0), 6.0);
        m_curBev -= m_lowBev;
        cv::reduce(m_curBev, m_colMean, 0, cv::REDUCE_AVG);
        for (int r = 0; r < m_curBev.rows; ++r) m_curBev.row(r) -= m_colMean;
        if (!m_prevBev.empty() && m_prevBev.size() == m_curBev.size() && m_prevBevT > 0) {
            const double dtb = t - m_prevBevT;
            const double truth = speed * dtb;
            if (dtb > 0.02 && dtb < 0.25 && truth > 0.1 && truth < 3.0) {
                if (m_win.size() != m_curBev.size()) cv::createHanningWindow(m_win, m_curBev.size(), CV_32F);
                double resp = 0.0;
                const cv::Point2d sh = cv::phaseCorrelate(m_prevBev, m_curBev, m_win, &resp);
                const double meas = sh.y * res;
                const double ratio = meas / truth;
                if (resp > m_p.flow_min_resp && std::abs(sh.x) < 0.5 / res && ratio > 0.4 && ratio < 2.5) {
                    m_flow.push_back(ratio);
                    if (m_flow.size() > 3000) m_flow.erase(m_flow.begin());
                    m_st.flow = int(m_flow.size());
                }
            }
        }
        m_curBev.copyTo(m_prevBev);
        m_prevBevT = t;
    }

    bool advance(CameraCalibration& c, const RoadGeometry& g) {
        (void)g;
        if (m_st.phase == 0) {
            if (hoodRecheck(c)) { m_auto.resetLanes(); m_st.vp = 0; return true; }
            double vy = -1, vx = -1, mad = 1;
            const bool vpOk = m_st.vp >= m_p.min_vp_samples && m_auto.laneVanishingPoint(vy, vx, mad) && mad < m_p.max_vp_mad;
            if (m_debug && ++m_dbgTick % 30 == 0) {
                double hv = -1, ch = -1;
                m_auto.vehicleFit(hv, ch);
                if (!vpOk) m_auto.laneVanishingPoint(vy, vx, mad);
                std::printf("[CalibMgr] A: vp %d (row %.3f col %.3f mad %.3f) veh-hz %.3f camH %.2f (%d) hood-frames %d valid %.1fs gate %d\n",
                            m_st.vp, vy, vx, mad, hv, ch, m_auto.vehicleSamples(), m_st.hood, m_st.valid_s, m_st.gate_ok);
            }
            const double hood = hoodEstimate(vpOk ? vy : c.horizon_y_ratio);
            double vehHz = -1, vehH = -1;
            const bool vehOk = m_p.geometry_only && !vpOk && m_auto.vehicleSamples() >= m_p.min_vehicle_samples &&
                               m_auto.vehicleFit(vehHz, vehH);
            const double pvp = std::max(m_st.vp / double(m_p.min_vp_samples),
                                        m_p.geometry_only ? m_auto.vehicleSamples() / double(m_p.min_vehicle_samples) : 0.0);
            const double pa = std::min({1.0, pvp, m_st.hood / double(m_p.min_hood_frames),
                                        m_st.valid_s / (0.5 * m_p.required_s)});
            m_st.progress = (m_p.geometry_only ? 1.0 : 0.5) * pa;
            if ((vpOk || vehOk) && hood > 0 && m_st.valid_s >= 0.5 * m_p.required_s) {
                c.horizon_y_ratio = vpOk ? vy : vehHz;
                if (vpOk) c.vanish_x_ratio = std::clamp(vx, 0.3, 0.7);
                c.hood_y_ratio = hood;
                std::printf("[CalibMgr] phase A done: horizon %.3fH (%s)  vp_x %.3fW  hood %.3fH  (%d VP, %d hood frames, %.0f s)\n",
                            c.horizon_y_ratio, vpOk ? "lane VP" : "vehicles", c.vanish_x_ratio, hood, m_st.vp, m_st.hood, m_st.valid_s);
                m_st.phase = 1;
                m_flow.clear(); m_lat.clear(); m_width.clear(); m_st.flow = m_st.lat = 0;
                m_prevBev.release();
                m_yaw.clear(); m_paintW.clear(); m_wN.clear(); m_cN.clear();
                return true;
            }
            return false;
        }
        if (m_p.geometry_only && hoodRecheck(c)) {
            m_auto.resetLanes();
            m_yaw.clear(); m_paintW.clear(); m_wN.clear(); m_cN.clear(); m_laneC.clear(); m_pairVy.clear(); m_pairVx.clear(); m_otherPx.clear();
            return true;
        }
        if (m_p.geometry_only) {
            const double W = 640.0;
            const int need = m_p.min_width_samples;
            if (!m_vpRefined) {
                const double py = std::min(1.0, m_yaw.size() / double(m_p.min_yaw_samples));
                const double pv = std::min(1.0, m_pairVy.size() / double(m_p.min_pair_samples));
                m_st.progress = 0.5 + 0.25 * std::max(py, pv);
                auto mad = [](const std::vector<double>& v, double m) {
                    std::vector<double> d; for (double x : v) d.push_back(std::abs(x - m)); return med(d);
                };
                const int nPair = int(m_pairVy.size());
                const double pvy = nPair ? med(m_pairVy) : 0, pvx = nPair ? med(m_pairVx) : 0;
                const bool pairsAgree = nPair >= m_p.min_pair_samples && mad(m_pairVy, pvy) < 0.015 && mad(m_pairVx, pvx) < 0.015;
                const bool pairsNear = std::abs(pvy - c.horizon_y_ratio) < 0.05 && std::abs(pvx - c.vanish_x_ratio) < 0.05;
                if (nPair >= m_p.min_pair_samples && !(pairsAgree && pairsNear) && nPair < 6 * m_p.min_pair_samples) {
                    return false;
                }
                if (nPair >= m_p.min_pair_samples && pairsAgree && pairsNear) {
                    const double vy = std::clamp(pvy, 0.3, 0.7), vx = std::clamp(pvx, 0.3, 0.7);
                    std::printf("[CalibMgr] vanishing point from lane pairs: %.3fH %.3fW (was %.3fH %.3fW, %zu)\n",
                                vy, vx, c.horizon_y_ratio, c.vanish_x_ratio, m_pairVy.size());
                    c.horizon_y_ratio = vy;
                    c.vanish_x_ratio = vx;
                } else if (nPair >= m_p.min_pair_samples) {
                    std::printf("[CalibMgr] lane pairs disagree (%.3fH %.3fW, %d) -> vanishing point kept\n", pvy, pvx, nPair);
                } else if (nPair > 0 && int(m_yaw.size()) < 12 * m_p.min_yaw_samples) {
                    return false;
                } else if (int(m_yaw.size()) >= 3 * m_p.min_yaw_samples) {
                    const double c1 = med(m_yaw);
                    const double vx = std::clamp(c.vanish_x_ratio + c.focalPx(int(W)) * c1 / W, 0.3, 0.7);
                    std::printf("[CalibMgr] yaw: median heading %+.4f rad (%zu) -> vp_x %.3fW -> %.3fW\n",
                                c1, m_yaw.size(), c.vanish_x_ratio, vx);
                    c.vanish_x_ratio = vx;
                } else {
                    return false;
                }
                m_vpRefined = true;
                m_yaw.clear(); m_paintW.clear(); m_wN.clear(); m_cN.clear(); m_pairVy.clear(); m_pairVx.clear(); m_laneC.clear();
                return true;
            }
            const bool wantWidth = m_p.lane_width_ref_m > 0;
            m_st.progress = 0.75 + 0.25 * (wantWidth ? std::min(1.0, m_paintW.size() / double(need)) : 1.0);
            if (wantWidth && int(m_paintW.size()) < need) {
                if (m_yaw.size() < size_t(6 * m_p.min_yaw_samples)) return false;
                std::printf("[CalibMgr] scale: only %zu lane widths -> camera height kept\n", m_paintW.size());
            } else if (wantWidth) {
                const double wMeas = med(m_paintW);
                const double h = std::clamp(c.camera_height_m * m_p.lane_width_ref_m / wMeas, 0.8, 2.5);
                std::printf("[CalibMgr] scale: lane width %.2f m (%zu) vs ref %.2f m -> camera height %.2f -> %.2f m\n",
                            wMeas, m_paintW.size(), m_p.lane_width_ref_m, c.camera_height_m, h);
                if (m_p.estimate_lateral_quick) {
                    const double centre = med(m_laneC) * h / c.camera_height_m;
                    c.camera_lateral_offset_m = std::clamp(-centre, -m_p.max_lat_quick_m, m_p.max_lat_quick_m);
                    std::printf("[CalibMgr] lateral: lane centre %+.2f m -> camera offset %+.2f m\n", centre, c.camera_lateral_offset_m);
                }
                c.camera_height_m = h;
            }
            finish(c);
            return true;
        }
        const bool needFlow = m_p.estimate_scale && m_st.speed_known && m_st.state != State::Recalibrating;
        const double pf = needFlow ? std::min(1.0, m_st.flow / double(m_p.min_flow_samples)) : 1.0;
        const double pl = m_p.estimate_lateral ? std::min(1.0, m_st.lat / double(m_p.min_lat_samples)) : 1.0;
        const double pb = std::min({pf, pl, m_st.valid_s / m_p.required_s});
        m_st.progress = 0.5 + 0.5 * pb;
        if (pb < 1.0) return false;

        double fh_scale = 1.0;
        if (needFlow) fh_scale = 1.0 / med(m_flow);
        double hz = -1, camH = -1;
        bool vehOk = m_auto.vehicleFit(hz, camH) && camH > 0.8 && camH < 3.2;
        const char* hSrc = vehOk ? "cars" : "kept";
        if (m_p.lane_width_ref_m > 0 && m_width.size() > 100) {
            const double hW = c.camera_height_m * m_p.lane_width_ref_m / med(m_width);
            if (hW > 0.8 && hW < 3.2) { camH = hW; vehOk = true; hSrc = "lane width"; }
        }
        const double W = 640.0;
        const double fCur = c.focalPx(int(W));
        const double fhTrue = fCur * c.camera_height_m * fh_scale;
        if (vehOk) c.camera_height_m = camH;
        const double fNew = fhTrue / c.camera_height_m;
        if (needFlow) c.hfov_deg = std::clamp(2.0 * std::atan(0.5 * W / fNew) * 180.0 / M_PI, 30.0, 160.0);
        if (m_p.estimate_lateral && !m_lat.empty()) {
            const double centre = med(m_lat) * (vehOk ? camH / m_ref.camera_height_m : 1.0);
            c.camera_lateral_offset_m = std::clamp(-centre, -1.0, 1.0);
        }
        std::printf("[CalibMgr] phase B done: f*h x%.3f (%d flow)  cam_h %.2f m (%s)  hfov %.1f deg  lat_off %+.2f m (%d)\n",
                    fh_scale, m_st.flow, c.camera_height_m, hSrc, c.hfov_deg,
                    c.camera_lateral_offset_m, m_st.lat);
        finish(c);
        return true;
    }

    void finish(CameraCalibration& c) {
        c.auto_calibrated = 1;
        if (!m_p.save_path.empty()) {
            const bool ok = c.save(m_p.save_path);
            std::printf("[CalibMgr] %s %s\n", ok ? "saved" : "SAVE FAILED:", m_p.save_path.c_str());
        }
        c.print();
        m_ref = c;
        m_st.state = State::Calibrated;
        m_st.phase = 2;
        m_st.progress = 1.0;
        m_auto.reset();
        m_driftS = 0.0;
    }

    bool monitor(double t, double dt, bool gate, const cv::Mat& laneMask, const RoadGeometry& g) {
        (void)t;
        if (!gate) return false;
        if (!laneMask.empty()) m_auto.addLaneMask(laneMask, g.hood);
        double vy, vx, mad;
        if (!m_auto.laneVanishingPoint(vy, vx, mad) || m_auto.laneSamples() < 120) return false;
        const bool off = std::abs(vy - m_ref.horizon_y_ratio) > m_p.drift_ratio ||
                         std::abs(vx - m_ref.vanish_x_ratio) > 2.0 * m_p.drift_ratio;
        m_driftS = off ? m_driftS + dt : std::max(0.0, m_driftS - 2.0 * dt);
        if (m_driftS > m_p.drift_hold_s) {
            std::printf("[CalibMgr] camera moved (VP %.3f/%.3f vs %.3f/%.3f) -> RECALIBRATING\n", vy, vx,
                        m_ref.horizon_y_ratio, m_ref.vanish_x_ratio);
            resetSamples();
            m_st = Status();
            m_st.state = State::Recalibrating;
        }
        return false;
    }

    void updateText() {
        char b[96];
        if (m_st.state == State::Calibrated) { m_st.text = "CALIB OK"; return; }
        if (!m_st.speed_known && !m_p.allow_no_speed) { m_st.text = "CALIB: CHUA CO TOC DO"; return; }
        std::snprintf(b, sizeof b, "%s %d%%%s", m_st.state == State::Recalibrating ? "RECALIB" : "CALIB",
                      int(std::round(100.0 * m_st.progress)), m_st.gate_ok ? "" : " (cho >30km/h, duong thang)");
        m_st.text = b;
    }
};
