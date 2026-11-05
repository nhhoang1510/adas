#pragma once

#include "road_geometry.hpp"
#include "bev_transformer.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct LaneModel {
    double c0 = 0.0, c1 = 0.0, c2 = 0.0;
    double x(double z) const { return c0 + c1 * z + c2 * z * z; }
};

struct LaneOutput {
    bool has_lane = false;
    bool left_valid = false, right_valid = false;
    bool left_synth = false, right_synth = false;
    bool held = false;
    LaneModel left, right;
    LaneModel left_trk, right_trk;
    double width_m = 3.5;
    double offset_m = 0.0;
    double lat_vel_mps = 0.0;
    double margin_left_m = 9.9;
    double margin_right_m = 9.9;
    double tlc_s = -1.0;
    int    tlc_side = 0;
    double conf_left = 0.0, conf_right = 0.0;
    int    pts_left = 0, pts_right = 0;

    int    road_shape = 0;
    double curve_radius_m = 0.0;

    int alert_level = 0;
    std::string status = "NO_LANE";
    std::string text = "KHONG THAY VACH LAN";

    std::vector<cv::Point> left_img, right_img, drivable_img;
};

inline void laneMarkingMask(const cv::Mat& frame, const RoadGeometry& g, cv::Mat& mask, cv::Mat& grayBuf,
                            const std::vector<cv::Rect>& exclude = {}, int contrast = 14, int above_median = 10,
                            int road_tol = 35, double max_paint_m = 0.45) {
    (void)above_median;
    const int H = frame.rows, W = frame.cols;
    mask.create(H, W, CV_8UC1);
    mask.setTo(0);
    if (!g.valid() || frame.empty()) return;
    const int v0 = std::clamp(int(std::ceil(g.vh + 3.0)), 0, H);
    const int v1 = std::clamp(g.hood, 0, H);
    if (v1 - v0 < 4) return;
    const int a0 = std::max(0, v0 - 1), a1 = std::min(H, v1 + 1);
    cv::Mat raw;
    if (frame.channels() == 3) cv::cvtColor(frame.rowRange(a0, a1), raw, cv::COLOR_BGR2GRAY);
    else raw = frame.rowRange(a0, a1).clone();
    grayBuf.create(v1 - v0, W, CV_8UC1);
    for (int v = v0; v < v1; ++v) {
        const uchar* p = raw.ptr<uchar>(std::max(v - 1, a0) - a0);
        const uchar* c = raw.ptr<uchar>(v - a0);
        const uchar* n = raw.ptr<uchar>(std::min(v + 1, a1 - 1) - a0);
        uchar* o = grayBuf.ptr<uchar>(v - v0);
        for (int u = 0; u < W; ++u) {
            const uchar x = p[u], y = c[u], z = n[u];
            o[u] = std::max(std::min(x, y), std::min(std::max(x, y), z));
        }
    }

    int rh[256];
    for (int v = v0; v < v1; ++v) {
        const uchar* r = grayBuf.ptr<uchar>(v - v0);
        uchar* m = mask.ptr<uchar>(v);
        const double z = g.zAtRow(v);
        const int d = std::clamp(int(std::lround(1.5 * 0.15 * g.f / z)), 2, 40);
        if (W <= 2 * d + 1) continue;
        const int half = std::max(3, int(0.8 * g.f / z));
        const int ua = std::max(0, int(g.cx) - half), ub = std::min(W, int(g.cx) + half);
        std::fill(rh, rh + 256, 0);
        int rn = 0;
        for (int u = ua; u < ub; ++u) {
            bool inVeh = false;
            for (const auto& b : exclude) if (v >= b.y && v < b.y + b.height && u >= b.x && u < b.x + b.width) { inVeh = true; break; }
            if (!inVeh) { ++rh[r[u]]; ++rn; }
        }
        int road = 128;
        if (rn >= 6) { int a2 = 0; road = 0; for (; road < 255; ++road) { a2 += rh[road]; if (a2 * 10 >= rn * 6) break; } }
        const int flankMax = road + road_tol;
        const int maxRun = int(max_paint_m * g.f / z) + 2;
        int runStart = -1;
        for (int u = d; u <= W - d; ++u) {
            bool on = false;
            if (u < W - d) {
                const int c = r[u], lf = r[u - d], rt = r[u + d];
                const int side = std::max(lf, rt);
                on = c - side > std::max(contrast, side / 5) &&
                     std::abs(lf - rt) <= road_tol &&
                     side <= flankMax;
            }
            if (on && runStart < 0) runStart = u;
            else if (!on && runStart >= 0) {
                if (u - runStart <= maxRun) std::fill(m + runStart, m + u, uchar(255));
                runStart = -1;
            }
        }
    }
    const cv::Rect img(0, 0, W, H);
    for (const auto& b : exclude) {
        const cv::Rect rr = b & img;
        if (rr.area() > 0) mask(rr).setTo(0);
    }
}

class StaticMaskFilter {
public:
    double tau_s = 8.0, warm_s = 4.0, max_rate = 0.75, near_frac = 0.3;

    void apply(const cv::Mat& in, cv::Mat& out, const RoadGeometry& g, double t, bool active = true) {
        if (in.empty()) { out = in; return; }
        if (in.size() != m_size) { m_size = in.size(); m_acc = cv::Mat::zeros(90, 160, CV_32F); m_on = 0.0; m_lastT = -1.0; }
        const double dt = m_lastT < 0 ? 0.0 : std::clamp(t - m_lastT, 0.0, 0.5);
        m_lastT = t;
        cv::resize(in, m_small, m_acc.size(), 0, 0, cv::INTER_AREA);
        cv::threshold(m_small, m_hit, 0.15 * 255, 1.0, cv::THRESH_BINARY);
        m_hit.convertTo(m_hitF, CV_32F);
        const double a = 1.0 - std::exp(-dt / tau_s);
        cv::accumulateWeighted(m_hitF, m_acc, a);
        m_on += dt;
        if (m_on < warm_s || !active) { out = in; return; }
        cv::compare(m_acc, max_rate, m_sup, cv::CMP_GT);
        const int r0 = std::clamp(int((g.vh + near_frac * (g.hood - g.vh)) * m_acc.rows / double(in.rows)), 0, m_acc.rows);
        m_sup.rowRange(0, r0).setTo(0);
        if (cv::countNonZero(m_sup) == 0) { out = in; return; }
        const int nc = cv::connectedComponents(m_sup, m_lbl, 8, CV_32S);
        if (nc > 1) {
            const double sx = m_acc.cols / double(in.cols), sy = m_acc.rows / double(in.rows);
            const double vx = g.cx * sx, vy = g.vh * sy;
            std::vector<cv::Vec6d> mo(nc, cv::Vec6d(0, 0, 0, 0, 0, 0));
            for (int y = 0; y < m_lbl.rows; ++y) {
                const int* l = m_lbl.ptr<int>(y);
                for (int x = 0; x < m_lbl.cols; ++x) if (l[x] > 0) {
                    cv::Vec6d& m = mo[l[x]];
                    m[0] += 1; m[1] += x; m[2] += y; m[3] += double(x) * x; m[4] += double(y) * y; m[5] += double(x) * y;
                }
            }
            m_keep.assign(nc, 0);
            for (int k = 1; k < nc; ++k) {
                const cv::Vec6d& m = mo[k];
                if (m[0] < 3) continue;
                const double cx = m[1] / m[0], cy = m[2] / m[0];
                const double a = m[3] / m[0] - cx * cx, b = m[5] / m[0] - cx * cy, c = m[4] / m[0] - cy * cy;
                const double tr = a + c, det = a * c - b * b, d = std::sqrt(std::max(0.0, tr * tr / 4 - det));
                const double l1 = tr / 2 + d, l2 = std::max(1e-6, tr / 2 - d);
                const double th = 0.5 * std::atan2(2 * b, a - c);
                const double tv = std::atan2(cy - vy, cx - vx);
                double da = std::fmod(std::abs(th - tv), CV_PI); da = std::min(da, CV_PI - da);
                if (l1 / l2 > 9.0 && da < 0.17) m_keep[k] = 1;
            }
            for (int y = 0; y < m_lbl.rows; ++y) {
                const int* l = m_lbl.ptr<int>(y);
                uchar* q = m_sup.ptr<uchar>(y);
                for (int x = 0; x < m_lbl.cols; ++x) if (l[x] > 0 && m_keep[l[x]]) q[x] = 0;
            }
            if (cv::countNonZero(m_sup) == 0) { out = in; return; }
        }
        cv::dilate(m_sup, m_sup, cv::Mat());
        cv::resize(m_sup, m_supBig, in.size(), 0, 0, cv::INTER_NEAREST);
        if (out.data == in.data) out = cv::Mat();
        in.copyTo(out);
        out.setTo(0, m_supBig);
    }
    void reset() { m_size = cv::Size(); }

private:
    cv::Size m_size;
    cv::Mat m_acc, m_small, m_hit, m_hitF, m_sup, m_supBig, m_lbl;
    std::vector<uchar> m_keep;
    double m_on = 0.0, m_lastT = -1.0;
};

class LaneDetector {
public:
    struct Params {
        double x_min = -6.0, x_max = 6.0, z_max = 40.0;
        double res_x = 0.05, res_z = 0.10;
        int    n_windows = 10;
        double search_len_m = 30.0;
        double win_half_cold_m = 0.50;
        double win_half_track_m = 0.35, win_half_per_m = 0.01;
        double max_shift_m = 0.30;
        int    min_win_pix = 6;
        double max_thickness_m = 0.45;
        bool   clean_bev = true;
        double min_blob_len_m = 0.8;
        double min_blob_elong = 2.5;
        int    min_windows = 3;
        int    min_windows_prior = 2;
        int    min_points = 40;
        double min_span_m = 6.0;
        double curvature_span_m = 15.0;
        double curv_sig = 3.0;
        double shape_window_s = 1.5;
        double curve_enter_r_m = 450.0, curve_exit_r_m = 800.0;
        double cold_lo_m = 0.8, cold_hi_m = 2.9;
        double centre_blank_m = 0.6;
        double cold_hist_len_m = 12.0;
        double min_hist_peak = 4.0;
        double fit_rms_max = 0.30, heading_max = 0.40, curv_max = 0.010;
        double lane_width_default = 3.5, lane_width_min = 2.8, lane_width_max = 4.3;
        double width_change_max = 0.9;
        double jump_m = 0.6; int jump_confirm = 3;
        double step_m = 0.25, step_rate = 1.0;
        double width_alpha = 0.05; int width_min_pts = 150, width_min_win = 4;
        double tau_out = 0.15;
        double kf_lat_acc = 0.6;
        double kf_q_c0 = 1e-4, kf_q_c1 = 1.5e-3, kf_q_c2 = 1e-6;
        double kf_rate_tau = 1.5;
        double kf_r0 = 0.03, kf_r_rms = 1.5;
        double kf_r_infl = 16.0;
        double conf_gain = 0.40, conf_decay_s = 0.7, conf_valid = 0.35, conf_drop = 0.05;
        double max_extrapolate_s = 0.0;
        double stale_s = 1.0;
        double hold_s = [] { const char* e = getenv("ADAS_LANE_HOLD_S"); return e ? std::max(0.0, atof(e)) : 2.5; }();
        double hold_drift_max_m = 0.4;
        double width_tol_m = 0.6;
        double car_half_width_m = 0.92;
        double ldw_ref_z_max = 8.0;
        double z_far_draw = 30.0;
        double drift_margin_m = 0.0, depart_margin_m = -0.30;
        double depart_rearm_m = 0.30, depart_hold_s = 2.5;
        double depart_deeper_m = 0.40;
        double lane_change_jump_m = 1.2, lane_change_quiet_s = 2.0;
        double opposite_block_s = 5.0;
        double tlc_warn_s = 0.0, tlc_margin_m = 0.35;
        double debounce_up_s = 0.25, debounce_down_s = 0.40;
    };

    LaneDetector() { alloc(); }
    explicit LaneDetector(const Params& p) : m_p(p) { alloc(); }
    const BevTransformer& bevTransformer() const { return m_bevT; }

    void roadShape(double t, LaneOutput& o) {
        const double k = 2.0 * 0.5 * (o.left.c2 + o.right.c2);
        m_kHist.emplace_back(t, k);
        while (!m_kHist.empty() && t - m_kHist.front().first > m_p.shape_window_s) m_kHist.erase(m_kHist.begin());
        m_kSort.clear();
        for (const auto& q : m_kHist) m_kSort.push_back(q.second);
        std::nth_element(m_kSort.begin(), m_kSort.begin() + m_kSort.size() / 2, m_kSort.end());
        m_kappa = m_kSort[m_kSort.size() / 2];
        const double r = std::abs(m_kappa) > 1e-6 ? 1.0 / std::abs(m_kappa) : 1e6;
        if (m_shape == 0 && r < m_p.curve_enter_r_m) m_shape = m_kappa > 0 ? 1 : -1;
        else if (m_shape != 0 && (r > m_p.curve_exit_r_m || (m_kappa > 0) != (m_shape > 0))) m_shape = 0;
        o.road_shape = m_shape;
        o.curve_radius_m = m_shape ? r : 0.0;
    }

    void reset() {
        m_kappa = 0.0; m_shape = 0; m_kHist.clear();
        m_side[0] = Side(); m_side[1] = Side();
        m_width = m_p.lane_width_default; m_widthN = 0;
        m_outT = -1.0; m_hadOut = false; m_holdT0 = -1.0; m_lastRate = 0.0; m_wasHeld = false;
        m_lastT = -1; m_lvl = 0; m_cand = -1;
        m_armed[0] = m_armed[1] = true; m_alertSide = -1; m_alertUntil = m_quietUntil = -1.0; m_alertT = -1e9; m_offHist.clear(); m_offAll.clear();
        m_status = "NO_LANE"; m_text = "KHONG THAY VACH LAN";
        m_bev.setTo(0);
        for (auto& w : m_dbgWin) w.clear();
    }

    void predict(double t) {
        const double dt = (m_lastT < 0) ? 0.0 : std::clamp(t - m_lastT, 0.0, 0.5);
        m_lastT = t;
        if (dt <= 0) return;
        for (auto& s : m_side) {
            if (!s.init) continue;
            s.predict(dt, m_p);
            s.conf *= std::exp(-dt / m_p.conf_decay_s);
            if (s.conf < m_p.conf_drop) s = Side();
        }
    }

    void update(const cv::Mat& mask, bool mask_valid, const RoadGeometry& g) {
        if (!g.valid() || mask.empty() || !mask_valid || mask.type() != CV_8UC1) return;
        if (!warp(mask, g)) return;
        const double t = m_lastT;

        Fit fit[2];
        for (int s = 0; s < 2; ++s) fit[s] = search(s, g);

        if (fit[0].ok && fit[0].m.c0 > g.car_x + 0.15 && !fit[1].ok) {
            fit[1] = fit[0]; fit[0].ok = false;
            m_side[1] = m_side[0]; m_side[0] = Side();
        } else if (fit[1].ok && fit[1].m.c0 < g.car_x - 0.15 && !fit[0].ok) {
            fit[0] = fit[1]; fit[1].ok = false;
            m_side[0] = m_side[1]; m_side[1] = Side();
        }

        if (m_geomCalibrated) {
            const double zn = nearZ(g);
            const bool apart = fit[0].ok && fit[1].ok && fit[1].m.x(zn) - fit[0].m.x(zn) > m_p.lane_width_min;
            if (fit[0].ok && fit[0].m.x(zn) > g.car_x + 0.8) {
                if (!fit[1].ok || apart) { fit[1] = fit[0]; m_side[1] = m_side[0]; m_side[0] = Side(); }
                fit[0].ok = false;
            } else if (fit[1].ok && fit[1].m.x(zn) < g.car_x - 0.8) {
                if (!fit[0].ok || apart) { fit[0] = fit[1]; m_side[0] = m_side[1]; m_side[1] = Side(); }
                fit[1].ok = false;
            }
        }

        if (fit[0].ok && fit[1].ok) {
            const double zn = nearZ(g), zf = zn + 15.0;
            const double wn = fit[1].m.x(zn) - fit[0].m.x(zn);
            const double wf = fit[1].m.x(zf) - fit[0].m.x(zf);
            const bool widthOff = m_widthN >= 5 && std::abs(wn - m_width) > m_p.width_tol_m;
            if (wn < m_p.lane_width_min || wn > m_p.lane_width_max || std::abs(wf - wn) > m_p.width_change_max || widthOff) {
                const double eL = m_side[0].init ? std::abs(fit[0].m.c0 - m_side[0].target.c0) : 9.0;
                const double eR = m_side[1].init ? std::abs(fit[1].m.c0 - m_side[1].target.c0) : 9.0;
                const int worse = (eL != eR) ? (eL > eR ? 0 : 1) : (fit[0].n < fit[1].n ? 0 : 1);
                fit[worse].ok = false;
            } else {
                if (std::min(fit[0].n, fit[1].n) >= m_p.width_min_pts && std::min(fit[0].win, fit[1].win) >= m_p.width_min_win) {
                    m_width += (m_widthN < 5 ? 0.3 : m_p.width_alpha) * (wn - m_width);
                    ++m_widthN;
                }
                if (!jointFit(fit[0], fit[1])) {
                    const double wl = fit[0].n / (fit[0].rms + 0.05), wr = fit[1].n / (fit[1].rms + 0.05);
                    const double c1 = (fit[0].m.c1 * wl + fit[1].m.c1 * wr) / (wl + wr);
                    const double c2 = (fit[0].m.c2 * wl + fit[1].m.c2 * wr) / (wl + wr);
                    fit[0].m.c1 = fit[1].m.c1 = c1;
                    fit[0].m.c2 = fit[1].m.c2 = c2;
                }
            }
        } else {
            for (int s = 0; s < 2; ++s) {
                if (!fit[s].ok) continue;
                const double ax = std::abs(fit[s].m.c0 - g.car_x);
                if (ax > m_p.lane_width_max - 0.3) { fit[s].ok = false; continue; }
                const Side& o = m_side[1 - s];
                if (m_widthN >= 5 && o.init && o.conf >= m_p.conf_valid && t - o.lastMeasT < m_p.stale_s) {
                    const double expect = o.target.c0 + (s == 0 ? -m_width : m_width);
                    if (std::abs(fit[s].m.c0 - expect) > m_p.width_tol_m) fit[s].ok = false;
                }
            }
        }
        for (int s = 0; s < 2; ++s) {
            m_lastPts[s] = fit[s].ok ? fit[s].n : 0;
            if (fit[s].ok) m_side[s].measure(fit[s], t, m_p);
        }
    }

    LaneOutput output(const RoadGeometry& g, double t, bool turn_left = false, bool turn_right = false) {
        LaneOutput o;
        o.width_m = m_width;
        o.conf_left = m_side[0].conf;
        o.conf_right = m_side[1].conf;
        o.pts_left = m_lastPts[0];
        o.pts_right = m_lastPts[1];
        const bool freshL = t - m_side[0].lastMeasT < m_p.stale_s, freshR = t - m_side[1].lastMeasT < m_p.stale_s;
        bool realL = m_side[0].init && m_side[0].conf >= m_p.conf_valid;
        bool realR = m_side[1].init && m_side[1].conf >= m_p.conf_valid;
        if (realL && realR && !freshL && freshR) realL = false;
        if (realL && realR && !freshR && freshL) realR = false;
        if (realL) o.left = m_side[0].shown;
        if (realR) o.right = m_side[1].shown;
        if (realL && !realR) { o.right = o.left; o.right.c0 += m_width; o.right_synth = true; }
        if (realR && !realL) { o.left = o.right; o.left.c0 -= m_width; o.left_synth = true; }
        o.left_trk = realL ? m_side[0].target : o.left;
        o.right_trk = realR ? m_side[1].target : o.right;
        if (realL && !realR) { o.right_trk = o.left_trk; o.right_trk.c0 += m_width; }
        if (realR && !realL) { o.left_trk = o.right_trk; o.left_trk.c0 -= m_width; }
        if (realL || realR) m_holdT0 = -1.0;
        else if (m_hadOut && !turn_left && !turn_right) {
            if (m_holdT0 < 0) { m_holdT0 = t; m_hold[0] = m_out[0]; m_hold[1] = m_out[1]; }
            const double th = t - m_holdT0;
            if (th < m_p.hold_s) {
                const double tau = m_p.kf_rate_tau;
                const double d = std::clamp(m_lastRate * tau * (1.0 - std::exp(-th / tau)), -m_p.hold_drift_max_m, m_p.hold_drift_max_m);
                o.left = m_hold[0]; o.right = m_hold[1];
                o.left.c0 += d; o.right.c0 += d;
                o.left_trk = o.left; o.right_trk = o.right;
                o.left_synth = o.right_synth = o.held = true;
                m_out[0] = o.left; m_out[1] = o.right;
            }
        }
        if (realL || realR) {
            const double dt = (m_outT < 0) ? 0.0 : std::clamp(t - m_outT, 0.0, 0.5);
            if (!m_hadOut || m_wasHeld || dt <= 0) { m_out[0] = o.left; m_out[1] = o.right; }
            else {
                const double a = 1.0 - std::exp(-dt / m_p.tau_out);
                LaneModel* tgt[2] = {&o.left, &o.right};
                for (int k = 0; k < 2; ++k) {
                    m_out[k].c0 += a * (tgt[k]->c0 - m_out[k].c0);
                    m_out[k].c1 += a * (tgt[k]->c1 - m_out[k].c1);
                    m_out[k].c2 += a * (tgt[k]->c2 - m_out[k].c2);
                }
            }
            o.left = m_out[0]; o.right = m_out[1];
        }
        m_hadOut = realL || realR || o.held;
        m_wasHeld = o.held;
        m_outT = t;
        o.left_valid = realL || o.left_synth;
        o.right_valid = realR || o.right_synth;
        o.has_lane = realL || realR || o.held;

        int rawLevel = 0;
        std::string rawStatus = "NO_LANE", rawText = "KHONG THAY VACH LAN";
        if (o.has_lane) {
            const double zr = std::clamp(nearZ(g), 0.0, m_p.ldw_ref_z_max);
            const double xl = o.left.x(zr), xr = o.right.x(zr);
            o.offset_m = g.car_x - 0.5 * (xl + xr);
            o.margin_left_m = (g.car_x - m_p.car_half_width_m) - xl;
            o.margin_right_m = xr - (g.car_x + m_p.car_half_width_m);

            double rate = 0.0; int n = 0;
            if (realL) { rate += m_side[0].rate; ++n; }
            if (realR) { rate += m_side[1].rate; ++n; }
            o.lat_vel_mps = n ? -rate / n : 0.0;
            if (n) m_lastRate = rate / n;
            if (o.lat_vel_mps > 0.05) { o.tlc_s = std::max(0.0, o.margin_right_m) / o.lat_vel_mps; o.tlc_side = +1; }
            else if (o.lat_vel_mps < -0.05) { o.tlc_s = std::max(0.0, o.margin_left_m) / -o.lat_vel_mps; o.tlc_side = -1; }

            rawStatus = "NORMAL"; rawText = "DI DUNG GIUA LAN";
            if (o.held) { rawStatus = "HOLD"; rawText = "VACH MO: GIU LAN DU DOAN"; }
            const bool widthKnown = m_widthN >= 5;
            const bool canL = (realL || (o.left_synth && realR && widthKnown)) && !turn_left;
            const bool canR = (realR || (o.right_synth && realL && widthKnown)) && !turn_right;
            const bool tlcL = canL && o.tlc_side < 0 && o.tlc_s >= 0 && o.tlc_s < m_p.tlc_warn_s && o.margin_left_m < m_p.tlc_margin_m;
            const bool tlcR = canR && o.tlc_side > 0 && o.tlc_s >= 0 && o.tlc_s < m_p.tlc_warn_s && o.margin_right_m < m_p.tlc_margin_m;

            while (!m_offHist.empty() && t - m_offHist.front().first > 0.5) m_offHist.erase(m_offHist.begin());
            if (!(realL && realR)) m_offHist.clear();
            int crossed = -1;
            const double off0 = m_offHist.empty() ? 0.0 : m_offHist.front().second;
            if (!m_offHist.empty() && std::abs(o.offset_m - off0) > m_p.lane_change_jump_m &&
                off0 * o.offset_m < 0 && std::min(std::abs(off0), std::abs(o.offset_m)) > 0.25 * o.width_m) {
                crossed = o.offset_m < off0 ? 1 : 0;
                m_quietUntil = t + m_p.lane_change_quiet_s;
                m_armed[0] = m_armed[1] = false;
                m_offHist.clear();
            }
            if (realL && realR) m_offHist.emplace_back(t, o.offset_m);
            if (o.margin_left_m > m_p.depart_margin_m + m_p.depart_rearm_m) m_armed[0] = true;
            if (o.margin_right_m > m_p.depart_margin_m + m_p.depart_rearm_m) m_armed[1] = true;
            const bool quiet = t < m_quietUntil;
            const bool blockOpp = t < m_alertT + m_p.opposite_block_s;
            const bool deeperL = !m_armed[0] && m_alertSide == 0 && o.margin_left_m < m_alertMargin - m_p.depart_deeper_m;
            const bool deeperR = !m_armed[1] && m_alertSide == 1 && o.margin_right_m < m_alertMargin - m_p.depart_deeper_m;
            const bool depL = canL && (m_armed[0] || deeperL) && !quiet && !(blockOpp && m_alertSide == 1) && o.lat_vel_mps <= 0.05 && (o.margin_left_m <= m_p.depart_margin_m || tlcL);
            const bool depR = canR && (m_armed[1] || deeperR) && !quiet && !(blockOpp && m_alertSide == 0) && o.lat_vel_mps >= -0.05 && (o.margin_right_m <= m_p.depart_margin_m || tlcR);
            const bool missed = crossed >= 0 && t - m_alertT > 3.0 && !(crossed == 0 ? turn_left : turn_right);
            while (!m_offAll.empty() && t - m_offAll.front().first > 1.5) m_offAll.erase(m_offAll.begin());
            m_offAll.emplace_back(t, o.offset_m);
            if (depL || depR || missed) {
                m_alertSide = missed ? crossed : depL ? 0 : 1;
                const double mSide = m_alertSide == 0 ? o.margin_left_m : o.margin_right_m;
                const double offA = m_offAll.front().second;
                if (!missed && mSide < -m_p.car_half_width_m && offA * o.offset_m < 0 && std::abs(offA) > 0.3)
                    m_alertSide = offA > 0 ? 1 : 0;
                m_alertUntil = t + m_p.depart_hold_s;
                m_alertT = t;
                m_alertMargin = missed ? -9.0 : m_alertSide == 0 ? o.margin_left_m : o.margin_right_m;
                m_armed[0] = m_armed[1] = false;
            }
            if (o.held) {
            } else if (t < m_alertUntil && m_alertSide == 0 && !turn_left) {
                rawLevel = 2; rawStatus = "DEPARTURE_LEFT"; rawText = "CANH BAO: LECH TRAI!";
            } else if (t < m_alertUntil && m_alertSide == 1 && !turn_right) {
                rawLevel = 2; rawStatus = "DEPARTURE_RIGHT"; rawText = "CANH BAO: LECH PHAI!";
            } else if (canL && o.margin_left_m < m_p.drift_margin_m) {
                rawLevel = 1; rawStatus = "DRIFT_LEFT"; rawText = "Bam mep trai";
            } else if (canR && o.margin_right_m < m_p.drift_margin_m) {
                rawLevel = 1; rawStatus = "DRIFT_RIGHT"; rawText = "Bam mep phai";
            }
            buildPolylines(g, o);
            roadShape(t, o);
        } else {
            m_kappa = 0.0; m_shape = 0; m_kHist.clear();
        }

        if (rawLevel == m_lvl) { m_cand = -1; m_status = rawStatus; m_text = rawText; }
        else {
            if (m_lvl == 0 && o.has_lane) { m_status = o.held ? "HOLD" : "NORMAL"; m_text = o.held ? "VACH MO: GIU LAN DU DOAN" : "DI DUNG GIUA LAN"; }
            if (m_cand != rawLevel) { m_cand = rawLevel; m_candT = t; }
            const double need = rawLevel > m_lvl ? m_p.debounce_up_s : m_p.debounce_down_s;
            if (t - m_candT >= need) { m_lvl = rawLevel; m_status = rawStatus; m_text = rawText; m_cand = -1; }
        }
        if (!o.has_lane) { m_lvl = 0; m_status = rawStatus; m_text = rawText; }
        o.alert_level = m_lvl; o.status = m_status; o.text = m_text;
        return o;
    }

    void drawDebug(cv::Mat& bgr, int inset_h = 0) const {
        if (bgr.empty() || m_bev.empty()) return;
        if (inset_h <= 0) inset_h = bgr.rows * 2 / 3;
        cv::Mat c;
        cv::cvtColor(m_bev, c, cv::COLOR_GRAY2BGR);
        c *= 0.6;
        const cv::Scalar winCol(0, 220, 255), fitCol[2] = {cv::Scalar(60, 60, 255), cv::Scalar(255, 120, 40)};
        for (int s = 0; s < 2; ++s) {
            for (const auto& r : m_dbgWin[s]) cv::rectangle(c, r, winCol, 1);
            if (!m_side[s].init) continue;
            std::vector<cv::Point> pl;
            for (double z = 0.5; z < m_p.z_max; z += 1.0) pl.push_back(toBev(m_side[s].shown.x(z), z));
            cv::polylines(c, std::vector<std::vector<cv::Point>>{pl}, false, fitCol[s], 2);
        }
        cv::line(c, toBev(0.0, 0.0), toBev(0.0, 3.0), cv::Scalar(0, 255, 0), 2);
        const int iw = std::max(8, int(std::lround(double(c.cols) * inset_h / c.rows)));
        cv::Mat small;
        cv::resize(c, small, cv::Size(iw, inset_h), 0, 0, cv::INTER_NEAREST);
        const cv::Rect dst(bgr.cols - iw - 4, 4, iw, inset_h);
        if ((dst & cv::Rect(0, 0, bgr.cols, bgr.rows)) == dst) {
            small.copyTo(bgr(dst));
            cv::rectangle(bgr, dst, cv::Scalar(200, 200, 200), 1);
        }
    }

    const cv::Mat& bev() const { return m_bev; }
    const Params& params() const { return m_p; }
    double laneWidth() const { return m_width; }
    void setGeometryCalibrated(bool c) { m_geomCalibrated = c; }

private:
    struct Fit { LaneModel m; double rms = 1.0; int n = 0; int rows = 0; double span = 0.0; int win = 0; bool ok = false;
                 cv::Matx33d cov = cv::Matx33d::eye();
                 const std::vector<cv::Point2d>* pts = nullptr; const std::vector<double>* w = nullptr; };

    struct Side {
        bool init = false;
        cv::Matx41d s = cv::Matx41d::zeros();
        cv::Matx44d P = cv::Matx44d::eye();
        LaneModel target, shown;
        double conf = 0.0, rate = 0.0, lastMeasT = -1.0, lastC0 = 0.0;
        int jumps = 0;

        LaneModel model() const { LaneModel m; m.c0 = s(0); m.c1 = s(2); m.c2 = s(3); return m; }

        void predict(double dt, const Params& p) {
            if (!init || dt <= 0) return;
            cv::Matx44d F = cv::Matx44d::eye(); F(0, 1) = dt;
            s = F * s;
            s(1) *= std::exp(-dt / p.kf_rate_tau);
            cv::Matx44d Q = cv::Matx44d::zeros();
            const double qa = p.kf_lat_acc * p.kf_lat_acc;
            Q(0, 0) = qa * dt * dt * dt * dt / 4.0 + p.kf_q_c0 * dt;
            Q(0, 1) = Q(1, 0) = qa * dt * dt * dt / 2.0;
            Q(1, 1) = qa * dt * dt;
            Q(2, 2) = p.kf_q_c1 * dt;
            Q(3, 3) = p.kf_q_c2 * dt;
            P = F * P * F.t() + Q;
            shown = model();
            rate = s(1);
        }

        void measure(const Fit& f, double t, const Params& p) {
            const double dt = t - lastMeasT;
            const double step = std::min(p.jump_m, p.step_m + p.step_rate * std::max(0.0, dt));
            const bool far = init && std::abs(f.m.c0 - s(0)) > step;
            if (far && jumps + 1 < p.jump_confirm) { ++jumps; return; }
            const double rows = std::max(4.0, double(f.rows));
            const double s0 = p.kf_r0 + p.kf_r_rms * f.rms / std::sqrt(rows);
            const double span = std::max(5.0, f.span);
            const double s1 = 2.0 * s0 / span + 0.002, s2 = 2.0 * s1 / span + 2e-5;
            if (!init || far) {
                s = cv::Matx41d(f.m.c0, 0.0, f.m.c1, f.m.c2);
                P = cv::Matx44d::zeros();
                P(0, 0) = s0 * s0; P(1, 1) = 0.25; P(2, 2) = s1 * s1; P(3, 3) = s2 * s2 + 1e-6;
                init = true; conf = std::max(conf, 0.4); jumps = 0;
                lastMeasT = t; lastC0 = f.m.c0;
                target = shown = model(); rate = 0.0;
                return;
            }
            jumps = 0;
            cv::Matx<double, 3, 4> Hm = cv::Matx<double, 3, 4>::zeros();
            Hm(0, 0) = 1; Hm(1, 2) = 1; Hm(2, 3) = 1;
            cv::Matx33d R = f.cov;
            R(0, 0) += p.kf_r0 * p.kf_r0; R(1, 1) += 1e-6; R(2, 2) += 1e-9;
            (void)s1; (void)s2;
            const cv::Matx31d z(f.m.c0, f.m.c1, f.m.c2);
            const cv::Matx31d y = z - Hm * s;
            const cv::Matx33d S = Hm * P * Hm.t() + R;
            const cv::Matx<double, 4, 3> K = P * Hm.t() * S.inv();
            s = s + K * y;
            P = (cv::Matx44d::eye() - K * Hm) * P;
            lastMeasT = t; lastC0 = f.m.c0;
            target = shown = model(); rate = s(1);
            conf = std::min(1.0, conf + p.conf_gain);
        }
    };

    Params m_p;
    Side   m_side[2];
    double m_width = 3.5;
    int    m_widthN = 0;
    double m_lastT = -1.0;
    int    m_lvl = 0, m_cand = -1;
    bool   m_armed[2] = {true, true};
    int    m_alertSide = -1;
    double m_alertUntil = -1.0, m_quietUntil = -1.0, m_alertT = -1e9, m_alertMargin = 0.0;
    bool m_geomCalibrated = true;
    double m_kappa = 0.0;
    std::vector<std::pair<double, double>> m_kHist;
    std::vector<double> m_kSort;
    int m_shape = 0;
    std::vector<std::pair<double, double>> m_offHist, m_offAll;
    double m_candT = 0.0;
    int    m_lastPts[2] = {0, 0};
    LaneModel m_out[2];
    double m_outT = -1.0;
    bool   m_hadOut = false;
    LaneModel m_hold[2];
    double m_holdT0 = -1.0;
    bool   m_wasHeld = false;
    double m_lastRate = 0.0;
    std::string m_status = "NO_LANE", m_text = "KHONG THAY VACH LAN";
    int    m_bw = 0, m_bh = 0;
    BevTransformer m_bevT;
    cv::Mat m_bev;
    std::vector<cv::Point2d> m_pts[2];
    std::vector<double> m_xs, m_wRow, m_wRob[2];
    std::vector<char> m_use;
    std::vector<cv::Rect> m_dbgWin[2];
    std::vector<float> m_hist, m_sm;
    cv::Mat m_lbl, m_stats, m_cent;
    std::vector<uchar> m_keep;
    int m_dropped = 0;

    void alloc() {
        BevTransformer::Params bp;
        bp.x_min = m_p.x_min; bp.x_max = m_p.x_max; bp.z_max = m_p.z_max; bp.res_x = m_p.res_x; bp.res_z = m_p.res_z;
        m_bevT = BevTransformer(bp);
        m_bw = m_bevT.width();
        m_bh = m_bevT.height();
        m_bev = cv::Mat::zeros(m_bh, m_bw, CV_8UC1);
        m_width = m_p.lane_width_default;
        const size_t cap = size_t(m_bw) * m_bh / 4;
        for (int k = 0; k < 2; ++k) { m_pts[k].reserve(cap); m_wRob[k].reserve(cap); }
        m_xs.reserve(cap); m_wRow.reserve(cap); m_use.reserve(cap);
        m_hist.reserve(m_bw); m_sm.reserve(m_bw); m_keep.reserve(4096);
        for (auto& w : m_dbgWin) w.reserve(16);
    }

    cv::Point toBev(double x, double z) const { return m_bevT.toBev(x, z); }
    int colOf(double x) const { return m_bevT.colOf(x); }
    int rowOf(double z) const { return m_bevT.rowOf(z); }
    double xOf(int c) const { return m_bevT.xOf(c); }
    double zOf(int r) const { return m_bevT.zOf(r); }
    double nearZ(const RoadGeometry& g) const { return BevTransformer::nearZ(g); }

    bool warp(const cv::Mat& mask, const RoadGeometry& g) {
        if (!m_bevT.prepare(g)) return false;
        m_bevT.warp(mask, m_bev, 64);
        if (m_p.centre_blank_m > 0) {
            const int ca = std::clamp(colOf(g.car_x - m_p.centre_blank_m), 0, m_bw);
            const int cb = std::clamp(colOf(g.car_x + m_p.centre_blank_m) + 1, 0, m_bw);
            if (cb > ca) m_bev.colRange(ca, cb).setTo(0);
        }
        if (m_p.clean_bev) cleanBev();
        return true;
    }

    void cleanBev() {
        const int n = cv::connectedComponentsWithStats(m_bev, m_lbl, m_stats, m_cent, 8, CV_32S);
        if (n <= 1) return;
        m_keep.assign(n, 0);
        int dropped = 0;
        for (int i = 1; i < n; ++i) {
            const double lenZ = m_stats.at<int>(i, cv::CC_STAT_HEIGHT) * m_p.res_z;
            const double widX = m_stats.at<int>(i, cv::CC_STAT_WIDTH) * m_p.res_x;
            const int area = m_stats.at<int>(i, cv::CC_STAT_AREA);
            const double thick = double(area) / std::max(1, m_stats.at<int>(i, cv::CC_STAT_HEIGHT)) * m_p.res_x;
            const bool keep = lenZ >= m_p.min_blob_len_m && lenZ >= m_p.min_blob_elong * std::min(widX, 3.0 * thick + 0.1) &&
                              thick <= m_p.max_thickness_m;
            m_keep[i] = keep ? 255 : 0;
            dropped += !keep;
        }
        m_dropped = dropped;
        for (int r = 0; r < m_bev.rows; ++r) {
            const int* l = m_lbl.ptr<int>(r);
            uchar* b = m_bev.ptr<uchar>(r);
            for (int c = 0; c < m_bev.cols; ++c) if (b[c]) b[c] = m_keep[l[c]];
        }
    }

    Fit search(int side, const RoadGeometry& g) {
        m_dbgWin[side].clear();
        const double zlo = nearZ(g), zhi = std::min(m_p.z_max - 1.0, zlo + m_p.search_len_m);
        const double sgn = side == 0 ? -1.0 : 1.0;

        bool havePrior = false;
        LaneModel prior;
        const Side& me = m_side[side];
        const Side& other = m_side[1 - side];
        bool corridor = false;
        double base = 0.0, want = 0.0;
        const double zref = zlo + 0.5 * std::min(m_p.cold_hist_len_m, zhi - zlo);
        if (me.init && me.conf > 0.2) { prior = me.target; havePrior = true; }
        else if (other.init && other.conf > 0.3) { corridor = true; base = other.target.x(zref); want = base + sgn * m_width; }

        double xc = 0.0;
        if (!havePrior) {
            const int r1 = std::clamp(rowOf(zlo), 0, m_bh), r0 = std::clamp(rowOf(std::min(zhi, zlo + m_p.cold_hist_len_m)), 0, m_bh);
            double lo = side == 0 ? g.car_x - m_p.cold_hi_m : g.car_x + m_p.cold_lo_m;
            double hi = side == 0 ? g.car_x - m_p.cold_lo_m : g.car_x + m_p.cold_hi_m;
            if (corridor) {
                lo = side == 0 ? base - m_p.lane_width_max : base + m_p.lane_width_min;
                hi = side == 0 ? base - m_p.lane_width_min : base + m_p.lane_width_max;
            }
            const int c0 = std::clamp(colOf(lo), 0, m_bw - 1), c1 = std::clamp(colOf(hi), 0, m_bw - 1);
            if (r1 - r0 < 5 || c1 - c0 < 5) return Fit();
            m_hist.assign(m_bw, 0.f);
            for (int r = r0; r < r1; ++r) {
                const uchar* p = m_bev.ptr<uchar>(r);
                const int dc = corridor ? int(std::lround((other.target.x(zOf(r)) - base) / m_p.res_x)) : 0;
                for (int c = c0; c <= c1; ++c) {
                    const int cc = c + dc;
                    if (cc >= 0 && cc < m_bw && p[cc]) m_hist[c] += 1.f;
                }
            }
            m_sm.assign(m_bw, 0.f);
            double best = 0.0;
            for (int c = c0 + 3; c <= c1 - 3; ++c) {
                double s = 0.0;
                for (int k = -3; k <= 3; ++k) s += m_hist[c + k];
                m_sm[c] = float(s / 7.0);
                best = std::max(best, double(m_sm[c]));
            }
            if (best < m_p.min_hist_peak) return Fit();
            const float thr = float(std::max(m_p.min_hist_peak, 0.5 * best));
            int bc = -1;
            if (corridor) {
                double bestD = 1e9;
                for (int c = c0 + 3; c <= c1 - 3; ++c) {
                    if (m_sm[c] < thr || m_sm[c] < m_sm[c - 1] || m_sm[c] < m_sm[c + 1]) continue;
                    const double d = std::abs(xOf(c) - want);
                    if (d < bestD) { bestD = d; bc = c; }
                }
                if (bc < 0) return Fit();
                prior = other.target;
                prior.c0 += xOf(bc) - base;
                havePrior = true;
            } else {
                const int dir = side == 0 ? -1 : 1;
                bc = side == 0 ? c1 - 3 : c0 + 3;
                while (bc >= c0 + 3 && bc <= c1 - 3 && m_sm[bc] < thr) bc += dir;
                if (bc < c0 + 3 || bc > c1 - 3) return Fit();
                while (bc + dir >= c0 + 3 && bc + dir <= c1 - 3 && m_sm[bc + dir] >= m_sm[bc]) bc += dir;
            }
            xc = xOf(bc);
        }

        std::vector<cv::Point2d>& pts = m_pts[side];
        pts.clear();
        const double dz = (zhi - zlo) / m_p.n_windows;
        double shift = 0.0;
        int empty = 0, hit = 0;
        for (int i = 0; i < m_p.n_windows; ++i) {
            const double za = zlo + i * dz, zb = za + dz, zm = 0.5 * (za + zb);
            double half;
            double centre;
            if (havePrior) { centre = prior.x(zm) + shift; half = m_p.win_half_track_m + m_p.win_half_per_m * zm; }
            else           { centre = xc; half = m_p.win_half_cold_m; }
            const int ra = std::clamp(rowOf(zb), 0, m_bh), rb = std::clamp(rowOf(za), 0, m_bh);
            const int ca = std::clamp(colOf(centre - half), 0, m_bw), cb = std::clamp(colOf(centre + half) + 1, 0, m_bw);
            if (rb <= ra || cb <= ca) { ++empty; continue; }
            m_dbgWin[side].emplace_back(ca, ra, cb - ca, rb - ra);
            const size_t before = pts.size();
            for (int r = ra; r < rb; ++r) {
                const uchar* p = m_bev.ptr<uchar>(r);
                const double z = zOf(r);
                for (int c = ca; c < cb; ++c) if (p[c]) pts.emplace_back(xOf(c), z);
            }
            const int got = int(pts.size() - before);
            if (got < m_p.min_win_pix) { pts.resize(before); ++empty; continue; }
            {
                int rows = 0; double lastZ = 1e9;
                for (size_t k = before; k < pts.size(); ++k) if (pts[k].y != lastZ) { ++rows; lastZ = pts[k].y; }
                if (rows > 0 && double(got) / rows * m_p.res_x > m_p.max_thickness_m) { pts.resize(before); ++empty; continue; }
            }
            ++hit;
            std::vector<double>& xs = m_xs; xs.clear();
            for (size_t k = before; k < pts.size(); ++k) xs.push_back(pts[k].x);
            std::nth_element(xs.begin(), xs.begin() + got / 2, xs.end());
            const double med = xs[got / 2];
            if (havePrior) shift = std::clamp(shift + 0.5 * (med - (prior.x(zm) + shift)), -m_p.max_shift_m, m_p.max_shift_m);
            else xc = med;
        }
        (void)empty;
        if (hit < (havePrior ? m_p.min_windows_prior : m_p.min_windows)) return Fit();
        Fit f = fitPoints(pts, side);
        f.win = hit;
        return f;
    }

    Fit fitPoints(const std::vector<cv::Point2d>& p, int side) {
        if (int(p.size()) < m_p.min_points) return Fit();
        double zmin = 1e9, zmax = -1e9;
        for (const auto& q : p) { zmin = std::min(zmin, q.y); zmax = std::max(zmax, q.y); }
        const double span = zmax - zmin;
        if (span < m_p.min_span_m) return Fit();
        Fit best;
        if (span >= m_p.curvature_span_m) {
            const Fit q = fitDeg(p, true, m_wRow, m_wRob[side]);
            const double bend = std::abs(q.m.c2) * span * span / 4.0;
            if (q.ok && bend > m_p.curv_sig * std::max(0.03, q.rms)) best = q;
        }
        if (!best.ok) best = fitDeg(p, false, m_wRow, m_wRob[side]);
        if (!best.ok) return Fit();
        best.span = span;
        best.ok = best.n >= m_p.min_points && best.rms < m_p.fit_rms_max && std::abs(best.m.c1) < m_p.heading_max &&
                  std::abs(best.m.c2) < m_p.curv_max;
        return best;
    }

    Fit fitDeg(const std::vector<cv::Point2d>& p, bool curved, std::vector<double>& wRow, std::vector<double>& wRob) const {
        Fit f;
        const size_t N = p.size();
        wRow.assign(N, 1.0); wRob.assign(N, 1.0);
        for (size_t i = 0; i < N;) {
            size_t j = i;
            while (j < N && p[j].y == p[i].y) ++j;
            const double w = 1.0 / double(j - i) / (1.0 + 0.05 * p[i].y);
            for (size_t k = i; k < j; ++k) wRow[k] = w;
            i = j;
        }
        double cT = 0.6;
        for (int pass = 0; pass < 3; ++pass) {
            cv::Matx33d A = cv::Matx33d::zeros();
            cv::Matx31d b = cv::Matx31d::zeros();
            int n = 0;
            for (size_t i = 0; i < N; ++i) {
                const double w = wRow[i] * wRob[i];
                if (w <= 0) continue;
                const double z = p[i].y;
                const double r[3] = {1.0, z, z * z};
                for (int a = 0; a < 3; ++a) { b(a) += w * r[a] * p[i].x; for (int c = 0; c < 3; ++c) A(a, c) += w * r[a] * r[c]; }
                ++n;
            }
            if (n < m_p.min_points) return Fit();
            const double tr = A(0, 0);
            A(2, 2) += curved ? 0.5 * tr : 1e9;
            A(1, 1) += 1e-3 * tr;
            const cv::Matx31d c = A.solve(b, cv::DECOMP_CHOLESKY);
            if (!std::isfinite(c(0)) || !std::isfinite(c(1)) || !std::isfinite(c(2))) return Fit();
            f.m.c0 = c(0); f.m.c1 = c(1); f.m.c2 = curved ? c(2) : 0.0;
            f.cov = A.inv(cv::DECOMP_CHOLESKY);
            double ss = 0.0, ws = 0.0; int m = 0, rows = 0; double lastY = 1e9;
            for (size_t i = 0; i < N; ++i) {
                const double e = p[i].x - f.m.x(p[i].y);
                const double w = wRow[i] * wRob[i];
                if (w > 0) { ss += w * e * e; ws += w; ++m; if (p[i].y != lastY) { ++rows; lastY = p[i].y; } }
            }
            f.rms = std::sqrt(ss / std::max(1e-9, ws));
            f.n = m; f.rows = rows;
            if (pass == 2) {
                const double s2 = std::max(f.rms, 0.03) * std::max(f.rms, 0.03) * m_p.kf_r_infl;
                f.cov = f.cov * s2;
                if (!curved) { f.cov(2, 2) = 1e-10; f.cov(0, 2) = f.cov(2, 0) = f.cov(1, 2) = f.cov(2, 1) = 0.0; }
            }
            cT = std::max(0.12, 3.0 * f.rms);
            for (size_t i = 0; i < N; ++i) {
                const double u = std::abs(p[i].x - f.m.x(p[i].y)) / cT;
                wRob[i] = u < 1.0 ? (1.0 - u * u) * (1.0 - u * u) : 0.0;
            }
        }
        for (size_t i = 0; i < N; ++i) wRob[i] *= wRow[i];
        f.pts = &p; f.w = &wRob;
        f.ok = true;
        return f;
    }

    bool jointFit(Fit& L, Fit& R) const {
        const bool curved = L.m.c2 != 0.0 || R.m.c2 != 0.0;
        cv::Matx44d A = cv::Matx44d::zeros();
        cv::Matx41d b = cv::Matx41d::zeros();
        auto acc = [&](const Fit& f, int side) {
            if (!f.pts || !f.w) return;
            const auto& P = *f.pts; const auto& Wt = *f.w;
            for (size_t i = 0; i < P.size(); ++i) {
                const double w = Wt[i];
                if (w <= 0) continue;
                const double z = P[i].y;
                const double r[4] = {side == 0 ? 1.0 : 0.0, side == 1 ? 1.0 : 0.0, z, z * z};
                for (int a = 0; a < 4; ++a) { b(a) += w * r[a] * P[i].x; for (int c = 0; c < 4; ++c) A(a, c) += w * r[a] * r[c]; }
            }
        };
        acc(L, 0); acc(R, 1);
        if (A(0, 0) <= 0 || A(1, 1) <= 0) return false;
        A(3, 3) += curved ? 0.5 * (A(0, 0) + A(1, 1)) : 1e9;
        A(2, 2) += 1e-3 * (A(0, 0) + A(1, 1));
        const cv::Matx41d c = A.solve(b, cv::DECOMP_CHOLESKY);
        if (!std::isfinite(c(0)) || !std::isfinite(c(1)) || !std::isfinite(c(2)) || !std::isfinite(c(3))) return false;
        const double c2 = curved ? c(3) : 0.0;
        double ss = 0.0, ws = 0.0;
        auto res = [&](const Fit& f, double c0) {
            const auto& P = *f.pts; const auto& Wt = *f.w;
            for (size_t i = 0; i < P.size(); ++i) {
                const double e = P[i].x - (c0 + c(2) * P[i].y + c2 * P[i].y * P[i].y);
                ss += Wt[i] * e * e; ws += Wt[i];
            }
        };
        res(L, c(0)); res(R, c(1));
        const double rms = std::sqrt(ss / std::max(1e-9, ws));
        if (rms > m_p.fit_rms_max) return false;
        const cv::Matx44d C = A.inv(cv::DECOMP_CHOLESKY) * (std::max(rms, 0.03) * std::max(rms, 0.03) * m_p.kf_r_infl);
        const int idx[2][3] = {{0, 2, 3}, {1, 2, 3}};
        Fit* F[2] = {&L, &R};
        for (int s = 0; s < 2; ++s) {
            F[s]->m.c0 = c(s); F[s]->m.c1 = c(2); F[s]->m.c2 = c2;
            for (int a = 0; a < 3; ++a) for (int k = 0; k < 3; ++k) F[s]->cov(a, k) = C(idx[s][a], idx[s][k]);
            if (!curved) { F[s]->cov(2, 2) = 1e-10; F[s]->cov(0, 2) = F[s]->cov(2, 0) = F[s]->cov(1, 2) = F[s]->cov(2, 1) = 0.0; }
            F[s]->rms = rms;
        }
        return true;
    }

    void buildPolylines(const RoadGeometry& g, LaneOutput& o) const {
        const double zn = std::max(2.0, g.nearZ());
        double zf = std::max(zn + 2.0, m_p.z_far_draw);
        while (zf > zn + 2.0 && g.rowAtZ(zf) - g.vh < 6.0) zf -= 1.0;
        const int N = 12;
        o.left_img.reserve(N); o.right_img.reserve(N);
        for (int i = 0; i < N; ++i) {
            const double z = zn * std::pow(zf / zn, double(i) / (N - 1));
            const cv::Point2f pl = g.toImage(o.left.x(z), z), pr = g.toImage(o.right.x(z), z);
            o.left_img.emplace_back(cvRound(pl.x), cvRound(pl.y));
            o.right_img.emplace_back(cvRound(pr.x), cvRound(pr.y));
        }
        o.drivable_img = o.left_img;
        o.drivable_img.insert(o.drivable_img.end(), o.right_img.rbegin(), o.right_img.rend());
    }
};

using LaneTracker = LaneDetector;
