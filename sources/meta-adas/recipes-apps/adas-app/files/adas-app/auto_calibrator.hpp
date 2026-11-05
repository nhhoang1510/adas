#pragma once

#include "road_geometry.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <random>
#include <string>
#include <vector>

class AutoCalibrator {
public:
    struct Params {
        int    window = 240;
        int    min_lane_samples = 12;
        int    min_vehicle_samples = 25;
        double car_width_m = 1.80;
        double min_ratio = 0.22, max_ratio = 0.68;
        double min_cam_h = 0.8, max_cam_h = 3.2;
        int    ransac_iters = 120;
        double inlier_px = 3.5;
        int    min_inliers = 14;
    };

    AutoCalibrator() = default;
    explicit AutoCalibrator(const Params& p) : m_p(p) {}

    void reset() { m_vp.clear(); m_vpx.clear(); m_veh.clear(); m_hz = -1; m_camH = -1; m_conf = 0; }
    void resetLanes() { m_vp.clear(); m_vpx.clear(); }

    bool addLaneMask(const cv::Mat& ll, int hood) {
        if (ll.empty() || ll.type() != CV_8UC1) return false;
        const int H = ll.rows, W = ll.cols;
        const int yTop = int(H * (m_hz > 0 ? std::max(0.15, m_hz + 0.04) : 0.30));
        const int yBot = std::min(H - 1, hood - 2);
        if (yBot - yTop < 30) return false;

        m_pts.clear();
        const int maxRun = std::max(6, W / 16);
        for (int y = yBot; y >= yTop; y -= 3) {
            const uchar* r = ll.ptr<uchar>(y);
            int s = -1;
            for (int x = 0; x <= W; ++x) {
                const bool on = x < W && r[x];
                if (on && s < 0) s = x;
                else if (!on && s >= 0) { if (x - s <= maxRun) m_pts.emplace_back(0.5f * (s + x - 1), float(y)); s = -1; }
            }
        }
        if (m_pts.size() < 40) return false;

        Line L, R;
        findSideLines(m_pts, W, yBot, L, R);
        if (!L.ok || !R.ok) return false;
        const double da = L.a - R.a;
        if (std::abs(da) < 0.25) return false;
        const double vy = (R.b - L.b) / da;
        const double vx = L.a * vy + L.b;
        if (vy < m_p.min_ratio * H || vy > m_p.max_ratio * H || vx < 0.15 * W || vx > 0.85 * W) return false;
        push(m_vp, vy / H);
        push(m_vpx, vx / W);
        return true;
    }

    bool laneVanishingPoint(double& vy, double& vx, double& mad_y) const {
        if (int(m_vp.size()) < m_p.min_lane_samples) return false;
        vy = median(m_vp); vx = median(m_vpx);
        std::deque<double> d; for (double v : m_vp) d.push_back(std::abs(v - vy));
        mad_y = median(d);
        return true;
    }
    bool vehicleFit(double& hz, double& camH) const {
        hz = camH = -1;
        if (int(m_veh.size()) < m_p.min_vehicle_samples) return false;
        fitVehicles(hz, camH);
        return hz > 0;
    }

    void addVehicle(const cv::Rect2f& box, int cls, int H, int hood) {
        if (cls != VC_CAR || H <= 0) return;
        const double by = box.y + box.height;
        if (box.width < 12 || by > hood - 3 || box.width > 0.6 * H) return;
        const double ar = box.height / std::max(1.0f, box.width);
        if (ar < 0.5 || ar > 1.3) return;
        m_veh.emplace_back(box.width / H, by / H);
        if (int(m_veh.size()) > m_p.window) m_veh.pop_front();
    }

    bool update(CameraCalibration& c, bool verbose = true) {
        double hzLane = -1, hzVeh = -1, camH = -1;
        int nL = int(m_vp.size()), nV = int(m_veh.size());
        if (nL >= m_p.min_lane_samples) hzLane = median(m_vp);
        if (nV >= m_p.min_vehicle_samples) fitVehicles(hzVeh, camH);

        double hz = -1, wsum = 0;
        if (hzLane > 0) { const double w = std::min(1.0, nL / 40.0) * 2.0; hz = hzLane * w; wsum += w; }
        if (hzVeh > 0)  { const double w = std::min(1.0, nV / 80.0);       hz = (hz < 0 ? 0 : hz) + hzVeh * w; wsum += w; }
        if (wsum <= 0) { m_conf = 0; return false; }
        hz = std::clamp(hz / wsum, m_p.min_ratio, m_p.max_ratio);
        if (hzLane > 0 && hzVeh > 0 && std::abs(hzLane - hzVeh) > 0.06) hz = hzLane;

        m_conf = std::min(1.0, wsum / 2.0);
        m_hz = hz;
        if (camH > 0) m_camH = camH;

        bool changed = false;
        if (std::abs(c.horizon_y_ratio - hz) > 0.004) { c.horizon_y_ratio = hz; changed = true; }
        if (m_camH > 0 && std::abs(c.camera_height_m - m_camH) > 0.05) { c.camera_height_m = m_camH; changed = true; }
        c.hood_y_ratio = std::max(c.hood_y_ratio, c.horizon_y_ratio + 0.10);
        if (verbose && changed)
            std::printf("[AutoCal] horizon %.3fH (lanes %d -> %s, cars %d -> %s)  cam_h %.2f m  conf %.2f\n",
                        hz, nL, hzLane > 0 ? fmt(hzLane).c_str() : "-", nV, hzVeh > 0 ? fmt(hzVeh).c_str() : "-",
                        c.camera_height_m, m_conf);
        return changed;
    }

    double confidence() const { return m_conf; }
    int laneSamples() const { return int(m_vp.size()); }
    int vehicleSamples() const { return int(m_veh.size()); }

private:
    struct Line { double a = 0, b = 0; int n = 0; bool ok = false; };
    Params m_p;
    std::deque<double> m_vp, m_vpx;
    std::deque<cv::Point2d> m_veh;
    std::vector<cv::Point2f> m_pts;
    std::mt19937 m_rng{1234};
    double m_hz = -1, m_camH = -1, m_conf = 0;

    void push(std::deque<double>& q, double v) { q.push_back(v); if (int(q.size()) > m_p.window) q.pop_front(); }
    static double median(std::deque<double> v) {
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    }
    static std::string fmt(double v) { char b[16]; std::snprintf(b, sizeof b, "%.3f", v); return b; }

    void findSideLines(std::vector<cv::Point2f> pts, int W, int yBot, Line& L, Line& R) {
        const double cx = W * 0.5;
        double bestL = -1e9, bestR = 1e9;
        for (int k = 0; k < 4 && pts.size() >= 20; ++k) {
            Line best; std::vector<int> bestIn;
            std::uniform_int_distribution<size_t> pick(0, pts.size() - 1);
            for (int it = 0; it < m_p.ransac_iters; ++it) {
                const auto& p1 = pts[pick(m_rng)];
                const auto& p2 = pts[pick(m_rng)];
                const double dy = p2.y - p1.y;
                if (std::abs(dy) < 20) continue;
                const double a = (p2.x - p1.x) / dy;
                if (std::abs(a) < 0.15 || std::abs(a) > 6.0) continue;
                const double b = p1.x - a * p1.y;
                int n = 0;
                for (const auto& q : pts) if (std::abs(q.x - (a * q.y + b)) < m_p.inlier_px) ++n;
                if (n > best.n) { best = {a, b, n, true}; }
            }
            if (!best.ok || best.n < m_p.min_inliers) break;
            double sy = 0, sx = 0, syy = 0, sxy = 0; int n = 0;
            std::vector<cv::Point2f> rest;
            for (const auto& q : pts) {
                if (std::abs(q.x - (best.a * q.y + best.b)) < m_p.inlier_px) { sy += q.y; sx += q.x; syy += q.y * q.y; sxy += q.x * q.y; ++n; }
                else rest.push_back(q);
            }
            const double den = n * syy - sy * sy;
            if (n >= m_p.min_inliers && std::abs(den) > 1e-6) {
                best.a = (n * sxy - sy * sx) / den;
                best.b = (sx - best.a * sy) / n;
                best.n = n;
            }
            const double xb = best.a * yBot + best.b;
            if (best.a < -0.15 && xb < cx && xb > bestL) { bestL = xb; L = best; }
            if (best.a > 0.15 && xb > cx && xb < bestR) { bestR = xb; R = best; }
            pts.swap(rest);
        }
    }

    void fitVehicles(double& hz, double& camH) const {
        double wmin = 1e9, wmax = -1e9;
        for (const auto& p : m_veh) { wmin = std::min(wmin, p.x); wmax = std::max(wmax, p.x); }
        if (wmax - wmin < 0.03) return;
        double a = 0.45, k = 0.75;
        for (int it = 0; it < 6; ++it) {
            double S = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
            for (const auto& p : m_veh) {
                const double r = p.y - (a + k * p.x);
                const double w = std::abs(r) < 0.02 ? 1.0 : 0.02 / std::abs(r);
                S += w; Sx += w * p.x; Sy += w * p.y; Sxx += w * p.x * p.x; Sxy += w * p.x * p.y;
            }
            const double den = S * Sxx - Sx * Sx;
            if (std::abs(den) < 1e-12) return;
            k = (S * Sxy - Sx * Sy) / den;
            a = (Sy - k * Sx) / S;
        }
        if (a < m_p.min_ratio || a > m_p.max_ratio || k <= 0) return;
        hz = a;
        const double h = k * m_p.car_width_m;
        if (h >= m_p.min_cam_h && h <= m_p.max_cam_h) camH = h;
    }
};
