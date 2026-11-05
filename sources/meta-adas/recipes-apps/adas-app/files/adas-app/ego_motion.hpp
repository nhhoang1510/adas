#pragma once

#include "road_geometry.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

struct EgoOutput {
    bool   valid = false;
    double yaw_rate = 0.0;
    double speed_mps = -1.0;
    int    speed_src = 0;
    double curvature = 0.0;
    int    turn = 0;
    int    inliers = 0;
    double foe_x = 0.0, foe_y = 0.0;

    double yawDegS() const { return yaw_rate * 180.0 / M_PI; }

    std::vector<cv::Point2d> path(double len, double step = 1.0) const {
        std::vector<cv::Point2d> p;
        const double k = curvature;
        for (double s = 0.0; s <= len + 1e-6; s += step) {
            if (std::abs(k) < 1e-5) { p.emplace_back(0.5 * k * s * s, s); continue; }
            const double a = k * s;
            if (std::abs(a) > 0.5 * M_PI) break;
            p.emplace_back((1.0 - std::cos(a)) / k, std::sin(a) / k);
        }
        return p;
    }

    double xAt(double z) const {
        const double k = curvature;
        if (std::abs(k) < 1e-5) return 0.5 * k * z * z;
        const double kz = std::min(1.0, std::abs(k * z));
        return (1.0 - std::sqrt(1.0 - kz * kz)) / k;
    }
};

class EgoMotion {
public:
    struct Params {
        double band_up = 0.22;
        int    grid_cols = 8, grid_rows = 6, per_cell = 5;
        double fb_max_px = 0.8;
        double min_eig = 3e-3;
        int    min_inliers = 20;
        double foe_max = 0.35;
        double foe_prior = 0.02;
        double yaw_tau_s = 0.25;
        double max_yaw = 1.2;
        double lost_s = 0.5;
        double turn_enter = 6.0 * M_PI / 180.0, turn_exit = 3.0 * M_PI / 180.0, turn_hold_s = 0.3;
        double min_speed = 2.0;
        double max_curv = 1.0 / 6.0;
        double assumed_speed = 8.0;
        double spd_xr = 3.0, spd_z_far = 25.0, spd_window_s = 0.6;
        double spd_hold_s = 5.0;
        int    spd_min_pts = 4;
        double spd_min_px = 0.7;
        double spd_hood_gap_m = 1.5;
        double move_px = 0.5, still_frac = 0.1;
        double act_up_s = 0.1, act_down_s = 3.0, act_min = 1.0;
    };

    EgoMotion() = default;
    explicit EgoMotion(const Params& p) : m_p(p) {}
    const Params& params() const { return m_p; }

    void reset() {
        m_prev.release(); m_prevT = -1.0; m_yaw = 0.0; m_lastGoodT = -1.0; m_turn = 0; m_turnCandT = -1.0;
        m_ex = m_ey = 0.0; m_spd.clear(); m_lastSpd = -1.0; m_lastSpdT = -1.0; m_out = EgoOutput(); m_act.release(); m_qPrev.release();
    }

    const EgoOutput& update(double t, const cv::Mat& gray, double scale, const RoadGeometry& g,
                            const std::vector<cv::Rect>& boxes, double ext_mps) {
        if (gray.empty() || !g.valid()) return m_out;
        const double dt = m_prevT > 0 ? t - m_prevT : 0.0;
        if (!m_prev.empty() && m_prev.size() == gray.size() && dt > 0.005 && dt < 0.3) {
            activity(gray, dt);
            double wy = 0.0, dz = -1.0;
            int inl = 0;
            if (measure(gray, scale, g, boxes, wy, dz, inl) && std::abs(wy / dt) < m_p.max_yaw) {
                const double a = 1.0 - std::exp(-dt / m_p.yaw_tau_s);
                m_yaw += a * (wy / dt - m_yaw);
                m_lastGoodT = t;
                m_out.inliers = inl;
                if (dz >= 0.0) m_spd.emplace_back(t, dz / dt);
            }
        } else if (dt >= 0.3) {
            m_yaw = 0.0; m_ex = m_ey = 0.0;
        }
        while (!m_spd.empty() && t - m_spd.front().first > m_p.spd_window_s) m_spd.pop_front();
        gray.copyTo(m_prev);
        m_prevT = t;
        m_out.valid = m_lastGoodT > 0 && t - m_lastGoodT < m_p.lost_s;
        if (!m_out.valid) m_yaw = 0.0;
        m_out.yaw_rate = m_yaw;
        m_out.foe_x = m_ex; m_out.foe_y = m_ey;

        if (ext_mps >= 0.0) { m_out.speed_mps = ext_mps; m_out.speed_src = 1; }
        else if (m_spd.size() >= 4) { m_out.speed_mps = m_lastSpd = medianSpeed(); m_lastSpdT = t; m_out.speed_src = 2; }
        else if (m_lastSpdT > 0 && t - m_lastSpdT < m_p.spd_hold_s) { m_out.speed_mps = m_lastSpd; m_out.speed_src = 2; }
        else { m_out.speed_mps = -1.0; m_out.speed_src = 0; }

        const double v = m_out.speed_mps >= 0.0 ? std::max(m_out.speed_mps, m_p.min_speed) : m_p.assumed_speed;
        m_out.curvature = m_out.valid ? std::clamp(m_yaw / v, -m_p.max_curv, m_p.max_curv) : 0.0;

        const double aw = std::abs(m_yaw);
        if (!m_out.valid) { m_turn = 0; m_turnCandT = -1.0; }
        else if (m_turn == 0) {
            if (aw > m_p.turn_enter) {
                if (m_turnCandT < 0) m_turnCandT = t;
                if (t - m_turnCandT >= m_p.turn_hold_s) m_turn = m_yaw > 0 ? 1 : -1;
            } else m_turnCandT = -1.0;
        } else if (aw < m_p.turn_exit || (m_yaw > 0) != (m_turn > 0)) { m_turn = 0; m_turnCandT = -1.0; }
        m_out.turn = m_turn;
        return m_out;
    }

    const EgoOutput& output() const { return m_out; }

    struct DebugPt { cv::Point2f p, flow; float w, dz; };
    bool debug = false;
    std::vector<DebugPt> dbg;

private:
    struct Pt { double x, y, fx, fy, w; };

    static void rotFlow(double x, double y, const cv::Vec3d& w, double& fx, double& fy) {
        fx = w[0] * x * y - w[1] * (1 + x * x) + w[2] * y;
        fy = w[0] * (1 + y * y) - w[1] * x * y - w[2] * x;
    }

    void detect(const RoadGeometry& g, double s, const std::vector<cv::Rect>& boxes) {
        const int W = m_prev.cols, H = m_prev.rows;
        const int y0 = std::clamp(int(g.vh * s - m_p.band_up * H), 0, H - 1);
        const int y1 = std::clamp(int(g.hood * s) - 2, y0 + 1, H);
        m_mask.create(H, W, CV_8UC1);
        m_mask.setTo(0);
        m_mask.rowRange(y0, y1).setTo(255);
        if (!m_act.empty()) {
            cv::resize(m_act < m_p.act_min, m_still, m_mask.size(), 0, 0, cv::INTER_NEAREST);
            m_mask.setTo(0, m_still);
        }
        for (const auto& b : boxes) {
            const cv::Rect r = cv::Rect(int(b.x * s) - 3, int(b.y * s) - 3, int(b.width * s) + 6, int(b.height * s) + 6)
                               & cv::Rect(0, 0, W, H);
            if (r.area() > 0) m_mask(r).setTo(0);
        }
        m_p0.clear();
        const int cw = W / m_p.grid_cols, ch = std::max(4, (y1 - y0) / m_p.grid_rows);
        for (int r = y0; r + 4 <= y1; r += ch)
            for (int c = 0; c + cw <= W; c += cw) {
                const cv::Rect cell(c, r, cw, std::min(ch, y1 - r));
                m_cell.clear();
                cv::goodFeaturesToTrack(m_prev(cell), m_cell, m_p.per_cell, 0.02, 4.0, m_mask(cell), 5);
                for (const auto& q : m_cell) m_p0.push_back(q + cv::Point2f(float(c), float(r)));
            }
    }

    bool measure(const cv::Mat& gray, double s, const RoadGeometry& g, const std::vector<cv::Rect>& boxes,
                 double& wy, double& dz, int& inliers) {
        detect(g, s, boxes);
        if (int(m_p0.size()) < m_p.min_inliers) return false;
        const cv::TermCriteria tc(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
        cv::calcOpticalFlowPyrLK(m_prev, gray, m_p0, m_p1, m_st, m_err, cv::Size(15, 15), 3, tc, 0, m_p.min_eig);
        cv::calcOpticalFlowPyrLK(gray, m_prev, m_p1, m_pb, m_stb, m_err, cv::Size(15, 15), 3, tc, 0, m_p.min_eig);
        const double f = g.f * s, cx = g.cx * s, vh = g.vh * s;
        m_pts.clear();
        for (size_t i = 0; i < m_p0.size(); ++i) {
            if (!m_st[i] || !m_stb[i] || cv::norm(m_pb[i] - m_p0[i]) > m_p.fb_max_px) continue;
            m_pts.push_back({(m_p0[i].x - cx) / f, (m_p0[i].y - vh) / f, (m_p1[i].x - m_p0[i].x) / f,
                             (m_p1[i].y - m_p0[i].y) / f, 1.0});
        }
        if (int(m_pts.size()) < m_p.min_inliers) return false;

        cv::Vec3d w(0, 0, 0);
        double ex = m_ex, ey = m_ey;
        const double noise = 0.15 / f;
        for (int outer = 0; outer < 3; ++outer) {
            for (auto& p : m_pts) p.w = 1.0;
            for (int it = 0; it < 4; ++it) {
                cv::Matx33d A = cv::Matx33d::zeros(); cv::Vec3d b(0, 0, 0);
                for (const auto& p : m_pts) {
                    const double rx = p.x - ex, ry = p.y - ey, r = std::hypot(rx, ry);
                    if (r < 0.04 || p.w <= 0) continue;
                    const double nx = -ry / r, ny = rx / r;
                    const cv::Vec3d c((nx * p.x * p.y + ny * (1 + p.y * p.y)),
                                      (-nx * (1 + p.x * p.x) - ny * p.x * p.y),
                                      (nx * p.y - ny * p.x));
                    const double m = nx * p.fx + ny * p.fy;
                    A += p.w * (c * c.t()); b += p.w * m * c;
                }
                A(0, 0) += 1e-6; A(1, 1) += 1e-6; A(2, 2) += 1e-4;
                w = A.inv(cv::DECOMP_CHOLESKY) * b;
                m_res.clear();
                for (const auto& p : m_pts) {
                    const double rx = p.x - ex, ry = p.y - ey, r = std::hypot(rx, ry);
                    double fx, fy; rotFlow(p.x, p.y, w, fx, fy);
                    m_res.push_back(r < 0.04 ? 0.0 : std::abs((-ry * (p.fx - fx) + rx * (p.fy - fy)) / r));
                }
                m_tmp = m_res;
                std::nth_element(m_tmp.begin(), m_tmp.begin() + m_tmp.size() / 2, m_tmp.end());
                const double c = 4.685 * std::max(noise, 1.4826 * m_tmp[m_tmp.size() / 2]);
                for (size_t i = 0; i < m_pts.size(); ++i) {
                    const double u = m_res[i] / c;
                    m_pts[i].w = u < 1.0 ? (1 - u * u) * (1 - u * u) : 0.0;
                }
            }
            cv::Matx22d A(1, 0, 0, 1); cv::Vec2d b(m_ex, m_ey);
            const double lam = 1.0 / (m_p.foe_prior * m_p.foe_prior);
            A *= lam; b *= lam;
            int nf = 0;
            for (const auto& p : m_pts) {
                if (p.w < 0.3) continue;
                double fx, fy; rotFlow(p.x, p.y, w, fx, fy);
                const double dx = p.fx - fx, dy = p.fy - fy, d = std::hypot(dx, dy);
                if (d < 3 * noise) continue;
                const cv::Vec2d a(dy / d, -dx / d);
                const double rhs = (p.x * dy - p.y * dx) / d;
                const double sig = std::hypot(p.x - ex, p.y - ey) * noise / d;
                const double wt = p.w / (sig * sig + 0.02 * 0.02);
                A += wt * (a * a.t()); b += wt * rhs * a; ++nf;
            }
            if (nf >= 10) {
                const cv::Vec2d e = A.inv() * b;
                ex = std::clamp(e[0], -m_p.foe_max, m_p.foe_max);
                ey = std::clamp(e[1], -m_p.foe_max, m_p.foe_max);
            }
        }
        inliers = 0;
        for (const auto& p : m_pts) inliers += p.w > 0.1;
        if (inliers < m_p.min_inliers) return false;
        m_ex = ex; m_ey = ey;
        wy = w[1];

        dz = -1.0;
        const double h = g.cam_h, zNear = g.nearZ();
        if (debug) {
            dbg.clear();
            for (const auto& p : m_pts)
                dbg.push_back({cv::Point2f(float(p.x * f + cx), float(p.y * f + vh)), cv::Point2f(float(p.fx * f), float(p.fy * f)), float(p.w), -1.f});
        }
        m_tmp.clear();
        for (const auto& p : m_pts) {
            const double yr = p.y - ey;
            if (p.w < 0.3 || yr < 0.02) continue;
            const double z0 = h / yr;
            if (z0 < zNear + m_p.spd_hood_gap_m || z0 > m_p.spd_z_far || std::abs(p.x * z0 - g.car_x) > m_p.spd_xr) continue;
            double fx, fy; rotFlow(p.x, p.y, w, fx, fy);
            const double y1 = yr + p.fy - fy;
            if (y1 <= 0.0) continue;
            m_tmp.push_back((y1 - yr) * f >= m_p.spd_min_px ? z0 - h / y1 : 0.0);
            if (debug) dbg.push_back({cv::Point2f(float(p.x * f + cx), float(p.y * f + vh)), cv::Point2f(float(p.fx * f), float(p.fy * f)),
                                      float(p.w), float(m_tmp.back())});
        }
        const size_t nRoad = m_tmp.size();
        m_tmp.erase(std::remove_if(m_tmp.begin(), m_tmp.end(), [](double d) { return d <= 0.0; }), m_tmp.end());
        if (int(m_tmp.size()) >= m_p.spd_min_pts) {
            std::nth_element(m_tmp.begin(), m_tmp.begin() + m_tmp.size() / 2, m_tmp.end());
            dz = m_tmp[m_tmp.size() / 2];
        } else if (int(nRoad) >= m_p.spd_min_pts) {
            int moving = 0, all = 0;
            for (const auto& p : m_pts) {
                if (p.w < 0.3) continue;
                double fx, fy; rotFlow(p.x, p.y, w, fx, fy);
                moving += std::hypot(p.fx - fx, p.fy - fy) * f > m_p.move_px;
                ++all;
            }
            if (moving < m_p.still_frac * all) dz = 0.0;
        }
        return true;
    }

    void activity(const cv::Mat& gray, double dt) {
        cv::resize(gray, m_q, cv::Size(gray.cols / 4, gray.rows / 4), 0, 0, cv::INTER_AREA);
        if (m_qPrev.size() != m_q.size()) { m_q.copyTo(m_qPrev); m_act = cv::Mat(m_q.size(), CV_32F, cv::Scalar(10.0)); return; }
        cv::absdiff(m_q, m_qPrev, m_diff);
        m_q.copyTo(m_qPrev);
        m_diff.convertTo(m_diffF, CV_32F);
        const float up = float(1.0 - std::exp(-dt / m_p.act_up_s)), down = float(1.0 - std::exp(-dt / m_p.act_down_s));
        for (int r = 0; r < m_act.rows; ++r) {
            float* a = m_act.ptr<float>(r);
            const float* d = m_diffF.ptr<float>(r);
            for (int c = 0; c < m_act.cols; ++c) a[c] += (d[c] > a[c] ? up : down) * (d[c] - a[c]);
        }
    }

    double medianSpeed() {
        m_tmp.clear();
        for (const auto& q : m_spd) m_tmp.push_back(q.second);
        std::nth_element(m_tmp.begin(), m_tmp.begin() + m_tmp.size() / 2, m_tmp.end());
        return m_tmp[m_tmp.size() / 2];
    }

    Params m_p;
    EgoOutput m_out;
    cv::Mat m_prev, m_mask, m_act, m_diff, m_diffF, m_q, m_qPrev, m_still;
    double m_prevT = -1.0, m_yaw = 0.0, m_lastGoodT = -1.0, m_turnCandT = -1.0;
    double m_ex = 0.0, m_ey = 0.0, m_lastSpd = -1.0, m_lastSpdT = -1.0;
    int m_turn = 0;
    std::vector<cv::Point2f> m_p0, m_p1, m_pb, m_cell;
    std::vector<uchar> m_st, m_stb;
    std::vector<float> m_err;
    std::vector<Pt> m_pts;
    std::vector<double> m_res, m_tmp;
    std::deque<std::pair<double, double>> m_spd;
};
