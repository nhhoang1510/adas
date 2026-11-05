#pragma once

#include "road_geometry.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

class BevTransformer {
public:
    struct Params {
        double x_min = -6.0, x_max = 6.0, z_max = 40.0;
        double res_x = 0.05, res_z = 0.10;
    };

    BevTransformer() : BevTransformer(Params{}) {}
    explicit BevTransformer(const Params& p) : m_p(p) {
        m_bw = int(std::lround((p.x_max - p.x_min) / p.res_x));
        m_bh = int(std::lround(p.z_max / p.res_z));
        m_mapXY.create(m_bh, m_bw, CV_32FC2);
        m_bevF.create(m_bh, m_bw, CV_8UC1);
    }

    int width() const { return m_bw; }
    int height() const { return m_bh; }
    const Params& params() const { return m_p; }

    int    colOf(double x) const { return int(std::floor((x - m_p.x_min) / m_p.res_x)); }
    int    rowOf(double z) const { return int(std::floor((m_p.z_max - z) / m_p.res_z)); }
    double xOf(int c) const { return m_p.x_min + (c + 0.5) * m_p.res_x; }
    double zOf(int r) const { return m_p.z_max - (r + 0.5) * m_p.res_z; }
    cv::Point toBev(double x, double z) const {
        return cv::Point(int((x - m_p.x_min) / m_p.res_x), int((m_p.z_max - z) / m_p.res_z));
    }
    static double nearZ(const RoadGeometry& g) { return std::clamp(g.nearZ(), 1.5, 15.0); }

    bool prepare(const RoadGeometry& g) {
        if (!g.valid()) return false;
        if (sameGeometry(g)) return true;
        m_g = g;
        const double zn = nearZ(g);
        m_rowNear = std::clamp(rowOf(zn), 0, m_bh);
        for (int r = 0; r < m_bh; ++r) {
            cv::Vec2f* m = m_mapXY.ptr<cv::Vec2f>(r);
            const double z = zOf(r);
            const bool usable = z > zn && z > 0.5 && (g.vh + g.f * g.cam_h / z) < g.hood;
            const float v = usable ? float(g.vh + g.f * g.cam_h / z) : -1e4f;
            const double k = usable ? g.f / z : 0.0;
            for (int c = 0; c < m_bw; ++c) {
                m[c][0] = usable ? float(g.cx + k * xOf(c)) : -1e4f;
                m[c][1] = v;
            }
        }
        cv::convertMaps(m_mapXY, cv::noArray(), m_map1, m_map2, CV_16SC2);
        return true;
    }

    void warp(const cv::Mat& src, cv::Mat& bev, int threshold = -1) const {
        if (m_map1.empty()) { bev.create(m_bh, m_bw, CV_8UC1); bev.setTo(0); return; }
        if (threshold < 0) {
            cv::remap(src, bev, m_map1, m_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        } else {
            cv::remap(src, m_bevF, m_map1, m_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
            cv::threshold(m_bevF, bev, threshold, 255, cv::THRESH_BINARY);
        }
        if (m_rowNear < m_bh) bev.rowRange(m_rowNear, m_bh).setTo(0);
    }

    const RoadGeometry& geometry() const { return m_g; }
    int rowNear() const { return m_rowNear; }

private:
    Params m_p;
    int m_bw = 0, m_bh = 0, m_rowNear = 0;
    RoadGeometry m_g;
    cv::Mat m_mapXY, m_map1, m_map2;
    mutable cv::Mat m_bevF;

    bool sameGeometry(const RoadGeometry& g) const {
        return m_g.valid() && m_g.W == g.W && m_g.H == g.H && m_g.hood == g.hood &&
               std::abs(m_g.f - g.f) < 0.05 && std::abs(m_g.cx - g.cx) < 0.05 &&
               std::abs(m_g.vh - g.vh) < 0.05 && std::abs(m_g.cam_h - g.cam_h) < 1e-4;
    }
};
