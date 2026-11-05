#pragma once

#include "adas_types.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <vector>

class FrameHistory {
public:
    explicit FrameHistory(int depth = 16, int width = 320) : m_depth(depth), m_width(width) {}

    void clear() { m_frames.clear(); }
    bool empty() const { return m_frames.empty(); }
    const cv::Mat& newest() const { return m_frames.back().second; }
    double scale() const { return m_scale; }

    void push(int64_t frame, const cv::Mat& bgr) {
        if (!m_frames.empty() && m_frames.back().first != frame - 1) m_frames.clear();
        m_scale = std::min(1.0, double(m_width) / bgr.cols);
        cv::Mat small, gray;
        if (m_scale < 1.0) cv::resize(bgr, small, cv::Size(), m_scale, m_scale, cv::INTER_AREA);
        else small = bgr;
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        m_frames.emplace_back(frame, gray);
        if (int(m_frames.size()) > m_depth) m_frames.erase(m_frames.begin());
    }

    bool carry(std::vector<VehicleDetection>& dets, int64_t from) const {
        if (m_frames.empty() || dets.empty() || from >= m_frames.back().first) return true;
        int k = 0;
        while (k < int(m_frames.size()) && m_frames[k].first != from) ++k;
        if (k == int(m_frames.size())) return false;

        const cv::Mat& g0 = m_frames[k].second;
        const cv::Rect img(0, 0, g0.cols, g0.rows);
        std::vector<cv::Point2f> p0;
        std::vector<int> owner;
        const float s = float(m_scale);
        for (int d = 0; d < int(dets.size()); ++d) {
            const auto& b = dets[d].box;
            const float w = s * (b.x2 - b.x1), h = s * (b.y2 - b.y1);
            const cv::Rect r = cv::Rect(int(s * b.x1 + 0.15f * w), int(s * b.y1 + 0.10f * h), int(0.70f * w), int(0.80f * h)) & img;
            if (r.width < 6 || r.height < 6) continue;
            std::vector<cv::Point2f> c;
            cv::goodFeaturesToTrack(g0(r), c, 20, 0.01, std::max(1.5, r.width / 12.0));
            for (const auto& q : c) { p0.push_back(q + cv::Point2f(float(r.x), float(r.y))); owner.push_back(d); }
        }
        if (p0.empty()) return true;
        std::vector<cv::Point2f> pts = p0, next;
        std::vector<char> alive(p0.size(), 1);
        std::vector<uchar> st;
        std::vector<float> err;
        for (; k + 1 < int(m_frames.size()); ++k) {
            cv::calcOpticalFlowPyrLK(m_frames[k].second, m_frames[k + 1].second, pts, next, st, err, cv::Size(11, 11), 2);
            for (size_t i = 0; i < pts.size(); ++i)
                if (!st[i] || err[i] > 20.f) alive[i] = 0;
            pts.swap(next);
        }
        auto median = [](std::vector<float>& v) {
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            return v[v.size() / 2];
        };
        for (int d = 0; d < int(dets.size()); ++d) {
            std::vector<int> idx;
            for (int i = 0; i < int(p0.size()); ++i) if (owner[i] == d && alive[i]) idx.push_back(i);
            if (idx.size() < 5) continue;
            std::vector<float> dx, dy, ratio;
            for (int i : idx) { dx.push_back(pts[i].x - p0[i].x); dy.push_back(pts[i].y - p0[i].y); }
            for (size_t a = 0; a < idx.size(); ++a) {
                const size_t b = (a + idx.size() / 2) % idx.size();
                const double d0 = cv::norm(p0[idx[a]] - p0[idx[b]]);
                if (d0 > 3.0) ratio.push_back(float(cv::norm(pts[idx[a]] - pts[idx[b]]) / d0));
            }
            const float sc = ratio.size() >= 3 ? std::clamp(median(ratio), 0.7f, 1.5f) : 1.f;
            auto& b = dets[d].box;
            const float cx = 0.5f * (b.x1 + b.x2) + median(dx) / s, cy = 0.5f * (b.y1 + b.y2) + median(dy) / s;
            const float hw = 0.5f * sc * (b.x2 - b.x1), hh = 0.5f * sc * (b.y2 - b.y1);
            b.x1 = cx - hw; b.x2 = cx + hw; b.y1 = cy - hh; b.y2 = cy + hh;
        }
        return true;
    }

private:
    int m_depth, m_width;
    double m_scale = 1.0;
    std::vector<std::pair<int64_t, cv::Mat>> m_frames;
};
