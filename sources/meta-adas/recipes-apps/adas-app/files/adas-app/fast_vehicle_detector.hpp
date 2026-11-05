#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include "road_geometry.hpp"

class FastVehicleDetector {
public:
    FastVehicleDetector(int max_w = 640, int max_h = 480)
        : m_maxW(max_w)
        , m_maxH(max_h)
    {
        m_roiGray = cv::Mat(max_h, max_w, CV_8UC1);
        m_shadowMask = cv::Mat(max_h, max_w, CV_8UC1);
        m_morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 3));
    }

    std::vector<VehicleDetection> detect(const cv::Mat& gray,
                                         double horizon_y,
                                         int left_bound,
                                         int right_bound)
    {
        std::vector<VehicleDetection> detections;
        int w = gray.cols;
        int h = gray.rows;

        int y_start = static_cast<int>(std::max(horizon_y + 10.0, h * 0.35));
        int y_end = static_cast<int>(h * 0.88);
        if (y_start >= y_end - 20) return detections;

        int x_start = std::max(10, left_bound - 35);
        int x_end = std::min(w - 10, right_bound + 35);
        if (x_start >= x_end - 30) {
            x_start = static_cast<int>(w * 0.15);
            x_end = static_cast<int>(w * 0.85);
        }

        cv::Rect roiRect(x_start, y_start, x_end - x_start, y_end - y_start);
        roiRect &= cv::Rect(0, 0, w, h);
        if (roiRect.width < 30 || roiRect.height < 30) return detections;

        cv::Mat roi = gray(roiRect);

        cv::Rect roadSampleRect(roi.cols * 0.2, roi.rows * 0.6, roi.cols * 0.6, roi.rows * 0.35);
        roadSampleRect &= cv::Rect(0, 0, roi.cols, roi.rows);
        cv::Scalar roadMean = cv::mean(roi(roadSampleRect));
        double road_intensity = roadMean[0];

        int shadow_thresh = static_cast<int>(std::min(70.0, std::max(25.0, road_intensity * 0.52)));

        cv::Mat shadowRoi;
        cv::threshold(roi, shadowRoi, shadow_thresh, 255, cv::THRESH_BINARY_INV);

        cv::morphologyEx(shadowRoi, shadowRoi, cv::MORPH_CLOSE, m_morphKernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(shadowRoi, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& cnt : contours) {
            cv::Rect b = cv::boundingRect(cnt);

            if (b.width < 18 || b.width > w * 0.65) continue;
            if (b.height < 3 || b.height > 40) continue;
            double aspect = (double)b.width / (double)b.height;
            if (aspect < 1.6 || aspect > 12.0) continue;

            double y_ground = y_start + b.y + b.height;
            double dy = std::max(10.0, y_ground - horizon_y);
            double expected_w = (1.82 * 850.0 * (w / 1280.0)) / (1.35 * 850.0 * (w / 1280.0) / dy);
            expected_w = expected_w * 0.85;

            if (b.width < expected_w * 0.35 || b.width > expected_w * 2.8) continue;

            int veh_w = b.width;
            int veh_h = static_cast<int>(b.width * 0.78);
            int veh_x = x_start + b.x;
            int veh_y = y_start + b.y + b.height - veh_h;

            if (veh_x < 0) { veh_w += veh_x; veh_x = 0; }
            if (veh_x + veh_w > w) veh_w = w - veh_x;
            if (veh_y < 0) { veh_h += veh_y; veh_y = 0; }
            if (veh_y + veh_h > h) veh_h = h - veh_y;
            if (veh_w < 18 || veh_h < 15) continue;

            cv::Rect vehRect(veh_x, veh_y, veh_w, veh_h);
            vehRect &= cv::Rect(0, 0, w, h);
            if (vehRect.width < 16 || vehRect.height < 14) continue;

            cv::Mat vehCrop = gray(vehRect);
            double symmetry_score = computeVerticalSymmetry(vehCrop);
            if (symmetry_score < 0.50) continue;

            VehicleDetection vd;
            vd.box.x1 = static_cast<float>(vehRect.x);
            vd.box.y1 = static_cast<float>(vehRect.y);
            vd.box.x2 = static_cast<float>(vehRect.x + vehRect.width);
            vd.box.y2 = static_cast<float>(vehRect.y + vehRect.height);
            vd.box.conf = static_cast<float>(0.45f + 0.35f * symmetry_score);
            vd.box.cls = 0;

            detections.push_back(vd);
        }

        return applyNms(detections, 0.40f);
    }

private:
    int m_maxW;
    int m_maxH;
    cv::Mat m_roiGray;
    cv::Mat m_shadowMask;
    cv::Mat m_morphKernel;

    double computeVerticalSymmetry(const cv::Mat& crop) {
        if (crop.empty() || crop.cols < 10 || crop.rows < 8) return 0.0;

        cv::Mat small;
        cv::resize(crop, small, cv::Size(32, 24), 0, 0, cv::INTER_NEAREST);

        double diff_sum = 0.0;
        double norm_sum = 0.0;
        int half_w = small.cols / 2;

        for (int r = 0; r < small.rows; ++r) {
            const uchar* ptr = small.ptr<uchar>(r);
            for (int c = 0; c < half_w; ++c) {
                int left_val = ptr[c];
                int right_val = ptr[small.cols - 1 - c];
                diff_sum += std::abs(left_val - right_val);
                norm_sum += (left_val + right_val + 1);
            }
        }

        if (norm_sum < 1e-3) return 0.0;
        double symmetry = 1.0 - (diff_sum / norm_sum);
        return std::max(0.0, std::min(1.0, symmetry));
    }

    std::vector<VehicleDetection> applyNms(const std::vector<VehicleDetection>& dets, float iou_thresh) {
        if (dets.empty()) return {};

        std::vector<VehicleDetection> sorted = dets;
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return a.box.conf > b.box.conf;
        });

        std::vector<bool> suppressed(sorted.size(), false);
        std::vector<VehicleDetection> result;

        for (size_t i = 0; i < sorted.size(); ++i) {
            if (suppressed[i]) continue;
            result.push_back(sorted[i]);

            float w1 = sorted[i].box.x2 - sorted[i].box.x1;
            float h1 = sorted[i].box.y2 - sorted[i].box.y1;
            float area1 = w1 * h1;

            for (size_t j = i + 1; j < sorted.size(); ++j) {
                if (suppressed[j]) continue;

                float inter_x1 = std::max(sorted[i].box.x1, sorted[j].box.x1);
                float inter_y1 = std::max(sorted[i].box.y1, sorted[j].box.y1);
                float inter_x2 = std::min(sorted[i].box.x2, sorted[j].box.x2);
                float inter_y2 = std::min(sorted[i].box.y2, sorted[j].box.y2);

                float inter_w = std::max(0.0f, inter_x2 - inter_x1);
                float inter_h = std::max(0.0f, inter_y2 - inter_y1);
                float inter_area = inter_w * inter_h;

                float w2 = sorted[j].box.x2 - sorted[j].box.x1;
                float h2 = sorted[j].box.y2 - sorted[j].box.y1;
                float area2 = w2 * h2;

                float iou = inter_area / (area1 + area2 - inter_area + 1e-5f);
                if (iou > iou_thresh) {
                    suppressed[j] = true;
                }
            }
        }
        return result;
    }
};
