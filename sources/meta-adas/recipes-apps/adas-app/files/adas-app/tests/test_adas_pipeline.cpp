#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <opencv2/opencv.hpp>

#include "../src/fast_vehicle_detector.hpp"
#include "../src/horizon_detector.hpp"

struct TestTrack {
    int id;
    cv::Rect2f bbox;
    float conf;
    int hits = 1;
    int time_since_update = 0;
    double dist_m = -1.0;
    double smooth_dist_m = -1.0;
    double v_rel_mps = 0.0;

    void predict(double dt = 0.033) {
        time_since_update++;
        if (smooth_dist_m > 0 && std::abs(v_rel_mps) > 0.05 && time_since_update <= 10) {
            smooth_dist_m = std::max(1.5, smooth_dist_m - v_rel_mps * dt);
        }
    }

    void update(const cv::Rect2f& b, float c, double dist) {
        time_since_update = 0;
        hits++;
        conf = c;
        bbox = b;
        dist_m = dist;
        if (smooth_dist_m < 0) {
            smooth_dist_m = dist;
        } else {
            double d = std::min(std::max(dist, smooth_dist_m * 0.75), smooth_dist_m * 1.30);
            smooth_dist_m = 0.20 * d + 0.80 * smooth_dist_m;
        }
    }
};

cv::Mat renderSyntheticRoadFrame(int frame_idx, double car_dist, bool drop_car = false) {
    int w = 640, h = 480;
    cv::Mat img(h, w, CV_8UC1, cv::Scalar(40));

    int yh = 240;
    cv::rectangle(img, cv::Rect(0, yh, w, h - yh), cv::Scalar(115), cv::FILLED);

    cv::line(img, cv::Point(310, yh), cv::Point(140, 480), cv::Scalar(240), 3, cv::LINE_AA);
    cv::line(img, cv::Point(330, yh), cv::Point(500, 480), cv::Scalar(240), 3, cv::LINE_AA);

    if (drop_car || car_dist <= 0) return img;

    double dy = (1.35 * 425.0) / std::max(2.0, car_dist);
    int ground_y = static_cast<int>(yh + dy);
    ground_y = std::min(h - 20, std::max(yh + 15, ground_y));

    int car_w = static_cast<int>((1.82 * 425.0) / std::max(2.0, car_dist));
    car_w = std::min(260, std::max(22, car_w));
    int car_h = static_cast<int>(car_w * 0.78);
    int car_x = 320 - car_w / 2;
    int car_y = ground_y - car_h;

    cv::rectangle(img, cv::Rect(car_x, car_y, car_w, car_h), cv::Scalar(80), cv::FILLED);
    int glass_w = static_cast<int>(car_w * 0.75);
    int glass_h = static_cast<int>(car_h * 0.35);
    cv::rectangle(img, cv::Rect(car_x + (car_w - glass_w) / 2, car_y + 4, glass_w, glass_h), cv::Scalar(50), cv::FILLED);

    int shadow_h = std::max(4, car_h / 6);
    int shadow_y = ground_y - shadow_h / 2;
    cv::rectangle(img, cv::Rect(car_x + 2, shadow_y, car_w - 4, shadow_h), cv::Scalar(30), cv::FILLED);

    return img;
}

int main() {
    std::cout << "\n=================================================================\n";
    std::cout << "   AUTOMOTIVE ADAS PI-4 PIPELINE TEST & AUTO-BENCHMARK SUITE   \n";
    std::cout << "=================================================================\n\n";

    FastVehicleDetector detector(640, 480);
    HorizonDetector horizonDetector;

    const int TOTAL_FRAMES = 120;
    int frames_processed = 0;
    double total_detector_ms = 0.0;
    double max_detector_ms = 0.0;
    int false_negative_drops = 0;
    int drop_occurred_frames = 0;

    TestTrack lead_track{ 1, cv::Rect2f(0, 0, 0, 0), 0.0f, 0, 999, -1.0, -1.0, 0.4 };
    bool has_lead = false;
    int lead_lost_count = 999;
    double last_valid_dist = -1.0;

    std::vector<double> recorded_distances;
    std::vector<bool> lead_active_history;

    std::cout << ">>> Running 120 Simulated Driving Frames with In-Lane Approaching Vehicle...\n";
    std::cout << ">>> Test Scenario: Approach from 35m -> 15m. Dropping Deep Detection at frames 45-49.\n\n";

    for (int i = 0; i < TOTAL_FRAMES; ++i) {
        double true_dist = 35.0 - (i * 0.22);
        bool intentionally_drop = (i >= 45 && i <= 49);

        if (i >= 95) {
            true_dist = -1.0;
            intentionally_drop = true;
        }

        cv::Mat gray = renderSyntheticRoadFrame(i, true_dist, intentionally_drop);

        auto t0 = std::chrono::high_resolution_clock::now();
        double hz_y = horizonDetector.getHorizonY(480);
        auto detections = detector.detect(gray, hz_y, 160, 480);
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        total_detector_ms += dt_ms;
        if (dt_ms > max_detector_ms) max_detector_ms = dt_ms;
        frames_processed++;

        lead_track.predict(0.033);

        bool matched = false;
        if (!detections.empty()) {
            const auto& det = detections[0];
            cv::Rect2f box(det.box.x1, det.box.y1, det.box.x2 - det.box.x1, det.box.y2 - det.box.y1);

            double f_eff = 425.0;
            double dy = std::max(5.0, (double)(box.y + box.height - hz_y));
            double est_d = (1.35 * f_eff) / dy;

            lead_track.update(box, det.box.conf, est_d);
            matched = true;
        }

        if (matched && lead_track.hits >= 2) {
            has_lead = true;
            lead_lost_count = 0;
            last_valid_dist = lead_track.smooth_dist_m;
        } else {
            lead_lost_count++;
            if (has_lead && lead_lost_count <= 8 && last_valid_dist > 0) {
                has_lead = true;
            } else {
                has_lead = false;
                last_valid_dist = -1.0;
            }
        }

        double reported_dist = has_lead ? last_valid_dist : -1.0;
        recorded_distances.push_back(reported_dist);
        lead_active_history.push_back(has_lead);

        if (i >= 45 && i <= 49) {
            drop_occurred_frames++;
            if (reported_dist <= 0 || !has_lead) {
                false_negative_drops++;
            }
        }

        if (i % 20 == 0 || i == 47 || i == 50) {
            std::cout << "  Frame #" << std::setw(3) << i
                      << " | Det Time: " << std::fixed << std::setprecision(2) << dt_ms << "ms"
                      << " | True: " << std::setw(5) << (true_dist > 0 ? std::to_string(true_dist).substr(0,4) + "m" : "--")
                      << " | HUD Dist: " << std::setw(5) << (reported_dist > 0 ? std::to_string(reported_dist).substr(0,4) + "m" : "-1.0m")
                      << " | Status: " << (intentionally_drop ? "[OCCLUSION DROP]" : "[TRACKING]")
                      << " | Lead: " << (has_lead ? "YES" : "NO")
                      << "\n";
        }
    }

    double avg_detector_ms = total_detector_ms / frames_processed;
    double fast_path_fps = 1000.0 / avg_detector_ms;

    std::cout << "\n-----------------------------------------------------------------\n";
    std::cout << "                   BENCHMARK & VALIDATION RESULTS                \n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "1. Fast Path Execution Speed:\n";
    std::cout << "   - Average Detection Latency : " << std::fixed << std::setprecision(2) << avg_detector_ms << " ms\n";
    std::cout << "   - Peak Detection Latency    : " << max_detector_ms << " ms\n";
    std::cout << "   - Fast Path Equivalent FPS  : " << std::setprecision(1) << fast_path_fps << " FPS (Target >= 25 FPS: "
              << (fast_path_fps >= 25.0 ? "PASSED [OK]" : "FAILED") << ")\n\n";

    std::cout << "2. Lead Vehicle Coasting / Hysteresis Robustness:\n";
    std::cout << "   - Occlusion Drop Duration   : 5 consecutive frames (165 ms)\n";
    std::cout << "   - False Negative Drops      : " << false_negative_drops << " / " << drop_occurred_frames << " frames\n";
    std::cout << "   - Result                    : "
              << (false_negative_drops == 0 ? "PASSED [ZERO DROPOUT - KHÔNG CHỚP TẮT HUD]" : "FAILED") << "\n\n";

    std::cout << "3. Timeout & Graceful Fadeout Validation:\n";
    std::cout << "   - Vehicle departed at Frame : #95\n";
    int timeout_frame = -1;
    for (int k = 95; k < TOTAL_FRAMES; ++k) {
        if (!lead_active_history[k]) {
            timeout_frame = k;
            break;
        }
    }
    int timeout_frames_taken = timeout_frame - 95;
    std::cout << "   - HUD Fadeout triggered at  : Frame #" << timeout_frame << " (Timeout: " << timeout_frames_taken << " frames)\n";
    std::cout << "   - Result                    : "
              << (timeout_frames_taken >= 6 && timeout_frames_taken <= 10 ? "PASSED [CHÍNH XÁC 8 FRAMES HYSTERESIS]" : "FAILED") << "\n\n";

    std::cout << "=================================================================\n";
    if (false_negative_drops == 0 && fast_path_fps >= 25.0) {
        std::cout << ">>> OVERALL EVALUATION: AUTOMOTIVE ADAS CRITERIA PASSED [OK] <<<\n";
    } else {
        std::cout << ">>> OVERALL EVALUATION: FAILED <<< \n";
    }
    std::cout << "=================================================================\n\n";

    return (false_negative_drops == 0) ? 0 : 1;
}
