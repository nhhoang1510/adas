#include "nanodet_engine.hpp"
#include "detection_filter.hpp"
#include "lane_detector.hpp"
#include "vehicle_tracker.hpp"
#include "box_flow.hpp"
#include <opencv2/videoio.hpp>
#include <cstdio>
#include <deque>

struct Score { double iouSum = 0; int matched = 0, ref = 0; double ms = 0; };

static void score(const std::vector<TrackedVehicle>& tr, const std::vector<VehicleDetection>& ref, Score& s) {
    std::vector<char> used(tr.size(), 0);
    s.ref += int(ref.size());
    for (const auto& d : ref) {
        const cv::Rect2f r(d.box.x1, d.box.y1, d.box.x2 - d.box.x1, d.box.y2 - d.box.y1);
        int best = -1; float bi = 0.f;
        for (size_t k = 0; k < tr.size(); ++k) {
            if (used[k]) continue;
            const float in = (tr[k].box & r).area(), u = tr[k].box.area() + r.area() - in;
            const float iou = u > 0 ? in / u : 0.f;
            if (iou > bi) { bi = iou; best = int(k); }
        }
        if (best >= 0 && bi >= 0.3f) { used[best] = 1; s.iouSum += bi; ++s.matched; }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("test_flow <video> [frames] [period] [lag]\n"); return 1; }
    const int N = argc > 2 ? atoi(argv[2]) : 600, period = argc > 3 ? atoi(argv[3]) : 6, lag = argc > 4 ? atoi(argv[4]) : 6;
    cv::VideoCapture cap(argv[1]);
    CameraCalibration c; c.hfov_deg = 70; c.camera_height_m = 1.35; c.horizon_y_ratio = 0.45; c.hood_y_ratio = 0.88;
    NanoDetEngine::Config dc; dc.onnx_path = getenv("NANODET") ? getenv("NANODET") : "models/nanodet-plus-m_320.onnx";
    NanoDetEngine det(dc);
    if (!det.usesRealModel()) { std::printf("no model %s\n", dc.onnx_path.c_str()); return 1; }

    VehicleTracker vt[2];
    Score sc[2];
    FrameHistory hist;
    std::deque<std::pair<int, std::vector<VehicleDetection>>> pending;
    int n = 0;
    for (int i = 0; i < N; ++i, ++n) {
        cv::Mat f;
        if (!cap.read(f)) break;
        if (f.cols != 640) cv::resize(f, f, cv::Size(640, 640 * f.rows / f.cols), 0, 0, cv::INTER_AREA);
        const double t = i / 30.0;
        const RoadGeometry g = RoadGeometry::make(c, f.cols, f.rows, c.horizon_y_ratio * f.rows, int(c.hood_y_ratio * f.rows));
        const std::vector<VehicleDetection> ref = filterDetections(det.infer(f), g);
        if (i % period == 0) pending.push_back({i, ref});
        std::vector<VehicleDetection> late, carried;
        const bool fresh = !pending.empty() && i - pending.front().first >= lag;
        int detFrame = -1;
        if (fresh) { late = pending.front().second; detFrame = pending.front().first; pending.pop_front(); }
        const int64 t0 = cv::getTickCount();
        hist.push(i, f);
        carried = late;
        if (fresh) hist.carry(carried, detFrame);
        sc[1].ms += (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();
        LaneOutput lo;
        for (int m = 0; m < 2; ++m) {
            const auto& tr = vt[m].step(t, fresh ? (m ? &carried : &late) : nullptr, g, lo, -1.0);
            if (i >= 60) score(tr, ref, sc[m]);
        }
    }
    const char* name[2] = {"ket qua tre dung nguyen  ", "day ket qua tre ve hien tai"};
    for (int m = 0; m < 2; ++m)
        std::printf("%s | IoU TB %.3f | bat duoc %.1f%% xe tham chieu | %.2f ms/khung\n", name[m],
                    sc[m].matched ? sc[m].iouSum / sc[m].matched : 0.0, 100.0 * sc[m].matched / std::max(1, sc[m].ref),
                    sc[m].ms / std::max(1, n));
}
