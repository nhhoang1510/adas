#include "nanodet_engine.hpp"
#include "detection_filter.hpp"
#include "lane_detector.hpp"
#include "vehicle_tracker.hpp"
#include <opencv2/imgcodecs.hpp>
int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : "frames";
    const int N = argc > 2 ? atoi(argv[2]) : 90;
    CameraCalibration c; c.hfov_deg = 100; c.camera_height_m = 1.3; c.horizon_y_ratio = 0.478; c.hood_y_ratio = 0.728;
    NanoDetEngine::Config dc; dc.onnx_path = getenv("NANODET") ? getenv("NANODET") : "nanodet.onnx"; NanoDetEngine det(dc);
    LaneDetector ld; VehicleTracker vt; cv::Mat mask, gray; std::vector<cv::Rect> ex;
    for (int i = 0; i < N; ++i) {
        char p[128]; std::snprintf(p, sizeof p, "%s/%03d.png", dir, i);
        cv::Mat f = cv::imread(p); if (f.empty()) break;
        const double t = i / 30.0;
        const RoadGeometry g = RoadGeometry::make(c, f.cols, f.rows, c.horizon_y_ratio * f.rows, int(c.hood_y_ratio * f.rows));
        std::vector<VehicleDetection> dets; bool fresh = (i % 2 == 0);
        if (fresh) { dets = det.infer(f); std::vector<VehicleDetection> k; for (auto& d : dets) if (d.box.y1 < g.hood) k.push_back(d); dets = filterDetections(k, g); }
        laneMarkingMask(f, g, mask, gray, ex);
        ld.predict(t); ld.update(mask, true, g); LaneOutput lo = ld.output(g, t);
        auto& tr = vt.step(t, fresh ? &dets : nullptr, g, lo, -1.0);
        ex.clear(); for (auto& d : dets) ex.emplace_back(cv::Point(int(d.box.x1) - 4, int(d.box.y1) - 4), cv::Point(int(d.box.x2) + 4, int(d.box.y2) + 4));
        for (auto& v : tr) ex.emplace_back(v.box);
        const auto& L = vt.lead();
        if (i % 6 == 0 || argc > 3) {
            std::printf("%2d lanes %s [%.2f %.2f] | lead %s", i, lo.has_lane ? "ok" : "--", lo.left.c0, lo.right.c0, L.has_lead ? "" : "none");
            if (L.has_lead) std::printf("corr@z [%.2f %.2f] ", lo.left_trk.x(L.lead.z), lo.right_trk.x(L.lead.z));
            if (L.has_lead) std::printf("#%d z %.1f x %+.2f w %.2f box[%.0f %.0f %.0f %.0f]", L.lead.id, L.lead.z, L.lead.x, L.lead.width_m, L.lead.box.x, L.lead.box.y, L.lead.box.width, L.lead.box.height);
            std::printf("\n");
            if (i % 30 == 0) for (auto& v : tr) std::printf("      #%d %-9s z %5.1f x %+.2f w %.2f %s box[%.0f %.0f %.0f %.0f]\n", v.id, vehicleClassName(v.cls), v.z, v.x, v.width_m, v.lane_pos.c_str(), v.box.x, v.box.y, v.box.width, v.box.height);
        }
        if (i == 30) {
            cv::Mat d = f.clone();
            for (size_t k = 1; k < lo.left_img.size(); ++k) { cv::line(d, lo.left_img[k-1], lo.left_img[k], {0,0,255}, 2); cv::line(d, lo.right_img[k-1], lo.right_img[k], {255,0,0}, 2); }
            for (auto& v : tr) { bool isL = L.has_lead && v.id == L.lead.id; cv::rectangle(d, v.box, isL ? cv::Scalar(0,0,255) : cv::Scalar(255,255,255), isL ? 2 : 1);
                char s[64]; std::snprintf(s, sizeof s, "#%d %.1fm %s", v.id, v.z, v.lane_pos.c_str()); cv::putText(d, s, v.box.tl() + cv::Point2f(0, -3), 0, 0.33, {255,255,255}, 1); }
            cv::imwrite("full30.png", d);
        }
    }
}
