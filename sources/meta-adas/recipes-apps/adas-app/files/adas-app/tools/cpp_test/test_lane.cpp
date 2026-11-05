#include "lane_detector.hpp"
#include "detection_filter.hpp"
#include "vehicle_tracker.hpp"
#include "auto_calibrator.hpp"
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <fstream>
#include <sstream>
int main(int argc, char** argv) {
    const double hz = 0.478, hood = 0.728;
    CameraCalibration c; c.hfov_deg = 100; c.camera_height_m = 1.3; c.horizon_y_ratio = hz; c.hood_y_ratio = hood;
    LaneDetector ld; VehicleTracker vt; AutoCalibrator ac;
    cv::Mat mask, gray; double tot = 0, totU = 0; int n = 0;
    std::ifstream bf("boxes.txt"); std::string line;
    double prevL = NAN, prevR = NAN, maxJ = 0;
    for (int i = 0; i < 90; ++i) {
        char p[64]; std::snprintf(p, sizeof p, "frames/%03d.png", i);
        cv::Mat f = cv::imread(p); if (f.empty()) break;
        const RoadGeometry g = RoadGeometry::make(c, f.cols, f.rows, hz * f.rows, int(hood * f.rows));
        auto t0 = std::chrono::steady_clock::now();
        std::vector<cv::Rect> ex; std::getline(bf, line); { std::istringstream is(line); int x,y,w,h; while (is >> x >> y >> w >> h) ex.emplace_back(x,y,w,h); }
        laneMarkingMask(f, g, mask, gray, ex);
        auto t1 = std::chrono::steady_clock::now();
        ld.predict(i / 30.0); ld.update(mask, true, g);
        LaneOutput o = ld.output(g, i / 30.0);
        auto t2 = std::chrono::steady_clock::now();
        ac.addLaneMask(mask, g.hood);
        tot += std::chrono::duration<double, std::milli>(t1 - t0).count();
        totU += std::chrono::duration<double, std::milli>(t2 - t1).count(); ++n;
        if (o.has_lane) {
            if (!std::isnan(prevL)) maxJ = std::max({maxJ, std::abs(o.left.c0 - prevL), std::abs(o.right.c0 - prevR)});
            prevL = o.left.c0; prevR = o.right.c0;
        }
        if (i % 9 == 0)
            std::printf("%2d %s L %.2f R %.2f %s%s w %.2f pts %d/%d off %+.2f %s\n", i, o.has_lane ? "ok" : "--", o.left.c0, o.right.c0,
                        o.left_synth ? "synL " : "", o.right_synth ? "synR" : "", o.width_m, o.pts_left, o.pts_right, o.offset_m, o.status.c_str());
        if (i == 0) { cv::Mat d = f.clone(); d.setTo(cv::Scalar(0, 255, 0), mask); ld.drawDebug(d, d.rows / 2);
            for (size_t k = 1; k < o.left_img.size(); ++k) { cv::line(d, o.left_img[k-1], o.left_img[k], {0,0,255}, 2); cv::line(d, o.right_img[k-1], o.right_img[k], {255,0,0}, 2); }
            cv::imwrite("cpp_dbg0.png", d); }
    }
    CameraCalibration c2 = c; ac.update(c2);
    std::printf("mask %.2f ms/frame, detector %.2f ms/frame, max jump %.3f m, autocal lane samples %d\n", tot / n, totU / n, maxJ, ac.laneSamples());
    std::vector<VehicleDetection> d(4);
    float B[4][6] = {{453,176,640,324,0.66f,0},{117,262,511,360,0.54f,0},{456,178,638,220,0.30f,0},{179,162,207,182,0.27f,0}};
    for (int k = 0; k < 4; ++k) { d[k].box = {B[k][0],B[k][1],B[k][2],B[k][3],B[k][4],int(B[k][5])}; }
    const RoadGeometry g = RoadGeometry::make(c, 640, 360, hz * 360, int(hood * 360));
    DetFilterStats st; auto kept = filterDetections(d, g, &st, true);
    std::printf("filter %d -> %d\n", st.in, st.kept);
    LaneOutput lo; auto& tr = vt.step(0.0, &kept, g, lo, -1.0); (void)tr;
    return 0;
}
