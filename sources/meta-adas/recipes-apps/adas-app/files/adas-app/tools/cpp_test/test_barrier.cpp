#include "lane_detector.hpp"
#include <opencv2/imgcodecs.hpp>
int main() {
    CameraCalibration c; c.hfov_deg = 100; c.camera_height_m = 1.3; c.horizon_y_ratio = 0.478; c.hood_y_ratio = 0.728;
    for (const char* fn : {"../day.png", "../barrier.png"}) {
        cv::Mat f = cv::imread(fn), m, gr;
        const RoadGeometry g = RoadGeometry::make(c, f.cols, f.rows, 0.478 * f.rows, int(0.728 * f.rows));
        laneMarkingMask(f, g, m, gr, {});
        LaneDetector ld; ld.predict(0); ld.update(m, true, g); auto o = ld.output(g, 0);
        std::printf("%-14s px %d  L %.2f R %.2f w %.2f pts %d/%d %s\n", fn, cv::countNonZero(m), o.left.c0, o.right.c0, o.width_m, o.pts_left, o.pts_right, o.status.c_str());
    }
}
