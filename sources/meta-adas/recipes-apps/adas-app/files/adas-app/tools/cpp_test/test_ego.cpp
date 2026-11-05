#include "ego_motion.hpp"
#include "nanodet_engine.hpp"
#include "detection_filter.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <cstdio>

static void drawPath(cv::Mat& img, const RoadGeometry& g, const EgoOutput& e) {
    const double len = e.speed_mps > 0 ? std::clamp(e.speed_mps * 3.0, 10.0, 40.0) : 25.0;
    const auto c = e.path(len, 0.5);
    for (int side = -1; side <= 1; side += 2) {
        std::vector<cv::Point> poly;
        for (size_t i = 0; i < c.size(); ++i) {
            const double th = std::atan2(i + 1 < c.size() ? c[i + 1].x - c[i].x : c[i].x - c[i - 1].x,
                                         i + 1 < c.size() ? c[i + 1].y - c[i].y : c[i].y - c[i - 1].y);
            const double x = c[i].x + side * 0.92 * std::cos(th) + g.car_x, z = c[i].y - side * 0.92 * std::sin(th);
            if (z < g.nearZ()) continue;
            poly.push_back(g.toImage(x, z));
        }
        if (poly.size() >= 2) cv::polylines(img, poly, false, cv::Scalar(255, 200, 0), 2, cv::LINE_AA);
    }
    char buf[128];
    std::snprintf(buf, sizeof buf, "yaw %+5.1f deg/s  v %s%4.1f m/s  R %s  %s", e.yawDegS(), e.speed_src == 2 ? "~" : "",
                  e.speed_mps, std::abs(e.curvature) > 1e-4 ? cv::format("%.0f m", 1 / std::abs(e.curvature)).c_str() : "-",
                  e.turn < 0 ? "TURN LEFT" : e.turn > 0 ? "TURN RIGHT" : "");
    cv::putText(img, buf, {10, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(img, buf, {10, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("test_ego <video> [frames] [out.avi]\n"); return 1; }
    const int N = argc > 2 ? atoi(argv[2]) : 0;
    const bool synth = getenv("SYNTH") != nullptr;
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) { std::printf("cannot open %s\n", argv[1]); return 1; }
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (!(fps > 1 && fps < 121)) fps = 30;
    CameraCalibration c;
    if (CameraCalibration::load(CameraCalibration::sidecarFor(argv[1]), c)) c.print();
    cv::VideoWriter vw;

    NanoDetEngine::Config dc; if (getenv("NANODET")) dc.onnx_path = getenv("NANODET");
    NanoDetEngine det(dc);
    const bool useDet = getenv("NANODET") && det.usesRealModel();
    EgoMotion em[2];
    em[0].debug = true;
    double errSum = 0, errMax = 0, refSum = 0; int errN = 0;
    double ms = 0; int i = 0;
    cv::Mat f, small, gray, rot, small2, gray2;
    for (; N <= 0 || i < N; ++i) {
        if (!cap.read(f)) break;
        if (f.cols != 640) cv::resize(f, f, cv::Size(640, 640 * f.rows / f.cols), 0, 0, cv::INTER_AREA);
        const double t = i / fps;
        const RoadGeometry g = RoadGeometry::make(c, f.cols, f.rows, c.horizon_y_ratio * f.rows, int(c.hood_y_ratio * f.rows));
        const double s = 0.5;
        cv::resize(f, small, cv::Size(), s, s, cv::INTER_AREA);
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> boxes;
        if (useDet)
            for (const auto& d : filterDetections(det.infer(f), g))
                boxes.emplace_back(cv::Point(int(d.box.x1), int(d.box.y1)), cv::Point(int(d.box.x2), int(d.box.y2)));
        const int64 t0 = cv::getTickCount();
        const EgoOutput e = em[0].update(t, gray, s, g, boxes, -1.0);
        ms += (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();

        if (synth) {
            const double A = (getenv("SYNTH_DEG") ? atof(getenv("SYNTH_DEG")) : 6.0) * M_PI / 180.0, T = 3.0;
            const double psi = A * std::sin(2 * M_PI * t / T), rate = A * 2 * M_PI / T * std::cos(2 * M_PI * t / T);
            const cv::Matx33d K(g.f, 0, g.cx, 0, g.f, g.vh, 0, 0, 1);
            const cv::Matx33d R(std::cos(psi), 0, -std::sin(psi), 0, 1, 0, std::sin(psi), 0, std::cos(psi));
            cv::warpPerspective(f, rot, cv::Mat(K * R * K.inv()), f.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            cv::resize(rot, small2, cv::Size(), s, s, cv::INTER_AREA);
            cv::cvtColor(small2, gray2, cv::COLOR_BGR2GRAY);
            if (getenv("SYNTH_OUT")) {
                static cv::VideoWriter sw(getenv("SYNTH_OUT"), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, rot.size());
                sw.write(rot);
            }
            const EgoOutput e2 = em[1].update(t, gray2, s, g, {}, -1.0);
            if (t > 1.0 && e.valid && e2.valid) {
                static double refF = 0.0; refF += (1 - std::exp(-1.0 / fps / 0.25)) * (rate - refF);
                const double err = (e2.yaw_rate - e.yaw_rate) - refF;
                errSum += err * err; errMax = std::max(errMax, std::abs(err)); refSum += refF * refF; ++errN;
            }
        }
        if (i % int(std::round(fps / 2)) == 0)
            std::printf("t %6.1f  foe %+5.2f %+5.2f  yaw %+6.1f deg/s  inl %3d  v %5.1f m/s%s  R %6.0f m  turn %+d\n", t, e.foe_x, e.foe_y, e.yawDegS(), e.inliers,
                        e.speed_mps, e.speed_src == 2 ? "(vis)" : "     ",
                        std::abs(e.curvature) > 1e-5 ? 1 / e.curvature : 0.0, e.turn);
        const int snap = getenv("SNAP") ? atoi(getenv("SNAP")) : 0;
        if ((argc > 3 && i % 2 == 0) || (snap > 0 && i % snap == 0)) {
            cv::Mat d = f.clone();
            cv::line(d, {0, int(g.vh)}, {d.cols, int(g.vh)}, cv::Scalar(80, 80, 80), 1);
            for (const auto& b : boxes) cv::rectangle(d, b, cv::Scalar(0, 0, 255), 1);
            for (const auto& q : em[0].dbg) {
                const cv::Point2f p = q.p / s, e = (q.p + 3.f * q.flow) / s;
                const cv::Scalar col = q.dz >= 0 ? cv::Scalar(255, 0, 255) : q.w > 0.3 ? cv::Scalar(0, 255, 255) : cv::Scalar(128, 128, 128);
                cv::circle(d, p, 2, col, -1);
                cv::line(d, p, e, col, 1);
            }
            cv::circle(d, cv::Point(int(g.cx + e.foe_x * g.f), int(g.vh + e.foe_y * g.f)), 5, cv::Scalar(0, 255, 0), 2);
            drawPath(d, g, e);
            if (argc > 3 && i % 2 == 0) {
                if (!vw.isOpened()) vw.open(argv[3], cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps / 2, f.size());
                vw.write(d);
            }
            if (snap > 0 && i % snap == 0) cv::imwrite(cv::format("ego_%05d.png", i), d);
        }
    }
    std::printf("frames %d, ego %.2f ms/frame\n", i, ms / std::max(1, i));
    if (synth && errN)
        std::printf("SYNTH: injected yaw rate RMS %.2f deg/s, error RMS %.2f deg/s, max %.2f deg/s (%d frames)\n",
                    std::sqrt(refSum / errN) * 180 / M_PI, std::sqrt(errSum / errN) * 180 / M_PI, errMax * 180 / M_PI, errN);
    return 0;
}
