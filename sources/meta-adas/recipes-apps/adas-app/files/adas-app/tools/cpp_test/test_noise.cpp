#include "road_geometry.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <random>
#include <cstring>
#include <chrono>
namespace v3 {
#include "lane_detector_v3.hpp"
}
namespace v4 {
#include "lane_detector.hpp"
}
static inline uint32_t hsh(int a, int b, int c = 0) { uint32_t h = uint32_t(a) * 374761393u + uint32_t(b) * 668265263u + uint32_t(c) * 2246822519u; h = (h ^ (h >> 13)) * 1274126177u; return h ^ (h >> 16); }
struct Scene { double hfov = 100, h = 1.3, hz = 0.45, hood = 0.80; double laneW = 3.5; };
static double offTruth(double t) { return 0.45 * std::sin(2 * M_PI * t / 9.0); }
static double headTruth(double t) { return 0.45 * 2 * M_PI / 9.0 * std::cos(2 * M_PI * t / 9.0) / 20.0; }
static double curvTruth(double t) { return 0.0012 * std::sin(2 * M_PI * t / 25.0); }
static void render(cv::Mat& img, const Scene& S, double t, double s, std::mt19937& rng, int level) {
    const int W = img.cols, H = img.rows;
    const double f = 0.5 * W / std::tan(S.hfov * M_PI / 360.0), cx = 0.5 * W, vh = S.hz * H;
    const double off = offTruth(t), hd = headTruth(t), cv2_ = curvTruth(t);
    std::normal_distribution<double> sn(0, level >= 1 ? 7.0 : 2.0);
    const int frame = int(t * 30);
    for (int v = 0; v < H; ++v) {
        cv::Vec3b* row = img.ptr<cv::Vec3b>(v);
        for (int u = 0; u < W; ++u) {
            if (v >= S.hood * H) { row[u] = cv::Vec3b(100, 100, 108); continue; }
            if (v <= vh + 0.5) { row[u] = cv::Vec3b(200, 180, 150); continue; }
            const double z = f * S.h / (v - vh), xc = (u - cx) * z / f;
            const double x = xc + off + hd * z - cv2_ * z * z;
            const double zw = z + s;
            int g = 88 + int(hsh(int(std::floor(x / 0.2)), int(std::floor(zw / 0.2))) % 23) - 11;
            if (level >= 1) {
                if (std::abs(std::fmod(x * 0.7 + 0.3 * std::sin(zw * 0.8) + 50.0, 2.3) - 1.0) < 0.03) g -= 35;
                const uint32_t pr = hsh(int(std::floor(x / 0.9)), int(std::floor(zw / 0.9)), 7);
                if (pr % 97 == 0) g += 70;
                if (std::sin(zw * 0.35 + 1.3 * std::sin(x * 0.4)) > 0.55) g = int(g * 0.45);
                if (std::fmod(zw, 180.0) < 3.0 && std::fmod(std::abs(x) + 10.0, 1.0) < 0.5) g = 215;
                if (std::fmod(zw + 60.0, 150.0) < 4.0 && std::abs(x) < 0.25) g = 210;
            }
            auto paint = [&](double xl, bool dashed) {
                if (std::abs(x - xl) > 0.075) return false;
                if (dashed && std::fmod(zw, 12.0) >= 3.0) return false;
                if (level >= 1 && (hsh(int(zw / 1.5), int(xl * 10), 3) % 100) < 25) return false;
                return true; };
            if (paint(-S.laneW / 2, false) || paint(S.laneW / 2, true) || paint(-1.5 * S.laneW, true) || paint(1.5 * S.laneW, false))
                g = (level >= 1) ? 200 : 215;
            if (std::abs(x) > 6.5) g = 70 + int(hsh(int(x * 3), int(zw * 3)) % 40);
            g += int(sn(rng));
            if (level >= 1 && (hsh(u, v, frame) % 1500) == 0) g = 250;
            g = std::clamp(g, 0, 255);
            row[u] = cv::Vec3b(g, g, g);
        }
    }
}
template <class LD, class LO, class MaskFn>
static void run(const char* name, int level, MaskFn maskFn, bool save) {
    const int W = 640, H = 360; const double fps = 30, speed = 20.0;
    Scene S; CameraCalibration c; c.hfov_deg = S.hfov; c.camera_height_m = S.h; c.horizon_y_ratio = S.hz; c.hood_y_ratio = S.hood;
    LD ld; cv::Mat img(H, W, CV_8UC3), mask, gray; std::mt19937 rng(42);
    double se0 = 0, se1 = 0, seW = 0, jit = 0, prev = NAN; int n = 0, have = 0, frames = 0, ldwFalse = 0; double maxErr = 0;
    double tms = 0;
    for (int i = 0; i < int(40 * fps); ++i) {
        const double t = i / fps, s = speed * t;
        render(img, S, t, s, rng, level);
        const RoadGeometry g = RoadGeometry::make(c, W, H, S.hz * H, int(S.hood * H));
        auto t0 = std::chrono::steady_clock::now();
        ld.predict(t);
        if (i % 2 == 0) { maskFn(img, g, mask, gray); ld.update(mask, true, g); }
        const LO o = ld.output(g, t);
        tms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (i < 30) continue;
        ++frames;
        if (!o.has_lane) continue;
        ++have;
        const double e0 = o.offset_m - offTruth(t);
        const double e1 = 0.5 * (o.left.c1 + o.right.c1) - (-headTruth(t));
        const double eW = o.width_m - S.laneW;
        se0 += e0 * e0; se1 += e1 * e1; seW += eW * eW; ++n; maxErr = std::max(maxErr, std::abs(e0));
        if (!std::isnan(prev)) { const double d = (o.offset_m - prev) - (offTruth(t) - offTruth(t - 1 / fps)); jit += d * d; }
        prev = o.offset_m;
        if (getenv("DUMP") && std::string(name) == getenv("DUMP") && i % 15 == 0) std::printf("  t %5.2f truth %+.3f est %+.3f  hd %+.4f c1 %+.4f  w %.2f L %+.2f R %+.2f\n", t, offTruth(t), o.offset_m, -headTruth(t), 0.5*(o.left.c1+o.right.c1), o.width_m, o.left.c0, o.right.c0);
        if (o.alert_level > 0 && std::abs(offTruth(t)) < 0.45) ++ldwFalse;
        if (save && i == 600) { cv::Mat d = img.clone(); d.setTo(cv::Scalar(0, 255, 0), mask);
            for (size_t k = 1; k < o.left_img.size(); ++k) { cv::line(d, o.left_img[k-1], o.left_img[k], {0,0,255}, 2); cv::line(d, o.right_img[k-1], o.right_img[k], {255,0,0}, 2); }
            ld.drawDebug(d, H / 2); cv::imwrite(std::string("noise_") + name + ".png", d); }
    }
    std::printf("%-4s noise=%d | lane %5.1f%% | offset RMSE %.3f m max %.2f | heading RMSE %.4f | width RMSE %.2f | jitter %.4f m/frame | false LDW %d | %.2f ms/frame\n",
                name, level, 100.0 * have / frames, std::sqrt(se0 / std::max(1, n)), maxErr, std::sqrt(se1 / std::max(1, n)), std::sqrt(seW / std::max(1, n)),
                std::sqrt(jit / std::max(1, n)), ldwFalse, tms / (40 * fps));
}
int main(int argc, char** argv) {
    const int level = argc > 1 ? atoi(argv[1]) : 1;
    run<v3::LaneDetector, v3::LaneOutput>("v3", level, [](const cv::Mat& f, const RoadGeometry& g, cv::Mat& m, cv::Mat& gr) { v3::laneMarkingMask(f, g, m, gr); }, true);
    run<v4::LaneDetector, v4::LaneOutput>("v4", level, [](const cv::Mat& f, const RoadGeometry& g, cv::Mat& m, cv::Mat& gr) { v4::laneMarkingMask(f, g, m, gr); }, true);
}
