#include "calibration_manager.hpp"
#include <opencv2/imgcodecs.hpp>
#include <random>
static inline uint32_t hsh(int a, int b) { uint32_t h = uint32_t(a) * 374761393u + uint32_t(b) * 668265263u; h = (h ^ (h >> 13)) * 1274126177u; return h ^ (h >> 16); }
struct Truth { double hfov = 105, h = 1.25, hz = 0.43, vpx = 0.53, hood = 0.82, cam_off = 0.30; };
static void render(cv::Mat& img, const Truth& T, double s, double pitchNoise) {
    const int W = img.cols, H = img.rows;
    const double f = 0.5 * W / std::tan(T.hfov * M_PI / 360.0), cx = T.vpx * W, vh = T.hz * H + pitchNoise;
    for (int v = 0; v < H; ++v) {
        cv::Vec3b* row = img.ptr<cv::Vec3b>(v);
        for (int u = 0; u < W; ++u) {
            if (v >= T.hood * H + 6.0 * std::pow((u - W / 2.0) / (W / 2.0), 2)) {
                const int g = 95 + int(20.0 * (v - T.hood * H) / (H * (1 - T.hood) + 1)); row[u] = cv::Vec3b(g, g, g + 8); continue; }
            if (v <= vh + 0.5) { const int b = 200 - int(60.0 * v / std::max(1.0, vh)); row[u] = cv::Vec3b(b, b - 20, b - 50); continue; }
            const double z = f * T.h / (v - vh), xc = (u - cx) * z / f;
            const double x = xc + T.cam_off;
            const double zw = z + s;
            int g = 85 + int(hsh(int(std::floor(x / 0.25)), int(std::floor(zw / 0.25))) % 21) - 10;
            auto line = [&](double xl, bool dashed) { if (std::abs(x - xl) > 0.075) return false; if (!dashed) return true; return std::fmod(zw, 12.0) < 3.0; };
            if (line(-1.75, false) || line(1.75, true) || line(-5.25, true) || line(5.25, false)) g = 215;
            if (std::abs(x) > 7.0) g = 60 + int(hsh(int(x * 3), int(zw * 3)) % 30);
            row[u] = cv::Vec3b(g, g, g);
        }
    }
}
int main() {
    const int W = 640, H = 360; const double fps = 30.0, speed = 60.0 / 3.6;
    Truth T;
    CameraCalibration c; c.hfov_deg = 90; c.camera_height_m = 1.35; c.horizon_y_ratio = 0.40; c.vanish_x_ratio = 0.5; c.hood_y_ratio = 0.75; c.camera_lateral_offset_m = 0.0;
    c.print();
    CalibrationManager::Params cp; cp.save_path = "auto_calib.json"; cp.lane_width_ref_m = getenv("LWREF") ? atof(getenv("LWREF")) : 0.0; CalibrationManager mgr(cp); mgr.begin(c, false);
    LaneDetector ld; cv::Mat img(H, W, CV_8UC3), mask, gray; std::mt19937 rng(1); std::normal_distribution<double> pn(0.0, 0.8);
    double s = 0.0; int lastPrint = -1;
    const double dur = getenv("BUMP") ? 520 : 260;
    for (int i = 0; i < int(dur * fps); ++i) {
        if (getenv("BUMP") && i == int(200 * fps)) { T.hz = 0.46; std::printf("--- camera knocked: true horizon now 0.460 ---\n"); }
        const double t = i / fps; s += speed / fps;
        render(img, T, s, pn(rng));
        const RoadGeometry g = RoadGeometry::make(c, W, H, c.horizon_y_ratio * H, int(c.hood_y_ratio * H));
        ld.predict(t);
        const bool upd = (i % 2 == 0);
        cv::Mat lm, bev;
        if (upd) { laneMarkingMask(img, g, mask, gray); ld.update(mask, true, g); lm = mask; bev = ld.bev(); }
        const LaneOutput lo = ld.output(g, t);
        if (mgr.step(t, img, g, lo, lm, bev, 0.10, 40.0, speed, false, c)) { ld.reset(); }
        const auto& st = mgr.status();
        if (int(t) % 40 == 0 && int(t) != lastPrint) { lastPrint = int(t);
            std::printf("t=%3.0fs %-34s valid %5.1fs vp %4d hood %4d flow %4d lat %4d | lanes %s w %.2f off %+.2f\n", t, st.text.c_str(), st.valid_s, st.vp, st.hood, st.flow, st.lat,
                        lo.has_lane ? "ok" : "--", lo.width_m, lo.offset_m); }
        if (i == 300) cv::imwrite("calib_frame.png", img);
    }
    const double fT = 0.5 * W / std::tan(T.hfov * M_PI / 360.0), fC = c.focalPx(W);
    std::printf("\nTRUTH : horizon %.3f vp_x %.3f hood %.3f  f*h %.1f  lat_off %+.2f\n", T.hz, T.vpx, T.hood, fT * T.h, T.cam_off);
    std::printf("RESULT: horizon %.3f vp_x %.3f hood %.3f  f*h %.1f  lat_off %+.2f  (hfov %.1f, h %.2f kept)\n", c.horizon_y_ratio, c.vanish_x_ratio, c.hood_y_ratio, fC * c.camera_height_m, c.camera_lateral_offset_m, c.hfov_deg, c.camera_height_m);
    const double vT = T.hz * H + fT * T.h / 20.0;
    const RoadGeometry g = RoadGeometry::make(c, W, H, c.horizon_y_ratio * H, int(c.hood_y_ratio * H));
    std::printf("range of a road point truly at 20.0 m -> %.2f m (before: %.2f m)\n", g.zAtRow(vT), 0.5 * W / std::tan(45 * M_PI / 180) * 1.35 / (vT - 0.40 * H));
}
