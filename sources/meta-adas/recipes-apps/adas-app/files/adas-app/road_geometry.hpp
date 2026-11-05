#pragma once

#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

enum VehicleClass : int {
    VC_CAR = 0,
    VC_MOTORBIKE = 1,
    VC_BICYCLE = 2,
    VC_BUS_TRUCK = 3,
    VC_IGNORE = -1
};

inline const char* vehicleClassName(int c) {
    switch (c) {
        case VC_CAR: return "car";
        case VC_MOTORBIKE: return "motorbike";
        case VC_BICYCLE: return "bicycle";
        case VC_BUS_TRUCK: return "bus/truck";
        default: return "?";
    }
}

inline double vehicleClassWidthM(int c) {
    switch (c) {
        case VC_MOTORBIKE: return 0.75;
        case VC_BICYCLE:   return 0.60;
        case VC_BUS_TRUCK: return 2.45;
        default:           return 1.80;
    }
}

struct CameraCalibration {
    double hfov_deg = 70.0;
    double camera_height_m = 1.35;
    double horizon_y_ratio = 0.42;
    double hood_y_ratio = 0.88;
    double camera_lateral_offset_m = 0.0;
    double vanish_x_ratio = 0.5;
    int    auto_calibrated = 0;
    std::string source = "default";

    double focalPx(int frame_w) const { return (frame_w * 0.5) / std::tan(hfov_deg * M_PI / 360.0); }
    static std::string sidecarFor(const std::string& video_path) { return video_path + ".calib.json"; }

    static bool load(const std::string& path, CameraCalibration& c) {
        std::ifstream f(path);
        if (!f) return false;
        std::stringstream ss; ss << f.rdbuf();
        const std::string s = ss.str();
        auto num = [&](const char* key, double& dst) {
            const std::string k = std::string("\"") + key + "\"";
            const size_t p = s.find(k);
            if (p == std::string::npos) return;
            const size_t colon = s.find(':', p + k.size());
            if (colon == std::string::npos) return;
            char* end = nullptr;
            const double v = std::strtod(s.c_str() + colon + 1, &end);
            if (end != s.c_str() + colon + 1) dst = v;
        };
        num("hfov_deg", c.hfov_deg);
        num("camera_height_m", c.camera_height_m);
        num("horizon_y_ratio", c.horizon_y_ratio);
        num("hood_y_ratio", c.hood_y_ratio);
        num("camera_lateral_offset_m", c.camera_lateral_offset_m);
        num("vanish_x_ratio", c.vanish_x_ratio);
        double ac = c.auto_calibrated; num("auto_calibrated", ac); c.auto_calibrated = int(ac);
        c.vanish_x_ratio = std::clamp(c.vanish_x_ratio, 0.3, 0.7);
        c.hfov_deg = std::clamp(c.hfov_deg, 20.0, 170.0);
        c.horizon_y_ratio = std::clamp(c.horizon_y_ratio, 0.15, 0.75);
        c.hood_y_ratio = std::clamp(c.hood_y_ratio, c.horizon_y_ratio + 0.10, 1.0);
        c.source = path;
        return true;
    }

    bool save(const std::string& path) const {
        const std::string tmp = path + ".tmp";
        FILE* fp = std::fopen(tmp.c_str(), "w");
        if (!fp) return false;
        std::fprintf(fp,
                     "{\n  \"hfov_deg\": %.2f,\n  \"camera_height_m\": %.3f,\n  \"horizon_y_ratio\": %.4f,\n"
                     "  \"hood_y_ratio\": %.4f,\n  \"camera_lateral_offset_m\": %.3f,\n  \"vanish_x_ratio\": %.4f,\n"
                     "  \"auto_calibrated\": %d\n}\n",
                     hfov_deg, camera_height_m, horizon_y_ratio, hood_y_ratio, camera_lateral_offset_m, vanish_x_ratio,
                     auto_calibrated);
        std::fflush(fp);
        fsync(fileno(fp));
        std::fclose(fp);
        return std::rename(tmp.c_str(), path.c_str()) == 0;
    }

    void print() const {
        std::printf("[Calib] %s: hfov=%.1f deg  cam_h=%.2f m  horizon=%.3fH  vp_x=%.3fW  hood=%.3fH  lat_off=%+.2f m%s\n",
                    source.c_str(), hfov_deg, camera_height_m, horizon_y_ratio, vanish_x_ratio, hood_y_ratio,
                    camera_lateral_offset_m, auto_calibrated ? "  [AUTO]" : "");
    }
};

struct RoadGeometry {
    double f = 0.0;
    double cx = 0.0;
    double vh = 0.0;
    double cam_h = 1.35;
    double car_x = 0.0;
    int W = 0, H = 0, hood = 0;

    static RoadGeometry make(const CameraCalibration& c, int W, int H, double horizon_y, int hood_y) {
        RoadGeometry g;
        g.f = c.focalPx(W);
        g.cx = W * c.vanish_x_ratio;
        g.vh = horizon_y;
        g.cam_h = c.camera_height_m;
        g.car_x = -c.camera_lateral_offset_m;
        g.W = W; g.H = H;
        g.hood = std::clamp(hood_y, 1, H);
        return g;
    }

    bool valid() const { return f > 1.0 && W > 0 && H > 0; }

    bool toGround(double u, double v, double& x, double& z) const {
        const double dv = v - vh;
        if (dv < 0.5) return false;
        z = f * cam_h / dv;
        x = (u - cx) * z / f;
        return true;
    }
    cv::Point2f toImage(double x, double z) const {
        return cv::Point2f(float(cx + f * x / z), float(vh + f * cam_h / z));
    }
    double zAtRow(double v) const { const double dv = v - vh; return dv > 0.5 ? f * cam_h / dv : 1e9; }
    double rowAtZ(double z) const { return vh + f * cam_h / z; }
    double nearZ() const { return zAtRow(hood - 1); }

    double estimateRange(const cv::Rect2f& b, int cls) const {
        const double bw = std::max(1.0f, b.width);
        const double z_w = vehicleClassWidthM(cls) * f / bw;
        const double by = b.y + b.height;
        const double dy = by - vh;
        const bool cutX = W > 0 && (b.x < 2 || b.x + b.width > W - 2);
        if (cutX) {
            if (dy >= 3.0 && by < hood - 2) return std::clamp(cam_h * f / dy, 2.0, 120.0);
            return std::clamp(std::max(z_w, nearZ()), 2.0, 120.0);
        }
        if (dy >= 3.0 && by < hood - 2) {
            const double z_g = cam_h * f / dy;
            const double var_w = std::pow(1.5 / bw, 2) + 0.01;
            const double var_g = std::pow(4.0 / dy, 2);
            return std::clamp((z_w / var_w + z_g / var_g) / (1.0 / var_w + 1.0 / var_g), 2.0, 120.0);
        }
        return std::clamp(z_w, 2.0, 120.0);
    }
};
