#pragma once

#include "road_geometry.hpp"
#include "adas_types.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

struct DetFilterStats {
    int in = 0, kept = 0, horizon = 0, hood = 0, aspect = 0, size = 0;
};

inline std::vector<VehicleDetection> filterDetections(const std::vector<VehicleDetection>& dets,
                                                      const RoadGeometry& g, DetFilterStats* st = nullptr,
                                                      bool verbose = false) {
    std::vector<VehicleDetection> out;
    out.reserve(dets.size());
    DetFilterStats s;
    s.in = int(dets.size());
    for (const auto& d : dets) {
        const auto& b = d.box;
        if (b.cls < 0) continue;
        const double w = b.x2 - b.x1, h = b.y2 - b.y1;
        if (w < 4 || h < 4) continue;
        const char* why = nullptr;
        char buf[48];
        if (b.y2 < g.vh + 2.0) { why = "above horizon"; ++s.horizon; }
        else if (b.y1 >= g.hood - 4) { why = "in hood"; ++s.hood; }
        else if (b.y2 > g.hood + 6 && w > 0.45 * g.W) { why = "ego hood"; ++s.hood; }
        else if (h / w < 0.35 || h / w > 2.5) { why = "aspect"; ++s.aspect; }
        else if (b.y2 <= g.hood + 2 && b.y2 - g.vh >= 6.0) {
            const double wm = w * g.cam_h / (b.y2 - g.vh);
            const double prior = vehicleClassWidthM(b.cls);
            const bool cut = b.x1 < 3 || b.x2 > g.W - 3;
            if (wm > 1.9 * prior || (wm < 0.45 * prior && !cut)) {
                std::snprintf(buf, sizeof buf, "size %.2f m", wm);
                why = buf; ++s.size;
            }
        }
        if (verbose)
            std::printf("    [DetFilter] %-9s %.2f [%4.0f %4.0f %4.0f %4.0f] %s\n", vehicleClassName(b.cls), b.conf,
                        b.x1, b.y1, b.x2, b.y2, why ? why : "keep");
        if (!why) out.push_back(d);
    }
    s.kept = int(out.size());
    if (st) *st = s;
    return out;
}
