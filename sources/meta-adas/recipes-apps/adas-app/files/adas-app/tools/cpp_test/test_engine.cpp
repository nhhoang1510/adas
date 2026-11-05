#include "dual_engine_pipeline.hpp"
#include "detection_filter.hpp"
#include <opencv2/imgcodecs.hpp>
int main(int argc, char** argv) {
    DualEnginePipeline::Config cfg;
    cfg.det.onnx_path = argc > 1 ? argv[1] : "nanodet.onnx";
    cfg.calib.hfov_deg = 100; cfg.calib.camera_height_m = 1.3; cfg.calib.horizon_y_ratio = 0.478; cfg.calib.hood_y_ratio = 0.728;
    cfg.log_every_s = 0;
    DualEnginePipeline p(cfg);
    cv::Mat f = cv::imread(argc > 2 ? argv[2] : "frame.png");
    DualEnginePipeline::Result r;
    for (int i = 0; i < 5 && !r.det_fresh; ++i) r = p.step(f);
    std::printf("det_fresh %d  %.1f ms  n=%zu\n", r.det_fresh, r.det_ms, r.vehicles.size());
    for (auto& v : r.vehicles) std::printf("  %-9s %.2f [%4.0f %4.0f %4.0f %4.0f]\n", vehicleClassName(v.box.cls), v.box.conf, v.box.x1, v.box.y1, v.box.x2, v.box.y2);
    const RoadGeometry g = p.geometry(r);
    DetFilterStats st; auto k = filterDetections(r.vehicles, g, &st, true);
    std::printf("filter %d -> %d\n", st.in, st.kept);
}
