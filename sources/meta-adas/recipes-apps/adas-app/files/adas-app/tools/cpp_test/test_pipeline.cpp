#include "adas_pipeline.hpp"
#include <opencv2/imgcodecs.hpp>
#include <thread>
int main(int argc, char** argv) {
    AdasPipeline::Config cfg;
    cfg.engine.det.onnx_path = getenv("NANODET") ? getenv("NANODET") : "nanodet.onnx";
    cfg.engine.calib.hfov_deg = 100; cfg.engine.calib.camera_height_m = 1.3; cfg.engine.calib.horizon_y_ratio = 0.478; cfg.engine.calib.hood_y_ratio = 0.728;
    cfg.engine.det_hz = 8; cfg.engine.log_every_s = 0;
    cfg.fast_cores = {0}; cfg.ai_cores = {1};
    AdasPipeline adas(cfg); adas.start(); adas.pinFastThread();
    const double fps = 30; double sum = 0, mx = 0; int n = 0, aiHits = 0; int lastAge = 999;
    const int loops = argc > 1 ? atoi(argv[1]) : 3;
    for (int L = 0; L < loops; ++L)
    for (int i = 0; i < 90; ++i) {
        char p[64]; std::snprintf(p, sizeof p, "frames/%03d.png", i);
        cv::Mat f = cv::imread(p); if (f.empty()) break;
        const auto t0 = std::chrono::steady_clock::now();
        const AdasOutput& o = adas.step(f, (L * 90 + i) / fps);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        sum += ms; mx = std::max(mx, ms); ++n;
        if (o.ai_age_frames < lastAge) ++aiHits;
        lastAge = o.ai_age_frames;
        if (i % 30 == 0) std::printf("[%d/%2d] %s off %+.2f | veh %d lead %s %s | step %.2f ms ai %.0f ms age %d\n", L, i, o.ldw_status, o.offset_m, o.n_veh, o.has_lead ? "yes" : "no", o.fcw_text, ms, o.ai_ms, o.ai_age_frames);
        if (L == loops - 1 && i == 60) { cv::Mat d = f.clone(); adas.drawHud(d, o); cv::imwrite("pipeline60.png", d); }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::printf("fast step: avg %.2f ms, max %.2f ms over %d frames; AI results received %d (%.1f Hz)\n", sum / n, mx, n, aiHits, aiHits / (n / fps));
    adas.stop();
}
