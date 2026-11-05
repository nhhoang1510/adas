#include "adas_pipeline.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cstdlib>
#include <string>

static double envd(const char* k, double d) { const char* e = getenv(k); return (e && *e) ? atof(e) : d; }
static std::string envs(const char* k, const char* d) { const char* e = getenv(k); return (e && *e) ? e : d; }

int main(int argc, char** argv) {
    const std::string src = argc > 1 ? argv[1] : "/dev/video0";
    const std::string out = argc > 2 ? argv[2] : "";
    const int maxFrames = argc > 3 ? atoi(argv[3]) : 0;

    AdasPipeline::Config cfg;
    cfg.engine.det.onnx_path = envs("ADAS_NANODET", "/opt/adas/models/nanodet.onnx");
    cfg.engine.det_hz = envd("ADAS_DET_HZ", 8.0);
    cfg.engine.log_every_s = envd("ADAS_ENGINE_LOG_S", 0.0);
    const bool isVideo = src.rfind("/dev/", 0) != 0;
    const std::string calibPath = isVideo ? CameraCalibration::sidecarFor(src) : envs("ADAS_CALIB", "/opt/adas/calib/picam.json");
    if (!CameraCalibration::load(calibPath, cfg.engine.calib)) {
        std::printf("[main] no calibration at %s -> defaults (self-calibration will take over)\n", calibPath.c_str());
        cfg.engine.calib.hfov_deg = envd("ADAS_HFOV", 100.0); cfg.engine.calib.camera_height_m = envd("ADAS_CAM_H", 1.3);
        cfg.engine.calib.horizon_y_ratio = envd("ADAS_HORIZON", 0.45); cfg.engine.calib.hood_y_ratio = envd("ADAS_HOOD", 0.80);
    }
    cfg.calib.save_path = isVideo ? calibPath : envs("ADAS_CALIB_SAVE", "/var/lib/adas/calib/picam_auto.json");
    cfg.calib.lane_width_ref_m = envd("ADAS_LANE_WIDTH_REF", 0.0);
    cfg.debug_view = envd("ADAS_DEBUG_VIEW", 1.0) > 0;

    cv::VideoCapture cap;
    if (isVideo) { if (!cap.open(src, cv::CAP_FFMPEG)) cap.open(src, cv::CAP_OPENCV_MJPEG); }
    else cap.open(src, cv::CAP_V4L2);
    if (!cap.isOpened()) { std::fprintf(stderr, "cannot open %s\n", src.c_str()); return 1; }
    if (!isVideo) { cap.set(cv::CAP_PROP_FRAME_WIDTH, 640); cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360); cap.set(cv::CAP_PROP_FPS, 30); }
    const double fps = cap.get(cv::CAP_PROP_FPS) > 1 ? cap.get(cv::CAP_PROP_FPS) : 30.0;
    const double assumeKph = envd("ADAS_CALIB_ASSUME_KPH", -1.0);

    AdasPipeline adas(cfg);
    adas.start();
    adas.pinFastThread();
    if (assumeKph > 0) adas.setEgoSpeedKph(assumeKph);

    cv::VideoWriter writer; bool writerTried = false;
    cv::Mat frame, small;
    int n = 0; double sumFast = 0; auto tStart = std::chrono::steady_clock::now();
    while (cap.read(frame)) {
        if (frame.cols != 640) { cv::resize(frame, small, cv::Size(640, int(std::lround(frame.rows * 640.0 / frame.cols))), 0, 0, cv::INTER_AREA); frame = small; }
        const double t = isVideo ? n / fps : std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        const AdasOutput& o = adas.step(frame, t);
        sumFast += o.fast_ms;
        if (!out.empty()) {
            if (!writerTried) { writerTried = true; if (!writer.open(out, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame.size())) std::fprintf(stderr, "cannot write %s (HUD disabled)\n", out.c_str()); }
            if (writer.isOpened()) { adas.drawHud(frame, o); writer.write(frame); }
        }
        if (n % int(fps) == 0)
            std::printf("[%5d] %s off %+.2f w %.2f | veh %d lead %s %s | fast %.2f ms ai %.0f ms age %d | %s\n", n, o.ldw_status, o.offset_m, o.width_m,
                        o.n_veh, o.has_lead ? "yes" : "no", o.fcw_text, o.fast_ms, o.ai_ms, o.ai_age_frames, o.calib_text);
        if (maxFrames && ++n >= maxFrames) break;
        if (!maxFrames) ++n;
        if (!isVideo) {   }
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count();
    std::printf("frames %d | fast loop %.2f ms/frame avg | wall %.1f s (%.1f FPS incl. decode/encode)\n", n, sumFast / std::max(1, n), wall, n / wall);
    adas.stop();
    return 0;
}
