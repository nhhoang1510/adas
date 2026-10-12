/**
 * =============================================================================
 * main.cpp - YOLOPv2 C++ Demo (OpenCV HighGUI + Inference Pipeline)
 * =============================================================================
 * Demo headless / OpenCV window chạy pipeline inference C++.
 * Tương đương main() + ADASPanopticApp._video_loop trong Python
 * nhưng không có Tkinter UI (thay bằng cv::imshow hoặc headless).
 *
 * CÁCH BUILD (Windows, MSVC + vcpkg):
 *   cmake -B build -G "Visual Studio 17 2022" -A x64 \
 *     -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
 *   cmake --build build --config Release
 *
 * CÁCH CHẠY:
 *   yolopv2_cpp.exe --yolop yolopv2_192x320.onnx --tsr tsr_480.onnx --src video.mp4
 *   yolopv2_cpp.exe --yolop yolopv2.onnx          --tsr tsr_480.onnx --src 0       (webcam)
 * =============================================================================
 */

#include "inference_pipeline.hpp"
#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <csignal>

static std::atomic<bool> g_quit{ false };

void sig_handler(int) { g_quit.store(true); }

struct Args {
    std::string yolop_weights = "yolopv2_192x320.onnx";
    std::string tsr_weights   = "tsr_480.onnx";
    std::string source        = "0";      // "0" = webcam, path = video file
    std::string device        = "cuda";   // "cuda"|"cpu"
    float  conf_yolop    = 0.30f;
    float  conf_sign     = 0.35f;
    float  iou_thres     = 0.45f;
    int    imgsz_yolop   = 192;           // sẽ tự detect từ ONNX
    int    imgsz_tsr     = 480;
    int    lane_cadence  = 5;
    int    sign_cadence  = 2;
    bool   no_display    = false;         // headless mode
    bool   turbo         = false;
    int    target_w      = 1280;
    int    target_h      = 720;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i],"--yolop") && i+1<argc)  a.yolop_weights = argv[++i];
        else if (!strcmp(argv[i],"--tsr") && i+1<argc) a.tsr_weights = argv[++i];
        else if (!strcmp(argv[i],"--src") && i+1<argc) a.source = argv[++i];
        else if (!strcmp(argv[i],"--device") && i+1<argc) a.device = argv[++i];
        else if (!strcmp(argv[i],"--conf-yolop") && i+1<argc) a.conf_yolop = std::stof(argv[++i]);
        else if (!strcmp(argv[i],"--conf-sign") && i+1<argc)  a.conf_sign  = std::stof(argv[++i]);
        else if (!strcmp(argv[i],"--iou") && i+1<argc)        a.iou_thres  = std::stof(argv[++i]);
        else if (!strcmp(argv[i],"--lane-cad") && i+1<argc)   a.lane_cadence = std::stoi(argv[++i]);
        else if (!strcmp(argv[i],"--sign-cad") && i+1<argc)   a.sign_cadence = std::stoi(argv[++i]);
        else if (!strcmp(argv[i],"--width") && i+1<argc)  a.target_w = std::stoi(argv[++i]);
        else if (!strcmp(argv[i],"--height") && i+1<argc) a.target_h = std::stoi(argv[++i]);
        else if (!strcmp(argv[i],"--no-display")) a.no_display = true;
        else if (!strcmp(argv[i],"--turbo"))      a.turbo = true;
        else {
            printf("Unknown arg: %s\n", argv[i]);
        }
    }
    return a;
}

int main(int argc, char** argv) {
    std::signal(SIGINT, sig_handler);
    std::signal(SIGTERM, sig_handler);

    Args args = parse_args(argc, argv);

    printf("=== YOLOPv2 C++ Inference Engine ===\n");
    printf("  YOLOPv2  : %s\n", args.yolop_weights.c_str());
    printf("  TSR      : %s\n", args.tsr_weights.c_str());
    printf("  Source   : %s\n", args.source.c_str());
    printf("  Device   : %s\n", args.device.c_str());
    printf("  Conf(Y)  : %.2f | Conf(S): %.2f | IOU: %.2f\n",
           args.conf_yolop, args.conf_sign, args.iou_thres);
    printf("  Lane cad : %d | Sign cad: %d\n",
           args.lane_cadence, args.sign_cadence);
    printf("  Target   : %dx%d\n", args.target_w, args.target_h);
    printf("====================================\n");

    // Mở nguồn video
    cv::VideoCapture cap;
    bool is_webcam = (args.source == "0" || args.source == "webcam");
    if (is_webcam) {
        cap.open(0, cv::CAP_DSHOW);
        if (!cap.isOpened()) cap.open(0);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    } else {
        cap.open(args.source);
    }

    if (!cap.isOpened()) {
        fprintf(stderr, "[ERROR] Khong the mo nguon video: %s\n", args.source.c_str());
        return 1;
    }

    // Pipeline config
    PipelineConfig cfg;
    cfg.yolop_weights  = args.yolop_weights;
    cfg.tsr_weights    = args.tsr_weights;
    cfg.device         = args.device;
    cfg.conf_yolop     = args.conf_yolop;
    cfg.conf_sign      = args.conf_sign;
    cfg.iou_thres      = args.iou_thres;
    cfg.lane_cadence   = args.lane_cadence;
    cfg.sign_cadence   = args.sign_cadence;
    cfg.turbo_mode     = args.turbo;
    cfg.target_w       = args.target_w;
    cfg.target_h       = args.target_h;
    cfg.enable_lane    = true;
    cfg.enable_vehicle = true;
    cfg.enable_sign    = true;

    // Thống kê FPS
    std::atomic<double> display_fps{ 0.0 };
    std::atomic<int>    total_frames{ 0 };
    int display_every = 1; // hiển thị mỗi frame (có thể tăng để giảm overhead)

    // Window
    if (!args.no_display) {
        cv::namedWindow("YOLOPv2 C++ | ADAS", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::resizeWindow("YOLOPv2 C++ | ADAS", args.target_w, args.target_h);
    }

    // Khởi động pipeline
    InferencePipeline pipeline(cfg);

    // Callback: nhận annotated frame từ inference thread
    // Hiển thị trên main thread (an toàn với OpenCV)
    std::mutex show_mtx;
    cv::Mat last_frame;
    bool has_frame = false;

    bool started = pipeline.start(cap, [&](AnnotatedFrame&& af) {
        ++total_frames;
        display_fps.store(af.real_fps);

        if (!args.no_display) {
            // Vẽ FPS lên góc dưới phải để dễ đọc
            char fps_buf[64];
            snprintf(fps_buf, sizeof(fps_buf), "%.1f FPS", af.real_fps);
            int baseline = 0;
            auto tsz = cv::getTextSize(fps_buf, cv::FONT_HERSHEY_SIMPLEX, 1.1, 2, &baseline);
            int fx = af.bgr.cols - tsz.width - 14;
            int fy = 36;
            cv::putText(af.bgr, fps_buf, {fx, fy},
                        cv::FONT_HERSHEY_SIMPLEX, 1.1,
                        af.real_fps >= 25 ? cv::Scalar{0,230,100} :
                        af.real_fps >= 15 ? cv::Scalar{0,160,255} :
                                            cv::Scalar{0,0,255}, 2, cv::LINE_AA);

            std::lock_guard<std::mutex> lk(show_mtx);
            last_frame = std::move(af.bgr);
            has_frame = true;
        }

        // In thống kê mỗi 30 frame
        if (total_frames.load() % 30 == 0) {
            printf("[Frame %5d] FPS: %5.1f | YOLOPv2: %5.1fms | TSR: %5.1fms | "
                   "Xe: %zu | Bien bao: %zu | %s (%d lux)\n",
                   af.frame_idx,
                   af.real_fps,
                   af.result.t_yolop_ms,
                   af.result.t_sign_ms,
                   af.result.vehicles.size(),
                   af.result.signs.size(),
                   af.env_label.c_str(),
                   af.env_brightness);
        }
    });

    if (!started) {
        fprintf(stderr, "[ERROR] Khong khoi dong duoc pipeline!\n");
        cap.release();
        return 1;
    }

    printf("[INFO] Pipeline dang chay. Nhan 'q' de thoat...\n");

    // Main loop: hiển thị frame + xử lý phím
    while (!g_quit.load()) {
        if (!args.no_display) {
            {
                std::lock_guard<std::mutex> lk(show_mtx);
                if (has_frame && !last_frame.empty()) {
                    cv::imshow("YOLOPv2 C++ | ADAS", last_frame);
                    has_frame = false;
                }
            }
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) break;  // q hoặc ESC
            if (key == ' ')  pipeline.pause(!pipeline.is_running());
            if (key == 't')  pipeline.set_turbo(true);
            if (key == 'n')  pipeline.set_turbo(false);
        } else {
            // Headless: check quit mỗi 10ms
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    printf("[INFO] Dang dung pipeline...\n");
    pipeline.stop();
    cap.release();

    if (!args.no_display)
        cv::destroyAllWindows();

    printf("[DONE] Tong frames xu ly: %d | FPS trung binh: %.1f\n",
           total_frames.load(), display_fps.load());
    return 0;
}
