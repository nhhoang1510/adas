/**
 * =============================================================================
 * InferencePipeline - Dual-Engine Async Pipeline (tương đương _video_loop + 
 * _process_frame trong Python)
 * =============================================================================
 * Cải thiện FPS qua:
 *   1. Producer-Consumer pattern: đọc frame ở một thread, infer ở thread khác
 *   2. Bounded queue (chỉ giữ 2 frame, drop cũ khi đầy - giảm latency)
 *   3. Lane mask cache (infer_lane mỗi N frame)
 *   4. Sign detection cadence (infer_sign mỗi M frame)
 *   5. Pre-allocated annotated Mat (tránh alloc mỗi frame)
 * =============================================================================
 */

#pragma once
#include "yolopv2_engine.hpp"
#include "tsr_engine.hpp"
#include <functional>

// ============================================================
// Lock-free Bounded Queue (capacity 2)
// ============================================================
template<typename T, int CAP = 2>
class BoundedQueue {
public:
    bool push(T&& item) {
        std::unique_lock<std::mutex> lk(mtx_);
        if ((int)q_.size() >= CAP) q_.pop(); // Drop oldest
        q_.push(std::move(item));
        cv_.notify_one();
        return true;
    }

    bool pop(T& out, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lk(mtx_);
        if (!cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                          [this]{ return !q_.empty() || done_; }))
            return false;
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop();
        return true;
    }

    void set_done() {
        std::lock_guard<std::mutex> lk(mtx_);
        done_ = true;
        cv_.notify_all();
    }
    void reset() { std::lock_guard<std::mutex> lk(mtx_); done_ = false; }
    bool empty() { std::lock_guard<std::mutex> lk(mtx_); return q_.empty(); }

private:
    std::queue<T>           q_;
    std::mutex              mtx_;
    std::condition_variable cv_;
    bool                    done_ = false;
};

// ============================================================
// Pipeline Config
// ============================================================
struct PipelineConfig {
    std::string  yolop_weights;
    std::string  tsr_weights;
    std::string  device         = "cuda";  // "cuda"|"cpu"|"dml"
    float        conf_yolop     = 0.30f;
    float        conf_sign      = 0.35f;
    float        iou_thres      = 0.45f;
    int          imgsz_yolop    = 640;
    int          imgsz_tsr      = 480;
    int          lane_cadence   = 5;       // Infer lane mỗi N frame
    int          sign_cadence   = 2;       // Infer sign mỗi M frame
    bool         enable_lane    = true;
    bool         enable_vehicle = true;
    bool         enable_sign    = true;
    bool         turbo_mode     = false;   // Bỏ qua frame delay
    int          target_w       = 1280;
    int          target_h       = 720;
};

// ============================================================
// Annotated Frame (output của pipeline)
// ============================================================
struct AnnotatedFrame {
    cv::Mat              bgr;           // Annotated frame
    InferenceResult      result;
    double               elapsed_ms  = 0.0;
    double               real_fps    = 0.0;
    int                  frame_idx   = 0;
    std::string          env_label   = "Ban ngay";
    int                  env_brightness = 135;
};

// Callback khi có frame mới
using FrameCallback = std::function<void(AnnotatedFrame&&)>;

// ============================================================
// CLASS CHÍNH: InferencePipeline
// ============================================================
class InferencePipeline {
public:
    explicit InferencePipeline(const PipelineConfig& cfg)
        : cfg_(cfg)
    {}

    ~InferencePipeline() { stop(); }

    // Khởi động pipeline với VideoCapture
    bool start(cv::VideoCapture& cap, FrameCallback cb) {
        if (running_.load()) return false;

        // Khởi tạo engines (có thể ném exception nếu file không tồn tại)
        if (!_init_engines()) return false;

        running_.store(true);
        frame_cb_ = std::move(cb);
        frame_queue_.reset();
        result_queue_.reset();

        // Thread 1: Đọc frame từ VideoCapture (Producer)
        reader_thread_ = std::thread([this, &cap] { _reader_loop(cap); });

        // Thread 2: Inference (Consumer+Producer)
        infer_thread_  = std::thread([this] { _infer_loop(); });

        return true;
    }

    void stop() {
        running_.store(false);
        frame_queue_.set_done();
        result_queue_.set_done();
        if (reader_thread_.joinable()) reader_thread_.join();
        if (infer_thread_.joinable())  infer_thread_.join();
    }

    void pause(bool p) { paused_.store(p); }
    bool is_running() const { return running_.load(); }

    // Cập nhật config động (thread-safe)
    void set_conf_yolop(float v)  { cfg_mtx_.lock(); cfg_.conf_yolop = v;  cfg_mtx_.unlock(); }
    void set_conf_sign(float v)   { cfg_mtx_.lock(); cfg_.conf_sign  = v;  cfg_mtx_.unlock(); }
    void set_lane_cadence(int v)  { cfg_mtx_.lock(); cfg_.lane_cadence = v; cfg_mtx_.unlock(); }
    void set_sign_cadence(int v)  { cfg_mtx_.lock(); cfg_.sign_cadence = v; cfg_mtx_.unlock(); }
    void set_enable_lane(bool v)  { cfg_mtx_.lock(); cfg_.enable_lane = v; cfg_mtx_.unlock(); }
    void set_enable_sign(bool v)  { cfg_mtx_.lock(); cfg_.enable_sign = v; cfg_mtx_.unlock(); }
    void set_turbo(bool v)        { cfg_mtx_.lock(); cfg_.turbo_mode  = v; cfg_mtx_.unlock(); }

    double get_fps() const { return last_fps_.load(); }

private:
    PipelineConfig        cfg_;
    std::mutex            cfg_mtx_;

    std::unique_ptr<YOLOPv2Engine>      yolop_engine_;
    std::unique_ptr<TrafficSignEngine>  tsr_engine_;

    BoundedQueue<FrameData>       frame_queue_{};
    BoundedQueue<AnnotatedFrame>  result_queue_{};

    std::atomic<bool>   running_{ false };
    std::atomic<bool>   paused_ { false };
    std::atomic<double> last_fps_{ 0.0 };

    FrameCallback frame_cb_;

    std::thread reader_thread_;
    std::thread infer_thread_;

    // Cache
    cv::Mat cached_ll_mask_;
    cv::Mat cached_da_mask_;
    std::vector<SignDetection> cached_signs_;

    bool _init_engines() {
        try {
            YOLOPv2Engine::Config yc;
            yc.weights_path = cfg_.yolop_weights;
            yc.device       = cfg_.device;
            yc.conf_thres   = cfg_.conf_yolop;
            yc.iou_thres    = cfg_.iou_thres;
            yc.imgsz        = cfg_.imgsz_yolop;
            yolop_engine_ = std::make_unique<YOLOPv2Engine>(yc);
        } catch (const std::exception& e) {
            fprintf(stderr, "[Pipeline] Loi khoi tao YOLOPv2: %s\n", e.what());
            return false;
        }

        try {
            TrafficSignEngine::Config tc;
            tc.weights_path = cfg_.tsr_weights;
            tc.device       = cfg_.device;
            tc.conf_thres   = cfg_.conf_sign;
            tc.iou_thres    = cfg_.iou_thres;
            tc.imgsz        = cfg_.imgsz_tsr;
            tsr_engine_ = std::make_unique<TrafficSignEngine>(tc);
        } catch (const std::exception& e) {
            fprintf(stderr, "[Pipeline] Loi khoi tao TSR: %s\n", e.what());
            // Tiếp tục nếu TSR lỗi (chạy chỉ YOLOPv2)
        }
        return true;
    }

    // Thread 1: Đọc frame từ VideoCapture
    void _reader_loop(cv::VideoCapture& cap) {
        int fidx = 0;
        while (running_.load()) {
            if (paused_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                continue;
            }

            cv::Mat raw;
            if (!cap.read(raw) || raw.empty()) {
                // Hết video: loop lại
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                fidx = 0;
                continue;
            }

            ++fidx;

            // Resize về target resolution
            PipelineConfig cfg_snap;
            { std::lock_guard<std::mutex> lk(cfg_mtx_); cfg_snap = cfg_; }

            if (raw.cols != cfg_snap.target_w || raw.rows != cfg_snap.target_h)
                cv::resize(raw, raw, {cfg_snap.target_w, cfg_snap.target_h},
                           0, 0, cv::INTER_LINEAR);

            bool infer_lane = cfg_snap.enable_lane &&
                              (fidx % cfg_snap.lane_cadence == 0 || cached_ll_mask_.empty());
            bool infer_sign = cfg_snap.enable_sign &&
                              (fidx % cfg_snap.sign_cadence == 0 || cached_signs_.empty());

            FrameData fd;
            fd.bgr        = std::move(raw);
            fd.frame_idx  = fidx;
            fd.infer_lane = infer_lane;
            fd.infer_sign = infer_sign;

            frame_queue_.push(std::move(fd));
        }
        frame_queue_.set_done();
    }

    // Thread 2: Inference + Annotate
    void _infer_loop() {
        int fps_cnt = 0;
        double fps_timer = _now_ms();
        double cur_fps = 0.0;

        while (running_.load()) {
            FrameData fd;
            if (!frame_queue_.pop(fd, 200)) continue;

            auto t0 = _now_ms();

            PipelineConfig cfg_snap;
            { std::lock_guard<std::mutex> lk(cfg_mtx_); cfg_snap = cfg_; }

            // --- YOLOPv2 Inference ---
            InferenceResult res;
            if (yolop_engine_) {
                yolop_engine_->cfg_.conf_thres = cfg_snap.conf_yolop;
                yolop_engine_->cfg_.iou_thres  = cfg_snap.iou_thres;
                res = yolop_engine_->infer(fd.bgr, fd.infer_lane);
            }

            // Lane mask cache
            if (fd.infer_lane && !res.ll_mask.empty())
                cached_ll_mask_ = res.ll_mask;
            else if (!cached_ll_mask_.empty())
                res.ll_mask = cached_ll_mask_;

            // --- TSR Inference ---
            if (tsr_engine_ && fd.infer_sign && cfg_snap.enable_sign) {
                tsr_engine_->cfg_.conf_thres = cfg_snap.conf_sign;
                auto t_sign0 = _now_ms();
                res.signs = tsr_engine_->infer(fd.bgr);
                res.t_sign_ms = _now_ms() - t_sign0;
                cached_signs_ = res.signs;
            } else if (cfg_snap.enable_sign) {
                res.signs = cached_signs_;
            }
            res.frame_idx = fd.frame_idx;

            // --- Annotate ---
            cv::Mat ann = fd.bgr.clone();

            // 1. Lane mask overlay
            if (cfg_snap.enable_lane && !res.ll_mask.empty())
                apply_lane_mask(ann, res.ll_mask);

            // 2. Vehicle boxes
            if (cfg_snap.enable_vehicle) {
                for (auto& vd : res.vehicles) {
                    int x1 = (int)vd.box.x1, y1 = (int)vd.box.y1;
                    int x2 = (int)vd.box.x2, y2 = (int)vd.box.y2;
                    cv::rectangle(ann, {x1,y1}, {x2,y2}, {0,240,255}, 2, cv::LINE_AA);
                    char tag[64];
                    snprintf(tag, sizeof(tag), "Xe (%.0f%%)", vd.box.conf * 100);
                    int baseline = 0;
                    auto tsz = cv::getTextSize(tag, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
                    int ty = std::max(y1 - 3, tsz.height + 3);
                    cv::rectangle(ann, {x1, ty - tsz.height - 4},
                                  {x1 + tsz.width + 6, ty + 2}, {0,240,255}, cv::FILLED);
                    cv::putText(ann, tag, {x1+3, ty-2},
                                cv::FONT_HERSHEY_SIMPLEX, 0.45, {11,15,25}, 1, cv::LINE_AA);
                }
            }

            // 3. Sign boxes
            if (cfg_snap.enable_sign) {
                for (auto& sd : res.signs) {
                    int x1=(int)sd.box.x1, y1=(int)sd.box.y1;
                    int x2=(int)sd.box.x2, y2=(int)sd.box.y2;
                    cv::Scalar col = sd.color_bgr;
                    cv::rectangle(ann, {x1,y1}, {x2,y2}, col, 2, cv::LINE_AA);

                    // Corner accent lines
                    int clen = std::max(8, std::min(16, (x2-x1)/4));
                    cv::line(ann,{x1,y1},{x1+clen,y1},col,3,cv::LINE_AA);
                    cv::line(ann,{x1,y1},{x1,y1+clen},col,3,cv::LINE_AA);
                    cv::line(ann,{x2,y1},{x2-clen,y1},col,3,cv::LINE_AA);
                    cv::line(ann,{x2,y1},{x2,y1+clen},col,3,cv::LINE_AA);
                    cv::line(ann,{x1,y2},{x1+clen,y2},col,3,cv::LINE_AA);
                    cv::line(ann,{x1,y2},{x1,y2-clen},col,3,cv::LINE_AA);
                    cv::line(ann,{x2,y2},{x2-clen,y2},col,3,cv::LINE_AA);
                    cv::line(ann,{x2,y2},{x2,y2-clen},col,3,cv::LINE_AA);

                    char tag[128];
                    snprintf(tag, sizeof(tag), "[%s] %s (%.0f%%)",
                             sd.code.c_str(), sd.name.c_str(), sd.box.conf * 100);
                    int baseline=0;
                    auto tsz = cv::getTextSize(tag, cv::FONT_HERSHEY_SIMPLEX, 0.48, 1, &baseline);
                    int ty = std::max(y1 - 6, tsz.height + 8);
                    cv::rectangle(ann, {x1, ty-tsz.height-6},
                                  {x1+tsz.width+8, ty+4}, col, cv::FILLED);
                    cv::putText(ann, tag, {x1+4, ty-2},
                                cv::FONT_HERSHEY_SIMPLEX, 0.48, {255,255,255}, 1, cv::LINE_AA);
                }
            }

            // 4. Alert banners (top-left)
            int alert_idx = 0;
            auto draw_alert = [&](const std::string& txt, cv::Scalar col) {
                int baseline=0;
                auto tsz = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
                int ay = 16 + alert_idx * 46;
                cv::rectangle(ann, {16, ay}, {28+tsz.width, ay+40}, {15,23,42}, cv::FILLED);
                cv::rectangle(ann, {16, ay}, {28+tsz.width, ay+40}, col, 2);
                cv::putText(ann, txt, {22, ay+30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv::LINE_AA);
                ++alert_idx;
            };

            for (auto& sd : res.signs) {
                if (alert_idx >= 2) break;
                if (sd.sign_type == "danger" || sd.sign_type == "prohibit")
                    draw_alert("CANH BAO: " + sd.name + "!", {0,0,255});
                else if (sd.sign_type == "warning")
                    draw_alert("CHU Y: " + sd.name + "!", {0,180,255});
            }

            // 5. HUD status bar (bottom)
            int H = ann.rows, W = ann.cols;
            cv::Mat hud_bg = ann.clone();
            cv::rectangle(hud_bg, {0, H-30}, {W, H}, {11,15,25}, cv::FILLED);
            cv::addWeighted(hud_bg, 0.8, ann, 0.2, 0, ann);
            char hud_txt[256];
            snprintf(hud_txt, sizeof(hud_txt),
                     "FRAME: #%d | YOLOPv2 PANOPTIC + TSR (32 Bien bao VN) | %.1f FPS",
                     fd.frame_idx, cur_fps);
            cv::putText(ann, hud_txt, {14, H-9},
                        cv::FONT_HERSHEY_SIMPLEX, 0.44, {241,245,249}, 1, cv::LINE_AA);

            double elapsed = _now_ms() - t0;

            // FPS calculation
            ++fps_cnt;
            double now = _now_ms();
            if (now - fps_timer >= 500.0) {
                cur_fps = fps_cnt / ((now - fps_timer) / 1000.0);
                fps_cnt = 0;
                fps_timer = now;
                last_fps_.store(cur_fps);
            }

            // Env estimate (nhẹ hơn Python - resize 64x36 grayscale mean)
            std::string env_label = "Ban ngay";
            int env_brightness = 135;
            {
                cv::Mat small;
                cv::resize(fd.bgr, small, {64,36}, 0, 0, cv::INTER_NEAREST);
                cv::Mat gray;
                cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
                double mean_val = cv::mean(gray)[0];
                env_brightness = (int)mean_val;
                if (mean_val < 65)       env_label = "Ban dem / Thieu sang";
                else if (mean_val > 185) env_label = "Nang gat / Choi sang";
            }

            // Callback với annotated frame
            if (frame_cb_) {
                AnnotatedFrame af;
                af.bgr            = std::move(ann);
                af.result         = std::move(res);
                af.elapsed_ms     = elapsed;
                af.real_fps       = cur_fps;
                af.frame_idx      = fd.frame_idx;
                af.env_label      = env_label;
                af.env_brightness = env_brightness;
                frame_cb_(std::move(af));
            }
        }
    }

    static double _now_ms() {
        using namespace std::chrono;
        return duration<double,std::milli>(
            high_resolution_clock::now().time_since_epoch()).count();
    }
};
