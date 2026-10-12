/**
 * =============================================================================
 * TrafficSignEngine C++ - Ultralytics YOLO11 TSR via ONNX Runtime
 * =============================================================================
 * Tương đương class TrafficSignEngine trong Python.
 * Sử dụng ONNX export của tsr.pt (yolo11n/s).
 * Để export: yolo export model=tsr.pt format=onnx imgsz=480
 * =============================================================================
 */

#pragma once
#include "yolopv2_inference.hpp"
#include <stdexcept>

class TrafficSignEngine {
public:
    struct Config {
        std::string weights_path;
        std::string device     = "cuda"; // "cuda" | "cpu"
        float       conf_thres = 0.35f;
        float       iou_thres  = 0.45f;
        int         imgsz      = 480;
    };

    explicit TrafficSignEngine(const Config& cfg) : cfg_(cfg) {
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        so.SetIntraOpNumThreads(2);

#ifdef ORT_HAS_CUDA
        if (cfg_.device != "cpu") {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            try {
                so.AppendExecutionProvider_CUDA(cuda_opts);
                printf("[TSR C++] CUDA EP.\n");
            } catch (...) {}
        }
#endif

        session_ = std::make_unique<Ort::Session>(env_,
            std::wstring(cfg_.weights_path.begin(), cfg_.weights_path.end()).c_str(), so);

        // Input shape
        auto shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 4 && shape[2] > 0) {
            fixed_h_ = (int)shape[2];
            fixed_w_ = (int)shape[3];
        } else {
            fixed_h_ = fixed_w_ = cfg_.imgsz;
        }

        Ort::AllocatorWithDefaultOptions alloc;
        input_name_  = session_->GetInputNameAllocated(0, alloc).get();
        output_name_ = session_->GetOutputNameAllocated(0, alloc).get();

        input_buf_.resize(1 * 3 * fixed_h_ * fixed_w_, 0.0f);
        _run_dummy();
        printf("[TSR C++] Ready! Size: %dx%d, Classes: %d\n",
               fixed_h_, fixed_w_, 32);
    }

    /**
     * Inference traffic signs từ frame BGR.
     */
    std::vector<SignDetection> infer(const cv::Mat& bgr_orig) {
        std::vector<SignDetection> results;
        int H0 = bgr_orig.rows, W0 = bgr_orig.cols;

        // 1. Letterbox
        float ratio, dw, dh;
        cv::Mat lb = letterbox(bgr_orig, fixed_h_, fixed_w_,
                               {114,114,114}, &ratio, &dw, &dh);

        // 2. BGR -> RGB CHW float32 /255
        _bgr_to_chw_float(lb, input_buf_.data(), fixed_h_, fixed_w_);

        // 3. Tensor
        std::array<int64_t,4> shape_in = {1, 3, fixed_h_, fixed_w_};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inp = Ort::Value::CreateTensor<float>(
            mem, input_buf_.data(), input_buf_.size(),
            shape_in.data(), shape_in.size());

        // 4. Run
        const char* in_names[]  = { input_name_.c_str() };
        const char* out_names[] = { output_name_.c_str() };
        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                     in_names, &inp, 1,
                                     out_names, 1);

        // 5. Parse - YOLO11 ONNX output: (1, NC+4, num_boxes) transposed
        auto& out_tensor = outputs[0];
        auto out_shape = out_tensor.GetTensorTypeAndShapeInfo().GetShape();
        // YOLO export shape: (1, 4+NC, num_anchors)
        if (out_shape.size() < 3) return results;

        int num_anchors = (int)out_shape[2];
        int num_fields  = (int)out_shape[1]; // 4 + NC
        int NC = num_fields - 4;
        if (NC <= 0 || NC > 128) return results;

        const float* data = out_tensor.GetTensorData<float>();

        // Parse boxes
        std::vector<BoundingBox> raw_boxes;
        for (int i = 0; i < num_anchors; ++i) {
            // box: cx,cy,w,h then classes
            float cx = data[0 * num_anchors + i];
            float cy = data[1 * num_anchors + i];
            float bw = data[2 * num_anchors + i];
            float bh = data[3 * num_anchors + i];

            float max_cls = -1e9f;
            int   max_c   = 0;
            for (int c = 0; c < NC; ++c) {
                float v = data[(4 + c) * num_anchors + i];
                if (v > max_cls) { max_cls = v; max_c = c; }
            }

            float conf = max_cls; // YOLO11 uses cls_score directly (sigmoid applied)
            if (conf < cfg_.conf_thres) continue;

            BoundingBox box;
            box.x1   = cx - bw * 0.5f;
            box.y1   = cy - bh * 0.5f;
            box.x2   = cx + bw * 0.5f;
            box.y2   = cy + bh * 0.5f;
            box.conf = conf;
            box.cls  = max_c;
            raw_boxes.push_back(box);
        }

        // NMS
        auto keep = nms(raw_boxes, cfg_.iou_thres);

        for (int ki : keep) {
            auto& b = raw_boxes[ki];
            if (b.conf < cfg_.conf_thres) continue;

            // Scale coords
            scale_coords(ratio, dw, dh, b, H0, W0);

            SignDetection sd;
            sd.box = b;
            sd.cls_id = std::min(b.cls, 31);

            const SignMeta& meta = SIGN_META_TABLE[sd.cls_id];
            sd.code      = meta.code;
            sd.name      = meta.name;
            sd.sign_type = meta.type;
            sd.color_bgr = sign_color(sd.sign_type);

            if (sd.sign_type == "danger" || sd.sign_type == "prohibit")
                sd.alert_tag = "NGUY HIEM";
            else if (sd.sign_type == "speed")
                sd.alert_tag = "TOC DO";
            else if (sd.sign_type == "warning")
                sd.alert_tag = "CHU Y";
            else
                sd.alert_tag = "CHI DAN";

            int x1 = std::max(0, (int)b.x1);
            int y1 = std::max(0, (int)b.y1);
            int x2 = std::min(W0, (int)b.x2);
            int y2 = std::min(H0, (int)b.y2);
            if (x2 > x1 && y2 > y1) {
                cv::Mat crop = bgr_orig(cv::Rect(x1,y1,x2-x1,y2-y1)).clone();
                cv::resize(crop, sd.crop_bgr, {96,76}, 0, 0, cv::INTER_AREA);
            }

            results.push_back(std::move(sd));
        }
        return results;
    }

    Config cfg_;

private:
    Ort::Env env_{ ORT_LOGGING_LEVEL_WARNING, "TSR" };
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_, output_name_;
    std::vector<float> input_buf_;
    int fixed_h_ = 480, fixed_w_ = 480;

    static void _bgr_to_chw_float(const cv::Mat& bgr, float* out, int H, int W) {
        int plane = H * W;
        cv::parallel_for_(cv::Range(0, H), [&](const cv::Range& r) {
            for (int y = r.start; y < r.end; ++y) {
                const uint8_t* row = bgr.ptr<uint8_t>(y);
                int off = y * W;
                for (int x = 0; x < W; ++x) {
                    out[0*plane+off+x] = row[x*3+2] / 255.0f; // R
                    out[1*plane+off+x] = row[x*3+1] / 255.0f; // G
                    out[2*plane+off+x] = row[x*3+0] / 255.0f; // B
                }
            }
        });
    }

    void _run_dummy() {
        std::fill(input_buf_.begin(), input_buf_.end(), 0.0f);
        std::array<int64_t,4> shape = {1,3,fixed_h_,fixed_w_};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inp = Ort::Value::CreateTensor<float>(
            mem, input_buf_.data(), input_buf_.size(), shape.data(), shape.size());
        const char* in[]  = { input_name_.c_str() };
        const char* out[] = { output_name_.c_str() };
        try { session_->Run(Ort::RunOptions{nullptr}, in, &inp, 1, out, 1); }
        catch (...) {}
    }
};
