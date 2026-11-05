#pragma once

#include "road_geometry.hpp"
#include "adas_types.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <unistd.h>
#include <vector>
#include <onnxruntime_cxx_api.h>

#if __has_include(<ncnn/net.h>)
#include <ncnn/allocator.h>
#include <ncnn/mat.h>
#include <ncnn/net.h>
#define HAVE_NCNN 1
#elif __has_include(<net.h>)
#include <allocator.h>
#include <mat.h>
#include <net.h>
#define HAVE_NCNN 1
#endif

class NanoDetEngine {
public:
    struct Config {
        std::string onnx_path;
        std::string param_path;
        std::string bin_path;
        std::string input_name = "data";
        std::string output_name = "output";
        int   input_w = 320;
        int   input_h = 320;
        int   num_classes = 80;
        int   reg_max = 7;
        std::vector<int> strides = {8, 16, 32, 64};
        float conf_thresh = 0.30f;
        float nms_thresh = 0.50f;
        int   num_threads = 2;
        int   min_box_px = 6;
        std::vector<int> class_map;
        bool  allow_simulation = false;
    };

    NanoDetEngine() : NanoDetEngine(Config{}) {}
    explicit NanoDetEngine(const Config& cfg) : m_cfg(cfg) {
        buildClassMap();
        init();
    }

    bool ready() const { return m_ready || m_cfg.allow_simulation; }
    bool usesRealModel() const { return m_ready; }
    const Config& config() const { return m_cfg; }

    void setConfThreshold(float t) { m_cfg.conf_thresh = std::clamp(t, 0.05f, 0.95f); }

    int freeRows(int roadW, int roadH) const {
        const float s = std::min(float(m_cfg.input_w) / roadW, float(m_cfg.input_h) / roadH);
        return m_cfg.input_h - farTop(int(std::lround(roadH * s)));
    }

    std::vector<VehicleDetection> infer(const cv::Mat& road, double* out_ms = nullptr, bool* ok = nullptr,
                                        const cv::Rect& far = cv::Rect()) {
        std::vector<VehicleDetection> dets;
        if (ok) *ok = false;
        if (road.empty() || road.type() != CV_8UC3) return dets;
        const auto t0 = std::chrono::steady_clock::now();

        const float scale = letterbox(road);
        const int mainH = int(std::lround(road.rows * scale));
        cv::Rect farR = far & cv::Rect(0, 0, road.cols, road.rows);
        const int fy0 = farTop(mainH);
        if (farR.width > 0 && farR.height > 0 && farR.width <= m_cfg.input_w && farR.height <= m_cfg.input_h - fy0)
            road(farR).copyTo(m_lb(cv::Rect(0, fy0, farR.width, farR.height)));
        else farR = cv::Rect();
        m_far = farR; m_farY0 = fy0; m_mainH = mainH;

        if (m_usesOrt && m_ortSession) {
            const int hw = m_cfg.input_h * m_cfg.input_w;
            m_input.resize(size_t(3) * hw);
            const float mean[3] = {103.53f, 116.28f, 123.675f};
            const float inv_std[3] = {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f};
            float* pb = m_input.data();
            float* pg = pb + hw;
            float* pr = pg + hw;
            for (int y = 0; y < m_cfg.input_h; ++y) {
                const cv::Vec3b* row = m_lb.ptr<cv::Vec3b>(y);
                const int o = y * m_cfg.input_w;
                for (int x = 0; x < m_cfg.input_w; ++x) {
                    pb[o + x] = (row[x][0] - mean[0]) * inv_std[0];
                    pg[o + x] = (row[x][1] - mean[1]) * inv_std[1];
                    pr[o + x] = (row[x][2] - mean[2]) * inv_std[2];
                }
            }
            const std::array<int64_t, 4> shape = {1, 3, m_cfg.input_h, m_cfg.input_w};
            Ort::Value input = Ort::Value::CreateTensor<float>(m_memInfo, m_input.data(), m_input.size(),
                                                               shape.data(), shape.size());
            const char* in_names[] = {m_cfg.input_name.c_str()};
            const char* out_names[] = {m_cfg.output_name.c_str()};
            try {
                auto outs = m_ortSession->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
                const auto shp = outs[0].GetTensorTypeAndShapeInfo().GetShape();
                if (shp.size() != 3) { logOnce("[NanoDet] unexpected output rank"); return dets; }
                decode(outs[0].GetTensorData<float>(), int(shp[1]), int(shp[2]), scale, road.cols, road.rows, dets);
                if (m_far.width > 0) dets = nmsMerged(std::move(dets), m_cfg.nms_thresh);
            } catch (const std::exception& e) {
                logOnce(e.what());
                return dets;
            }
            dets = nmsPerClass(std::move(dets), m_cfg.nms_thresh);
            if (out_ms) *out_ms = msSince(t0);
            if (ok) *ok = true;
            return dets;
        }

#ifdef HAVE_NCNN
        if (m_ready && m_net) {
            ncnn::Mat in = ncnn::Mat::from_pixels(m_lb.data, ncnn::Mat::PIXEL_BGR, m_lb.cols, m_lb.rows,
                                                  int(m_lb.step[0]), &m_blobPool);
            const float mean[3] = {103.53f, 116.28f, 123.675f};
            const float norm[3] = {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f};
            in.substract_mean_normalize(mean, norm);
            ncnn::Extractor ex = m_net->create_extractor();
            ex.input(m_cfg.input_name.c_str(), in);
            ncnn::Mat out;
            if (ex.extract(m_cfg.output_name.c_str(), out) != 0) { logOnce("[NanoDet] extract failed"); return dets; }
            decode(reinterpret_cast<const float*>(out.data), out.h, out.w, scale, road.cols, road.rows, dets);
            dets = nmsPerClass(std::move(dets), m_cfg.nms_thresh);
            if (out_ms) *out_ms = msSince(t0);
            if (ok) *ok = true;
            return dets;
        }
#endif

        if (m_cfg.allow_simulation) {
            VehicleDetection v;
            v.box.x1 = road.cols * 0.45f; v.box.x2 = road.cols * 0.55f;
            v.box.y1 = road.rows * 0.30f; v.box.y2 = road.rows * 0.42f;
            v.box.conf = 0.80f; v.box.cls = VC_CAR;
            dets.push_back(v);
            if (out_ms) *out_ms = msSince(t0);
            if (ok) *ok = true;
        }
        return dets;
    }

private:
    Config m_cfg;
    bool   m_ready = false;
    bool   m_usesOrt = false;
    bool   m_logged = false;
    std::vector<int> m_map;
    cv::Mat m_lb;
    cv::Rect m_far;
    int m_farY0 = 0, m_mainH = 0;
    static int farTop(int mainH) { return ((mainH + 8 + 31) / 32) * 32; }
    std::vector<float> m_input;

    std::unique_ptr<Ort::Env> m_ortEnv;
    std::unique_ptr<Ort::Session> m_ortSession;
    Ort::MemoryInfo m_memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

#ifdef HAVE_NCNN
    std::unique_ptr<ncnn::Net> m_net;
    ncnn::UnlockedPoolAllocator m_blobPool;
    ncnn::PoolAllocator m_workPool;
#endif

    float letterbox(const cv::Mat& src) {
        const float s = std::min(float(m_cfg.input_w) / src.cols, float(m_cfg.input_h) / src.rows);
        const int nw = std::max(1, int(std::lround(src.cols * s)));
        const int nh = std::max(1, int(std::lround(src.rows * s)));
        if (m_lb.rows != m_cfg.input_h || m_lb.cols != m_cfg.input_w) m_lb.create(m_cfg.input_h, m_cfg.input_w, CV_8UC3);
        m_lb.setTo(cv::Scalar(0, 0, 0));
        cv::Mat roi = m_lb(cv::Rect(0, 0, nw, nh));
        cv::resize(src, roi, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
        return s;
    }

    void buildClassMap() {
        if (!m_cfg.class_map.empty()) { m_map = m_cfg.class_map; return; }
        m_map.assign(m_cfg.num_classes, VC_IGNORE);
        if (m_cfg.num_classes == 80) {
            m_map[1] = VC_BICYCLE; m_map[2] = VC_CAR; m_map[3] = VC_MOTORBIKE;
            m_map[5] = VC_BUS_TRUCK; m_map[7] = VC_BUS_TRUCK;
        } else {
            for (int i = 0; i < std::min(4, m_cfg.num_classes); ++i) m_map[i] = i;
        }
    }

    void init() {
        if (!m_cfg.onnx_path.empty() && access(m_cfg.onnx_path.c_str(), R_OK) == 0) {
            try {
                Ort::SessionOptions so;
                so.SetIntraOpNumThreads(m_cfg.num_threads);
                so.SetInterOpNumThreads(1);
                so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
                so.AddConfigEntry("session.intra_op.allow_spinning", "0");
                m_ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "NanoDet");
                m_ortSession = std::make_unique<Ort::Session>(*m_ortEnv, m_cfg.onnx_path.c_str(), so);
                m_ready = m_usesOrt = true;
                std::printf("[NanoDet-Plus] ONNX model loaded: %s (%dx%d)\n", m_cfg.onnx_path.c_str(),
                            m_cfg.input_w, m_cfg.input_h);
                return;
            } catch (const std::exception& e) {
                std::printf("[NanoDet-Plus] Failed to load ONNX: %s\n", e.what());
            }
        }
#ifdef HAVE_NCNN
        if (!m_cfg.param_path.empty() && !m_cfg.bin_path.empty()) {
            m_net = std::make_unique<ncnn::Net>();
            ncnn::Option& o = m_net->opt;
            o.use_vulkan_compute = false;
            o.num_threads = m_cfg.num_threads;
            o.lightmode = true;
            o.use_packing_layout = true;
            o.use_fp16_storage = true;
            o.use_fp16_arithmetic = false;
            o.openmp_blocktime = 0;
            o.blob_allocator = &m_blobPool;
            o.workspace_allocator = &m_workPool;
            if (m_net->load_param(m_cfg.param_path.c_str()) == 0 && m_net->load_model(m_cfg.bin_path.c_str()) == 0) {
                m_ready = true;
                std::printf("[NanoDet-Plus] NCNN model loaded: %s\n", m_cfg.bin_path.c_str());
                return;
            }
            m_net.reset();
        }
#endif
        std::printf("[NanoDet-Plus] Model NOT loaded%s\n", m_cfg.allow_simulation ? " -> simulation (DEV ONLY)" : "");
    }

    static bool plausibleAspect(int vc, float w, float h) {
        const float r = w / std::max(1.f, h);
        switch (vc) {
            case VC_CAR:       return r > 0.65f && r < 3.2f;
            case VC_BUS_TRUCK: return r > 0.45f && r < 3.5f;
            case VC_MOTORBIKE: return r > 0.20f && r < 1.7f;
            case VC_BICYCLE:   return r > 0.20f && r < 1.8f;
            default:           return false;
        }
    }

    void decode(const float* out, int rows, int cols, float scale, int cw, int ch,
                std::vector<VehicleDetection>& dets) {
        const int nc = m_cfg.num_classes, bins = m_cfg.reg_max + 1;
        if (cols < nc + 4 * bins || bins > 32) { logOnce("[NanoDet] output width != num_classes + 4*(reg_max+1)"); return; }
        const float inv = 1.f / scale;
        float prob[32];
        int row = 0;
        for (int stride : m_cfg.strides) {
            const int fw = (m_cfg.input_w + stride - 1) / stride;
            const int fh = (m_cfg.input_h + stride - 1) / stride;
            for (int y = 0; y < fh; ++y) {
                for (int x = 0; x < fw; ++x, ++row) {
                    if (row >= rows) return;
                    const float* r = out + size_t(row) * cols;
                    int best = -1; float score = m_cfg.conf_thresh;
                    for (int c = 0; c < nc; ++c) if (r[c] > score) { score = r[c]; best = c; }
                    if (best < 0) continue;
                    const int vc = best < int(m_map.size()) ? m_map[best] : VC_IGNORE;
                    if (vc == VC_IGNORE) continue;

                    float d[4];
                    for (int k = 0; k < 4; ++k) {
                        const float* b = r + nc + k * bins;
                        float mx = b[0];
                        for (int i = 1; i < bins; ++i) mx = std::max(mx, b[i]);
                        float s = 0.f, e = 0.f;
                        for (int i = 0; i < bins; ++i) { prob[i] = std::exp(b[i] - mx); s += prob[i]; }
                        for (int i = 0; i < bins; ++i) e += prob[i] * i;
                        d[k] = (s > 1e-6f ? e / s : 0.f) * stride;
                    }
                    const float cx = float(x * stride), cy = float(y * stride);
                    VehicleDetection v;
                    const float ix1 = cx - d[0], iy1 = cy - d[1], ix2 = cx + d[2], iy2 = cy + d[3];
                    if (m_far.width > 0 && iy2 > m_mainH + 2) {
                        const float fx2 = float(m_far.width), fy2 = float(m_farY0 + m_far.height);
                        if (iy1 < m_farY0 - 1 || iy2 > fy2 - 1 || (ix1 < 1 && m_far.x > 0) ||
                            (ix2 > fx2 - 1 && m_far.x + m_far.width < cw) || ix2 > fx2 + 1) continue;
                        v.box.x1 = std::clamp(ix1 + m_far.x, 0.f, float(cw));
                        v.box.y1 = std::clamp(iy1 - m_farY0 + m_far.y, 0.f, float(ch));
                        v.box.x2 = std::clamp(ix2 + m_far.x, 0.f, float(cw));
                        v.box.y2 = std::clamp(iy2 - m_farY0 + m_far.y, 0.f, float(ch));
                        v.far = true;
                    } else {
                        v.box.x1 = std::clamp(ix1 * inv, 0.f, float(cw));
                        v.box.y1 = std::clamp(iy1 * inv, 0.f, float(ch));
                        v.box.x2 = std::clamp(ix2 * inv, 0.f, float(cw));
                        v.box.y2 = std::clamp(iy2 * inv, 0.f, float(ch));
                    }
                    v.box.conf = score;
                    v.box.cls = vc;
                    const float bw = v.box.x2 - v.box.x1, bh = v.box.y2 - v.box.y1;
                    if (bw < m_cfg.min_box_px || bh < m_cfg.min_box_px) continue;
                    if (!plausibleAspect(vc, bw, bh)) continue;
                    dets.push_back(v);
                }
            }
        }
    }

    static float iou(const BoundingBox& a, const BoundingBox& b) {
        const float iw = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
        const float ih = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
        const float in = iw * ih;
        const float u = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - in;
        return u > 0 ? in / u : 0.f;
    }
    static std::vector<VehicleDetection> nmsMerged(std::vector<VehicleDetection> d, float thr) {
        std::vector<VehicleDetection> far, keep;
        for (auto& v : d) if (v.far) far.push_back(v);
        far = nmsPerClass(std::move(far), thr);
        for (const auto& v : d) {
            if (v.far) continue;
            bool dup = false;
            for (const auto& f : far) if (iou(v.box, f.box) > 0.4f) { dup = true; break; }
            if (!dup) keep.push_back(v);
        }
        keep.insert(keep.end(), far.begin(), far.end());
        return keep;
    }
    static std::vector<VehicleDetection> nmsPerClass(std::vector<VehicleDetection> d, float thr) {
        std::sort(d.begin(), d.end(), [](const auto& a, const auto& b) { return a.box.conf > b.box.conf; });
        std::vector<VehicleDetection> keep;
        for (const auto& c : d) {
            bool drop = false;
            for (const auto& k : keep) {
                const float o = iou(c.box, k.box);
                if ((c.box.cls == k.box.cls && o > thr) || o > 0.80f) { drop = true; break; }
            }
            if (!drop) keep.push_back(c);
        }
        return keep;
    }

    void logOnce(const char* msg) {
        if (!m_logged) { std::fprintf(stderr, "%s\n", msg); m_logged = true; }
    }
    static double msSince(std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
};
