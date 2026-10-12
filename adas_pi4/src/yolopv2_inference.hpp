/**
 * =============================================================================
 * YOLOPv2 + Traffic Sign Recognition (TSR) - C++ Inference Engine
 * =============================================================================
 * Phiên bản C++ cao hiệu năng của app_yolopv2.py
 * Tối ưu hóa FPS thông qua:
 *   1. ONNX Runtime C++ API (tránh Python overhead)
 *   2. Multi-threading: Inference thread riêng, UI thread riêng
 *   3. Double-buffer frame queue (lock-free ring buffer)
 *   4. OpenCV parallel_for_ thay thế numpy Python loops
 *   5. Lane mask cache + cadence skip (mỗi N frame mới infer lane)
 *   6. Sign detection cadence (skip N frame)
 *   7. Pre-allocated Mat buffers (tránh alloc/dealloc mỗi frame)
 *   8. CUDA EP hoặc DirectML EP tự động chọn
 * =============================================================================
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <array>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <queue>
#include <condition_variable>
#include <chrono>
#include <functional>
#include <cmath>
#include <algorithm>
#include <numeric>

// ============================================================
// CẤU TRÚC DỮ LIỆU KẾT QUẢ
// ============================================================

struct BoundingBox {
    float x1, y1, x2, y2;
    float conf;
    int   cls;
};

struct VehicleDetection {
    BoundingBox box;
    cv::Mat     crop_bgr;   // thumbnail 96x76
};

struct SignDetection {
    BoundingBox  box;
    int          cls_id;
    std::string  code;
    std::string  name;
    std::string  sign_type;  // "prohibit","danger","speed","warning","guide"
    std::string  alert_tag;
    cv::Scalar   color_bgr;
    cv::Mat      crop_bgr;
};

struct InferenceResult {
    std::vector<VehicleDetection> vehicles;
    std::vector<SignDetection>    signs;
    cv::Mat                       ll_mask;   // lane line mask (H,W) uint8
    cv::Mat                       da_mask;   // drivable area mask (H,W) uint8
    double                        t_yolop_ms = 0.0;
    double                        t_sign_ms  = 0.0;
    int                           frame_idx  = 0;
};

struct FrameData {
    cv::Mat        bgr;
    int            frame_idx = 0;
    bool           infer_lane = true;
    bool           infer_sign = true;
};

// ============================================================
// UTILITY: letterbox (giống Python letterbox())
// ============================================================
inline cv::Mat letterbox(const cv::Mat& src,
                         int new_h, int new_w,
                         cv::Scalar pad_color = {114, 114, 114},
                         float* ratio_out = nullptr,
                         float* dw_out = nullptr,
                         float* dh_out = nullptr)
{
    int h = src.rows, w = src.cols;
    float r = std::min((float)new_h / h, (float)new_w / w);

    int unpad_w = (int)std::round(w * r);
    int unpad_h = (int)std::round(h * r);
    float dw = (new_w - unpad_w) / 2.0f;
    float dh = (new_h - unpad_h) / 2.0f;

    cv::Mat resized;
    cv::resize(src, resized, {unpad_w, unpad_h}, 0, 0, cv::INTER_LINEAR);

    int top    = (int)std::round(dh - 0.1f);
    int bottom = (int)std::round(dh + 0.1f);
    int left   = (int)std::round(dw - 0.1f);
    int right  = (int)std::round(dw + 0.1f);

    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right,
                       cv::BORDER_CONSTANT, pad_color);
    if (ratio_out) *ratio_out = r;
    if (dw_out)    *dw_out = dw;
    if (dh_out)    *dh_out = dh;
    return out;
}

// ============================================================
// UTILITY: NMS (Non-Maximum Suppression)
// ============================================================
inline float iou_box(const BoundingBox& a, const BoundingBox& b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float ua = std::max(1.0f, (a.x2-a.x1)*(a.y2-a.y1)) +
               std::max(1.0f, (b.x2-b.x1)*(b.y2-b.y1)) - inter;
    return inter / ua;
}

inline std::vector<int> nms(std::vector<BoundingBox>& boxes, float iou_thr) {
    // Sắp xếp theo conf giảm dần
    std::vector<int> idx(boxes.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b){ return boxes[a].conf > boxes[b].conf; });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<int> keep;

    for (size_t i = 0; i < idx.size(); ++i) {
        int ii = idx[i];
        if (suppressed[ii]) continue;
        keep.push_back(ii);
        for (size_t j = i + 1; j < idx.size(); ++j) {
            int jj = idx[j];
            if (!suppressed[jj] && iou_box(boxes[ii], boxes[jj]) > iou_thr)
                suppressed[jj] = true;
        }
    }
    return keep;
}

// ============================================================
// UTILITY: scale_coords (giống Python scale_coords())
// ============================================================
inline void scale_coords(float ratio, float dw, float dh,
                         BoundingBox& box,
                         int orig_h, int orig_w)
{
    box.x1 = std::max(0.0f, (box.x1 - dw) / ratio);
    box.y1 = std::max(0.0f, (box.y1 - dh) / ratio);
    box.x2 = std::min((float)orig_w, (box.x2 - dw) / ratio);
    box.y2 = std::min((float)orig_h, (box.y2 - dh) / ratio);
}

// ============================================================
// UTILITY: apply_lane_mask (giống Python apply_seg_masks())
// ============================================================
inline void apply_lane_mask(cv::Mat& bgr, const cv::Mat& ll_mask) {
    if (ll_mask.empty()) return;
    int H = bgr.rows, W = bgr.cols;

    cv::Mat mask_resized;
    if (ll_mask.rows != H || ll_mask.cols != W)
        cv::resize(ll_mask, mask_resized, {W, H}, 0, 0, cv::INTER_NEAREST);
    else
        mask_resized = ll_mask;

    // Tô màu đỏ (BGR: 0,0,255) cho vạch kẻ lane với alpha 0.55
    // Dùng OpenCV parallel để tăng tốc
    cv::parallel_for_(cv::Range(0, H), [&](const cv::Range& r) {
        for (int y = r.start; y < r.end; ++y) {
            const uint8_t* mrow = mask_resized.ptr<uint8_t>(y);
            cv::Vec3b* brow = bgr.ptr<cv::Vec3b>(y);
            for (int x = 0; x < W; ++x) {
                if (mrow[x] > 0) {
                    brow[x][0] = (uint8_t)(brow[x][0] * 0.45f);           // B
                    brow[x][1] = (uint8_t)(brow[x][1] * 0.45f);           // G
                    brow[x][2] = (uint8_t)(brow[x][2] * 0.45f + 255 * 0.55f); // R
                }
            }
        }
    });
}


// ============================================================
// SIGN METADATA (32 class Vietnamese TSR)
// ============================================================
struct SignMeta {
    const char* code;
    const char* name;
    const char* type; // "prohibit","danger","speed","warning","guide"
};

static const SignMeta SIGN_META_TABLE[32] = {
    {"R.434",  "Ben xe buyt",             "guide"},    // 0
    {"P.103a", "Cam o to",                "prohibit"}, // 1
    {"R.301",  "Chi dan huong di",        "guide"},    // 2
    {"R.415",  "Gop lan theo xe",         "guide"},    // 3
    {"P.102",  "Cam di nguoc chieu",      "danger"},   // 4
    {"P.137",  "Cam thang & re phai",     "prohibit"}, // 5
    {"P.131a", "Cam do xe",               "prohibit"}, // 6
    {"P.130",  "Cam dung va do xe",       "danger"},   // 7
    {"P.124a", "Cam quay dau",            "prohibit"}, // 8
    {"P.123b", "Cam re phai",             "prohibit"}, // 9
    {"P.123a", "Cam re trai",             "prohibit"}, // 10
    {"P.137",  "Cam re trai & phai",      "prohibit"}, // 11
    {"P.124c", "Cam re trai & quay dau",  "prohibit"}, // 12
    {"W.201",  "Cho ngoa nguy hiem",      "warning"},  // 13
    {"R.409",  "Cho quay xe",             "guide"},    // 14
    {"W.233",  "Chu y chuong ngai vat",   "warning"},  // 15
    {"P.245a", "Di cham",                 "warning"},  // 16
    {"W.203",  "Duong bi thu hep",        "warning"},  // 17
    {"W.205",  "Duong giao nhau",         "warning"},  // 18
    {"R.407a", "Duong mot chieu",         "guide"},    // 19
    {"W.224",  "Nguoi di bo cat ngang",   "warning"},  // 20
    {"W.207",  "Giao duong khong uu tien","warning"},  // 21
    {"W.210",  "Duong sat co rao chan",   "warning"},  // 22
    {"P.117",  "Gioi han chieu cao",      "prohibit"}, // 23
    {"P.127",  "Gioi han toc do",         "speed"},    // 24
    {"P.115",  "Gioi han trong luong",    "prohibit"}, // 25
    {"W.221",  "Go giam toc phia truoc",  "warning"},  // 26
    {"P.129",  "Tram kiem tra",           "danger"},   // 27
    {"R.303",  "Vong xuyen",              "guide"},    // 28
    {"R.302a", "Di vong sang ben phai",   "guide"},    // 29
    {"W.225",  "Tre em qua duong",        "warning"},  // 30
    {"I.423",  "Vach nguoi di bo",        "guide"},    // 31
};

inline cv::Scalar sign_color(const std::string& type) {
    if (type == "prohibit" || type == "danger") return {0, 0, 235};
    if (type == "speed")   return {30, 80, 255};
    if (type == "warning") return {0, 180, 255};
    if (type == "guide")   return {220, 150, 0};
    return {0, 212, 170};
}
