#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <future>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <onnxruntime_cxx_api.h>

enum SignType { SIGN_PROHIBIT = 0, SIGN_DANGER = 1, SIGN_SPEED = 2, SIGN_MANDATORY = 3, SIGN_GUIDE = 4,
                SIGN_END = 5,
                SIGN_OTHER = 6 };

struct SignInfo { const char* ascii; const char* display; SignType type; };

static const std::array<SignInfo, 17> kSignTable = {{
    {"Gioi han toc do",                      "Giới hạn tốc độ",                     SIGN_SPEED},
    {"Het han che toc do",                   "Hết hạn chế tốc độ",                  SIGN_END},
    {"Het moi lenh cam",                     "Hết mọi lệnh cấm",                    SIGN_END},
    {"Khu dong dan cu",                      "Khu đông dân cư",                     SIGN_GUIDE},
    {"Het khu dong dan cu",                  "Hết khu đông dân cư",                 SIGN_GUIDE},
    {"Duong cao toc",                        "Đường cao tốc",                       SIGN_GUIDE},
    {"Het duong cao toc",                    "Hết đường cao tốc",                   SIGN_GUIDE},
    {"Cam di nguoc chieu",                   "Cấm đi ngược chiều",                  SIGN_PROHIBIT},
    {"Dung lai",                             "Dừng lại",                            SIGN_PROHIBIT},
    {"Giao nhau voi duong uu tien",          "Giao với đường ưu tiên",              SIGN_DANGER},
    {"Duong nguoi di bo cat ngang",          "Người đi bộ cắt ngang",               SIGN_DANGER},
    {"Tre em",                               "Trẻ em",                              SIGN_DANGER},
    {"Giao nhau voi duong sat",              "Giao với đường sắt",                  SIGN_DANGER},
    {"Cho ngoat nguy hiem",                  "Chỗ ngoặt nguy hiểm",                 SIGN_DANGER},
    {"Giao nhau voi duong khong uu tien",    "Giao với đường không ưu tiên",        SIGN_DANGER},
    {"Go giam toc phia truoc",               "Gờ giảm tốc phía trước",              SIGN_DANGER},
    {"Bien bao",                             "Biển báo",                            SIGN_OTHER},
}};

inline const char* signTypeName(int t) {
    switch (t) {
        case SIGN_PROHIBIT:  return "prohibit";
        case SIGN_DANGER:    return "danger";
        case SIGN_SPEED:     return "speed";
        case SIGN_MANDATORY: return "mandatory";
        case SIGN_END:       return "end";
        case SIGN_OTHER:     return "other";
        default:             return "guide";
    }
}
inline const SignInfo& signInfo(int cls) {
    static const SignInfo unknown{"Bien bao", "Biển báo", SIGN_GUIDE};
    return cls >= 0 && cls < int(kSignTable.size()) ? kSignTable[cls] : unknown;
}

constexpr int kSpeedValues = 24, kSpeedOther = 24;

struct SignDetection {
    cv::Rect2f box;
    int cls = 0;
    float conf = 0.f;
    bool hasValue = false;
    std::array<float, kSpeedValues + 1> valueP{};
    cv::Mat patch;
};

class SpeedValueReader {
public:
    explicit SpeedValueReader(const std::string& path) {
        if (path.empty() || access(path.c_str(), R_OK) != 0) return;
        try {
            Ort::SessionOptions so;
            so.SetIntraOpNumThreads(1);
            so.SetInterOpNumThreads(1);
            so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
            so.AddConfigEntry("session.intra_op.allow_spinning", "0");
            m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "SpeedValue");
            m_session = std::make_unique<Ort::Session>(*m_env, path.c_str(), so);
            std::printf("[TSR] speed value reader loaded: %s\n", path.c_str());
        } catch (const std::exception& e) {
            std::printf("[TSR] speed value reader failed: %s\n", e.what());
            m_session.reset();
        }
    }
    bool ready() const { return m_session != nullptr; }

    bool read(const cv::Mat& bgr, const cv::Rect2f& box, std::array<float, kSpeedValues + 1>& p) {
        if (!m_session) return false;
        const cv::Rect r = cv::Rect(box) & cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (r.width < 6 || r.height < 6) return false;
        if (bgr.channels() == 1) bgr(r).copyTo(m_gray);
        else cv::cvtColor(bgr(r), m_gray, cv::COLOR_BGR2GRAY);
        cv::resize(m_gray, m_in, cv::Size(kIn, kIn), 0, 0, cv::INTER_LINEAR);
        float x[kIn * kIn];
        for (int i = 0; i < kIn * kIn; ++i) x[i] = m_in.data[i];
        const int64_t shp[4] = {1, 1, kIn, kIn};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in = Ort::Value::CreateTensor<float>(mem, x, kIn * kIn, shp, 4);
        const char* inN[] = {"crop"};
        const char* outN[] = {"logits"};
        try {
            auto out = m_session->Run(Ort::RunOptions{nullptr}, inN, &in, 1, outN, 1);
            const float* l = out[0].GetTensorData<float>();
            const float mx = *std::max_element(l, l + kSpeedValues + 1);
            float sum = 0.f;
            for (int i = 0; i <= kSpeedValues; ++i) sum += (p[i] = std::exp(l[i] - mx));
            for (auto& v : p) v /= sum;
            return true;
        } catch (const std::exception&) { return false; }
    }

private:
    static constexpr int kIn = 32;
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;
    cv::Mat m_gray, m_in;
};

class TrafficSignEngine {
public:
    struct Config {
        std::string onnx_path;
        float conf_thresh = 0.40f;
        float nms_thresh = 0.45f;
        int   num_threads = 2;
        int   min_box_px = 6;
    };

    explicit TrafficSignEngine(const Config& cfg) : m_cfg(cfg) {
        if (cfg.onnx_path.empty() || access(cfg.onnx_path.c_str(), R_OK) != 0) return;
        try {
            Ort::SessionOptions so;
            so.SetIntraOpNumThreads(cfg.num_threads);
            so.SetInterOpNumThreads(1);
            so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
            so.AddConfigEntry("session.intra_op.allow_spinning", "0");
            m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "TSR");
            m_session = std::make_unique<Ort::Session>(*m_env, cfg.onnx_path.c_str(), so);
            Ort::AllocatorWithDefaultOptions al;
            m_inName = m_session->GetInputNameAllocated(0, al).get();
            m_outName = m_session->GetOutputNameAllocated(0, al).get();
            const auto shp = m_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            if (shp.size() == 4 && shp[2] > 0 && shp[3] > 0) { m_inH = int(shp[2]); m_inW = int(shp[3]); }
            m_ready = true;
            std::printf("[TSR] ONNX model loaded: %s (%dx%d, %d threads)\n", cfg.onnx_path.c_str(), m_inW, m_inH, cfg.num_threads);
        } catch (const std::exception& e) {
            std::printf("[TSR] Failed to load ONNX: %s\n", e.what());
        }
    }

    bool ready() const { return m_ready; }
    int inputW() const { return m_inW; }
    int inputH() const { return m_inH; }

    std::vector<SignDetection> infer(const cv::Mat& band, double* out_ms = nullptr) {
        std::vector<SignDetection> dets;
        if (!m_ready || band.empty() || band.type() != CV_8UC3) return dets;
        const auto t0 = std::chrono::steady_clock::now();

        const float s = std::min(float(m_inW) / band.cols, float(m_inH) / band.rows);
        const int nw = std::max(1, int(std::lround(band.cols * s))), nh = std::max(1, int(std::lround(band.rows * s)));
        if (m_lb.rows != m_inH || m_lb.cols != m_inW) m_lb.create(m_inH, m_inW, CV_8UC3);
        m_lb.setTo(cv::Scalar(114, 114, 114));
        cv::Mat roi = m_lb(cv::Rect(0, 0, nw, nh));
        if (nw == band.cols && nh == band.rows) band.copyTo(roi);
        else cv::resize(band, roi, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

        const int hw = m_inW * m_inH;
        m_input.resize(size_t(3) * hw);
        float* pr = m_input.data(); float* pg = pr + hw; float* pb = pg + hw;
        for (int y = 0; y < m_inH; ++y) {
            const cv::Vec3b* row = m_lb.ptr<cv::Vec3b>(y);
            const int o = y * m_inW;
            for (int x = 0; x < m_inW; ++x) {
                pr[o + x] = row[x][2] * (1.f / 255.f);
                pg[o + x] = row[x][1] * (1.f / 255.f);
                pb[o + x] = row[x][0] * (1.f / 255.f);
            }
        }
        const std::array<int64_t, 4> shape = {1, 3, m_inH, m_inW};
        Ort::Value input = Ort::Value::CreateTensor<float>(m_memInfo, m_input.data(), m_input.size(), shape.data(), shape.size());
        const char* in_names[] = {m_inName.c_str()};
        const char* out_names[] = {m_outName.c_str()};
        try {
            auto outs = m_session->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
            const auto shp = outs[0].GetTensorTypeAndShapeInfo().GetShape();
            if (shp.size() == 3 && shp[1] > 4) decode(outs[0].GetTensorData<float>(), int(shp[1]) - 4, int(shp[2]), 1.f / s, band.cols, band.rows, dets);
        } catch (const std::exception& e) {
            if (!m_logged) { std::fprintf(stderr, "[TSR] %s\n", e.what()); m_logged = true; }
        }
        dets = nms(std::move(dets));
        if (out_ms) *out_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return dets;
    }

private:
    Config m_cfg;
    bool m_ready = false, m_logged = false;
    int m_inW = 640, m_inH = 224;
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;
    std::string m_inName, m_outName;
    Ort::MemoryInfo m_memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    cv::Mat m_lb;
    std::vector<float> m_input;

    void decode(const float* o, int nc, int n, float inv, int bw, int bh, std::vector<SignDetection>& dets) const {
        for (int i = 0; i < n; ++i) {
            int best = -1; float score = m_cfg.conf_thresh;
            for (int c = 0; c < nc; ++c) { const float v = o[size_t(4 + c) * n + i]; if (v > score) { score = v; best = c; } }
            if (best < 0) continue;
            const float cx = o[i] * inv, cy = o[size_t(n) + i] * inv, w = o[size_t(2) * n + i] * inv, h = o[size_t(3) * n + i] * inv;
            if (w < m_cfg.min_box_px || h < m_cfg.min_box_px) continue;
            const float x1 = std::clamp(cx - w / 2, 0.f, float(bw)), y1 = std::clamp(cy - h / 2, 0.f, float(bh));
            const float x2 = std::clamp(cx + w / 2, 0.f, float(bw)), y2 = std::clamp(cy + h / 2, 0.f, float(bh));
            if (x2 - x1 < 2 || y2 - y1 < 2) continue;
            dets.push_back({cv::Rect2f(x1, y1, x2 - x1, y2 - y1), best, score});
        }
    }
    static float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
        const float in = (a & b).area();
        const float u = a.area() + b.area() - in;
        return u > 0 ? in / u : 0.f;
    }
    std::vector<SignDetection> nms(std::vector<SignDetection> d) const {
        std::sort(d.begin(), d.end(), [](const auto& a, const auto& b) { return a.conf > b.conf; });
        std::vector<SignDetection> keep;
        for (const auto& c : d) {
            bool drop = false;
            for (const auto& k : keep) if (iou(c.box, k.box) > m_cfg.nms_thresh) { drop = true; break; }
            if (!drop) keep.push_back(c);
            if (keep.size() >= 20) break;
        }
        return keep;
    }
};

class SignTracker {
public:
    struct Track {
        int id = 0, cls = 0, hits = 0;
        float conf = 0.f;
        cv::Rect2f box;
        double firstT = 0, lastT = 0;
        bool confirmed = false;
        std::array<float, kSpeedValues + 1> votes{};
        int value = 0;
        cv::Mat patch;
        double followT = 0;
    };
    struct Params {
        int    confirm_hits = 2;
        double confirm_window_s = 3.0;
        float  confirm_conf = 0.75f;
        float  end_conf = 0.6f;
        double drop_s = 2.5;
        double hold_s = 5.0;
        float  value_min = 0.9f;
        float  value_margin = 2.0f;
        float  other_min = 1.2f;
        float  value_min_px = 12.f;
        float  follow_min = 0.6f;
        float  follow_weight = 0.5f;
        bool   speed_needs_value = false;
    };

    SignTracker() = default;
    explicit SignTracker(const Params& p) : m_p(p) {}

    void requireSpeedValue(bool on) { m_p.speed_needs_value = on; }

    void reset() { m_tracks.clear(); m_active = -1; m_activeT = -1e9; m_speed = -1; m_speedValue = 0; m_speedValueT = -1e9; m_aux = -1; m_auxT = -1e9; }

    void update(const std::vector<SignDetection>& dets, double t) {
        std::vector<bool> used(m_tracks.size(), false);
        for (const auto& d : dets) {
            if (signInfo(d.cls).type == SIGN_OTHER) continue;
            int best = -1; float bestCost = 1e9f;
            const cv::Point2f c(d.box.x + d.box.width / 2, d.box.y + d.box.height / 2);
            for (size_t i = 0; i < m_tracks.size(); ++i) {
                const auto& tr = m_tracks[i];
                if (used[i] || tr.cls != d.cls) continue;
                const cv::Point2f q(tr.box.x + tr.box.width / 2, tr.box.y + tr.box.height / 2);
                const float slow = float(std::max(1.0, (t - tr.lastT) / 0.7));
                const float gate = slow * (25.f + 2.5f * std::max({tr.box.width, tr.box.height, d.box.width, d.box.height}));
                const float dist = float(cv::norm(c - q));
                if (dist < gate && dist < bestCost) { bestCost = dist; best = int(i); }
            }
            if (best < 0) {
                Track tr; tr.id = ++m_nextId; tr.cls = d.cls; tr.conf = d.conf; tr.box = d.box;
                tr.firstT = tr.lastT = tr.followT = t; tr.hits = 1; tr.patch = d.patch;
                m_tracks.push_back(tr); used.push_back(true);
                best = int(m_tracks.size()) - 1;
            } else {
                auto& tr = m_tracks[best];
                used[best] = true;
                if (t - tr.lastT > m_p.confirm_window_s) { tr.hits = 0; tr.firstT = t; }
                tr.hits++; tr.conf = std::max(tr.conf * 0.8f, d.conf);
                tr.lastT = std::max(tr.lastT, t);
                if (t >= tr.followT) { tr.box = d.box; tr.patch = d.patch; tr.followT = t; }
            }
            auto& tr = m_tracks[best];
            if (d.hasValue) vote(tr, d, 1.f);
            if (tr.value < 0) continue;
            confirm(tr, d.conf);
            if (tr.confirmed) publish(tr, t);
        }
        m_tracks.erase(std::remove_if(m_tracks.begin(), m_tracks.end(),
                                      [&](const Track& tr) { return t - tr.lastT > m_p.drop_s; }), m_tracks.end());
    }

    void follow(const cv::Mat& bgr, double t, SpeedValueReader& reader) {
        const cv::Rect frame(0, 0, bgr.cols, bgr.rows);
        for (auto& tr : m_tracks) {
            if (signInfo(tr.cls).type != SIGN_SPEED || tr.value != 0 || tr.patch.empty()) continue;
            if (t <= tr.followT || t - tr.followT > 0.5) continue;
            const cv::Rect b = cv::Rect(tr.box) & frame;
            if (b.width < 8 || b.height < 8) { tr.patch.release(); continue; }
            const int m = std::max(8, int(0.6f * std::max(b.width, b.height)));
            const cv::Rect area = cv::Rect(b.x - m, b.y - m, b.width + 2 * m, b.height + 2 * m) & frame;
            cv::Mat grey;
            cv::cvtColor(bgr(area), grey, cv::COLOR_BGR2GRAY);
            double bestScore = -1; cv::Rect bestBox, bestIn;
            for (float sc : {1.f, 1.06f, 1.12f}) {
                const int g = int(std::lround(std::max(b.width, b.height) * (sc - 1.f)));
                if (sc > 1.f && g == 0) continue;
                const cv::Size sz(b.width + g, b.height + g);
                if (sz.width > area.width || sz.height > area.height) continue;
                cv::Mat tpl, res;
                if (sz == tr.patch.size()) tpl = tr.patch; else cv::resize(tr.patch, tpl, sz, 0, 0, cv::INTER_LINEAR);
                cv::matchTemplate(grey, tpl, res, cv::TM_CCOEFF_NORMED);
                double mx; cv::Point at;
                cv::minMaxLoc(res, nullptr, &mx, nullptr, &at);
                if (mx > bestScore) { bestScore = mx; bestIn = cv::Rect(at, sz); bestBox = bestIn + area.tl(); }
            }
            if (bestScore >= m_p.follow_min) {
                cv::Scalar mean, sd;
                cv::meanStdDev(grey(bestIn), mean, sd);
                if (sd[0] < 15.0) bestScore = 0;
            }
            if (bestScore < m_p.follow_min) { tr.patch.release(); continue; }
            bestBox = snapToRing(bgr, bestBox);
            tr.box = bestBox; tr.followT = t; tr.lastT = std::max(tr.lastT, t);
            SignDetection d; d.box = bestBox; d.cls = tr.cls;
            if (reader.read(bgr, d.box, d.valueP)) vote(tr, d, m_p.follow_weight);
            if (tr.value > 0) confirm(tr, tr.conf);
            if (tr.confirmed && tr.value > 0) publish(tr, t);
        }
    }

    bool speedPending(double t, double fresh_s = 1.0) const {
        for (const auto& tr : m_tracks)
            if (signInfo(tr.cls).type == SIGN_SPEED && tr.value == 0 && t - tr.lastT <= fresh_s) return true;
        return false;
    }

    std::vector<Track> visible(double t, double fresh_s = 1.6) const {
        std::vector<Track> v;
        for (const auto& tr : m_tracks) if (tr.confirmed && t - tr.lastT <= fresh_s) v.push_back(tr);
        return v;
    }
    int active(double t) const { return t - m_activeT <= m_p.hold_s ? m_active : m_speed; }
    float activeConf() const { return m_activeConf; }
    int speedSign() const { return m_speed; }
    int speedValue() const { return m_speed >= 0 ? m_speedValue : 0; }
    int auxSign(double  ) const { return m_aux; }

private:
    Params m_p;
    std::vector<Track> m_tracks;
    int m_nextId = 0, m_active = -1, m_speed = -1, m_speedValue = 0, m_aux = -1;
    float m_activeConf = 0.f;

    static cv::Rect snapToRing(const cv::Mat& bgr, const cv::Rect& box) {
        const int m = std::max(3, box.width / 3);
        const cv::Rect roi = cv::Rect(box.x - m, box.y - m, box.width + 2 * m, box.height + 2 * m) & cv::Rect(0, 0, bgr.cols, bgr.rows);
        int x0 = INT_MAX, y0 = INT_MAX, x1 = -1, y1 = -1, n = 0;
        for (int y = 0; y < roi.height; ++y) {
            const cv::Vec3b* row = bgr.ptr<cv::Vec3b>(roi.y + y) + roi.x;
            for (int x = 0; x < roi.width; ++x) {
                const int b = row[x][0], g = row[x][1], r = row[x][2];
                if (r - std::max(g, b) < 30 || r * 4 < std::max(g, b) * 5) continue;
                x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y); ++n;
            }
        }
        if (n < box.width) return box;
        const cv::Rect ring = cv::Rect(roi.x + x0, roi.y + y0, x1 - x0 + 1, y1 - y0 + 1);
        const float k = float(ring.width) / box.width, ar = float(ring.width) / ring.height;
        const cv::Point2f dc = (ring.tl() + ring.br()) * 0.5f - (box.tl() + box.br()) * 0.5f;
        if (k < 0.75f || k > 1.4f || ar < 0.7f || ar > 1.4f || cv::norm(dc) > 0.35f * box.width) return box;
        return ring & cv::Rect(0, 0, bgr.cols, bgr.rows);
    }

    void confirm(Track& tr, float detConf) {
        if (tr.confirmed) return;
        if (m_p.speed_needs_value && signInfo(tr.cls).type == SIGN_SPEED && tr.value <= 0) return;
        const bool end = signInfo(tr.cls).type == SIGN_END;
        const bool sure = detConf >= m_p.confirm_conf && !end;
        const bool enough = tr.hits >= m_p.confirm_hits && (!end || tr.conf >= m_p.end_conf);
        if (enough || sure) { tr.confirmed = true; m_active = tr.cls; m_activeConf = tr.conf; }
    }

    void publish(Track& tr, double t) {
        m_activeT = t; m_active = tr.cls; m_activeConf = tr.conf;
        if (signInfo(tr.cls).type == SIGN_SPEED) {
            m_speed = tr.cls;
            if (tr.value > 0) { m_speedValue = tr.value; m_speedValueT = t; }
            else if (t - m_speedValueT > m_p.drop_s) m_speedValue = 0;
        }
        else {
            m_aux = tr.cls; m_auxT = t;
            if (signInfo(tr.cls).type == SIGN_END) { m_speed = -1; m_speedValue = 0; m_speedValueT = -1e9; }
        }
    }

    void vote(Track& tr, const SignDetection& d, float scale) {
        if (std::min(d.box.width, d.box.height) < m_p.value_min_px) return;
        const float w = scale * std::clamp(std::min(d.box.width, d.box.height) / 24.f, 0.25f, 1.5f);
        for (int i = 0; i <= kSpeedValues; ++i) tr.votes[i] += w * d.valueP[i];
        std::array<float, kSpeedValues + 1> v = tr.votes;
        for (const auto& o : m_tracks)
            if (&o != &tr && signInfo(o.cls).type == SIGN_SPEED && o.value >= 0 && std::abs(o.lastT - tr.lastT) <= 1.0)
                for (int i = 0; i < kSpeedValues; ++i) v[i] += o.votes[i];
        int bi = 0;
        for (int i = 1; i < kSpeedValues; ++i) if (v[bi] < v[i]) bi = i;
        float second = 0.f;
        for (int i = 0; i < kSpeedValues; ++i) if (i != bi) second = std::max(second, v[i]);
        const float best = v[bi], other = v[kSpeedOther];
        if (best >= m_p.value_min && best >= m_p.value_margin * second && best > other) tr.value = (bi + 1) * 5;
        else if (other >= m_p.other_min && other > m_p.value_margin * best) tr.value = -1;
    }
    double m_activeT = -1e9, m_auxT = -1e9, m_speedValueT = -1e9;
};

class TrafficSignRunner {
public:
    struct Result {
        std::vector<SignDetection> dets;
        double t = 0;
        double ms = 0;
        uint64_t seq = 0;
    };

    ~TrafficSignRunner() { stop(); }

    bool start(const TrafficSignEngine::Config& cfg, const std::vector<int>& cores, int niceInc,
               const std::string& valuePath = "") {
        stop();
        m_valuePath = valuePath;
        std::promise<bool> loaded;
        auto ok = loaded.get_future();
        m_run = true;
        m_thread = std::thread([this, cfg, cores, niceInc, &loaded] { loop(cfg, cores, niceInc, loaded); });
        if (ok.get()) return true;
        stop();
        return false;
    }
    void stop() {
        { std::lock_guard<std::mutex> lk(m_mtx); m_run = false; }
        m_cv.notify_all();
        if (m_thread.joinable()) m_thread.join();
        m_engine.reset();
        m_reader.reset();
    }
    bool ready() const { return m_engine != nullptr; }
    bool readsValues() const { return m_reader && m_reader->ready(); }
    bool busy() const { return m_busy.load(); }
    int inputW() const { return m_engine ? m_engine->inputW() : 640; }
    int inputH() const { return m_engine ? m_engine->inputH() : 224; }

    void submit(cv::Mat band, int x0, int y0, float scale, double t) {
        if (!m_engine || m_busy.load()) return;
        {
            std::lock_guard<std::mutex> lk(m_mtx);
            m_band = std::move(band); m_x0 = x0; m_y0 = y0; m_scale = scale; m_t = t; m_has = true;
        }
        m_busy = true;
        m_cv.notify_one();
    }
    bool poll(Result& out, uint64_t seq) {
        std::lock_guard<std::mutex> lk(m_mtx);
        if (m_res.seq <= seq) return false;
        out = m_res;
        return true;
    }

private:
    std::unique_ptr<TrafficSignEngine> m_engine;
    std::unique_ptr<SpeedValueReader> m_reader;
    std::string m_valuePath;
    std::thread m_thread;
    std::mutex m_mtx;
    std::condition_variable m_cv;
    bool m_run = false, m_has = false;
    std::atomic<bool> m_busy{false};
    cv::Mat m_band;
    int m_x0 = 0, m_y0 = 0;
    float m_scale = 1.f;
    double m_t = 0;
    Result m_res;

    static float glare(const cv::Mat& band, const cv::Rect2f& box) {
        const cv::Rect r = cv::Rect(int(box.x + 0.2f * box.width), int(box.y + 0.2f * box.height),
                                    int(0.6f * box.width), int(0.6f * box.height)) & cv::Rect(0, 0, band.cols, band.rows);
        if (r.area() <= 0) return 0.f;
        cv::Mat g;
        cv::cvtColor(band(r), g, cv::COLOR_BGR2GRAY);
        return float(cv::countNonZero(g >= 245)) / r.area();
    }

    void loop(TrafficSignEngine::Config cfg, std::vector<int> cores, int niceInc, std::promise<bool>& loaded) {
        if (!cores.empty()) {
            cpu_set_t set; CPU_ZERO(&set);
            for (int c : cores) CPU_SET(c, &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        }
        if (niceInc > 0) (void)!nice(niceInc);
        pthread_setname_np(pthread_self(), "adas-tsr");
        auto engine = std::make_unique<TrafficSignEngine>(cfg);
        if (!engine->ready()) { loaded.set_value(false); return; }
        m_engine = std::move(engine);
        m_reader = std::make_unique<SpeedValueReader>(m_valuePath);
        loaded.set_value(true);
        uint64_t seq = 0;
        for (;;) {
            cv::Mat band; int x0, y0; float sc; double t;
            {
                std::unique_lock<std::mutex> lk(m_mtx);
                m_cv.wait(lk, [&] { return !m_run || m_has; });
                if (!m_run) return;
                band = std::move(m_band); x0 = m_x0; y0 = m_y0; sc = m_scale; t = m_t; m_has = false;
            }
            Result r; r.t = t; r.seq = ++seq;
            try {
                r.dets = m_engine->infer(band, &r.ms);
                r.dets.erase(std::remove_if(r.dets.begin(), r.dets.end(), [&](const SignDetection& d) {
                    return signInfo(d.cls).type == SIGN_END && glare(band, d.box) >= 0.3f; }), r.dets.end());
                for (auto& d : r.dets)
                    if (signInfo(d.cls).type == SIGN_SPEED) {
                        d.hasValue = m_reader->read(band, d.box, d.valueP);
                        const cv::Rect pr = cv::Rect(d.box) & cv::Rect(0, 0, band.cols, band.rows);
                        if (sc == 1.f && pr.width > 0 && pr.height > 0) {
                            cv::cvtColor(band(pr), d.patch, cv::COLOR_BGR2GRAY);
                            d.box = pr;
                        }
                    }
                for (auto& d : r.dets) {
                    d.box.x = x0 + d.box.x * sc; d.box.y = y0 + d.box.y * sc;
                    d.box.width *= sc; d.box.height *= sc;
                }
                { std::lock_guard<std::mutex> lk(m_mtx); m_res = std::move(r); }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[TSR] infer error: %s\n", e.what());
            } catch (...) {
                std::fprintf(stderr, "[TSR] infer error\n");
            }
            m_busy = false;
        }
    }
};
