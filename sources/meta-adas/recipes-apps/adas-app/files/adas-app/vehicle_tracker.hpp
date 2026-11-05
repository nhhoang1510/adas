#pragma once

#include "road_geometry.hpp"
#include "lane_detector.hpp"
#include "adas_types.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

struct TrackedVehicle {
    int    id = 0;
    int    cls = VC_CAR;
    float  conf = 0.f;
    cv::Rect2f box;
    double z = -1.0;
    double vz = 0.0;
    double x = 0.0;
    double vx = 0.0;
    double x_lane = 0.0;
    double width_m = 1.8;
    bool   confirmed = false;
    bool   in_path = false;
    bool   cut_in = false;
    std::string lane_pos = "IN_LANE";
    std::vector<cv::Point> trail;
};

struct LeadOutput {
    bool   has_lead = false;
    TrackedVehicle lead;
    int    fcw_level = 0;
    std::string fcw_text = "AN TOAN";
};

class VehicleTracker {
public:
    struct Params {
        double assoc_min_score = 0.25;
        int    confirm_hits = 2;
        int    max_miss_tentative = 1;
        int    max_miss_confirmed = 3;
        double max_coast_s = 1.5;
        double max_range_m = 90.0;
        double lane_fit_z_m = 40.0;
        double in_path_margin_m = 0.15;
        double no_lane_half_m = 1.2;
        double path_enter = 0.7, path_exit = 0.3;
        double vrel_window_s = 1.5;
        double min_closing = 0.5;
        double cutin_lat_vel = 0.7, cutin_gap_m = 0.3;
        double cutin_hold_s = 0.4;
        double vx_tau_s = 0.5;
        double lead_switch_margin_m = 3.0, lead_switch_hold_s = 0.3, lead_loss_hold_s = 0.4;
        double dist_warn_m = 18.0, dist_brake_m = 10.0;
        double headway_warn_s = 0.8;
        double close_warn_m = 6.0, close_brake_m = 3.5;
        double fcw_down_s = 0.5;
        double closing_min_age_s = 0.8;
        double side_settle_s = 1.0;
    };

    VehicleTracker() = default;
    explicit VehicleTracker(const Params& p) : m_p(p) {}

    void reset() { m_tracks.clear(); m_leadId = -1; m_lastT = -1; m_fcwLvl = 0; }
    void setPathCurvature(double k) { m_pathK = k; }

    const std::vector<TrackedVehicle>& step(double t, const std::vector<VehicleDetection>* dets,
                                            const RoadGeometry& g, const LaneOutput& lanes,
                                            double ego_speed_mps = -1.0) {
        const double dt = (m_lastT < 0) ? 0.0 : std::clamp(t - m_lastT, 0.0, 0.5);
        m_lastT = t;
        for (auto& tr : m_tracks) tr.predict(dt);
        m_frameW = g.W;
        if (dets) associate(*dets, t, g);
        prune(t, g);
        updateGeometry(dt, g, lanes);
        selectLead(t);
        computeFcw(t, ego_speed_mps);

        m_view.clear();
        for (const auto& tr : m_tracks) if (tr.view.confirmed) m_view.push_back(tr.view);
        return m_view;
    }

    const LeadOutput& lead() const { return m_lead; }

private:
    static constexpr int LANE_MEM = 10;
    struct Track {
        TrackedVehicle view;
        cv::Matx<double, 8, 1> s;
        cv::Matx<double, 8, 8> P;
        static constexpr int HIST = 48;
        struct RangeSample { double t = 0, z = 0, w = 0, h = 0; };
        RangeSample hist[HIST];
        int histN = 0;
        cv::Matx21d r = cv::Matx21d::zeros();
        cv::Matx22d Pr = cv::Matx22d::eye();
        bool   rangeInit = false;
        int    rangeRejects = 0;
        int    hits = 0;
        int    misses = 0;
        double born = 0.0, lastSeen = 0.0;
        double pathScore = 0.0;
        unsigned char laneMem[10] = {0};
        int    laneMemN = 0;
        bool   ahead = true;
        bool   closing = false;
        double cutSince = -1.0;
        bool   edgeCut = false;
        bool   sideBorn = false;
        double edgeFreeT = -1.0;
        int    clsVotes[4] = {0, 0, 0, 0};
        bool settled(double t, double hold) const { return !sideBorn || (!edgeCut && edgeFreeT >= 0 && t - edgeFreeT >= hold); }

        void initBox(const BoundingBox& b, double t) {
            const double w = b.x2 - b.x1, h = b.y2 - b.y1;
            s = cv::Matx<double, 8, 1>(0.5 * (b.x1 + b.x2), 0.5 * (b.y1 + b.y2), w, h, 0, 0, 0, 0);
            P = cv::Matx<double, 8, 8>::eye() * 25.0;
            P(4, 4) = P(5, 5) = 400.0; P(6, 6) = P(7, 7) = 100.0;
            born = lastSeen = t; hits = 1;
            vote(b.cls);
            refreshBox();
        }
        void predict(double dt) {
            if (dt <= 0) return;
            cv::Matx<double, 8, 8> F = cv::Matx<double, 8, 8>::eye();
            F(0, 4) = F(1, 5) = F(2, 6) = F(3, 7) = dt;
            s = F * s;
            const double q[4] = {150.0 * 150.0, 80.0 * 80.0, 60.0 * 60.0, 60.0 * 60.0};
            cv::Matx<double, 8, 8> Q = cv::Matx<double, 8, 8>::zeros();
            for (int i = 0; i < 4; ++i) {
                Q(i, i) = q[i] * dt * dt * dt * dt / 4.0;
                Q(i, i + 4) = Q(i + 4, i) = q[i] * dt * dt * dt / 2.0;
                Q(i + 4, i + 4) = q[i] * dt * dt;
            }
            P = F * P * F.t() + Q;
            s(6) *= std::exp(-dt / 0.3);
            s(7) *= std::exp(-dt / 0.3);
            s(2) = std::max(4.0, s(2));
            s(3) = std::max(4.0, s(3));
            if (rangeInit) {
                cv::Matx22d Fr(1, dt, 0, 1);
                r = Fr * r;
                const double qa = 9.0;
                cv::Matx22d Qr(qa * dt * dt * dt * dt / 4, qa * dt * dt * dt / 2, qa * dt * dt * dt / 2, qa * dt * dt);
                Pr = Fr * Pr * Fr.t() + Qr;
            }
            refreshBox();
        }
        void updateBox(const BoundingBox& b, double t) {
            const double w = b.x2 - b.x1, h = b.y2 - b.y1;
            cv::Matx41d z(0.5 * (b.x1 + b.x2), 0.5 * (b.y1 + b.y2), w, h);
            cv::Matx<double, 4, 8> Hm = cv::Matx<double, 4, 8>::zeros();
            Hm(0, 0) = Hm(1, 1) = Hm(2, 2) = Hm(3, 3) = 1;
            const double sp = 2.0 + 0.04 * w;
            cv::Matx44d R = cv::Matx44d::zeros();
            R(0, 0) = sp * sp; R(1, 1) = sp * sp; R(2, 2) = (1.5 * sp) * (1.5 * sp); R(3, 3) = (1.5 * sp) * (1.5 * sp);
            const cv::Matx41d y = z - Hm * s;
            const cv::Matx44d S = Hm * P * Hm.t() + R;
            const cv::Matx<double, 8, 4> K = P * Hm.t() * S.inv();
            s = s + K * y;
            P = (cv::Matx<double, 8, 8>::eye() - K * Hm) * P;
            lastSeen = t; ++hits; misses = 0;
            view.conf = b.conf;
            vote(b.cls);
            refreshBox();
        }
        void pushRange(double t, double z) {
            hist[histN % HIST] = {t, z, double(view.box.width), double(view.box.height)};
            ++histN;
        }
        bool regressVrel(double t, double window_s, double& vrel, double& zfit, double* vse = nullptr) const {
            const int n = std::min(histN, HIST);
            double St = 0, Sz = 0, Stt = 0, Stz = 0; int m = 0;
            for (int i = 0; i < n; ++i) {
                const auto& h = hist[i];
                const double dt = h.t - t;
                if (dt < -window_s) continue;
                St += dt; Sz += h.z; Stt += dt * dt; Stz += dt * h.z; ++m;
            }
            if (m < 4) return false;
            const double den = m * Stt - St * St;
            if (den < 1e-6) return false;
            vrel = (m * Stz - St * Sz) / den;
            zfit = (Sz - vrel * St) / m;
            if (vse) {
                double ssr = 0;
                for (int i = 0; i < n; ++i) {
                    const double dt = hist[i].t - t;
                    if (dt < -window_s) continue;
                    const double e = hist[i].z - (zfit + vrel * dt);
                    ssr += e * e;
                }
                *vse = m > 2 ? std::sqrt(ssr / (m - 2) / (Stt - St * St / m)) : 1e9;
            }
            return true;
        }
        void updateRange(double zm) {
            const double sr = 0.5 + 0.08 * zm;
            const double R = sr * sr;
            if (!rangeInit) { r = cv::Matx21d(zm, 0.0); Pr = cv::Matx22d(R, 0, 0, 16.0); rangeInit = true; return; }
            const double y = zm - r(0);
            const double S = Pr(0, 0) + R;
            if (y * y > 16.0 * S) {
                if (++rangeRejects >= 3) { rangeInit = false; rangeRejects = 0; updateRange(zm); }
                return;
            }
            rangeRejects = 0;
            const cv::Matx21d K(Pr(0, 0) / S, Pr(1, 0) / S);
            r = r + K * y;
            Pr = (cv::Matx22d::eye() - cv::Matx22d(K(0), 0, K(1), 0)) * Pr;
        }
        bool growth(double t, double lag, double& gw, double& gz) const {
            const int n = std::min(histN, HIST);
            if (n < 2) return false;
            const RangeSample& now = hist[(histN - 1) % HIST];
            const RangeSample* then = nullptr;
            for (int i = 0; i < n; ++i) {
                const auto& h = hist[i];
                if (t - h.t >= lag && (!then || h.t > then->t)) then = &h;
            }
            if (!then || then->w < 1 || then->h < 1 || now.z < 0.5) return false;
            gw = std::min(now.w / then->w, now.h / then->h); gz = then->z / now.z;
            return true;
        }
        void vote(int c) { if (c >= 0 && c < 4) ++clsVotes[c]; }
        int cls() const { return int(std::max_element(clsVotes, clsVotes + 4) - clsVotes); }
        void refreshBox() {
            const double w = s(2), h = s(3);
            view.box = cv::Rect2f(float(s(0) - w / 2), float(s(1) - h / 2), float(w), float(h));
        }
    };

    Params m_p;
    double m_pathK = 0.0;
    std::vector<Track> m_tracks;
    std::vector<TrackedVehicle> m_view;
    LeadOutput m_lead;
    int    m_nextId = 1;
    int    m_leadId = -1, m_candId = -1;
    double m_candT = 0.0, m_leadLostT = -1.0;
    double m_lastT = -1.0;
    int    m_fcwLvl = 0;
    int    m_frameW = 640;
    double m_fcwDownT = -1.0;

    static float iou(const cv::Rect2f& a, const BoundingBox& b) {
        const float ix = std::max(0.f, std::min(a.x + a.width, b.x2) - std::max(a.x, b.x1));
        const float iy = std::max(0.f, std::min(a.y + a.height, b.y2) - std::max(a.y, b.y1));
        const float in = ix * iy;
        const float u = a.area() + (b.x2 - b.x1) * (b.y2 - b.y1) - in;
        return u > 0 ? in / u : 0.f;
    }
    static bool sideCut(const cv::Rect2f& b, const RoadGeometry& g) {
        if (b.x < 8 || b.x + b.width > g.W - 8) return true;
        return b.y + b.height > std::min(g.hood, g.H) - 4 && std::abs(b.x + b.width / 2 - g.W / 2) > 0.2 * g.W;
    }
    static bool compatible(int a, int b) {
        if (a == b) return true;
        const bool big = (a == VC_CAR || a == VC_BUS_TRUCK) && (b == VC_CAR || b == VC_BUS_TRUCK);
        const bool two = (a == VC_MOTORBIKE || a == VC_BICYCLE) && (b == VC_MOTORBIKE || b == VC_BICYCLE);
        return big || two;
    }

    void associate(const std::vector<VehicleDetection>& dets, double t, const RoadGeometry& g) {
        struct Pair { float iou; int ti, di; };
        std::vector<Pair> pairs;
        for (int i = 0; i < int(m_tracks.size()); ++i)
            for (int j = 0; j < int(dets.size()); ++j) {
                if (!compatible(m_tracks[i].cls(), dets[j].box.cls)) continue;
                const auto& tb = m_tracks[i].view.box;
                const auto& db = dets[j].box;
                const float o = iou(tb, db);
                const float dx = (tb.x + tb.width / 2) - 0.5f * (db.x1 + db.x2);
                const float dy = (tb.y + tb.height / 2) - 0.5f * (db.y1 + db.y2);
                const float scale = 0.7f * std::max(tb.width, db.x2 - db.x1);
                const float prox = std::max(0.f, 1.f - std::sqrt(dx * dx + dy * dy) / std::max(4.f, scale));
                const float score = o + 0.5f * prox;
                if (score >= m_p.assoc_min_score) pairs.push_back({score, i, j});
            }
        std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.iou > b.iou; });
        std::vector<char> tUsed(m_tracks.size(), 0), dUsed(dets.size(), 0);
        for (const auto& p : pairs) {
            if (tUsed[p.ti] || dUsed[p.di]) continue;
            tUsed[p.ti] = dUsed[p.di] = 1;
            Track& tr = m_tracks[p.ti];
            tr.updateBox(dets[p.di].box, t);
            const bool cut = sideCut(tr.view.box, g);
            if (tr.edgeCut && !cut) {
                tr.rangeInit = false; tr.histN = 0; tr.rangeRejects = 0; tr.edgeFreeT = t;
            }
            tr.edgeCut = cut;
            const double zm = g.estimateRange(tr.view.box, tr.cls());
            tr.updateRange(zm);
            tr.pushRange(t, tr.rangeInit ? tr.r(0) : zm);
        }
        for (int j = 0; j < int(dets.size()); ++j) {
            if (dUsed[j]) continue;
            Track tr;
            tr.view.id = m_nextId++;
            tr.initBox(dets[j].box, t);
            tr.view.conf = dets[j].box.conf;
            const auto& nb = tr.view.box;
            tr.edgeCut = sideCut(nb, g);
            tr.sideBorn = tr.edgeCut || (nb.y + nb.height > g.hood - 0.08 * g.H && std::abs(nb.x + nb.width / 2 - g.W / 2) > 0.25 * g.W);
            if (!tr.edgeCut) tr.edgeFreeT = t;
            tr.updateRange(g.estimateRange(tr.view.box, tr.cls()));
            tr.pushRange(t, tr.r(0));
            m_tracks.push_back(tr);
        }
        for (int i = 0; i < int(m_tracks.size()); ++i) {
            Track& tr = m_tracks[i];
            if (i < int(tUsed.size()) && !tUsed[i]) ++tr.misses;
            if (!tr.view.confirmed && tr.hits >= m_p.confirm_hits) tr.view.confirmed = true;
        }
    }

    void prune(double t, const RoadGeometry& g) {
        std::vector<Track> keep;
        keep.reserve(m_tracks.size());
        for (auto& tr : m_tracks) {
            const auto& b = tr.view.box;
            const bool outside = b.x + b.width < 0 || b.x > g.W || b.y > g.H || b.y + b.height < 0;
            const bool stale = t - tr.lastSeen > m_p.max_coast_s;
            const bool missed = tr.view.confirmed ? tr.misses > m_p.max_miss_confirmed
                                                  : tr.misses > m_p.max_miss_tentative;
            if (outside || stale || missed) continue;
            keep.push_back(std::move(tr));
        }
        m_tracks.swap(keep);
        std::vector<char> dead(m_tracks.size(), 0);
        for (size_t i = 0; i < m_tracks.size(); ++i)
            for (size_t j = i + 1; j < m_tracks.size(); ++j) {
                if (dead[i] || dead[j]) continue;
                const auto& a = m_tracks[i].view.box; const auto& b = m_tracks[j].view.box;
                const float in = (a & b).area(), u = a.area() + b.area() - in;
                if (u <= 0 || in / u < 0.5f) continue;
                const bool iOlder = m_tracks[i].view.id < m_tracks[j].view.id;
                const bool iWins = m_tracks[i].view.confirmed != m_tracks[j].view.confirmed ? m_tracks[i].view.confirmed : iOlder;
                dead[iWins ? j : i] = 1;
            }
        if (std::find(dead.begin(), dead.end(), 1) != dead.end()) {
            std::vector<Track> k2;
            for (size_t i = 0; i < m_tracks.size(); ++i) if (!dead[i]) k2.push_back(std::move(m_tracks[i]));
            m_tracks.swap(k2);
        }
    }

    void corridor(const LaneOutput& lanes, const RoadGeometry& g, double z, double& lo, double& hi) const {
        if (lanes.has_lane && lanes.left_valid && lanes.right_valid) {
            const double zf = std::min(z, m_p.lane_fit_z_m), dz = z - zf;
            auto at = [&](const LaneModel& m) { return m.x(zf) + (m.c1 + 2.0 * m.c2 * zf) * dz; };
            lo = at(lanes.left_trk); hi = at(lanes.right_trk);
            const double mid = 0.5 * (lo + hi);
            if (lanes.left_synth)  lo = std::max(lo, mid - m_p.no_lane_half_m);
            if (lanes.right_synth) hi = std::min(hi, mid + m_p.no_lane_half_m);
        } else {
            double xc = 0.0;
            if (std::abs(m_pathK) > 1e-5) {
                const double kz = std::min(1.0, std::abs(m_pathK * z));
                xc = (1.0 - std::sqrt(1.0 - kz * kz)) / m_pathK;
            }
            lo = g.car_x + xc - m_p.no_lane_half_m;
            hi = g.car_x + xc + m_p.no_lane_half_m;
        }
    }

    void updateGeometry(double dt, const RoadGeometry& g, const LaneOutput& lanes) {
        for (auto& tr : m_tracks) {
            auto& v = tr.view;
            v.cls = tr.cls();
            v.z = tr.rangeInit ? std::max(1.5, tr.r(0)) : g.estimateRange(v.box, v.cls);
            double vReg = 0.0, zReg = 0.0, vSe = 1e9;
            if (tr.regressVrel(m_lastT, m_p.vrel_window_s, vReg, zReg, &vSe) && tr.histN >= 6) v.vz = std::clamp(vReg, -60.0, 60.0);
            else v.vz = tr.rangeInit ? tr.r(1) : 0.0;
            tr.closing = tr.histN >= 8 && m_lastT - tr.born >= m_p.closing_min_age_s && v.vz < -m_p.min_closing &&
                         v.vz + 2.0 * vSe < 0.0;
            double gw = 1, gz = 1;
            if (tr.closing && tr.growth(m_lastT, 0.6, gw, gz))
                tr.closing = gw >= 1.03 && gw - 1.0 >= 0.4 * (gz - 1.0);
            const bool settled = tr.settled(m_lastT, m_p.side_settle_s);
            const double prior_w = vehicleClassWidthM(v.cls);
            const double rearPx = prior_w * g.f / std::max(1.0, v.z);
            const bool cutL = v.box.x < 2, cutR = v.box.x + v.box.width > g.W - 2;
            const double x1 = v.box.x, x2 = v.box.x + v.box.width, bc = 0.5 * (x1 + x2);
            double cxPix = bc;
            if (cutR && !cutL)      cxPix = x1 + rearPx / 2;
            else if (cutL && !cutR) cxPix = x2 - rearPx / 2;
            else if (v.box.width > 1.15 * rearPx) cxPix = (bc > g.cx) ? x2 - rearPx / 2 : x1 + rearPx / 2;
            const double xNew = (cxPix - g.cx) * v.z / g.f;
            v.width_m = std::min<double>(v.box.width, rearPx) * v.z / g.f;
            v.x = xNew;

            double lo, hi; corridor(lanes, g, v.z, lo, hi);
            const bool haveLanes = lanes.has_lane && lanes.left_valid && lanes.right_valid;
            const double m = haveLanes ? m_p.in_path_margin_m : 0.0;
            const bool raw = v.x > lo + m && v.x < hi - m;
            tr.laneMem[tr.laneMemN % LANE_MEM] = raw ? 1 : 0;
            ++tr.laneMemN;
            int inCnt = 0; const int nm = std::min(tr.laneMemN, LANE_MEM);
            for (int k = 0; k < nm; ++k) inCnt += tr.laneMem[k];
            tr.pathScore = nm ? double(inCnt) / nm : 0.0;
            if (!v.in_path && (tr.pathScore >= m_p.path_enter || (nm < 3 && raw))) v.in_path = true;
            if (v.in_path && tr.pathScore <= m_p.path_exit) v.in_path = false;

            const double xl = v.x - 0.5 * (lo + hi);
            if (dt > 0 && tr.hits > 1) v.vx += (1.0 - std::exp(-dt / m_p.vx_tau_s)) * (std::clamp((xl - v.x_lane) / dt, -3.0, 3.0) - v.vx);
            v.x_lane = xl;

            const double vl = v.x - v.width_m / 2, vr = v.x + v.width_m / 2;
            const double gapL = lo - vr, gapR = vl - hi;
            const bool ahead = v.box.y + v.box.height <= g.hood + 4 || std::abs(v.x - g.car_x) < 0.6;
            tr.ahead = ahead;
            const bool cutRaw = ahead && settled && !v.in_path && v.z < 40.0 && tr.hits >= 6 &&
                                ((gapL < m_p.cutin_gap_m && v.x < lo && v.vx > m_p.cutin_lat_vel) ||
                                 (gapR < m_p.cutin_gap_m && v.x > hi && v.vx < -m_p.cutin_lat_vel));
            if (!cutRaw) tr.cutSince = -1.0;
            else if (tr.cutSince < 0) tr.cutSince = m_lastT;
            v.cut_in = cutRaw && m_lastT - tr.cutSince >= m_p.cutin_hold_s;
            v.lane_pos = v.cut_in ? "CUT_IN" : v.in_path ? "IN_LANE" : (v.x < 0.5 * (lo + hi) ? "LEFT_LANE" : "RIGHT_LANE");

            v.trail.emplace_back(cvRound(cxPix), cvRound(v.box.y + v.box.height));
            if (v.trail.size() > 20) v.trail.erase(v.trail.begin());
        }
    }

    const Track* find(int id) const {
        for (const auto& t : m_tracks) if (t.view.id == id) return &t;
        return nullptr;
    }

    void selectLead(double t) {
        const Track* nearest = nullptr;
        for (const auto& tr : m_tracks) {
            const auto& v = tr.view;
            if (!v.confirmed || !(v.in_path || v.cut_in) || v.z > m_p.max_range_m) continue;
            const bool atEdge = v.box.x < 2 || v.box.x + v.box.width > m_frameW - 2;
            if (atEdge && std::abs(v.x) > 0.6) continue;
            if (!tr.ahead) continue;
            if (!tr.settled(t, m_p.side_settle_s)) continue;
            if (!nearest || v.z < nearest->view.z) nearest = &tr;
        }
        const Track* cur = find(m_leadId);
        const bool curOk = cur && cur->view.confirmed && (cur->view.in_path || cur->view.cut_in);

        if (!cur) {
            m_leadId = nearest ? nearest->view.id : -1;
            m_leadLostT = -1; m_candId = -1;
        } else if (!curOk) {
            if (m_leadLostT < 0) m_leadLostT = t;
            if (t - m_leadLostT > m_p.lead_loss_hold_s) { m_leadId = nearest ? nearest->view.id : -1; m_leadLostT = -1; }
        } else {
            m_leadLostT = -1;
            if (nearest && nearest->view.id != m_leadId && nearest->view.z < cur->view.z - m_p.lead_switch_margin_m) {
                if (m_candId != nearest->view.id) { m_candId = nearest->view.id; m_candT = t; }
                if (t - m_candT >= m_p.lead_switch_hold_s) { m_leadId = m_candId; m_candId = -1; }
            } else {
                m_candId = -1;
            }
        }
        const Track* L = find(m_leadId);
        m_lead.has_lead = L != nullptr;
        if (L) m_lead.lead = L->view;
    }

    void computeFcw(double t, double ego_speed) {
        int raw = 0;
        if (m_lead.has_lead) {
            const auto& v = m_lead.lead;
            const Track* LT = find(m_leadId);
            const bool closing = LT && LT->closing;
            const double headway = ego_speed > 1.0 ? v.z / ego_speed : 1e9;
            const bool moving = ego_speed > 2.0 || (ego_speed < 0 && closing);
            if ((moving && v.z <= m_p.dist_brake_m) ||
                (closing && v.z < m_p.close_brake_m)) raw = 2;
            else if ((moving && v.z <= m_p.dist_warn_m) ||
                     headway < m_p.headway_warn_s || (closing && v.z < m_p.close_warn_m)) raw = 1;
        }
        if (raw > m_fcwLvl) { m_fcwLvl = raw; m_fcwDownT = -1; }
        else if (raw < m_fcwLvl) {
            if (m_fcwDownT < 0) m_fcwDownT = t;
            if (t - m_fcwDownT >= m_p.fcw_down_s) { m_fcwLvl = raw; m_fcwDownT = -1; }
        } else m_fcwDownT = -1;

        m_lead.fcw_level = m_fcwLvl;
        const std::string tag = (m_lead.has_lead && m_lead.lead.cut_in) ? " [TAT DAU!]" : "";
        m_lead.fcw_text = m_fcwLvl == 2 ? "NGUY HIEM: PHANH GAP!" + tag
                        : m_fcwLvl == 1 ? "CHU Y: GIU KHOANG CACH" + tag : "AN TOAN";
    }
};
