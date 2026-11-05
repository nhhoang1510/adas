#!/usr/bin/env python3
"""
adas_lab.py — run the Pi ADAS perception pipeline on a laptop, frame by frame.

It is a line-by-line Python port of the C++ code in this folder:
    lane_detector.hpp     (laneMarkingMask + LaneDetector: BEV sliding windows)
    detection_filter.hpp  (hood / horizon / aspect / size-vs-ground filter)
    nanodet_engine.hpp    (NanoDet-Plus decode: BGR, letterbox, x*stride priors)
so what you see here is what the Pi does. Use it to check your models and your
calibration BEFORE flashing the Pi.

    pip install onnxruntime opencv-python numpy

Lanes come only from the classic CV marking mask (laneMarkingMask); NanoDet is
the only neural network (vehicles).

    # 1) what is inside the NanoDet ONNX file?
    python3 adas_lab.py info --nanodet nanodet.onnx

    # 2) full pipeline -> annotated video (+ a sidecar calibration suggestion)
    python3 adas_lab.py run --video test.mp4 --nanodet nanodet.onnx \
                            --out out.mp4 [--hfov 110 --height 1.3 --horizon 0.40 --hood 0.80]

    # 3) single image
    python3 adas_lab.py run --image frame.png --nanodet nanodet.onnx --out frame_out.png
"""
import argparse, json, math, os, sys, time
import numpy as np
import cv2

class Geo:
    def __init__(self, W, H, hfov, cam_h, horizon_y, hood_y, lateral=0.0):
        self.W, self.H = W, H
        self.f = (W * 0.5) / math.tan(math.radians(hfov) / 2)
        self.cx = W * 0.5
        self.vh = float(horizon_y)
        self.cam_h = cam_h
        self.hood = int(hood_y)
        self.car_x = -lateral
    def toImage(self, x, z):
        return (self.cx + self.f * x / z, self.vh + self.f * self.cam_h / z)
    def zAtRow(self, v):
        dv = v - self.vh
        return self.f * self.cam_h / dv if dv > 0.5 else 1e9
    def rowAtZ(self, z): return self.vh + self.f * self.cam_h / z
    def nearZ(self): return self.zAtRow(self.hood - 1)

CLASS_W = {0: 1.80, 1: 0.75, 2: 0.60, 3: 2.45}
CLASS_N = {0: 'car', 1: 'motorbike', 2: 'bicycle', 3: 'bus/truck'}
COCO2VC = {2: 0, 3: 1, 1: 2, 5: 3, 7: 3}

def lane_marking_mask(bgr, g, exclude=(), contrast=14, above_median=10, road_tol=35, max_paint_m=0.45):
    H, W = bgr.shape[:2]
    m = np.zeros((H, W), np.uint8)
    v0 = int(min(max(math.ceil(g.vh + 3), 0), H)); v1 = int(min(max(g.hood, 0), H))
    if v1 - v0 < 4: return m
    gray = cv2.cvtColor(bgr[v0:v1], cv2.COLOR_BGR2GRAY).astype(np.int16)
    ex = np.zeros((H, W), bool)
    for (x, y, w, h) in exclude: ex[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = True
    for v in range(v0, v1):
        r = gray[v - v0]
        z = g.zAtRow(v)
        d = int(min(max(round(1.5 * 0.15 * g.f / z), 2), 40))
        if W <= 2 * d + 1: continue
        med = int(np.median(r[::2]))

        half = max(3, int(0.8 * g.f / z)); ua = max(0, int(g.cx - half)); ub = min(W, int(g.cx + half))
        seg = r[ua:ub][~ex[v, ua:ub]]
        road = int(np.percentile(seg, 40)) if seg.size >= 6 else med
        c = r[d:W - d]; lf = r[:W - 2 * d]; rt = r[2 * d:]
        side = np.maximum(lf, rt)
        on = (c > max(med, road) + above_median) & (c - side > contrast)\
             & (np.abs(lf - road) <= road_tol) & (np.abs(rt - road) <= road_tol)
        row = np.zeros(W, bool); row[d:W - d] = on

        maxrun = max_paint_m * g.f / z + 2
        dd = np.diff(np.r_[0, row.astype(np.int8), 0]); st = np.nonzero(dd == 1)[0]; en = np.nonzero(dd == -1)[0]
        for a_, b_ in zip(st, en):
            if b_ - a_ > maxrun: row[a_:b_] = False
        m[v] = row.astype(np.uint8) * 255
    m[ex] = 0
    return m

class LaneDetector:
    P = dict(x_min=-6.0, x_max=6.0, z_max=40.0, res_x=0.05, res_z=0.10, n_windows=10, search_len_m=30.0,
             win_half_cold_m=0.5, win_half_track_m=0.35, win_half_per_m=0.01, max_shift_m=0.30, min_win_pix=6, max_thickness_m=0.45, min_windows=3, min_points=40,
             min_span_m=6.0, curvature_span_m=15.0, curv_gain=0.80, cold_lo_m=0.3, cold_hi_m=3.2, cold_hist_len_m=12.0,
             min_hist_peak=4.0, fit_rms_max=0.30, heading_max=0.40, curv_max=0.010, lane_width_default=3.5,
             lane_width_min=2.6, lane_width_max=4.4, width_change_max=0.9, jump_m=0.6, jump_confirm=3, step_m=0.25, step_rate=1.0, width_alpha=0.05, width_min_pts=150, width_min_win=4,
             tau_c0=0.12, tau_c1=0.25, tau_c2=0.50, conf_gain=0.40, conf_decay_s=0.7, conf_valid=0.35,
             conf_drop=0.05, max_extrapolate_s=0.0, tau_out=0.15, stale_s=0.6, width_tol_m=0.6, car_half_width_m=0.92, z_far_draw=30.0)

    class Side:
        def __init__(s):
            s.init = False; s.target = np.zeros(3); s.shown = np.zeros(3)
            s.conf = 0.0; s.rate = 0.0; s.lastMeasT = -1.0; s.lastC0 = 0.0; s.jumps = 0

    def __init__(s):
        p = s.P
        s.bw = int(round((p['x_max'] - p['x_min']) / p['res_x'])); s.bh = int(round(p['z_max'] / p['res_z']))
        s.reset()

    def reset(s):
        s.side = [s.Side(), s.Side()]; s.width = s.P['lane_width_default']; s.widthN = 0; s.lastT = -1.0
        s.bev = np.zeros((s.bh, s.bw), np.uint8); s.win = [[], []]; s.pts = [0, 0]; s.out = None; s.outT = -1.0; s.hadOut = False

    @staticmethod
    def X(c, z): return c[0] + c[1] * z + c[2] * z * z
    def col(s, x): return int(math.floor((x - s.P['x_min']) / s.P['res_x']))
    def row(s, z): return int(math.floor((s.P['z_max'] - z) / s.P['res_z']))
    def nearZ(s, g): return min(max(g.nearZ(), 1.5), 15.0)

    def predict(s, t):
        p = s.P
        dt = 0.0 if s.lastT < 0 else min(max(t - s.lastT, 0.0), 0.5)
        s.lastT = t
        if dt <= 0: return
        for i, sd in enumerate(s.side):
            if not sd.init: continue
            if p['max_extrapolate_s'] > 0 and t - sd.lastMeasT < p['max_extrapolate_s']: sd.target[0] += min(max(sd.rate, -1.5), 1.5) * dt
            for k, tau in enumerate((p['tau_c0'], p['tau_c1'], p['tau_c2'])):
                sd.shown[k] += (sd.target[k] - sd.shown[k]) * (1 - math.exp(-dt / tau))
            sd.conf *= math.exp(-dt / p['conf_decay_s'])
            if sd.conf < p['conf_drop']: s.side[i] = s.Side()

    def warp(s, mask, g):
        p = s.P
        zA = max(s.nearZ(g) + 1.0, 6.0); zB = max(zA + 10.0, 25.0)
        while g.rowAtZ(zB) - g.vh < 4.0 and zB > zA + 4.0: zB -= 2.0
        src = np.float32([g.toImage(-2, zA), g.toImage(2, zA), g.toImage(-2, zB), g.toImage(2, zB)])
        dst = np.float32([[(-2 - p['x_min']) / p['res_x'], (p['z_max'] - zA) / p['res_z']],
                          [(2 - p['x_min']) / p['res_x'], (p['z_max'] - zA) / p['res_z']],
                          [(-2 - p['x_min']) / p['res_x'], (p['z_max'] - zB) / p['res_z']],
                          [(2 - p['x_min']) / p['res_x'], (p['z_max'] - zB) / p['res_z']]])
        Hm = cv2.getPerspectiveTransform(src, dst)
        b = cv2.warpPerspective(mask, Hm, (s.bw, s.bh), flags=cv2.INTER_LINEAR)
        _, s.bev = cv2.threshold(b, 64, 255, cv2.THRESH_BINARY)
        rN = min(max(s.row(s.nearZ(g)), 0), s.bh)
        s.bev[rN:] = 0

    def search(s, side, g):
        p = s.P; s.win[side] = []
        zlo = s.nearZ(g); zhi = min(p['z_max'] - 1.0, zlo + p['search_len_m'])
        sgn = -1.0 if side == 0 else 1.0
        me, other = s.side[side], s.side[1 - side]
        prior = None
        if me.init and me.conf > 0.2: prior = me.target.copy()
        elif other.init and other.conf > 0.3: prior = other.target.copy(); prior[0] += sgn * s.width
        xc = 0.0
        if prior is None:
            r1 = min(max(s.row(zlo), 0), s.bh); r0 = min(max(s.row(min(zhi, zlo + p['cold_hist_len_m'])), 0), s.bh)
            lo = g.car_x - p['cold_hi_m'] if side == 0 else g.car_x + p['cold_lo_m']
            hi = g.car_x - p['cold_lo_m'] if side == 0 else g.car_x + p['cold_hi_m']
            c0 = min(max(s.col(lo), 0), s.bw - 1); c1 = min(max(s.col(hi), 0), s.bw - 1)
            if r1 - r0 < 5 or c1 - c0 < 5: return None
            hist = (s.bev[r0:r1, c0:c1 + 1] > 0).sum(0).astype(float)
            sm = np.convolve(hist, np.ones(7) / 7, 'valid')
            if sm.size == 0 or sm.max() < p['min_hist_peak']: return None

            thr = max(p['min_hist_peak'], 0.5 * sm.max())
            order = range(sm.size - 1, -1, -1) if side == 0 else range(sm.size)
            k = next(k for k in order if sm[k] >= thr)
            while 0 <= k + (-1 if side == 0 else 1) < sm.size and sm[k + (-1 if side == 0 else 1)] >= sm[k]:
                k += -1 if side == 0 else 1
            xc = p['x_min'] + (c0 + 3 + k + 0.5) * p['res_x']
        xs, zs = [], []
        dz = (zhi - zlo) / p['n_windows']; shift = 0.0; hit = 0
        for i in range(p['n_windows']):
            za = zlo + i * dz; zb = za + dz; zm = 0.5 * (za + zb)
            if prior is not None:
                centre = s.X(prior, zm) + shift; half = p['win_half_track_m'] + p['win_half_per_m'] * zm
            else:
                centre = xc; half = p['win_half_cold_m']
            ra = min(max(s.row(zb), 0), s.bh); rb = min(max(s.row(za), 0), s.bh)
            ca = min(max(s.col(centre - half), 0), s.bw); cb = min(max(s.col(centre + half) + 1, 0), s.bw)
            if rb <= ra or cb <= ca: continue
            s.win[side].append((ca, ra, cb - ca, rb - ra))
            yy, xx = np.nonzero(s.bev[ra:rb, ca:cb])
            if len(xx) < p['min_win_pix']: continue
            if len(xx) / len(np.unique(yy)) * p['res_x'] > p['max_thickness_m']: continue
            px = p['x_min'] + (xx + ca + 0.5) * p['res_x']; pz = p['z_max'] - (yy + ra + 0.5) * p['res_z']
            xs.append(px); zs.append(pz); hit += 1
            med = float(np.sort(px)[len(px) // 2])
            if prior is not None: shift = min(max(shift + 0.5 * (med - (s.X(prior, zm) + shift)), -p['max_shift_m']), p['max_shift_m'])
            else: xc = med
        if hit < p['min_windows']: return None
        f = s.fit(np.concatenate(xs), np.concatenate(zs))
        if f: f['win'] = hit
        return f

    def fit(s, x, z):
        p = s.P
        if len(x) < p['min_points']: return None
        span = z.max() - z.min()
        if span < p['min_span_m']: return None
        lin = s.fit_deg(x, z, False)
        if lin is None: return None
        best = lin
        if span >= p['curvature_span_m']:
            q = s.fit_deg(x, z, True)
            if q is not None and q['rms'] < p['curv_gain'] * lin['rms']: best = q
        c = best['c']
        ok = best['n'] >= p['min_points'] and best['rms'] < p['fit_rms_max'] and abs(c[1]) < p['heading_max'] and abs(c[2]) < p['curv_max']
        return dict(c=c, rms=best['rms'], n=best['n'], span=span) if ok else None

    def fit_deg(s, x, z, curved):
        p = s.P
        use = np.ones(len(x), bool)
        for ps in range(3):
            if use.sum() < p['min_points']: return None
            w = 1.0 / (1.0 + 0.05 * z[use]); Z = z[use]; Xu = x[use]
            R = np.stack([np.ones_like(Z), Z, Z * Z], 1)
            A = (R * w[:, None]).T @ R; b = (R * w[:, None]).T @ Xu
            A[2, 2] += 50.0 if curved else 1e9; A[1, 1] += 1e-3
            c = np.linalg.solve(A, b)
            if not curved: c[2] = 0.0
            e = Xu - s.X(c, Z); rms = math.sqrt(float(np.mean(e * e))); n = int(use.sum())
            if ps < 2:
                lim = max(0.15, 2.5 * rms); use = np.abs(x - s.X(c, z)) <= lim
        return dict(c=c, rms=rms, n=n)

    def measure(s, sd, f, t):
        p = s.P; c = f['c']
        dt = t - sd.lastMeasT
        step = min(p['jump_m'], p['step_m'] + p['step_rate'] * max(0.0, dt))
        far = sd.init and abs(c[0] - sd.target[0]) > step
        if far and sd.jumps + 1 < p['jump_confirm']: sd.jumps += 1; return
        if not sd.init or far:
            sd.target = c.copy(); sd.shown = c.copy(); sd.init = True; sd.conf = max(sd.conf, 0.4)
            sd.rate = 0.0; sd.jumps = 0; sd.lastMeasT = t; sd.lastC0 = c[0]; return
        sd.jumps = 0
        if sd.lastMeasT > 0 and 0.02 < dt < 0.6:
            sd.rate = 0.7 * sd.rate + 0.3 * min(max((c[0] - sd.lastC0) / dt, -2.0), 2.0)
        sd.lastMeasT = t; sd.lastC0 = c[0]
        sd.target[0] = c[0]; sd.target[1] += 0.8 * (c[1] - sd.target[1]); sd.target[2] += 0.35 * (c[2] - sd.target[2])
        sd.conf = min(1.0, sd.conf + p['conf_gain'])

    def update(s, mask, g):
        p = s.P; t = s.lastT
        s.warp(mask, g)
        fit = [s.search(0, g), s.search(1, g)]
        if fit[0] and fit[0]['c'][0] > g.car_x + 0.15 and not fit[1]:
            fit[1] = fit[0]; fit[0] = None; s.side[1] = s.side[0]; s.side[0] = s.Side()
        elif fit[1] and fit[1]['c'][0] < g.car_x - 0.15 and not fit[0]:
            fit[0] = fit[1]; fit[1] = None; s.side[0] = s.side[1]; s.side[1] = s.Side()
        if fit[0] and fit[1]:
            zn = s.nearZ(g); zf = zn + 15
            wn = s.X(fit[1]['c'], zn) - s.X(fit[0]['c'], zn); wf = s.X(fit[1]['c'], zf) - s.X(fit[0]['c'], zf)
            widthOff = s.widthN >= 5 and abs(wn - s.width) > p['width_tol_m']
            if wn < p['lane_width_min'] or wn > p['lane_width_max'] or abs(wf - wn) > p['width_change_max'] or widthOff:
                eL = abs(fit[0]['c'][0] - s.side[0].target[0]) if s.side[0].init else 9.0
                eR = abs(fit[1]['c'][0] - s.side[1].target[0]) if s.side[1].init else 9.0
                worse = (0 if eL > eR else 1) if eL != eR else (0 if fit[0]['n'] < fit[1]['n'] else 1)
                fit[worse] = None
            else:
                if min(fit[0]['n'], fit[1]['n']) >= p['width_min_pts'] and min(fit[0]['win'], fit[1]['win']) >= p['width_min_win']:
                    a = 0.3 if s.widthN < 5 else p['width_alpha']
                    s.width += a * (wn - s.width); s.widthN += 1
                wl = fit[0]['n'] / (fit[0]['rms'] + 0.05); wr = fit[1]['n'] / (fit[1]['rms'] + 0.05)
                for k in (1, 2):
                    v = (fit[0]['c'][k] * wl + fit[1]['c'][k] * wr) / (wl + wr); fit[0]['c'][k] = fit[1]['c'][k] = v
        else:
            for i in (0, 1):
                if not fit[i]: continue
                if abs(fit[i]['c'][0] - g.car_x) > p['lane_width_max'] - 0.3: fit[i] = None; continue
                o = s.side[1 - i]
                if s.widthN >= 5 and o.init and o.conf >= p['conf_valid'] and t - o.lastMeasT < p['stale_s']:
                    expect = o.target[0] + (-s.width if i == 0 else s.width)
                    if abs(fit[i]['c'][0] - expect) > p['width_tol_m']: fit[i] = None
        for i in (0, 1):
            s.pts[i] = fit[i]['n'] if fit[i] else 0
            if fit[i]: s.measure(s.side[i], fit[i], t)

    def output(s, g):
        p = s.P; t = s.lastT
        realL = s.side[0].init and s.side[0].conf >= p['conf_valid']
        realR = s.side[1].init and s.side[1].conf >= p['conf_valid']
        fL = t - s.side[0].lastMeasT < p['stale_s']; fR = t - s.side[1].lastMeasT < p['stale_s']
        if realL and realR and not fL and fR: realL = False
        if realL and realR and not fR and fL: realR = False
        if not (realL or realR): s.hadOut = False; return None
        L = s.side[0].shown.copy() if realL else None; R = s.side[1].shown.copy() if realR else None
        if L is None: L = R.copy(); L[0] -= s.width
        if R is None: R = L.copy(); R[0] += s.width

        dt = 0.0 if s.outT < 0 else min(max(t - s.outT, 0.0), 0.5); s.outT = t
        if s.out is None or dt <= 0 or not s.hadOut: s.out = [L.copy(), R.copy()]
        else:
            a = 1 - math.exp(-dt / p['tau_out'])
            for k, v in enumerate((L, R)): s.out[k] += a * (v - s.out[k])
        s.hadOut = True
        L, R = s.out[0].copy(), s.out[1].copy()
        off = g.car_x - 0.5 * (L[0] + R[0])
        return dict(L=L, R=R, realL=realL, realR=realR, offset=off, width=s.width,
                    mL=(g.car_x - p['car_half_width_m']) - L[0], mR=R[0] - (g.car_x + p['car_half_width_m']))

    def polylines(s, g, o):
        zn = max(2.0, g.nearZ()); zf = max(zn + 2.0, s.P['z_far_draw'])
        while zf > zn + 2 and g.rowAtZ(zf) - g.vh < 6: zf -= 1
        zz = [zn * (zf / zn) ** (i / 11) for i in range(12)]
        return [np.int32([g.toImage(s.X(o[k], z), z) for z in zz]) for k in ('L', 'R')]

    def debug_inset(s, h):
        c = cv2.cvtColor(s.bev, cv2.COLOR_GRAY2BGR) // 2 + 0
        for side, col in ((0, (60, 60, 255)), (1, (255, 120, 40))):
            for (x, y, w, hh) in s.win[side]: cv2.rectangle(c, (x, y), (x + w, y + hh), (0, 220, 255), 1)
            if s.side[side].init:
                pts = np.int32([((s.X(s.side[side].shown, z) - s.P['x_min']) / s.P['res_x'], (s.P['z_max'] - z) / s.P['res_z'])
                                for z in np.arange(0.5, s.P['z_max'], 1.0)])
                cv2.polylines(c, [pts], False, col, 2)
        return cv2.resize(c, (int(c.shape[1] * h / c.shape[0]), h), interpolation=cv2.INTER_NEAREST)

def filter_dets(dets, g):
    out, why_all = [], []
    for (x1, y1, x2, y2, sc, vc) in dets:
        w, h = x2 - x1, y2 - y1; why = None
        if w < 4 or h < 4: continue
        if y2 < g.vh + 2: why = 'above horizon'
        elif y1 >= g.hood - 4: why = 'in hood'
        elif y2 > g.hood + 6 and w > 0.45 * g.W: why = 'ego hood'
        elif h / w < 0.35 or h / w > 2.5: why = 'aspect'
        elif y2 <= g.hood + 2 and y2 - g.vh >= 6:
            wm = w * g.cam_h / (y2 - g.vh); pr = CLASS_W[vc]; cut = x1 < 3 or x2 > g.W - 3
            if wm > 1.9 * pr or (wm < 0.45 * pr and not cut): why = 'size %.2f m' % wm
        why_all.append(why)
        if why is None: out.append((x1, y1, x2, y2, sc, vc))
    return out, why_all

def ort_session(path, threads=2):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3; so.intra_op_num_threads = threads
    return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])

def print_io(name, s):
    print('== %s' % name)
    for i in s.get_inputs(): print('   input  %-12s %s %s' % (i.name, i.shape, i.type))
    for o in s.get_outputs(): print('   output %-12s %s %s' % (o.name, o.shape, o.type))

class NanoDet:
    MEAN = np.array([103.53, 116.28, 123.675], np.float32); STD = np.array([57.375, 57.12, 58.395], np.float32)
    def __init__(s, path, conf=0.30):
        s.s = ort_session(path); s.inp = s.s.get_inputs()[0]; s.conf = conf
        shp = s.inp.shape; s.ih = shp[2] if isinstance(shp[2], int) else 320; s.iw = shp[3] if isinstance(shp[3], int) else 320
    def __call__(s, bgr):
        H, W = bgr.shape[:2]; sc = min(s.iw / W, s.ih / H); nw, nh = round(W * sc), round(H * sc)
        lb = np.zeros((s.ih, s.iw, 3), np.uint8); lb[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        x = ((lb.astype(np.float32) - s.MEAN) / s.STD).transpose(2, 0, 1)[None]
        out = s.s.run(None, {s.inp.name: x})[0][0]
        nc = out.shape[1] - 32; cls = out[:, :nc]; reg = out[:, nc:].reshape(-1, 4, 8)
        dets = []; row = 0
        for st in (8, 16, 32, 64):
            fw, fh = (s.iw + st - 1) // st, (s.ih + st - 1) // st
            n = fw * fh; c = cls[row:row + n]; best = c.argmax(1); score = c[np.arange(n), best]
            for k in np.nonzero(score > s.conf)[0]:
                b = int(best[k])
                if b not in COCO2VC: continue
                r = reg[row + k]; pr = np.exp(r - r.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
                d = (pr * np.arange(8)).sum(1) * st; cx, cy = (k % fw) * st, (k // fw) * st
                dets.append([(cx - d[0]) / sc, (cy - d[1]) / sc, (cx + d[2]) / sc, (cy + d[3]) / sc, float(score[k]), COCO2VC[b]])
            row += n
        keep = []
        for vc in set(d[5] for d in dets):
            ds = [d for d in dets if d[5] == vc]
            idx = cv2.dnn.NMSBoxes([[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in ds], [d[4] for d in ds], s.conf, 0.5)
            keep += [ds[i] for i in np.array(idx).flatten()]
        return [(max(0, a), max(0, b), min(W - 1, c), min(H - 1, d), e, f) for a, b, c, d, e, f in keep]

def frames(args):
    if args.image:
        yield 0, cv2.imread(args.image); return
    cap = cv2.VideoCapture(args.video); i = 0
    while True:
        ok, f = cap.read()
        if not ok or (args.frames and i >= args.frames): break
        if args.width and f.shape[1] != args.width:
            f = cv2.resize(f, (args.width, int(round(f.shape[0] * args.width / f.shape[1]))), interpolation=cv2.INTER_AREA)
        yield i, f; i += 1

def load_calib(args):
    c = dict(hfov_deg=110.0, camera_height_m=1.30, horizon_y_ratio=0.40, hood_y_ratio=0.80, camera_lateral_offset_m=0.0)
    src = args.calib or ((args.video or args.image) + '.calib.json')
    if os.path.exists(src):
        try: c.update(json.load(open(src))); print('[calib] %s' % src)
        except Exception as e: print('[calib] %s unreadable: %s' % (src, e))
    for k, a in (('hfov_deg', args.hfov), ('camera_height_m', args.height), ('horizon_y_ratio', args.horizon), ('hood_y_ratio', args.hood)):
        if a is not None: c[k] = a
    return c

def cmd_info(args):
    if args.nanodet: print_io('NanoDet ' + args.nanodet, ort_session(args.nanodet))

def cmd_run(args):
    cal = load_calib(args)
    det = NanoDet(args.nanodet, args.conf) if args.nanodet else None
    lanes = LaneDetector(); writer = None; tracks_boxes = []
    vp_rows, stats = [], dict(frames=0, lane=0, dets=0, kept=0)
    fps = 30.0
    for i, f in frames(args):
        H, W = f.shape[:2]; t = i / fps
        g = Geo(W, H, cal['hfov_deg'], cal['camera_height_m'], cal['horizon_y_ratio'] * H, cal['hood_y_ratio'] * H,
                cal.get('camera_lateral_offset_m', 0.0))
        t0 = time.time()
        mask = lane_marking_mask(f, g, tracks_boxes)
        lanes.predict(t); lanes.update(mask, g); o = lanes.output(g)
        dets, kept, why = [], [], []
        if det and i % args.det_every == 0:
            dets = det(f); kept, why = filter_dets(dets, g)
            tracks_boxes = [(int(a), int(b), int(c - a), int(d - b)) for a, b, c, d, _, _ in kept]
        ms = 1000 * (time.time() - t0)
        stats['frames'] += 1; stats['lane'] += o is not None; stats['dets'] += len(dets); stats['kept'] += len(kept)

        if o and o['realL'] and o['realR']:
            pl, pr = lanes.polylines(g, o)
            (x1, y1), (x2, y2) = pl[0], pl[-1]; (x3, y3), (x4, y4) = pr[0], pr[-1]
            a1 = (x2 - x1) / max(1e-6, y2 - y1); a2 = (x4 - x3) / max(1e-6, y4 - y3)
            if abs(a1 - a2) > 0.25:
                vy = (x3 - a2 * y3 - x1 + a1 * y1) / (a1 - a2)
                if 0.2 * H < vy < 0.7 * H: vp_rows.append(vy / H)

        vis = f.copy(); vis[mask > 0] = (0, 200, 0)
        cv2.line(vis, (0, int(g.vh)), (W, int(g.vh)), (0, 255, 255), 1); cv2.line(vis, (0, g.hood), (W, g.hood), (0, 128, 255), 1)
        if o:
            pl, pr = lanes.polylines(g, o)
            cv2.polylines(vis, [pl], False, (0, 0, 255) if o['realL'] else (120, 120, 255), 2)
            cv2.polylines(vis, [pr], False, (255, 0, 0) if o['realR'] else (255, 160, 160), 2)
            txt = 'off %+.2f m  w %.2f m  mL %.2f mR %.2f' % (o['offset'], o['width'], o['mL'], o['mR'])
        else: txt = 'NO_LANE'
        for (a, b, c, d, sc, vc), w in zip(dets, why):
            col = (255, 255, 255) if w is None else (80, 80, 80)
            cv2.rectangle(vis, (int(a), int(b)), (int(c), int(d)), col, 1)
            if w is None:
                z = g.f * g.cam_h / max(1, min(d, g.hood) - g.vh)
                cv2.putText(vis, '%s %.2f %.0fm' % (CLASS_N[vc], sc, z), (int(a), int(b) - 3), 0, 0.35, col, 1)
        cv2.putText(vis, '%s | pts %d/%d | %.1f ms' % (txt, lanes.pts[0], lanes.pts[1], ms), (6, H - 8), 0, 0.42, (255, 255, 255), 1)
        ins = lanes.debug_inset(H // 2); vis[4:4 + ins.shape[0], W - ins.shape[1] - 4:W - 4] = ins
        if args.image: cv2.imwrite(args.out, vis); break
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
        writer.write(vis)
        if i % 30 == 0: print('frame %5d  %s  dets %d->%d  %.1f ms' % (i, txt, len(dets), len(kept), ms))
    if writer: writer.release()
    print('\nlanes found in %d / %d frames, detections %d -> %d after filter' % (stats['lane'], stats['frames'], stats['dets'], stats['kept']))
    if len(vp_rows) >= 10:
        hz = float(np.median(vp_rows))
        print('lane vanishing point -> horizon_y_ratio %.3f (now %.3f) from %d frames' % (hz, cal['horizon_y_ratio'], len(vp_rows)))
        if args.write_calib:
            cal['horizon_y_ratio'] = round(hz, 3); p = (args.video or args.image) + '.calib.json'
            json.dump(cal, open(p, 'w'), indent=2); print('wrote', p)
    print('wrote', args.out)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['info', 'run'])
    ap.add_argument('--video'); ap.add_argument('--image'); ap.add_argument('--out', default='adas_out.mp4')
    ap.add_argument('--nanodet'); ap.add_argument('--calib')
    ap.add_argument('--hfov', type=float); ap.add_argument('--height', type=float)
    ap.add_argument('--horizon', type=float); ap.add_argument('--hood', type=float)
    ap.add_argument('--width', type=int, default=640, help='resize video frames to this width (Pi pipeline uses 640)')
    ap.add_argument('--frames', type=int, default=0); ap.add_argument('--conf', type=float, default=0.30)
    ap.add_argument('--det-every', type=int, default=2)
    ap.add_argument('--write-calib', action='store_true', help='save the suggested horizon to <video>.calib.json')
    a = ap.parse_args()
    if a.cmd == 'run' and not (a.video or a.image): ap.error('run needs --video or --image')
    (cmd_info if a.cmd == 'info' else cmd_run)(a)

if __name__ == '__main__':
    main()
