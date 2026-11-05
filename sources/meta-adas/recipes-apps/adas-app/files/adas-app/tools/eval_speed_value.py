import sys, random, collections, numpy as np, cv2, onnxruntime as ort
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import speed_sign_synth as g

tsr = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
val = ort.InferenceSession(sys.argv[2], providers=['CPUExecutionProvider'])
ti = tsr.get_inputs()[0].name
cap = cv2.VideoCapture(sys.argv[3]); frames = []
while len(frames) < 400:
    ok, f = cap.read()
    if not ok: break
    frames.append(cv2.resize(f, (640, 360)))
N = int(sys.argv[4]) if len(sys.argv) > 4 else 200

def paste(frame, sign_bgr, D, cx, cy, size):
    n = sign_bgr.shape[0]; k = size / D; m = int(round(n * k))
    s = cv2.resize(sign_bgr, (m, m), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:m, 0:m]; a = (np.hypot(xx - m / 2 + .5, yy - m / 2 + .5) <= size / 2 + .5).astype(np.float32)[..., None]
    x0, y0 = int(cx - m / 2), int(cy - m / 2)
    if x0 < 0 or y0 < 0 or x0 + m > 640 or y0 + m > 360: return frame
    out = frame.copy(); r = out[y0:y0 + m, x0:x0 + m].astype(np.float32)
    out[y0:y0 + m, x0:x0 + m] = (r * (1 - a) + s * a).astype(np.uint8)
    out = cv2.GaussianBlur(out, (0, 0), 0.6)
    ok, b = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return cv2.imdecode(b, cv2.IMREAD_COLOR)

def detect(frame, y0=40):
    band = frame[y0:y0 + 224]
    x = cv2.cvtColor(band, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    o = tsr.run(None, {ti: x.transpose(2, 0, 1)[None]})[0][0]
    sc = o[4:]; cls = sc.argmax(0); conf = sc.max(0)
    return [(o[0, i], o[1, i] + y0, o[2, i], o[3, i], cls[i], conf[i]) for i in np.where(conf > 0.35)[0]], band

def read(frame, cx, cy, w, h):
    x1, y1 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
    c = frame[y1:int(cy + h / 2), x1:int(cx + w / 2)]
    if c.shape[0] < 6 or c.shape[1] < 6: return None
    x = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    l = val.run(None, {'crop': x[None, None]})[0][0]; p = np.exp(l - l.max()); return p / p.sum()

def vote(votes, p, w):
    votes += min(1.5, max(0.25, w / 24)) * p
    bi = int(votes[:24].argmax()); best = votes[bi]; sec = np.sort(votes[:24])[-2]
    if best >= 0.9 and best >= 2 * sec and best > votes[24]: return (bi + 1) * 5
    if votes[24] >= 1.2 and votes[24] > 2 * best: return -1
    return 0

random.seed(12345); np.random.seed(12345)
g.init([], [])
res = collections.Counter(); wrong = collections.Counter(); det_looks = tot_looks = 0
for trial in range(N):
    v = random.choice(g.VALUES)
    sign, D = g.sign(str(v), 160)
    side = random.choice([-1, 1]); fi = random.randint(0, len(frames) - 80)
    votes = np.zeros(25); value = 0; seen = False
    for k, size in enumerate([14, 20, 30, 44, 60]):
        cx = 320 + side * (150 + 2.2 * size * random.uniform(0.9, 1.1)); cy = 140 - 0.9 * size
        fr = paste(frames[fi + 15 * k], sign, D, cx, cy, size)
        dets, _ = detect(fr); tot_looks += 1
        hit = [d for d in dets if abs(d[0] - cx) < max(8, size * 0.5) and abs(d[1] - cy) < max(8, size * 0.5)]
        if not hit: continue
        d = max(hit, key=lambda d: d[5]); det_looks += 1
        if d[4] != 0: continue
        seen = True
        p = read(fr, d[0], d[1], d[2], d[3]) if min(d[2], d[3]) >= 12 else None
        if p is not None: value = vote(votes, p, min(d[2], d[3])) or value
    if not seen: res['detector: khong thay la bien toc do'] += 1
    elif value == v: res['doc dung'] += 1
    elif value == 0: res['chua quyet dinh'] += 1
    elif value < 0: res['bao khong phai bien toc do'] += 1
    else: res['doc SAI'] += 1; wrong['%d->%d' % (v, value)] += 1
print('%d bien, detector thay %.0f%% luot nhin' % (N, 100 * det_looks / tot_looks))
for k, n in res.most_common(): print('  %-38s %3d  (%.0f%%)' % (k, n, 100 * n / N))
seen = N - res['detector: khong thay la bien toc do']
print('  trong cac bien detector thay: doc dung %.0f%%, sai %.0f%%' % (100 * res['doc dung'] / max(1, seen), 100 * res['doc SAI'] / max(1, seen)))
if wrong: print('  doc sai:', dict(wrong.most_common(10)))
