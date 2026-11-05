import sys, glob, random, math
import numpy as np, cv2
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont

VALUES = list(range(5, 125, 5))
OTHER = len(VALUES)
S = 32
FONTS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf',
    '/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf',
    '/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Bold.otf',
    '/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Regular.otf',
    '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
]
NEG, BG = [], []

def rnd(a, b): return random.uniform(a, b)

def text_img(s, font_path, h_px):
    f = ImageFont.truetype(font_path, 200)
    w, h = f.getsize(s)
    im = Image.new('L', (w + 40, h + 40), 0)
    ImageDraw.Draw(im).text((20, 20), s, font=f, fill=255)
    a = np.array(im); ys, xs = np.nonzero(a)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    k = h_px / a.shape[0]
    return cv2.resize(a, (max(1, int(a.shape[1] * k)), max(1, int(h_px))), interpolation=cv2.INTER_AREA)

def background(n):
    if BG and random.random() < 0.85:
        b = cv2.imread(random.choice(BG))
        if b is not None: return cv2.resize(b, (n, n))
    c = np.random.randint(40, 220, 3)
    return np.clip(np.full((n, n, 3), c, np.float32) + np.random.randn(n, n, 3) * 12, 0, 255).astype(np.uint8)

def sign(content, n=128):
    img = background(n).astype(np.float32)
    D = n * rnd(0.74, 0.86); R = D / 2; c = (n / 2, n / 2)
    red = np.array([rnd(10, 70), rnd(10, 60), rnd(150, 245)])
    field = np.array([rnd(200, 255)] * 3) if random.random() < 0.9 else np.array([rnd(0, 60), rnd(150, 220), rnd(200, 255)])
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - c[0] + 0.5, yy - c[1] + 0.5)
    ring = rnd(0.10, 0.19)
    disc = np.clip(R - r + 0.5, 0, 1)[..., None]
    inner = np.clip(R * (1 - ring) - r + 0.5, 0, 1)[..., None]
    img = img * (1 - disc) + red * disc
    img = img * (1 - inner) + field * inner
    ink = np.array([rnd(0, 50)] * 3)
    ri = R * (1 - ring)
    if isinstance(content, str):
        font = random.choice(FONTS)
        h = ri * 2 * rnd(0.42, 0.66)
        t = text_img(content, font, h)

        if random.random() < 0.7:
            t = cv2.resize(t, (max(2, int(t.shape[1] * rnd(0.5, 0.85))), t.shape[0]), interpolation=cv2.INTER_AREA)
            if random.random() < 0.5:
                k = max(1, int(round(t.shape[0] * rnd(0.02, 0.05))))
                t = cv2.dilate(t, np.ones((k, k), np.uint8))
        maxw = ri * 2 * rnd(0.70, 0.84)
        if t.shape[1] > maxw: t = cv2.resize(t, (int(maxw), t.shape[0]), interpolation=cv2.INTER_AREA)
        if random.random() < 0.15 and len(content) <= 2:
            k = text_img('km/h', font, h * 0.28)
            if k.shape[1] < t.shape[1] * 1.2:
                pad = np.zeros((k.shape[0] + int(h * 0.12), t.shape[1]), np.uint8)
                x0 = (t.shape[1] - k.shape[1]) // 2; pad[-k.shape[0]:, max(0, x0):max(0, x0) + min(k.shape[1], t.shape[1])] = k[:, :t.shape[1]]
                t = np.vstack([t, pad])
        th, tw = t.shape
        x0 = int(c[0] - tw / 2 + rnd(-0.03, 0.03) * D); y0 = int(c[1] - th / 2 + rnd(-0.04, 0.04) * D)
        a = (t.astype(np.float32) / 255)[..., None]
        sub = img[y0:y0 + th, x0:x0 + tw]
        img[y0:y0 + th, x0:x0 + tw] = sub * (1 - a[:sub.shape[0], :sub.shape[1]]) + ink * a[:sub.shape[0], :sub.shape[1]]
    else:
        content(img, c, ri, ink)
    return np.clip(img, 0, 255).astype(np.uint8), D

def neg_text(img, c, ri, ink):
    s = random.choice(['3.5m', '2.5m', '4.2m', '10T', '5T', '13T', '30T', '2,7m', 'P', 'T', '7m', 'km/h', '1.8m', '15T', 'STOP'])
    t = text_img(s, random.choice(FONTS), ri * 2 * rnd(0.3, 0.5))
    if t.shape[1] > ri * 1.7: t = cv2.resize(t, (int(ri * 1.7), t.shape[0]))
    th, tw = t.shape; x0 = int(c[0] - tw / 2); y0 = int(c[1] - th / 2)
    a = (t.astype(np.float32) / 255)[..., None]
    img[y0:y0 + th, x0:x0 + tw] = img[y0:y0 + th, x0:x0 + tw] * (1 - a) + ink * a

def neg_symbol(img, c, ri, ink):
    k = random.randint(0, 4); u = ri
    if k == 0:   cv2.arrowedLine(img, (int(c[0]), int(c[1] + u * 0.6)), (int(c[0]), int(c[1] - u * 0.6)), ink.tolist(), max(2, int(u * 0.2)), tipLength=0.4)
    elif k == 1: cv2.rectangle(img, (int(c[0] - u * 0.6), int(c[1] - u * 0.25)), (int(c[0] + u * 0.6), int(c[1] + u * 0.25)), ink.tolist(), -1)
    elif k == 2: cv2.circle(img, (int(c[0]), int(c[1])), int(u * 0.35), ink.tolist(), -1)
    elif k == 3: cv2.line(img, (int(c[0] - u * 0.7), int(c[1] - u * 0.7)), (int(c[0] + u * 0.7), int(c[1] + u * 0.7)), (40, 40, 200), max(2, int(u * 0.18)))

def _limit_text(img, c, ri, ink, s, hk):
    t = text_img(s, random.choice(FONTS), ri * 2 * hk)
    if random.random() < 0.6: t = cv2.resize(t, (max(2, int(t.shape[1] * rnd(0.6, 0.9))), t.shape[0]), interpolation=cv2.INTER_AREA)
    if t.shape[1] > ri * 1.5: t = cv2.resize(t, (int(ri * 1.5), t.shape[0]), interpolation=cv2.INTER_AREA)
    th, tw = t.shape; x0 = int(c[0] - tw / 2); y0 = int(c[1] - th / 2 + rnd(-0.03, 0.03) * ri)
    a = (t.astype(np.float32) / 255)[..., None]
    img[y0:y0 + th, x0:x0 + tw] = img[y0:y0 + th, x0:x0 + tw] * (1 - a) + ink * a
    return th, tw

def _metres():
    v = random.choice(['2', '2,0', '2,2', '2,5', '2,7', '2,8', '3', '3,2', '3,5', '4', '4,2', '4,5', '4,75', '5'])
    return (v.replace(',', '.') if random.random() < 0.25 else v) + random.choice(['m', 'm', ' m', 'M'])

def neg_height(img, c, ri, ink):
    th, _ = _limit_text(img, c, ri, ink, _metres(), rnd(0.26, 0.40))
    w, h = ri * rnd(0.35, 0.55), ri * rnd(0.22, 0.34)
    for sg in (-1, 1):
        tip = c[1] + sg * (th / 2 + ri * rnd(0.03, 0.10)); base = tip + sg * h
        cv2.fillConvexPoly(img, np.array([[c[0], tip], [c[0] - w / 2, base], [c[0] + w / 2, base]], np.int32), ink.tolist())

def neg_width(img, c, ri, ink):
    _, tw = _limit_text(img, c, ri, ink, _metres(), rnd(0.22, 0.32))
    w, h = ri * rnd(0.22, 0.32), ri * rnd(0.35, 0.5)
    for sg in (-1, 1):
        tip = c[0] + sg * (tw / 2 + ri * rnd(0.03, 0.08)); base = tip + sg * w
        cv2.fillConvexPoly(img, np.array([[tip, c[1]], [base, c[1] - h / 2], [base, c[1] + h / 2]], np.int32), ink.tolist())

def neg_weight(img, c, ri, ink):
    s = random.choice(['5', '7', '10', '13', '15', '18', '20', '30', '2,5', '3,5', '8']) + random.choice(['t', 't', 'T', ' t'])
    if random.random() < 0.3:
        u = ri; y = c[1] + u * 0.25
        cv2.rectangle(img, (int(c[0] - u * 0.55), int(y)), (int(c[0] + u * 0.35), int(y + u * 0.3)), ink.tolist(), -1)
        cv2.rectangle(img, (int(c[0] + u * 0.38), int(y + u * 0.08)), (int(c[0] + u * 0.6), int(y + u * 0.3)), ink.tolist(), -1)
        c = (c[0], c[1] - u * 0.25)
    _limit_text(img, c, ri, ink, s, rnd(0.3, 0.45))

def neg_end_of_limit(img, c, ri, ink):
    v = str(random.choice(VALUES))
    t = text_img(v, random.choice(FONTS), ri * 2 * rnd(0.42, 0.55))
    if t.shape[1] > ri * 1.6: t = cv2.resize(t, (int(ri * 1.6), t.shape[0]))
    th, tw = t.shape; x0 = int(c[0] - tw / 2); y0 = int(c[1] - th / 2)
    a = (t.astype(np.float32) / 255)[..., None] * 0.6
    img[y0:y0 + th, x0:x0 + tw] = img[y0:y0 + th, x0:x0 + tw] * (1 - a) + ink * a
    for d in (-0.25, 0, 0.25):
        o = d * ri
        cv2.line(img, (int(c[0] - ri * 0.7 + o), int(c[1] + ri * 0.7 + o)), (int(c[0] + ri * 0.7 + o), int(c[1] - ri * 0.7 + o)), ink.tolist(), max(1, int(ri * 0.06)))

def degrade(img, D):
    n = img.shape[0]

    sx = rnd(0.72, 1.0) if random.random() < 0.6 else 1.0
    M = cv2.getRotationMatrix2D((n / 2, n / 2), rnd(-8, 8), 1.0); M[0] *= sx; M[0, 2] += n / 2 * (1 - sx)
    img = cv2.warpAffine(img, M, (n, n), borderMode=cv2.BORDER_REFLECT)

    bw, bh = D * sx, D
    sc = rnd(0.9, 1.22); cx = n / 2 + rnd(-0.1, 0.1) * bw; cy = n / 2 + rnd(-0.1, 0.1) * bh
    x1, y1 = int(max(0, cx - bw * sc / 2)), int(max(0, cy - bh * sc / 2))
    x2, y2 = int(min(n, cx + bw * sc / 2)), int(min(n, cy + bh * sc / 2))
    crop = img[y1:y2, x1:x2]

    s = int(np.clip(random.choice([rnd(11, 22), rnd(14, 34), rnd(20, 64)]), 10, 64))
    ar = crop.shape[1] / max(1, crop.shape[0])
    crop = cv2.GaussianBlur(crop, (0, 0), rnd(0.2, 1.0) * crop.shape[0] / s)
    small = cv2.resize(crop, (max(6, int(s * ar)), s), interpolation=cv2.INTER_AREA)
    if random.random() < 0.3:
        k = random.randint(2, 4); ker = np.zeros((k, k), np.float32); ker[k // 2] = 1 / k
        small = cv2.filter2D(small, -1, ker)
    f = small.astype(np.float32)
    f = f * rnd(0.5, 1.3) + rnd(-35, 35)
    if random.random() < 0.3: f = 255 * (np.clip(f, 0, 255) / 255) ** rnd(0.6, 1.8)
    f += np.random.randn(*f.shape) * rnd(0, 9)
    small = np.clip(f, 0, 255).astype(np.uint8)
    if random.random() < 0.7:
        ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, random.randint(15, 90)])
        small = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if random.random() < 0.3:
        q = rnd(0.5, 0.8); h0, w0 = small.shape[:2]
        small = cv2.resize(cv2.resize(small, (max(3, int(w0 * q)), max(3, int(h0 * q))), interpolation=cv2.INTER_AREA), (w0, h0), interpolation=cv2.INTER_LINEAR)
    return to_input(small)

def to_input(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (S, S), interpolation=cv2.INTER_LINEAR)

def real_neg():
    im = cv2.imread(random.choice(NEG))
    if im is None or im.shape[0] < 6: return None
    s = int(np.clip(rnd(10, 48), 8, max(10, im.shape[0])))
    im = cv2.resize(im, (max(6, int(s * im.shape[1] / im.shape[0])), s), interpolation=cv2.INTER_AREA)
    f = np.clip(im.astype(np.float32) * rnd(0.5, 1.3) + rnd(-30, 30), 0, 255).astype(np.uint8)
    return to_input(f)

def sample(cls):
    if cls < OTHER:
        img, D = sign(str(VALUES[cls]))
        return degrade(img, D)
    k = random.random()
    if NEG and k < 0.2:
        x = real_neg()
        if x is not None: return x
    if BG and k < 0.35:
        return to_input(cv2.imread(random.choice(BG)))
    fn = random.choice([neg_text, neg_height, neg_height, neg_width, neg_weight, neg_symbol, neg_end_of_limit])
    img, D = sign(fn)
    return degrade(img, D)

def work(args):
    seed, classes = args
    random.seed(seed); np.random.seed(seed % (2 ** 32))
    return np.stack([sample(c) for c in classes]), np.array(classes, np.int64)

def init(neg, bg):
    NEG[:] = neg; BG[:] = bg

if __name__ == '__main__':
    out, per = sys.argv[1], int(sys.argv[2])
    neg = glob.glob(sys.argv[3] + '/*.png') if len(sys.argv) > 3 else []
    bg = glob.glob(sys.argv[4] + '/*.png') if len(sys.argv) > 4 else []
    seed0 = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    classes = [c for c in range(OTHER) for _ in range(per)] + [OTHER] * (per * 3)
    random.seed(seed0); random.shuffle(classes)
    jobs = [(seed0 * 100000 + i, classes[i::64]) for i in range(64)]
    with Pool(10, initializer=init, initargs=(neg, bg)) as p:
        parts = p.map(work, jobs)
    X = np.concatenate([a for a, _ in parts]); y = np.concatenate([b for _, b in parts])
    np.savez_compressed(out, X=X, y=y)
    print(out, X.shape, 'neg', len(neg), 'bg', len(bg))
