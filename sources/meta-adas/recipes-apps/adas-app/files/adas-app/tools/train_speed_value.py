import sys, glob, os, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(10)
NC = 25

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        s.f = nn.Sequential(blk(1, 16), blk(16, 16), nn.MaxPool2d(2),
                            blk(16, 32), blk(32, 32), nn.MaxPool2d(2),
                            blk(32, 64), blk(64, 64), nn.MaxPool2d(2))
        s.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 16, 128), nn.ReLU(inplace=True), nn.Linear(128, NC))

    def forward(s, x):
        m = x.mean(dim=(2, 3), keepdim=True)
        sd = ((x - m) ** 2).mean(dim=(2, 3), keepdim=True).add(1.0).sqrt()
        return s.h(s.f((x - m) / sd))

def load(p):
    d = np.load(p); return torch.from_numpy(d['X']).float()[:, None], torch.from_numpy(d['y'])

def label(v): return v // 5 - 1

if __name__ == '__main__':
    Xt, yt = load(sys.argv[1]); Xv, yv = load(sys.argv[2]); out = sys.argv[3]
    real = sorted(glob.glob(sys.argv[4] + '/*.png')) if len(sys.argv) > 4 else []
    net = Net(); ep = 14
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=ep * ((len(Xt) + 255) // 256))
    for e in range(ep):
        net.train(); perm = torch.randperm(len(Xt)); tl = 0
        for i in range(0, len(Xt), 256):
            idx = perm[i:i + 256]; x = Xt[idx]

            dx, dy = np.random.randint(-2, 3, 2); x = torch.roll(x, (int(dy), int(dx)), (2, 3))
            loss = F.cross_entropy(net(x), yt[idx], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(Xv[i:i + 1024]) for i in range(0, len(Xv), 1024)]).argmax(1)
        acc = (pv == yv).float().mean().item()
        sv = yv < 24; accv = (pv[sv] == yv[sv]).float().mean().item()
        fa = ((pv < 24) & (yv == 24)).float().sum().item() / max(1, (yv == 24).sum().item())
        print('epoch %2d loss %.3f | val acc %.3f, values %.3f, other->value %.3f' % (e, tl / len(Xt), acc, accv, fa), flush=True)

    with torch.no_grad():
        P = torch.cat([F.softmax(net(Xv[i:i + 1024]), 1) for i in range(0, len(Xv), 1024)])
    conf, pv = P.max(1)
    for th in (0.0, 0.5, 0.7, 0.85):
        k = (conf >= th) & (yv < 24)
        print('  values, conf >= %.2f: kept %.0f%%, acc %.3f' % (th, 100 * k.float().sum() / (yv < 24).sum(), (pv[k] == yv[k]).float().mean()))
    if real:
        xr = torch.stack([torch.from_numpy(cv2.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_LINEAR)).float()[None] for f in real])
        with torch.no_grad(): pr = F.softmax(net(xr), 1)
        c, k = pr.max(1)
        names = [(i + 1) * 5 if i < 24 else 'x' for i in k.tolist()]
        print('real crops:', ' '.join('%s(%.2f)' % (n, v) for n, v in zip(names, c.tolist())))
    torch.onnx.export(net, torch.zeros(1, 1, 32, 32), out, input_names=['crop'], output_names=['logits'],
                      dynamic_axes={'crop': {0: 'n'}, 'logits': {0: 'n'}}, opset_version=13)
    print('saved', out, sum(p.numel() for p in net.parameters()), 'params')
