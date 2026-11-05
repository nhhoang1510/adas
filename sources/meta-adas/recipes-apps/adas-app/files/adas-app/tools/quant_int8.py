import sys, glob, cv2, numpy as np, onnx
from onnxruntime.quantization import (quantize_static, CalibrationDataReader, QuantType, QuantFormat,
                                      CalibrationMethod)
from onnxruntime.quantization.shape_inference import quant_pre_process

kind, src, dst, fdir = sys.argv[1:5]
frames = sorted(glob.glob(fdir + '/*.jpg'))[::int(sys.argv[5]) if len(sys.argv) > 5 else 8]

def letterbox(f, W, H):
    s = min(W / f.shape[1], H / f.shape[0])
    nw, nh = round(f.shape[1] * s), round(f.shape[0] * s)
    lb = np.full((H, W, 3), 114 if kind == 'tsr' else 0, np.uint8)
    lb[:nh, :nw] = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return lb

def prep(f):
    if kind == 'nanodet':
        x = (letterbox(f, 320, 320).astype(np.float32) - [103.53, 116.28, 123.675]) / [57.375, 57.12, 58.395]
        return {'data': x.transpose(2, 0, 1)[None].astype(np.float32)}
    band = f[16:240]
    x = cv2.cvtColor(letterbox(band, 640, 224), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    return {'images': x.transpose(2, 0, 1)[None].astype(np.float32)}

class R(CalibrationDataReader):
    def __init__(s): s.it = iter(frames)
    def get_next(s):
        p = next(s.it, None)
        return None if p is None else prep(cv2.imread(p))

pre = dst + '.pre.onnx'
m = onnx.load(src)

inits = {i.name for i in m.graph.initializer}
keep = [i for i in m.graph.input if i.name not in inits]
del m.graph.input[:]; m.graph.input.extend(keep)
if max(o.version for o in m.opset_import if o.domain in ('', 'ai.onnx')) < 13:
    from onnx import version_converter
    m = version_converter.convert_version(m, 13)
src = dst + '.clean.onnx'
onnx.save(m, src)
quant_pre_process(src, pre)

g = onnx.load(pre).graph
prod = {o: n for n in g.node for o in n.output}
excl = set()
def back(t):
    n = prod.get(t)
    if n is None or n.name in excl: return
    excl.add(n.name)
    if n.op_type == 'Conv': return
    for i in n.input: back(i)
for o in g.output: back(o.name)
print('float nodes:', len(excl))
quantize_static(pre, dst, R(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8, per_channel=True, nodes_to_exclude=sorted(excl),
                calibrate_method=CalibrationMethod.Percentile)
print('ok', dst, len(frames), 'calib frames')
