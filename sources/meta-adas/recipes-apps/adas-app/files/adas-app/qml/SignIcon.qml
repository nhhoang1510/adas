import QtQuick 2.12

Canvas {
    id: si
    property int   code: -1
    property int   value: 0
    property color ghost: "#070D10"

    readonly property color red:    "#FF0000"
    readonly property color yellow: "#FFE000"
    readonly property color blue:   "#0038FF"
    readonly property color white:  "#FFFFFF"
    readonly property color black:  "#000000"

    renderStrategy: Canvas.Immediate
    antialiasing: true
    onCodeChanged: requestPaint()
    onValueChanged: requestPaint()
    onGhostChanged: requestPaint()
    onWidthChanged: requestPaint()
    onHeightChanged: requestPaint()

    function disc(ctx, r, col) {
        ctx.fillStyle = col; ctx.beginPath(); ctx.arc(50, 50, r, 0, 2 * Math.PI); ctx.fill()
    }
    function poly(ctx, pts, col) {
        ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1])
        for (var i = 1; i < pts.length; ++i) ctx.lineTo(pts[i][0], pts[i][1])
        ctx.closePath(); ctx.fill()
    }
    function line(ctx, pts, col, lw) {
        ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.lineCap = "butt"; ctx.lineJoin = "miter"
        ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1])
        for (var i = 1; i < pts.length; ++i) ctx.lineTo(pts[i][0], pts[i][1])
        ctx.stroke()
    }
    function arrow(ctx, pts, col, lw) {
        var n = pts.length, a = pts[n - 1], b = pts[n - 2]
        var dx = a[0] - b[0], dy = a[1] - b[1], l = Math.sqrt(dx * dx + dy * dy)
        dx /= l; dy /= l
        var h = lw * 2.0, w = lw * 1.5
        var base = [a[0] - dx * h, a[1] - dy * h]
        var p = pts.slice(0, n - 1); p.push(base)
        line(ctx, p, col, lw)
        poly(ctx, [a, [base[0] - dy * w, base[1] + dx * w], [base[0] + dy * w, base[1] - dx * w]], col)
    }
    function text(ctx, s, y, px, col) {
        ctx.fillStyle = col; ctx.font = "bold " + px + "px sans-serif"
        ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(s, 50, y)
    }
    function number(ctx, s) {
        var D = 96, h = 0.396 * D
        var px = h / 0.729
        ctx.save()
        ctx.fillStyle = black
        ctx.font = "bold " + px + "px 'DejaVu Sans'"
        ctx.textAlign = "center"; ctx.textBaseline = "alphabetic"
        var w = ctx.measureText(s).width
        var target = (s.length >= 3 ? 0.66 : s.length === 2 ? 0.5 : 0.25) * D
        ctx.translate(50, 50 + h / 2)
        ctx.scale(target / w, 1)
        ctx.fillText(s, 0, 0)
        ctx.restore()
    }
    function disc2(ctx, x, y, r, col) {
        ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill()
    }

    function prohibit(ctx) { disc(ctx, 48, red); disc(ctx, 48 - 96 * 10 / 70, white) }
    function danger(ctx) {
        ctx.fillStyle = red; ctx.strokeStyle = red; ctx.lineWidth = 5; ctx.lineJoin = "round"
        ctx.beginPath(); ctx.moveTo(50, 5); ctx.lineTo(95, 86); ctx.lineTo(5, 86); ctx.closePath()
        ctx.fill(); ctx.stroke()
        poly(ctx, [[50, 22], [81.4, 77], [18.6, 77]], yellow)
    }

    function endDisc(ctx) { disc(ctx, 48, blue); disc(ctx, 40, white) }
    function endStripes(ctx) {
        ctx.save(); ctx.beginPath(); ctx.arc(50, 50, 40, 0, 2 * Math.PI); ctx.clip()
        for (var k = -2; k <= 2; ++k) {
            var o = k * 3.4
            line(ctx, [[12 + o, 88 + o], [88 + o, 12 + o]], black, 2.4)
        }
        ctx.restore()
    }
    function town(ctx) {
        ctx.fillStyle = blue; ctx.fillRect(3, 10, 94, 80)
        poly(ctx, [[9, 77], [9, 59], [19, 49], [27, 57], [29, 57], [29, 40], [35, 19], [41, 40], [41, 50],
                   [53, 50], [53, 24], [64, 24], [64, 35], [69, 35], [69, 52], [80, 42], [91, 53], [91, 77]], white)
    }
    function expressway(ctx) {
        ctx.fillStyle = blue; ctx.strokeStyle = blue; ctx.lineWidth = 8; ctx.lineJoin = "round"
        ctx.beginPath(); ctx.rect(16, 7, 68, 86); ctx.fill(); ctx.stroke(); ctx.lineJoin = "miter"
        ctx.strokeStyle = white; ctx.lineWidth = 1.5; ctx.strokeRect(15.5, 6.5, 69, 87)
        poly(ctx, [[41, 15], [47, 15], [46, 87], [23, 87]], white)
        poly(ctx, [[53, 15], [59, 15], [77, 87], [54, 87]], white)
        ctx.fillStyle = blue; ctx.fillRect(27, 45, 46, 4)
        ctx.fillStyle = white
        ctx.fillRect(21, 39, 58, 6)
        ctx.fillRect(23, 45, 4, 5); ctx.fillRect(73, 45, 4, 5)
    }
    function stop(ctx) {
        var o = [], q = []
        for (var j = 0; j < 8; ++j) {
            var a = (22.5 + 45 * j) * Math.PI / 180
            o.push([50 + 48 * Math.cos(a), 50 + 48 * Math.sin(a)]); q.push([50 + 44 * Math.cos(a), 50 + 44 * Math.sin(a)])
        }
        poly(ctx, o, white); poly(ctx, q, red)
        ctx.save(); ctx.translate(50, 51); ctx.scale(0.86, 1)
        ctx.fillStyle = white; ctx.font = "bold 30px 'DejaVu Sans'"; ctx.textAlign = "center"; ctx.textBaseline = "middle"
        ctx.fillText("STOP", 0, 0); ctx.restore()
    }

    function walker(ctx) {
        var outline = [
                [46.9, 38], [45.6, 39.2], [45.5, 40.3], [45.6, 41.3], [47.1, 42.5], [47.1, 43.2], [45, 45.5], [43.2, 49.4],
                [40.8, 51.6], [38.5, 52.1], [38.5, 53.5], [39.1, 54], [42.3, 53], [44.8, 51.4], [46.4, 49.2], [47.1, 49.8],
                [47.4, 51.8], [47.1, 56], [45.3, 61.4], [44.2, 66.7], [43.7, 67.1], [42.4, 67], [42, 67.7], [43.8, 68.3],
                [43.5, 70.8], [43.1, 71.3], [40.1, 71.7], [44.5, 73.5], [45.9, 68.3], [47.7, 67.8], [47.9, 67.1], [46.9, 67.1],
                [46.5, 66.5], [49.6, 58.7], [50.2, 58.3], [53.1, 63.3], [55.8, 66.7], [55.7, 67.9], [57.1, 68], [58.5, 69.3],
                [57.5, 72.1], [58.7, 72], [59.5, 71.4], [60.6, 69], [60.1, 68.2], [61.4, 68], [61.5, 67.6], [61, 67],
                [59.2, 67.1], [58.7, 66.6], [54.8, 60.2], [53.6, 55.8], [53.4, 48.4], [55, 49.5], [56, 52.3], [55.7, 54.2],
                [56.2, 54.9], [58.1, 53.9], [58.1, 53.3], [57.4, 52.7], [57.1, 49.9], [55.8, 47.2], [53.5, 44.4], [52, 43.3],
                [49.5, 42.4], [50, 40.8], [49.5, 38.8], [48.5, 38]]
        ctx.save(); ctx.translate(50, 58); ctx.scale(1.15, 1.15); ctx.translate(-50, -60)
        ctx.fillStyle = black
        for (var i = 0; i < 11; ++i) ctx.fillRect(32 + 3.25 * i, 66.6, 2.3, 1.8)
        for (var j = 0; j < 11; ++j) ctx.fillRect(26.9 + 4.25 * j, 74.1, 3.3, 1.8)
        poly(ctx, outline, black)
        ctx.restore()
    }
    function children(ctx) {
        var outline = [
                [41.7, 42.6], [40.4, 44.8], [40.5, 46.9], [41.4, 48.1], [41.4, 48.6], [39.4, 49.8], [37.8, 53.1], [35.7, 56.1],
                [32.1, 59], [32.1, 59.6], [32.6, 60.1], [34.7, 59.3], [36.9, 57.6], [39.3, 54.7], [39.6, 54.9], [38.5, 58.6],
                [36.8, 62], [37.8, 63.3], [35.8, 70.9], [35.2, 71.6], [32.8, 72.2], [36, 73.2], [37, 72.9], [37.5, 71],
                [39.4, 67.3], [39.8, 63.9], [40.4, 63.3], [44.2, 63.6], [50, 68.9], [50.5, 69.6], [50.2, 72.9], [51.3, 72.4],
                [54.4, 73], [54.9, 72.8], [55.8, 68.3], [57.9, 63.2], [61.2, 66.3], [65.9, 68], [66.4, 68.7], [66.6, 70.5],
                [67, 70.6], [67.7, 66.3], [64, 64.7], [61.9, 63], [60.4, 59.8], [60.1, 56.9], [60.6, 56.7], [61.7, 57.4],
                [63, 59], [63.4, 60.1], [63.1, 62.3], [63.7, 63], [64.6, 62.9], [64.9, 62.4], [64.8, 59.1], [64.4, 57.5],
                [63, 55.2], [61.6, 54], [59.6, 53.2], [57.7, 53.1], [57.3, 52.7], [58.3, 50.7], [57.9, 48.7], [56.3, 47.7],
                [55.2, 47.9], [54.3, 48.6], [53.9, 49.4], [53.9, 50.7], [55.4, 53.2], [52, 58.4], [51.2, 59.2], [50, 58.9],
                [49.2, 58.3], [49.1, 57.6], [50.5, 53.6], [50.3, 52.8], [48.2, 50.5], [45.6, 48.8], [43.8, 48.6], [45.4, 46.8],
                [45.4, 44.5], [44.1, 42.7], [43.4, 42.4]]
        var gap = [
                [45.6, 52.2], [47.9, 53.2], [48.5, 53.9], [48.4, 55.2], [47.2, 57.9], [47.2, 58.5], [48, 59.5], [49.5, 60.2],
                [52.4, 60.8], [53.1, 60.4], [55, 58.3], [55.4, 58.7], [55.3, 60.8], [54.3, 64.5], [53.1, 70.9], [52.7, 71.3],
                [52.1, 71], [52.6, 68.9], [51.3, 67.7], [49.4, 64.9], [46.5, 62.7], [47.3, 61.8], [45.7, 57.3], [45.4, 55],
                [45.4, 52.4]]
        ctx.save(); ctx.translate(50, 58); ctx.scale(1.1, 1.1); ctx.translate(-50, -58)
        poly(ctx, outline, black); poly(ctx, gap, yellow)
        ctx.restore()
    }
    function roundTri(ctx, pts, r, col) {
        ctx.fillStyle = col; ctx.strokeStyle = col; ctx.lineWidth = 2 * r; ctx.lineJoin = "round"
        ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); ctx.lineTo(pts[1][0], pts[1][1]); ctx.lineTo(pts[2][0], pts[2][1])
        ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.lineJoin = "miter"
    }
    function giveWay(ctx) {
        roundTri(ctx, [[12.2, 14], [87.8, 14], [50, 81.7]], 6, red)
        roundTri(ctx, [[18.2, 17.5], [81.8, 17.5], [50, 74.5]], 2, yellow)
    }

    function drawSign(ctx, c) {
        switch (c) {
        case 0:  prohibit(ctx)
                 if (value > 0) number(ctx, value.toString())
                 else text(ctx, "km/h", 52, 22, black)
                 break
        case 1:  endDisc(ctx)
                 if (value > 0) number(ctx, value.toString())
                 else text(ctx, "km/h", 52, 22, black)
                 endStripes(ctx); break
        case 2:  endDisc(ctx); endStripes(ctx); break
        case 3:  town(ctx); break
        case 4:  town(ctx); line(ctx, [[6, 87], [94, 13]], red, 6); break
        case 5:  expressway(ctx); break
        case 6:  expressway(ctx)
                 ctx.strokeStyle = red; ctx.lineWidth = 9; ctx.lineCap = "round"
                 ctx.beginPath(); ctx.moveTo(19, 89); ctx.lineTo(81, 11); ctx.stroke(); ctx.lineCap = "butt"; break
        case 7:  disc(ctx, 48, red)
                 ctx.fillStyle = white; ctx.fillRect(18, 41, 64, 18); break
        case 8:  stop(ctx); break
        case 9:  giveWay(ctx); break
        case 10: danger(ctx); walker(ctx); break
        case 11: danger(ctx); children(ctx); break
        case 12: danger(ctx)
                 ctx.fillStyle = black; ctx.fillRect(31, 50, 38, 6); ctx.fillRect(31, 62, 38, 6)
                 ctx.fillRect(36, 44, 6, 31); ctx.fillRect(58, 44, 6, 31); break
        case 13: {
                 danger(ctx)
                 var pts = [[44, 76], [44, 60]]
                 for (var j = 1; j <= 6; ++j) { var t = Math.PI + j * Math.PI / 12; pts.push([60 + 16 * Math.cos(t), 60 + 16 * Math.sin(t)]) }
                 pts.push([74, 44]); arrow(ctx, pts, black, 7); break
        }
        case 14: danger(ctx)
                 line(ctx, [[30, 59], [70, 59]], black, 4)
                 poly(ctx, [[50, 37], [58, 47], [55.5, 47], [55.5, 62], [58, 72], [50, 68.5], [42, 72],
                            [44.5, 62], [44.5, 47], [42, 47]], black); break
        case 15: danger(ctx)
                 line(ctx, [[26, 73], [74, 73]], black, 6)
                 ctx.save(); ctx.translate(50, 73); ctx.scale(1, 0.85)
                 ctx.fillStyle = black; ctx.beginPath(); ctx.arc(0, 0, 15, Math.PI, 2 * Math.PI); ctx.fill(); ctx.restore(); break
        default: danger(ctx)
                 ctx.fillStyle = black; ctx.fillRect(45, 40, 10, 22); ctx.fillRect(45, 66, 10, 8); break
        }
    }

    onPaint: {
        var ctx = getContext("2d")
        ctx.reset()
        ctx.clearRect(0, 0, width, height)
        ctx.scale(width / 100, height / 100)
        if (code < 0) {
            disc(ctx, 48, ghost); disc(ctx, 38, black)
        } else {
            drawSign(ctx, code)
            snapToPalette(ctx)
        }
    }

    function snapToPalette(ctx) {
        var pal = ["#FF0000", "#FFE000", "#0038FF", "#FFFFFF", "#000000"]
        var rgb = [[255, 0, 0], [255, 224, 0], [0, 56, 255], [255, 255, 255], [0, 0, 0]]
        ctx.resetTransform()
        var w = Math.round(width), h = Math.round(height)
        var d = ctx.getImageData(0, 0, w, h).data
        var used = []
        for (var q = 0; q < w * h * 4; q += 4) {
            if (d[q + 3] !== 255) continue
            for (var m = 0; m < rgb.length; ++m)
                if (used.indexOf(m) < 0 && d[q] === rgb[m][0] && d[q + 1] === rgb[m][1] && d[q + 2] === rgb[m][2]) used.push(m)
        }
        var idx = []
        for (var i = 0; i < w * h; ++i) {
            var p = i * 4
            if (d[p + 3] < 128) { idx.push(-1); continue }
            var best = 0, bd = 1e9
            for (var n = 0; n < used.length; ++n) {
                var k = used[n]
                var dr = d[p] - rgb[k][0], dg = d[p + 1] - rgb[k][1], db = d[p + 2] - rgb[k][2]
                var e = dr * dr + dg * dg + db * db
                if (e < bd) { bd = e; best = k }
            }
            idx.push(best)
        }
        ctx.clearRect(0, 0, w, h)
        for (var y = 0; y < h; ++y) {
            var x = 0
            while (x < w) {
                var c = idx[y * w + x], x1 = x + 1
                while (x1 < w && idx[y * w + x1] === c) ++x1
                if (c >= 0) { ctx.fillStyle = pal[c]; ctx.fillRect(x, y, x1 - x, 1) }
                x = x1
            }
        }
    }
}
