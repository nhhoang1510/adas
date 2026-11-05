import QtQuick 2.12

Canvas {
    id: cell

    property string digit: "8"
    property color  onColor: "#00E5FF"
    property color  offColor: "#070D10"
    property bool   showDot: false
    property bool   dotOn: showDot
    property color  dotColor: onColor
    property real   skew: 0.20
    property real   thickness: Math.max(2, height * 0.15)

    implicitWidth: 30
    implicitHeight: 50
    antialiasing: false
    renderStrategy: Canvas.Immediate

    readonly property var _map: ({
        "0": "abcdef", "1": "bc",    "2": "abdeg",  "3": "abcdg",  "4": "bcfg",
        "5": "acdfg",  "6": "acdefg","7": "abc",    "8": "abcdefg","9": "abcdfg",
        "-": "g",      " ": "",      "E": "adefg",  "r": "eg",     "A": "abcefg",
        "C": "adef",   "F": "aefg",  "H": "bcefg",  "L": "def",    "P": "abefg",
        "o": "cdeg",   "n": "ceg",   "U": "bcdef",  "_": "d"
    })
    property int    sweep: -1
    readonly property string segs: sweep >= 0 ? "abcdefg".slice(0, sweep)
                                 : _map[digit] !== undefined ? _map[digit] : ""

    readonly property real _dotW: showDot ? thickness * 1.5 : 0
    readonly property real _w: width - skew * height - _dotW
    readonly property real _h: height
    readonly property real _t: thickness
    readonly property real _g: Math.max(1, thickness * 0.12)
    readonly property real _k: thickness * 0.7

    onSegsChanged: requestPaint()
    onOnColorChanged: requestPaint()
    onOffColorChanged: requestPaint()
    onDotOnChanged: requestPaint()
    onDotColorChanged: requestPaint()
    onWidthChanged: requestPaint()
    onHeightChanged: requestPaint()

    function _segments(w, h, t, g, k) {
        var m = h / 2
        var a = [[k, 0], [w - k, 0], [w - (k + g) / 2, (k - g) / 2], [w - t - g, t], [t + g, t], [(k + g) / 2, (k - g) / 2]]
        var f = [[0, k], [(k - g) / 2, (k + g) / 2], [t, t + g], [t, m - t / 2 - g], [t / 2, m - g], [0, m - g]]
        var mx = function (pts) { return pts.map(function (p) { return [w - p[0], p[1]] }) }
        var my = function (pts) { return pts.map(function (p) { return [p[0], h - p[1]] }) }
        return {
            "a": a, "d": my(a), "f": f, "b": mx(f), "e": my(f), "c": mx(my(f)),
            "g": [[g + t / 2, m], [t + g, m - t / 2], [w - t - g, m - t / 2], [w - g - t / 2, m],
                  [w - t - g, m + t / 2], [t + g, m + t / 2]]
        }
    }

    onPaint: {
        var ctx = getContext("2d")
        ctx.reset()
        ctx.clearRect(0, 0, width, height)
        var seg = _segments(_w, _h, _t, _g, _k)
        ctx.save()
        ctx.transform(1, 0, -skew, 1, skew * _h, 0)
        for (var pass = 0; pass < 2; ++pass) {
            ctx.beginPath()
            for (var name in seg) {
                if ((segs.indexOf(name) >= 0) !== (pass === 1)) continue
                var p = seg[name]
                ctx.moveTo(p[0][0], p[0][1])
                for (var i = 1; i < p.length; ++i) ctx.lineTo(p[i][0], p[i][1])
                ctx.closePath()
            }
            ctx.fillStyle = pass === 1 ? onColor : offColor
            ctx.fill()
        }
        ctx.restore()
        if (showDot) {
            var d = _t * 1.05, x0 = width - _dotW + (_dotW - d) / 2
            ctx.beginPath()
            ctx.moveTo(x0 + skew * d, height - d); ctx.lineTo(x0 + d + skew * d, height - d)
            ctx.lineTo(x0 + d, height); ctx.lineTo(x0, height); ctx.closePath()
            ctx.fillStyle = dotOn ? dotColor : offColor
            ctx.fill()
        }
    }
}
