import QtQuick 2.12

Item {
    id: disp

    property string text: ""
    property int    digits: 3
    property real   digitWidth: 30
    property real   digitHeight: 50
    property real   spacing: 6
    property color  onColor: "#00E5FF"
    property color  offColor: "#070D10"
    property bool   dots: true
    property real   skew: 0.20
    property bool   dash: false
    property real   dashY: digitHeight / 2
    property real   dashThick: 0.09
    property real   dashLen: 0.65
    property int    sweep: -1
    readonly property real _over: skew * digitHeight

    readonly property bool showsNumber: !dash && (sweep >= 0 || /[0-9]/.test(text))

    width: digits * digitWidth + (digits - 1) * spacing + _over
    height: digitHeight

    readonly property var cells: {
        var out = []
        for (var i = 0; i < text.length; ++i) {
            var ch = text.charAt(i)
            if (ch === ".") {
                if (out.length > 0) out[out.length - 1].dot = true
                else out.push({ c: " ", dot: true })
            } else {
                out.push({ c: ch, dot: false })
            }
        }
        while (out.length < digits) out.unshift({ c: " ", dot: false })
        if (out.length > digits) out = out.slice(out.length - digits)
        return out
    }

    Row {
        spacing: disp.spacing - disp._over
        Repeater {
            model: disp.digits
            SevenSegmentDigit {
                width: disp.digitWidth + disp._over
                height: disp.digitHeight
                skew: disp.skew
                digit: disp.dash ? " " : disp.cells[index].c
                sweep: disp.sweep
                showDot: disp.dots
                dotOn: disp.cells[index].dot
                onColor: disp.onColor
                offColor: disp.showsNumber ? "#000000" : disp.offColor
            }
        }
    }

    Canvas {
        anchors.fill: parent
        visible: disp.dash
        antialiasing: false
        renderStrategy: Canvas.Immediate
        readonly property color c: disp.onColor
        onCChanged: requestPaint()
        onVisibleChanged: requestPaint()
        onWidthChanged: requestPaint()
        readonly property real dy: disp.dashY
        onDyChanged: requestPaint()
        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            ctx.clearRect(0, 0, width, height)
            var h = height, t = Math.max(2, Math.round(h * disp.dashThick)), m = Math.round(dy), sh = disp.skew * (h - m)
            var x1 = width - disp._over + sh - t * 0.3
            var x0 = x1 - (x1 - (sh + t * 0.3)) * disp.dashLen
            ctx.beginPath()
            ctx.moveTo(x0, m); ctx.lineTo(x0 + t / 2, m - t / 2); ctx.lineTo(x1 - t / 2, m - t / 2)
            ctx.lineTo(x1, m); ctx.lineTo(x1 - t / 2, m + t / 2); ctx.lineTo(x0 + t / 2, m + t / 2)
            ctx.closePath()
            ctx.fillStyle = c
            ctx.fill()
        }
    }
}
