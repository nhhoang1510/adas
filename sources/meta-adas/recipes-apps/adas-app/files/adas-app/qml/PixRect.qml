import QtQuick 2.12

Canvas {
    id: pr
    property color color: "#FFFFFF"
    property real  radius: 0
    property color borderColor: "transparent"
    property real  borderWidth: 0
    property real  shapeWidth: width
    property real  shapeHeight: height
    property real  rotationDeg: 0

    antialiasing: false
    renderStrategy: Canvas.Immediate
    onColorChanged: requestPaint()
    onBorderColorChanged: requestPaint()
    onWidthChanged: requestPaint()
    onHeightChanged: requestPaint()
    onRadiusChanged: requestPaint()
    onRotationDegChanged: requestPaint()

    function roundRect(ctx, x, y, w, h, r) {
        r = Math.max(0, Math.min(r, w / 2, h / 2))
        if (r <= 0) { ctx.rect(x, y, w, h); return }
        ctx.roundedRect(x, y, w, h, r, r)
    }

    onPaint: {
        var ctx = getContext("2d")
        ctx.reset()
        ctx.clearRect(0, 0, width, height)
        ctx.translate(width / 2, height / 2)
        if (rotationDeg !== 0) ctx.rotate(rotationDeg * Math.PI / 180)
        var w = shapeWidth, h = shapeHeight
        ctx.beginPath()
        if (borderWidth > 0) {
            ctx.fillStyle = borderColor
            ctx.fillRule = Qt.OddEvenFill
            roundRect(ctx, -w / 2, -h / 2, w, h, radius)
            roundRect(ctx, -w / 2 + borderWidth, -h / 2 + borderWidth, w - 2 * borderWidth, h - 2 * borderWidth,
                      Math.max(0, radius - borderWidth))
        } else {
            ctx.fillStyle = pr.color
            roundRect(ctx, -w / 2, -h / 2, w, h, radius)
        }
        ctx.fill()
    }
}
