import QtQuick 2.12

Canvas {
    id: tt
    property color c: "#070D10"
    property size  design: Qt.size(48, 34)
    property var   draw: null

    renderStrategy: Canvas.Immediate
    antialiasing: false
    onCChanged: requestPaint()
    onWidthChanged: requestPaint()
    onHeightChanged: requestPaint()

    onPaint: {
        var ctx = getContext("2d")
        ctx.reset()
        ctx.clearRect(0, 0, width, height)
        ctx.scale(width / design.width, height / design.height)
        ctx.fillStyle = tt.c
        ctx.beginPath()
        if (tt.draw) tt.draw(ctx)
    }
}
