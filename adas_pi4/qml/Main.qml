import QtQuick
import QtQuick.Window
import QtQuick.Shapes

// Giao diện táp-lô ADAS cho màn hình 3.5 inch NGANG (480x320)
Window {
    id: root
    width: 480
    height: 320
    visible: true
    color: "#0b0d10"
    title: "ADAS Dashboard"

    // ===== Dữ liệu (nối với backend C++/CAN/GPS ở đây) =====
    property int speed: 49              // km/h
    property int speedLimit: 60         // biển giới hạn tốc độ
    property real distance: 5.0         // khoảng cách tới xe phía trước (m)
    property bool laneDeparture: false  // true khi xe lệch làn

    readonly property real showCarDistance: 5.0   // từ <= 5 m mới hiện biểu tượng ô tô
    readonly property real warnDistance: 2.0      // < 2 m: cảnh báo đỏ

    readonly property color okColor: "#2ecc40"
    readonly property color warnColor: "#ff3b30"
    readonly property color laneColor: laneDeparture ? warnColor : okColor
    readonly property bool carVisible: distance > 0 && distance <= showCarDistance
    readonly property bool collisionWarning: carVisible && distance < warnDistance
    readonly property bool overSpeed: speed > speedLimit

    // Hệ số co giãn theo kích thước màn hình (thiết kế gốc 480x320)
    readonly property real u: Math.min(width / 480, height / 320)

    // ===== Hai vạch kẻ làn =====
    Canvas {
        id: laneCanvas
        anchors { left: parent.left; right: parent.right; top: parent.top; bottom: distanceText.top }

        property color lane: root.laneColor
        onLaneChanged: requestPaint()
        onWidthChanged: requestPaint()
        onHeightChanged: requestPaint()

        onPaint: {
            var ctx = getContext("2d")
            var w = width, h = height
            ctx.reset()

            var vx = w / 2              // điểm tụ (nằm phía trên canvas cho đỡ chụm)
            var vy = -h * 0.60
            var topY = h * 0.06         // điểm kết thúc của vạch
            var laneOffset = w * 0.34   // khoảng cách từ tâm tới vạch ở đáy
            var laneThick = w * 0.075   // độ dày vạch ở đáy

            function xAt(side, y) {
                return vx + side * laneOffset * (y - vy) / (h - vy)
            }
            function thickAt(y) {
                return laneThick * (y - vy) / (h - vy)
            }

            // gradient trong suốt dần về phía xa
            var grad = ctx.createLinearGradient(0, h, 0, topY)
            grad.addColorStop(0.0, Qt.rgba(lane.r, lane.g, lane.b, 1.0))
            grad.addColorStop(0.55, Qt.rgba(lane.r, lane.g, lane.b, 0.85))
            grad.addColorStop(1.0, Qt.rgba(lane.r, lane.g, lane.b, 0.0))
            ctx.fillStyle = grad

            for (var s = -1; s <= 1; s += 2) {
                var tb = thickAt(h), tt = thickAt(topY)
                ctx.beginPath()
                ctx.moveTo(xAt(s, h) - tb / 2, h)
                ctx.lineTo(xAt(s, topY) - tt / 2, topY)
                ctx.lineTo(xAt(s, topY) + tt / 2, topY)
                ctx.lineTo(xAt(s, h) + tb / 2, h)
                ctx.closePath()
                ctx.fill()
            }
        }
    }

    // ===== Biểu tượng ô tô (nét viền, vector): chỉ hiện khi khoảng cách <= 5 m =====
    Item {
        id: carIcon
        property color carColor: root.collisionWarning ? root.warnColor : root.okColor

        width: laneCanvas.height * 0.72
        height: width * 0.8
        x: (root.width - width) / 2
        y: laneCanvas.height * 0.20

        opacity: root.carVisible ? 1 : 0
        visible: opacity > 0
        Behavior on opacity { NumberAnimation { duration: 200 } }

        layer.enabled: true     // khử răng cưa
        layer.samples: 4

        Shape {
            width: 100
            height: 80
            scale: carIcon.width / 100
            transformOrigin: Item.TopLeft

            // thân xe, mui xe, đèn hậu, biển số
            ShapePath {
                strokeColor: carIcon.carColor
                strokeWidth: 5
                fillColor: "transparent"
                capStyle: ShapePath.RoundCap
                joinStyle: ShapePath.RoundJoin

                PathSvg { path: "M 17 32 L 83 32 Q 92 32 92 41 L 92 54 Q 92 63 83 63 L 17 63 Q 8 63 8 54 L 8 41 Q 8 32 17 32 Z" }
                PathSvg { path: "M 20 32 L 27 13 Q 28 9 33 9 L 67 9 Q 72 9 73 13 L 80 32" }
                PathSvg { path: "M 17 45 L 31 45 M 69 45 L 83 45" }
                PathSvg { path: "M 40 52 L 60 52" }
            }

            // bánh xe
            ShapePath {
                strokeColor: carIcon.carColor
                strokeWidth: 9
                fillColor: "transparent"
                capStyle: ShapePath.RoundCap

                PathSvg { path: "M 24 67 L 24 71 M 76 67 L 76 71" }
            }
        }
    }

    // ===== Biển giới hạn tốc độ (trái) =====
    Rectangle {
        width: 96 * root.u
        height: width
        radius: width / 2
        color: "white"
        border { color: "#e3170a"; width: 10 * root.u }
        anchors { left: parent.left; leftMargin: 24 * root.u; top: parent.top; topMargin: 20 * root.u }

        Text {
            anchors.centerIn: parent
            text: root.speedLimit
            color: "black"
            font { pixelSize: 44 * root.u; bold: true }
        }
    }

    // ===== Tốc độ hiện tại (phải) =====
    Column {
        anchors { right: parent.right; rightMargin: 24 * root.u; top: parent.top; topMargin: 20 * root.u }
        spacing: -6 * root.u

        Text {
            anchors.right: parent.right
            text: root.speed
            color: root.overSpeed ? root.warnColor : "white"
            font { pixelSize: 72 * root.u; bold: true }
        }
        Text {
            anchors.right: parent.right
            text: "km/h"
            color: "#cfd3d8"
            font.pixelSize: 20 * root.u
        }
    }

    // ===== Cảnh báo va chạm: chỉ hiện khi khoảng cách quá gần =====
    Text {
        anchors { left: parent.left; leftMargin: 24 * root.u; bottom: parent.bottom; bottomMargin: 16 * root.u }
        text: "FCW"
        color: root.warnColor
        font { pixelSize: 24 * root.u; bold: true }
        visible: root.collisionWarning
    }

    // ===== Khoảng cách tới xe phía trước (m) =====
    Row {
        id: distanceText
        anchors { horizontalCenter: parent.horizontalCenter; bottom: parent.bottom; bottomMargin: 6 * root.u }
        spacing: 6 * root.u

        Text {
            id: distanceValue
            text: root.distance.toFixed(1)
            color: root.collisionWarning ? root.warnColor : "white"
            font { pixelSize: 60 * root.u; bold: true }
        }
        Text {
            text: "m"
            color: "#cfd3d8"
            font.pixelSize: 22 * root.u
            topPadding: 32 * root.u
        }
    }

    // ===== Demo: giả lập số liệu (xoá khi nối dữ liệu thật) =====
    Timer {
        interval: 1000
        running: true
        repeat: true
        onTriggered: {
            root.speed = Math.max(0, Math.min(120, root.speed + Math.round(Math.random() * 6 - 3)))
            root.distance = Math.max(0.5, Math.min(9.9, root.distance + (Math.random() * 1.6 - 0.8)))
        }
    }
}
