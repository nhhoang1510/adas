import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ApplicationWindow {
    id: root
    visible: true
    width: 480
    height: 320
    minimumWidth: 480
    minimumHeight: 320
    title: "ADAS 3.5\" HMI Cockpit"
    color: "#05070d"
    flags: Qt.Window

    // ─── Bảng màu chuẩn ───────────────────────────────────────────────────────
    readonly property color colGreen : "#00e676"
    readonly property color colAmber : "#f59e0b"
    readonly property color colRed   : "#ef4444"
    readonly property color colCyan  : "#00f0ff"
    readonly property color colDim   : "#94a3b8"

    // Lệch làn: trả về màu thanh tương ứng
    function laneBarColor(side) {
        var st = adasBackend.laneStatus
        if (st === "DEPARTURE_LEFT"  && side === "left")  return colRed
        if (st === "DEPARTURE_RIGHT" && side === "right") return colRed
        if (st === "DRIFT_LEFT"      && side === "left")  return colAmber
        if (st === "DRIFT_RIGHT"     && side === "right") return colAmber
        return colGreen
    }

    // Thanh sáng lên hay không
    function laneBarActive(side) {
        var st = adasBackend.laneStatus
        if (side === "left"  && (st === "DEPARTURE_LEFT"  || st === "DRIFT_LEFT"))  return true
        if (side === "right" && (st === "DEPARTURE_RIGHT" || st === "DRIFT_RIGHT")) return true
        return false
    }

    property int frameCounter: 0

    Connections {
        target: adasWorker
        function onFrame_ready(frameId) {
            root.frameCounter = frameId
            videoFeed.source = "image://camera/frame?id=" + frameId
            if (adasBackend.hasLead) {
                cropThumb.source = "image://camera/crop?id=" + frameId
            }
        }
    }

    // ─── BỐ CỤC: TRÁI 240 px CAMERA | PHẢI 240 px HUD ────────────────────────
    RowLayout {
        anchors.fill: parent
        spacing: 0

        // ─── NỬA TRÁI: CAMERA ────────────────────────────────────────────────
        Item {
            Layout.preferredWidth: 240
            Layout.fillHeight: true

            Image {
                id: videoFeed
                anchors.fill: parent
                fillMode: Image.PreserveAspectCrop
                cache: false
                source: "image://camera/frame?id=0"
            }

            // Thanh trên: nhãn CAMERA + FPS
            Rectangle {
                anchors.top: parent.top
                anchors.left: parent.left
                anchors.right: parent.right
                height: 22
                color: "#bb05070d"

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 6
                    anchors.rightMargin: 6
                    spacing: 4

                    Rectangle { width: 5; height: 5; radius: 2.5; color: colCyan }
                    Text {
                        text: "CAMERA AI"
                        color: colCyan
                        font.pixelSize: 8
                        font.bold: true
                    }
                    Item { Layout.fillWidth: true }
                    Text {
                        text: adasBackend.fps.toFixed(0) + " FPS"
                        color: adasBackend.fps >= 25 ? colGreen : (adasBackend.fps >= 15 ? colAmber : colRed)
                        font.pixelSize: 8
                        font.bold: true
                    }
                }
            }

            // Cảnh báo va chạm / lệch làn ở đáy camera
            Rectangle {
                visible: adasBackend.threatLevel > 0 || adasBackend.laneStatus !== "NORMAL"
                anchors.bottom: parent.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                height: 22
                color: adasBackend.threatLevel === 2 ? "#cc dc2626" : "#cc d97706"

                Text {
                    anchors.centerIn: parent
                    text: adasBackend.warningText !== "" ? adasBackend.warningText : adasBackend.laneText
                    color: "#ffffff"
                    font.pixelSize: 8
                    font.bold: true
                    elide: Text.ElideRight
                }
            }
        }

        // ─── ĐƯỜNG KẺ NGĂN ───────────────────────────────────────────────────
        Rectangle {
            Layout.preferredWidth: 1
            Layout.fillHeight: true
            color: "#1e293b"
        }

        // ─── NỬA PHẢI: HUD ADAS ──────────────────────────────────────────────
        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true

            // Nền đen AMOLED
            Rectangle { anchors.fill: parent; color: "#020408" }

            // ─── THANH TRÊN: BIỂN BÁO + TỐC ĐỘ ─────────────────────────────
            RowLayout {
                id: topBar
                anchors.top: parent.top
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.topMargin: 6
                anchors.leftMargin: 8
                anchors.rightMargin: 8
                height: 48
                spacing: 6

                // Biển báo tốc độ (hình tròn đỏ trắng)
                Rectangle {
                    visible: adasBackend.hasSign
                    width: 42; height: 42; radius: 21
                    color: "#ffffff"
                    border.color: colRed
                    border.width: 5

                    Text {
                        anchors.centerIn: parent
                        text: {
                            var c = adasBackend.signCode
                            if (c.indexOf("P.") !== -1) return c.replace("P.", "")
                            return c !== "" ? c : "60"
                        }
                        color: "#000000"
                        font.pixelSize: 14
                        font.bold: true
                    }
                }

                // Placeholder khi chưa có biển
                Item {
                    visible: !adasBackend.hasSign
                    width: 42; height: 42
                    Rectangle {
                        anchors.centerIn: parent
                        width: 36; height: 36; radius: 18
                        color: "#0f172a"
                        border.color: "#334155"; border.width: 2
                        Text {
                            anchors.centerIn: parent
                            text: "🚫"
                            font.pixelSize: 14
                        }
                    }
                }

                // Tên biển báo
                ColumnLayout {
                    spacing: 1
                    Layout.fillWidth: true

                    Text {
                        text: adasBackend.hasSign ? adasBackend.signName : "Chưa phát hiện biển"
                        color: adasBackend.hasSign ? "#ffffff" : colDim
                        font.pixelSize: 9
                        font.bold: true
                        elide: Text.ElideRight
                        Layout.fillWidth: true
                    }
                    Text {
                        text: adasBackend.hasSign ? ("Code: " + adasBackend.signCode) : "Đang quét..."
                        color: colDim
                        font.pixelSize: 7
                        Layout.fillWidth: true
                    }
                }

                // Tốc độ xe
                ColumnLayout {
                    spacing: -2
                    Text {
                        text: adasBackend.speed > 0 ? adasBackend.speed : "--"
                        color: "#ffffff"
                        font.pixelSize: 26
                        font.bold: true
                        font.italic: true
                        Layout.alignment: Qt.AlignRight
                    }
                    Text {
                        text: "km/h"
                        color: colDim
                        font.pixelSize: 8
                        font.italic: true
                        Layout.alignment: Qt.AlignRight
                    }
                }
            }

            // ─── VÙNG GIỮA: THANH LÀN TRÁI + XE + THANH LÀN PHẢI ───────────
            Item {
                id: midZone
                anchors.top: topBar.bottom
                anchors.bottom: bottomBar.top
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.topMargin: 4

                // ── THANH LÀN TRÁI ──────────────────────────────────────────
                Column {
                    id: leftLaneBars
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.left: parent.left
                    anchors.leftMargin: 6
                    spacing: 4

                    Repeater {
                        model: 5
                        Rectangle {
                            width: 14 + index * 3
                            height: 6
                            radius: 3
                            // Sáng dần từ trên xuống (index 0 mờ nhất)
                            opacity: laneBarActive("left") ? (0.4 + index * 0.15) : 0.18
                            color: laneBarColor("left")

                            Behavior on color    { ColorAnimation  { duration: 200 } }
                            Behavior on opacity  { NumberAnimation { duration: 200 } }

                            // Hiệu ứng nháy khi cảnh báo đỏ
                            SequentialAnimation on opacity {
                                running: adasBackend.laneStatus === "DEPARTURE_LEFT"
                                loops: Animation.Infinite
                                NumberAnimation { to: 1.0; duration: 180 }
                                NumberAnimation { to: 0.2; duration: 180 }
                            }
                        }
                    }

                    // Nhãn L
                    Text {
                        text: "L"
                        color: laneBarColor("left")
                        font.pixelSize: 9
                        font.bold: true
                        Behavior on color { ColorAnimation { duration: 200 } }
                    }
                }

                // ── THANH LÀN PHẢI ──────────────────────────────────────────
                Column {
                    id: rightLaneBars
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.right: parent.right
                    anchors.rightMargin: 6
                    spacing: 4

                    Repeater {
                        model: 5
                        Rectangle {
                            width: 14 + (4 - index) * 3   // Rộng dần từ dưới lên
                            height: 6
                            radius: 3
                            anchors.right: parent ? parent.right : undefined
                            opacity: laneBarActive("right") ? (0.4 + (4 - index) * 0.15) : 0.18
                            color: laneBarColor("right")

                            Behavior on color    { ColorAnimation  { duration: 200 } }
                            Behavior on opacity  { NumberAnimation { duration: 200 } }

                            SequentialAnimation on opacity {
                                running: adasBackend.laneStatus === "DEPARTURE_RIGHT"
                                loops: Animation.Infinite
                                NumberAnimation { to: 1.0; duration: 180 }
                                NumberAnimation { to: 0.2; duration: 180 }
                            }
                        }
                    }

                    Text {
                        text: "R"
                        color: laneBarColor("right")
                        font.pixelSize: 9
                        font.bold: true
                        Behavior on color { ColorAnimation { duration: 200 } }
                    }
                }

                // ── TRUNG TÂM: ICON XE + KHOẢNG CÁCH ────────────────────────
                Column {
                    anchors.centerIn: parent
                    spacing: 4

                    // Hình xe phía trước
                    Item {
                        width: 54; height: 50
                        anchors.horizontalCenter: parent.horizontalCenter

                        Rectangle {
                            anchors.fill: parent
                            radius: 10
                            color: "transparent"
                            border.color: {
                                if (!adasBackend.hasLead)           return "#334155"
                                if (adasBackend.threatLevel === 2)  return colRed
                                if (adasBackend.threatLevel === 1)  return colAmber
                                return colGreen
                            }
                            border.width: 2

                            Behavior on border.color { ColorAnimation { duration: 200 } }

                            SequentialAnimation on border.color {
                                running: adasBackend.threatLevel === 2
                                loops: Animation.Infinite
                                ColorAnimation { from: colRed; to: "#ffffff"; duration: 200 }
                                ColorAnimation { from: "#ffffff"; to: colRed; duration: 200 }
                            }
                        }

                        // Thumbnail xe thật nếu có
                        Image {
                            id: cropThumb
                            anchors.fill: parent
                            anchors.margins: 4
                            fillMode: Image.PreserveAspectCrop
                            visible: adasBackend.hasLead
                            source: "image://camera/crop?id=0"
                            opacity: 0.9
                        }

                        // Icon xe mặc định khi chưa phát hiện
                        Text {
                            anchors.centerIn: parent
                            visible: !adasBackend.hasLead
                            text: "🚗"
                            font.pixelSize: 22
                            opacity: 0.4
                        }
                    }

                    // Khoảng cách to
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: {
                            if (adasBackend.hasLead && adasBackend.leadDist > 0)
                                return adasBackend.leadDist.toFixed(1) + " m"
                            return "-- m"
                        }
                        color: {
                            if (!adasBackend.hasLead)              return colDim
                            if (adasBackend.leadDist <= 12)        return colRed
                            if (adasBackend.leadDist <= 22)        return colAmber
                            return "#ffffff"
                        }
                        font.pixelSize: 28
                        font.bold: true
                        font.family: "Segoe UI, Arial"

                        Behavior on color { ColorAnimation { duration: 200 } }
                    }

                    // TTC nhỏ bên dưới
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: adasBackend.leadTtc !== "--" ? "TTC " + adasBackend.leadTtc : "KHOẢNG CÁCH"
                        color: colDim
                        font.pixelSize: 8
                        font.bold: true
                    }
                }
            }

            // ─── THANH DƯỚI: TRẠNG THÁI LÀN + NÚT TẠM DỪNG ─────────────────
            Rectangle {
                id: bottomBar
                anchors.bottom: parent.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                height: 24
                color: {
                    var st = adasBackend.laneStatus
                    if (st === "DEPARTURE_LEFT" || st === "DEPARTURE_RIGHT") return "#b91c1c"
                    if (st === "DRIFT_LEFT"     || st === "DRIFT_RIGHT")     return "#92400e"
                    return "#0f172a"
                }

                Behavior on color { ColorAnimation { duration: 250 } }

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 8
                    anchors.rightMargin: 8
                    spacing: 6

                    Text {
                        text: {
                            var st = adasBackend.laneStatus
                            if (st === "DEPARTURE_LEFT")  return "⚠ LỆCH TRÁI"
                            if (st === "DEPARTURE_RIGHT") return "⚠ LỆCH PHẢI"
                            if (st === "DRIFT_LEFT")      return "↙ Bám mép trái"
                            if (st === "DRIFT_RIGHT")     return "↘ Bám mép phải"
                            if (st === "NO_LANE")         return "Mất vạch làn"
                            return "✓ Giữ làn tốt"
                        }
                        color: "#ffffff"
                        font.pixelSize: 8
                        font.bold: true
                        Layout.fillWidth: true
                        elide: Text.ElideRight
                    }

                    // Nút tạm dừng
                    Rectangle {
                        width: 22; height: 18; radius: 4; color: "#1e293b"
                        Text { anchors.centerIn: parent; text: "⏯"; color: "#fff"; font.pixelSize: 10 }
                        MouseArea { anchors.fill: parent; onClicked: adasBackend.togglePlay() }
                    }
                }
            }
        }
    }

    // ─── VIỀN ĐỎ NHẤP NHÁY TOÀN MÀN HÌNH KHI CẤP 2 ────────────────────────
    Rectangle {
        id: dangerBorder
        anchors.fill: parent
        color: "transparent"
        border.color: colRed
        border.width: 4
        visible: adasBackend.threatLevel === 2
        opacity: 0.8

        SequentialAnimation on opacity {
            running: dangerBorder.visible
            loops: Animation.Infinite
            NumberAnimation { from: 0.2; to: 1.0; duration: 220 }
            NumberAnimation { from: 1.0; to: 0.2; duration: 220 }
        }
    }
}
