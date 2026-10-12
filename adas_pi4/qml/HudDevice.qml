import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: hudRoot

    // Bảng màu Automotive HUD
    readonly property color colNeonGreen: "#00e676"
    readonly property color colAmber: "#f59e0b"
    readonly property color colRed: "#ef4444"
    readonly property color colCyan: "#00f0ff"
    readonly property color colTextMuted: "#94a3b8"

    // Animation nhấp nháy khi cảnh báo
    property bool blinkState: false
    Timer {
        interval: 350
        running: true
        repeat: true
        onTriggered: hudRoot.blinkState = !hudRoot.blinkState
    }

    // =========================================================================
    // KHUNG THIẾT BỊ VẬT LÝ KIM LOẠI BẠC (METALLIC SILVER CASING)
    // Tự động co giãn theo kích thước thực tế của khung nhúng, không bao giờ bị khuất
    // =========================================================================
    readonly property real availW: Math.max(100, hudRoot.width - 24)
    readonly property real availH: Math.max(100, hudRoot.height - 24)

    // Tỷ lệ thiết bị chuẩn ~ 1:1.06 như ảnh chụp thực tế
    readonly property real casingW: Math.min(availW, availH * 0.92)
    readonly property real casingH: Math.min(availH, casingW * 1.05)

    // Hệ số tỷ lệ động (chuẩn hóa theo 340px)
    readonly property real sc: Math.max(0.5, Math.min(1.3, casingW / 340.0))

    Rectangle {
        id: casing
        anchors.centerIn: parent
        width: Math.round(casingW)
        height: Math.round(casingH)
        radius: Math.round(24 * sc)
        color: "#282e3a"
        border.color: "#9ca3af"
        border.width: Math.max(2, Math.round(4 * sc))

        // Đổ bóng & viền kim loại sáng bên trong
        Rectangle {
            anchors.fill: parent
            anchors.margins: Math.max(2, Math.round(3 * sc))
            radius: Math.round(20 * sc)
            color: "transparent"
            border.color: "#475569"
            border.width: 1
        }

        // =====================================================================
        // 3 NÚT VẬT LÝ TRÊN ĐỈNH MÁY: [-]  [::]  [+]
        // =====================================================================
        RowLayout {
            anchors.top: parent.top
            anchors.topMargin: Math.round(6 * sc)
            anchors.horizontalCenter: parent.horizontalCenter
            spacing: Math.round(40 * sc)

            // Nút [-]
            Rectangle {
                width: Math.round(18 * sc)
                height: Math.max(2, Math.round(3 * sc))
                radius: 1.5
                color: "#e2e8f0"
            }

            // Nút [::] Menu lưới 4 chấm
            Grid {
                columns: 2
                spacing: Math.max(1, Math.round(2 * sc))
                Rectangle { width: Math.max(2, Math.round(3 * sc)); height: width; radius: 1; color: "#e2e8f0" }
                Rectangle { width: Math.max(2, Math.round(3 * sc)); height: width; radius: 1; color: "#e2e8f0" }
                Rectangle { width: Math.max(2, Math.round(3 * sc)); height: width; radius: 1; color: "#e2e8f0" }
                Rectangle { width: Math.max(2, Math.round(3 * sc)); height: width; radius: 1; color: "#e2e8f0" }
            }

            // Nút [+]
            Item {
                width: Math.round(12 * sc)
                height: Math.round(12 * sc)
                Rectangle {
                    anchors.centerIn: parent
                    width: Math.round(12 * sc)
                    height: Math.max(2, Math.round(2.5 * sc))
                    radius: 1
                    color: "#e2e8f0"
                }
                Rectangle {
                    anchors.centerIn: parent
                    width: Math.max(2, Math.round(2.5 * sc))
                    height: Math.round(12 * sc)
                    radius: 1
                    color: "#e2e8f0"
                }
            }
        }

        // =====================================================================
        // MẶT KÍNH MÀN HÌNH ĐEN SÂU AMOLED (DEEP BLACK SCREEN)
        // =====================================================================
        Rectangle {
            id: screen
            anchors.fill: parent
            anchors.topMargin: Math.round(20 * sc)
            anchors.bottomMargin: Math.round(10 * sc)
            anchors.leftMargin: Math.round(10 * sc)
            anchors.rightMargin: Math.round(10 * sc)
            radius: Math.round(16 * sc)
            color: "#020408"
            clip: true

            // Lưới hiển thị Canvas: Vẽ vạch làn đường bậc thang (Ladder Rungs)
            Canvas {
                id: roadCanvas
                anchors.fill: parent

                onPaint: {
                    var ctx = getContext("2d");
                    ctx.reset();

                    var w = width;
                    var h = height;

                    // Điểm tụ vanishing point
                    var vX = w * 0.5;
                    var vY = h * 0.36;

                    // Màu sắc làn đường trái/phải theo trạng thái LDW
                    var isDepartLeft = adasBackend.laneStatus === "DEPART_LEFT";
                    var isDepartRight = adasBackend.laneStatus === "DEPART_RIGHT";

                    var colLeft = isDepartLeft ? (hudRoot.blinkState ? "#ff1744" : "#ff5252") : "#00e676";
                    var colRight = isDepartRight ? (hudRoot.blinkState ? "#ff1744" : "#ff5252") : "#00e676";

                    // Vẽ các thanh ngang bậc thang
                    var steps = 14;
                    for (var i = 0; i < steps; i++) {
                        var t = i / (steps - 1.0); // 0 (xa) -> 1 (gần)
                        var t_cubed = t * t * 0.8 + t * 0.2;

                        var y = vY + t_cubed * (h * 0.86 - vY);

                        // Độ mở của 2 làn
                        var spread = (28 + t_cubed * 82) * sc;
                        var barWidth = (6 + t_cubed * 20) * sc;
                        var barHeight = Math.max(1.5, (2.0 + t_cubed * 4.0) * sc);

                        // Thanh bên trái (nghiêng phối cảnh)
                        ctx.fillStyle = colLeft;
                        ctx.save();
                        ctx.translate(vX - spread, y);
                        ctx.rotate(0.32);
                        ctx.fillRect(-barWidth / 2, -barHeight / 2, barWidth, barHeight);
                        ctx.restore();

                        // Thanh bên phải (nghiêng phối cảnh)
                        ctx.fillStyle = colRight;
                        ctx.save();
                        ctx.translate(vX + spread, y);
                        ctx.rotate(-0.32);
                        ctx.fillRect(-barWidth / 2, -barHeight / 2, barWidth, barHeight);
                        ctx.restore();
                    }
                }
            }

            // =================================================================
            // 1. GÓC TRÊN TRÁI: BIỂN BÁO TRÒN GIỚI HẠN TỐC ĐỘ (60)
            // =================================================================
            Rectangle {
                id: signCircle
                visible: adasBackend.hasSign
                anchors.top: parent.top
                anchors.topMargin: Math.round(8 * sc)
                anchors.left: parent.left
                anchors.leftMargin: Math.round(8 * sc)
                width: Math.round(50 * sc)
                height: width
                radius: width / 2
                color: "#ffffff"
                border.color: "#dc2626"
                border.width: Math.max(3, Math.round(5 * sc))

                Text {
                    anchors.centerIn: parent
                    text: {
                        if (adasBackend.hasSign && adasBackend.signCode.length > 0) {
                            var code = adasBackend.signCode;
                            var m = code.match(/\d+/);
                            return m ? m[0] : "60";
                        }
                        return "60";
                    }
                    color: "#0f172a"
                    font.family: "Arial"
                    font.pixelSize: Math.max(10, Math.round(20 * sc))
                    font.bold: true
                }
            }

            // =================================================================
            // 2. GÓC TRÊN GIỮA: ICON TRẠNG THÁI (RADAR 📡, ● REC, MIC 🎙️)
            // =================================================================
            RowLayout {
                anchors.top: parent.top
                anchors.topMargin: Math.round(10 * sc)
                anchors.horizontalCenter: parent.horizontalCenter
                spacing: Math.round(8 * sc)

                // Icon Radar/GPS xanh lá
                Text {
                    text: "📡"
                    font.pixelSize: Math.max(8, Math.round(12 * sc))
                    color: colNeonGreen
                }

                // Chữ ● REC đỏ
                RowLayout {
                    spacing: Math.max(1, Math.round(2 * sc))
                    Rectangle {
                        width: Math.max(3, Math.round(5 * sc))
                        height: width
                        radius: width / 2
                        color: colRed
                        opacity: hudRoot.blinkState ? 1.0 : 0.35
                    }
                    Text {
                        text: "REC"
                        color: colRed
                        font.pixelSize: Math.max(6, Math.round(9 * sc))
                        font.bold: true
                    }
                }

                // Icon Mic xanh lá
                Text {
                    text: "🎙️"
                    font.pixelSize: Math.max(8, Math.round(11 * sc))
                    color: colNeonGreen
                }
            }

            // =================================================================
            // 3. GÓC TRÊN PHẢI: VẬN TỐC (49 km/h)
            // =================================================================
            ColumnLayout {
                anchors.top: parent.top
                anchors.topMargin: Math.round(6 * sc)
                anchors.right: parent.right
                anchors.rightMargin: Math.round(10 * sc)
                spacing: -2

                Text {
                    Layout.alignment: Qt.AlignRight
                    text: Math.round(adasBackend.speed).toString()
                    color: "#ffffff"
                    font.family: "Arial"
                    font.pixelSize: Math.max(16, Math.round(28 * sc))
                    font.bold: true
                    font.italic: true
                }

                Text {
                    Layout.alignment: Qt.AlignRight
                    text: "km/h"
                    color: colTextMuted
                    font.pixelSize: Math.max(7, Math.round(9 * sc))
                    font.bold: true
                }
            }

            // =================================================================
            // 4. XE DẪN ĐẦU Ở TRUNG TÂM LÀN ĐƯỜNG (LEAD VEHICLE CONTOUR)
            // =================================================================
            Item {
                id: leadCarContainer
                anchors.centerIn: parent
                anchors.verticalCenterOffset: Math.round(8 * sc)
                width: Math.round(85 * sc)
                height: Math.round(58 * sc)

                // Viền xe ô tô dạ quang neon
                Rectangle {
                    id: carBody
                    anchors.centerIn: parent
                    width: Math.round(74 * sc)
                    height: Math.round(46 * sc)
                    radius: Math.round(10 * sc)
                    color: "#182234"
                    border.width: Math.max(1.5, Math.round(2.5 * sc))
                    border.color: {
                        if (adasBackend.threatLevel === "DANGER") {
                            return hudRoot.blinkState ? "#ff1744" : "#ff5252";
                        } else if (adasBackend.threatLevel === "WARNING") {
                            return "#f59e0b";
                        }
                        return colNeonGreen;
                    }

                    // Kính sau xe ô tô (Rear Windshield)
                    Rectangle {
                        anchors.top: parent.top
                        anchors.topMargin: Math.round(4 * sc)
                        anchors.horizontalCenter: parent.horizontalCenter
                        width: Math.round(48 * sc)
                        height: Math.round(16 * sc)
                        radius: Math.round(4 * sc)
                        color: "#0c1322"
                        border.color: carBody.border.color
                        border.width: 1
                    }

                    // Đèn hậu ô tô (Rear Lights)
                    RowLayout {
                        anchors.bottom: parent.bottom
                        anchors.bottomMargin: Math.round(6 * sc)
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.leftMargin: Math.round(6 * sc)
                        anchors.rightMargin: Math.round(6 * sc)

                        Rectangle {
                            width: Math.round(13 * sc)
                            height: Math.round(5 * sc)
                            radius: 2
                            color: colRed
                        }

                        Item { Layout.fillWidth: true }

                        Rectangle {
                            width: Math.round(13 * sc)
                            height: Math.round(5 * sc)
                            radius: 2
                            color: colRed
                        }
                    }

                    // Biển số xe / Badge
                    Rectangle {
                        anchors.bottom: parent.bottom
                        anchors.bottomMargin: Math.round(5 * sc)
                        anchors.horizontalCenter: parent.horizontalCenter
                        width: Math.round(18 * sc)
                        height: Math.round(5 * sc)
                        radius: 1.5
                        color: "#ffffff"
                    }
                }
            }

            // =================================================================
            // 5. CỰ LY KHOẢNG CÁCH KHỔNG LỒ (4,9 m) DƯỚI ĐÍT XE
            // =================================================================
            Text {
                id: distDisplay
                anchors.top: leadCarContainer.bottom
                anchors.topMargin: Math.round(2 * sc)
                anchors.horizontalCenter: parent.horizontalCenter
                text: {
                    if (adasBackend.hasLead) {
                        return adasBackend.leadDist.toFixed(1).replace(".", ",");
                    }
                    return "--";
                }
                color: {
                    if (adasBackend.threatLevel === "DANGER") {
                        return hudRoot.blinkState ? "#ff1744" : "#ffffff";
                    }
                    return "#ffffff";
                }
                font.family: "Arial"
                font.pixelSize: Math.max(18, Math.round(38 * sc))
                font.bold: true
            }

            // =================================================================
            // 6. TEM KIỂM ĐỊNH QC PASS GÓC DƯỚI PHẢI (CHÂN THỰC THEO ẢNH)
            // =================================================================
            Rectangle {
                anchors.bottom: parent.bottom
                anchors.bottomMargin: Math.round(6 * sc)
                anchors.right: parent.right
                anchors.rightMargin: Math.round(6 * sc)
                width: Math.round(26 * sc)
                height: Math.round(16 * sc)
                radius: Math.round(8 * sc)
                rotation: -18
                color: "#166534"
                border.color: "#4ade80"
                border.width: 1

                Text {
                    anchors.centerIn: parent
                    text: "QC\nPASS"
                    horizontalAlignment: Text.AlignHCenter
                    color: "#bbf7d0"
                    font.pixelSize: Math.max(4, Math.round(5 * sc))
                    font.bold: true
                }
            }
        }
    }

    // Khi backend gửi dữ liệu mới, yêu cầu vẽ lại vạch đường Canvas
    Connections {
        target: adasBackend
        function onDataChanged() {
            roadCanvas.requestPaint();
        }
    }

    // Khi kích thước Item thay đổi, vẽ lại vạch đường Canvas
    onWidthChanged: roadCanvas.requestPaint();
    onHeightChanged: roadCanvas.requestPaint();
}
