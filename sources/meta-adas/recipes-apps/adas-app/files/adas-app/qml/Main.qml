import QtQuick 2.12
import QtQuick.Window 2.12

Window {
    id: root
    width: 480
    height: 320
    visible: true
    color: "#000000"
    title: "ADAS Segment Cluster"

    readonly property bool hasBackend: typeof adasBackend !== "undefined"
    readonly property int    speed:       mock ? mk.s.speed : hasBackend ? adasBackend.speed : 0
    readonly property real   distance:    mock ? mk.s.dist : hasBackend ? adasBackend.leadDist : -1.0
    readonly property string laneStatus:  mock ? mk.s.lane : hasBackend ? adasBackend.laneStatus : "NORMAL"
    readonly property string laneOffset:  hasBackend ? adasBackend.laneOffset : "+0.00m"
    readonly property int    threat:      mock ? mk.s.threat : hasBackend ? adasBackend.threatLevel : 0
    readonly property string sourceState: mock ? mk.s.source : (hasBackend && typeof adasBackend.sourceState !== "undefined") ? adasBackend.sourceState : "LIVE"
    readonly property bool   aiHealthy:   mock ? mk.s.ai : (hasBackend && typeof adasBackend.aiHealthy !== "undefined") ? adasBackend.aiHealthy : true
    readonly property bool   hasSign:     hasBackend && adasBackend.hasSign === true
    readonly property string signName:    hasBackend ? adasBackend.signName : ""
    readonly property int    signCode:    hasBackend && adasBackend.signCode !== "" ? parseInt(adasBackend.signCode) : -1
    readonly property string signType:    hasBackend ? adasBackend.signType : "guide"
    readonly property int    speedSign:   mock ? (mk.s.limit > 0 ? 0 : -1) : hasBackend && typeof adasBackend.speedSign !== "undefined" ? adasBackend.speedSign : -1
    readonly property int    auxSign:     mock ? mk.s.aux : hasBackend && typeof adasBackend.auxSign !== "undefined" ? adasBackend.auxSign : -1
    readonly property int    speedValue:  mock ? mk.s.limit : hasBackend && typeof adasBackend.speedValue !== "undefined" ? adasBackend.speedValue : 0
    readonly property real   roadBend:    mock ? mk.s.bend : hasBackend && typeof adasBackend.roadBend !== "undefined" ? adasBackend.roadBend : 0.0
    readonly property string obdState:    mock ? mk.s.obd : hasBackend && typeof adasBackend.obdState !== "undefined" ? adasBackend.obdState : "ok"
    readonly property bool   wifiOn:      mock ? mk.s.wifi : hasBackend && typeof adasBackend.wifiOn !== "undefined" ? adasBackend.wifiOn : false
    readonly property bool   btOn:        mock ? mk.s.bt : hasBackend && typeof adasBackend.btOn !== "undefined" ? adasBackend.btOn : false
    readonly property bool   btConnected: mock ? mk.s.bt : hasBackend && typeof adasBackend.btConnected !== "undefined" ? adasBackend.btConnected : false
    readonly property int    calibProgress: mock ? mk.s.calib : hasBackend && typeof adasBackend.calibProgress !== "undefined" ? adasBackend.calibProgress : -1

    readonly property bool debugKeys: Qt.application.arguments.indexOf("--debug-keys") >= 0
    readonly property bool signDemo: Qt.application.arguments.indexOf("--sign-demo") >= 0
    property int lastLimit: 0
    onSpeedValueChanged: if (speedValue > 0) lastLimit = speedValue
    property int demoSign: -1
    Timer {
        interval: 3000; repeat: true; running: root.signDemo
        onTriggered: root.demoSign = root.demoSign >= 15 ? -1 : root.demoSign === -1 ? 1 : root.demoSign + 1
    }
    readonly property bool mock: Qt.application.arguments.indexOf("--mock") >= 0
    QtObject {
        id: mk
        property real t: 0
        readonly property var s: state(t)
        function state(t) {
            var o = { speed: 0, dist: -1, threat: 0, lane: "NORMAL", source: "LIVE", ai: true, obd: "ok",
                      calib: -1, turnL: false, turnR: false, limit: 0, aux: -1, wifi: true, bt: false, bend: 0 }
            o.speed = Math.max(0, Math.round(55 + 25 * Math.sin(t / 4) + 8 * Math.sin(t / 1.3)))
            var lim = [60, 50, 80, 40, 100, 0, 70]
            o.limit = lim[Math.floor(t / 6) % lim.length]
            var lp = t % 13
            if (lp < 10) {
                o.dist = 70 - 64 * lp / 10
                o.threat = o.dist < 12 ? 2 : o.dist < 22 ? 1 : 0
            }
            var lanes = ["NORMAL", "NORMAL", "DRIFT_LEFT", "DEPARTURE_LEFT", "NORMAL", "NORMAL",
                         "DRIFT_RIGHT", "DEPARTURE_RIGHT", "NORMAL", "HOLD", "NORMAL", "DEPARTURE_LEFT"]
            var lk = Math.floor(t / 1.5) % lanes.length
            o.lane = lanes[lk]
            o.turnL = lk === 11
            o.turnR = t % 17 < 2.5
            var a = Math.floor(t / 4) % 18
            o.aux = a < 15 ? a + 1 : -1
            var f = t % 60
            if (f >= 40 && f < 43) o.source = "CAMERA_LOST"
            else if (f >= 43 && f < 45) o.ai = false
            else if (f >= 45 && f < 50) o.calib = Math.round((f - 45) / 5 * 100)
            if (t % 47 >= 42) o.obd = "no_elm"
            o.bt = t % 20 < 12
            o.wifi = t % 33 < 28
            var bp = t % 24
            o.bend = bp < 6 ? 0 : bp < 12 ? 0.05 * Math.sin((bp - 6) / 6 * Math.PI) : bp < 18 ? 0 : -0.01 * Math.sin((bp - 18) / 6 * Math.PI)
            return o
        }
    }
    Timer {
        interval: 100; repeat: true; running: root.mock && !root.bulbCheck
        onTriggered: mk.t += 0.1
    }

    property bool dbgTurnLeft: false
    property bool dbgTurnRight: false
    readonly property bool turnLeftActive:  (mock && mk.s.turnL) || (hasBackend && adasBackend.turnLeft === true)  || dbgTurnLeft
    readonly property bool turnRightActive: (mock && mk.s.turnR) || (hasBackend && adasBackend.turnRight === true) || dbgTurnRight

    property int displayRotation: 0

    readonly property real u: Math.min(width / 480, height / 320)

    readonly property color colPrimary:   "#FFFFFF"
    readonly property color colRoad:      "#6E7C88"
    readonly property color colHold:      "#404040"
    readonly property color colSecondary: "#87919A"
    readonly property color colSafe:      "#00B000"
    readonly property color colLane:      colSafe
    readonly property color colCaution:   "#FF7A00"
    readonly property color colDanger:    "#FF0000"
    readonly property color colFrame:     "#282828"
    readonly property color colTellOff:   colIdle
    readonly property color colGhost:     "#080808"
    readonly property color colIdle:      "#181818"
    readonly property color colWait:      colPrimary
    readonly property color colBtLink:    "#0038FF"
    readonly property color colOff:       "#000000"
    readonly property color readoutColor: leadThreat >= 2 ? colDanger : leadThreat === 1 ? colCaution : colPrimary

    readonly property bool rawLead: distance > 0 && distance <= 80.0
    readonly property bool hasLead: rawLead || shownBars > 0
    property real lastDist: 0
    property int lastThreat: 0
    onDistanceChanged: if (distance > 0 && distance <= 80.0) lastDist = distance
    onThreatChanged: if (distance > 0 && distance <= 80.0) lastThreat = threat
    readonly property real leadDist: rawLead ? distance : lastDist
    readonly property int leadThreat: rawLead ? threat : lastThreat
    readonly property bool hasLane: laneStatus !== "NO_LANE"
    readonly property bool hasVideo: (mock || hasBackend) && sourceState !== "NO_CAMERA" && sourceState !== "CAMERA_LOST"

    readonly property int leftLevel:  turnLeftActive  ? 0 : laneStatus === "DEPARTURE_LEFT"  ? 2 : laneStatus === "DRIFT_LEFT"  ? 1 : 0
    readonly property int rightLevel: turnRightActive ? 0 : laneStatus === "DEPARTURE_RIGHT" ? 2 : laneStatus === "DRIFT_RIGHT" ? 1 : 0
    readonly property int laneLevel: Math.max(leftLevel, rightLevel)

    readonly property bool collision: threat >= 2
    readonly property bool obdOk: obdState === "ok"
    readonly property int overSpeed: bulbCheck || !obdOk || speedSign !== 0 || speedValue <= 0 ? 0
                                   : speed >= speedValue + 10 ? 2 : speed > speedValue ? 1 : 0
    readonly property color speedColor: overSpeed === 2 ? colDanger : overSpeed === 1 ? colCaution : colPrimary

    property int bootStep: 0
    readonly property bool bulbCheck: bootStep < 25
    readonly property int bootLevel: bootStep <= 7 ? bootStep : bootStep <= 17 ? 7 : Math.max(0, 24 - bootStep)
    readonly property int bootSweep: bulbCheck ? bootLevel : -1
    function bootLit(step) { return bulbCheck && bootLevel >= step }
    Timer { interval: root.bootStep < 17 ? 80 : 40; repeat: true; running: root.bulbCheck; onTriggered: root.bootStep++ }

    readonly property bool showCalib: !bulbCheck && hasVideo && aiHealthy && calibProgress >= 0
    readonly property bool degraded: !hasVideo || !aiHealthy

    readonly property color threatColor: threat >= 2 ? colDanger : threat === 1 ? colCaution : colRoad

    readonly property int targetBars: !rawLead ? 0
                                    : distance < 12 ? 1 : distance < 25 ? 2
                                    : distance < 40 ? 3 : distance < 60 ? 4 : 5
    property int shownBars: 0
    readonly property int gapBars: shownBars
    onRawLeadChanged: if (rawLead) leadGrace.stop(); else leadGrace.restart()
    Timer { id: leadGrace; interval: 200 }
    Timer {
        interval: 60; repeat: true
        running: root.shownBars !== root.targetBars && !(root.targetBars === 0 && leadGrace.running)
        onTriggered: root.shownBars += root.targetBars > root.shownBars ? 1 : -1
    }

    property bool blink: true
    Timer {
        interval: 200; repeat: true
        running: root.collision || root.laneLevel === 2 || root.overSpeed === 2
        onTriggered: root.blink = !root.blink
        onRunningChanged: if (!running) root.blink = true
    }
    property bool blinkSlow: true
    Timer {
        interval: 500; repeat: true
        running: root.showCalib || (!root.bulbCheck && root.hasVideo && !root.aiHealthy)
        onTriggered: root.blinkSlow = !root.blinkSlow
        onRunningChanged: if (!running) root.blinkSlow = true
    }
    property bool blinkTurn: true
    Timer {
        interval: 333; repeat: true
        running: root.turnLeftActive || root.turnRightActive
        onTriggered: root.blinkTurn = !root.blinkTurn
        onRunningChanged: if (!running) root.blinkTurn = true
    }

    function sideColor(level) {
        if (root.bulbCheck) return root.bootLit(1) ? root.colSafe : root.colIdle
        if (!root.hasLane || root.degraded) return root.colWait

        if (root.laneStatus === "HOLD") return root.colHold
        if (level === 2) return root.blink ? root.colDanger : root.colGhost
        if (level === 1) return root.colCaution
        return root.colLane
    }

    readonly property string speedText: Math.max(0, Math.min(999, speed)).toString()
    readonly property string distText: !hasLead ? "" : Math.round(leadDist).toString()
    readonly property string distErrText: !hasLead ? "" : "±" + Math.max(1, Math.round(0.5 + 0.08 * leadDist))

    Item {
        id: hud
        anchors.fill: parent
        rotation: root.displayRotation
        transformOrigin: Item.Center

        Telltale {
            id: turnL
            x: 14 * root.u; y: 10 * root.u; width: 44 * root.u; height: 30 * root.u
            design: Qt.size(44, 30)
            c: root.bootLit(1) || (root.turnLeftActive && root.blinkTurn) ? root.colSafe : root.colTellOff
            draw: function(ctx) {
                ctx.moveTo(1, 15); ctx.lineTo(19, 1); ctx.lineTo(19, 9); ctx.lineTo(43, 9)
                ctx.lineTo(43, 21); ctx.lineTo(19, 21); ctx.lineTo(19, 29); ctx.closePath(); ctx.fill()
            }
        }
        Telltale {
            id: turnR
            x: root.width - 58 * root.u; y: 10 * root.u; width: 44 * root.u; height: 30 * root.u
            design: Qt.size(44, 30)
            c: root.bootLit(5) || (root.turnRightActive && root.blinkTurn) ? root.colSafe : root.colTellOff
            draw: function(ctx) {
                ctx.moveTo(43, 15); ctx.lineTo(25, 1); ctx.lineTo(25, 9); ctx.lineTo(1, 9)
                ctx.lineTo(1, 21); ctx.lineTo(25, 21); ctx.lineTo(25, 29); ctx.closePath(); ctx.fill()
            }
        }

        Telltale {
            id: fcwIcon
            x: 337 * root.u; y: 8 * root.u; width: 48 * root.u; height: 34 * root.u
            design: Qt.size(48, 34)
            c: root.bulbCheck ? (root.bootLit(2) ? root.colDanger : root.colTellOff)
               : root.collision ? (root.blink ? root.colDanger : root.colTellOff)
               : root.threat === 1 ? root.colCaution : root.colTellOff
            draw: function(ctx) {
                ctx.moveTo(24, 1); ctx.lineTo(47, 33); ctx.lineTo(1, 33); ctx.closePath()
                ctx.fill()
                ctx.fillStyle = "#000000"
                ctx.beginPath()
                ctx.moveTo(24, 8.7); ctx.lineTo(38.2, 28.5); ctx.lineTo(9.8, 28.5); ctx.closePath()
                ctx.fill()
                ctx.fillStyle = fcwIcon.c
                ctx.beginPath()
                ctx.rect(22, 13, 4, 9.5)
                ctx.rect(22, 24, 4, 3)
                ctx.fill()
            }
        }

        Telltale {
            id: camIcon
            x: 277 * root.u; y: 8 * root.u; width: 48 * root.u; height: 34 * root.u
            design: Qt.size(48, 34)
            c: root.bulbCheck ? (root.bootLit(4) ? root.colCaution : root.colTellOff)
               : root.sourceState === "CAMERA_LOST" ? root.colDanger
               : !root.hasVideo ? root.colCaution
               : root.showCalib ? (root.blinkSlow ? root.colCaution : root.colTellOff)
               : root.colTellOff
            draw: function(ctx) {
                ctx.rect(6, 10, 36, 20)
                ctx.rect(17, 6, 14, 4)
                ctx.fill()
                ctx.fillStyle = "#000000"
                ctx.beginPath(); ctx.arc(24, 20, 7, 0, 2 * Math.PI); ctx.fill()
                ctx.fillStyle = camIcon.c
                ctx.beginPath(); ctx.arc(24, 20, 3.5, 0, 2 * Math.PI); ctx.fill()
            }
        }
        Telltale {
            id: aiIcon
            x: 216 * root.u; y: 8 * root.u; width: 48 * root.u; height: 34 * root.u
            design: Qt.size(48, 34)
            c: root.bulbCheck ? (root.bootLit(4) ? root.colCaution : root.colTellOff)
               : root.hasVideo && !root.aiHealthy ? (root.blinkSlow ? root.colCaution : root.colTellOff)
               : root.colTellOff
            draw: function(ctx) {
                ctx.rect(13, 6, 22, 22)
                for (var k = 0; k < 3; ++k) {
                    ctx.rect(9, 9 + 6 * k, 4, 2.5); ctx.rect(35, 9 + 6 * k, 4, 2.5)
                    ctx.rect(16 + 6 * k, 2, 2.5, 4); ctx.rect(16 + 6 * k, 28, 2.5, 4)
                }
                ctx.fill()
                ctx.fillStyle = "#000000"
                ctx.font = "bold 12px sans-serif"
                ctx.textAlign = "center"; ctx.textBaseline = "middle"
                ctx.fillText("AI", 24, 17.5)
            }
        }
        Telltale {
            id: wifiIcon
            x: 101 * root.u; y: 10 * root.u; width: 36 * root.u; height: 28 * root.u
            design: Qt.size(28, 22)
            c: root.bootLit(4) || (!root.bulbCheck && root.wifiOn) ? root.colPrimary : root.colTellOff
            draw: function(ctx) {
                var r = [[6, 9], [11, 14], [16, 19]], a0 = -0.75 * Math.PI, a1 = -0.25 * Math.PI
                for (var k = 0; k < 3; ++k) {
                    ctx.beginPath()
                    ctx.arc(14, 21, r[k][1], a0, a1, false)
                    ctx.arc(14, 21, r[k][0], a1, a0, true)
                    ctx.closePath(); ctx.fill()
                }
                ctx.beginPath(); ctx.arc(14, 19.5, 2.5, 0, 2 * Math.PI); ctx.fill()
            }
        }
        Telltale {
            id: btIcon
            x: 170 * root.u; y: 9 * root.u; width: 18 * root.u; height: 31 * root.u
            design: Qt.size(14, 24)
            c: root.bootLit(4) ? root.colPrimary : root.bulbCheck || !root.btOn ? root.colTellOff
               : root.btConnected ? root.colBtLink : root.colPrimary
            draw: function(ctx) {
                ctx.strokeStyle = btIcon.c; ctx.lineWidth = 2.2; ctx.lineJoin = "miter"
                ctx.moveTo(2, 7); ctx.lineTo(12, 16.5); ctx.lineTo(7, 21.5); ctx.lineTo(7, 2.5)
                ctx.lineTo(12, 7.5); ctx.lineTo(2, 17)
                ctx.stroke()
            }
        }

        Rectangle { antialiasing: false; x: 14 * root.u; y: 50 * root.u; width: 452 * root.u; height: Math.max(1, root.u); color: root.colFrame }

        SegmentDisplay {
            id: distDisp
            x: 172 * root.u; y: 194 * root.u
            digits: 2
            digitWidth: 36 * root.u; digitHeight: 88 * root.u; spacing: 6 * root.u
            dots: false
            text: root.distText
            sweep: root.bootSweep
            dash: !root.bulbCheck && !root.hasLead
            onColor: root.bulbCheck || root.hasLead ? root.readoutColor : root.colPrimary; offColor: root.colOff
        }
        Row {
            id: distUnitRow
            x: distDisp.x + distDisp.width - distDisp._over - width
            anchors.top: distDisp.bottom; anchors.topMargin: 3 * root.u
            spacing: 4 * root.u
            Text {
                id: distUnit
                text: "m"; color: distDisp.onColor
                font.pixelSize: 20 * root.u; font.bold: true
            }
            Text {
                anchors.baseline: distUnit.baseline
                text: root.bootLit(7) ? "±8" : root.bulbCheck ? "" : root.distErrText; color: distDisp.onColor
                font.pixelSize: 16 * root.u; font.bold: true
                visible: text !== ""
            }
        }

        Rectangle { antialiasing: false; x: 14 * root.u; y: 182 * root.u; width: 256 * root.u; height: Math.max(1, root.u); color: root.colFrame }

        SegmentDisplay {
            id: speedDisp
            x: 14 * root.u; y: 194 * root.u
            digits: 3
            digitWidth: 36 * root.u; digitHeight: 88 * root.u; spacing: 6 * root.u
            dashLen: 0.65 * 78 / 120
            dots: false
            text: root.speedText
            sweep: root.bootSweep
            dash: !root.bulbCheck && !root.obdOk
            onColor: root.speedColor; offColor: root.colOff
        }
        Text {
            id: speedUnit
            x: speedDisp.x + speedDisp.width - speedDisp._over - width
            anchors.top: speedDisp.bottom; anchors.topMargin: 3 * root.u
            text: "km/h"; color: root.speedColor
            font.pixelSize: 20 * root.u; font.bold: true
        }

        SignIcon {
            id: signIcon
            x: 31 * root.u; y: 56 * root.u; width: 120 * root.u; height: 120 * root.u
            code: root.bootLit(7) || root.signDemo ? 0 : root.bulbCheck ? -1
                : root.overSpeed === 2 && !root.blink ? -1 : root.speedSign
            value: root.bulbCheck ? 88 : root.signDemo ? 80 : root.speedValue
            ghost: root.colOff
        }

        readonly property real roadX: 374
        readonly property real roadY0: 314
        readonly property real roadYh: 14
        readonly property real roadHW0: 90
        readonly property real roadTop: 64
        function rz(z) { return roadYh + (roadY0 - roadYh) / z }
        function rhw(z) { return roadHW0 / z }
        readonly property real bend: Math.round(root.roadBend / 0.0005) * 0.0005
        function pathX(k, d) {
            if (Math.abs(k) < 1e-5) return 0
            var kd = Math.min(1, Math.abs(k * d))
            return (1 - Math.sqrt(1 - kd * kd)) / k
        }
        function bx(z) {
            var raw = (pathX(bend, 6 * z) - pathX(bend, 6)) * 51.4 / z
            return 95 * Math.tanh(raw / 95)
        }
        function barZ(i) { return 1.07 + i * 0.45 }
        function laneHW(z) { return 7 / z }

        readonly property color colCentre:   "#7B797B"
        readonly property real zTop: (roadY0 - roadYh) / (roadTop - roadYh)
        Canvas {
            id: roadCentre
            x: 270 * root.u; y: 56 * root.u; width: 210 * root.u; height: 262 * root.u
            antialiasing: false
            renderStrategy: Canvas.Immediate
            readonly property real bend: hud.bend
            onBendChanged: requestPaint()
            onPaint: {
                var ctx = getContext("2d"), u = root.u, ox = 270, oy = 56
                ctx.reset()
                ctx.clearRect(0, 0, width, height)
                ctx.fillStyle = hud.colCentre
                for (var z0 = 1.0; z0 < hud.zTop; z0 += 0.5) {
                    var z1 = Math.min(hud.zTop, z0 + 0.25)
                    var top = Math.round((hud.rz(z1) - oy) * u), bot = Math.round((hud.rz(z0) - oy) * u)
                    var w = Math.max(1, Math.round(3 / (z0 + 0.125) * u))
                    ctx.fillRect(Math.round((hud.roadX + hud.bx(z0 + 0.125) - ox) * u - w / 2), top, w, Math.max(1, bot - top))
                }
            }
        }

        Canvas {
            id: laneLines
            readonly property color cl: root.sideColor(root.leftLevel)
            readonly property color cr: root.sideColor(root.rightLevel)
            x: 270 * root.u; y: 56 * root.u; width: 210 * root.u; height: 262 * root.u
            antialiasing: false
            renderStrategy: Canvas.Immediate
            readonly property real bend: hud.bend
            onClChanged: requestPaint()
            onCrChanged: requestPaint()
            onBendChanged: requestPaint()
            onPaint: {
                var ctx = getContext("2d")
                ctx.reset()
                ctx.clearRect(0, 0, width, height)
                var u = root.u, ox = 270, oy = 56
                function quad(side, zn, zf) {
                    var yn = hud.rz(zn), yf = hud.rz(zf)
                    var xn = hud.roadX + hud.bx(zn) + side * hud.rhw(zn), xf = hud.roadX + hud.bx(zf) + side * hud.rhw(zf)
                    var wn = hud.laneHW(zn), wf = hud.laneHW(zf)
                    var s0 = side * (hud.rhw(zf) - hud.rhw(zn)) / (yf - yn), s1 = (xf - xn) / (yf - yn)
                    var kw = Math.sqrt((1 + s1 * s1) / (1 + s0 * s0))
                    wn *= kw; wf *= kw
                    ctx.beginPath()
                    ctx.moveTo((xn - wn - ox) * u, (yn - oy) * u); ctx.lineTo((xn + wn - ox) * u, (yn - oy) * u)
                    ctx.lineTo((xf + wf - ox) * u, (yf - oy) * u); ctx.lineTo((xf - wf - ox) * u, (yf - oy) * u)
                    ctx.closePath(); ctx.fill()
                }
                var zTop = (hud.roadY0 - hud.roadYh) / (hud.roadTop - hud.roadYh)
                var n = Math.abs(hud.bend) < 1e-5 ? 1 : 12
                for (var side = -1; side <= 1; side += 2) {
                    ctx.fillStyle = side < 0 ? cl : cr
                    for (var k = 0; k < n; ++k) {
                        var yA = hud.rz(1.0) + (hud.rz(zTop) - hud.rz(1.0)) * k / n
                        var yB = hud.rz(1.0) + (hud.rz(zTop) - hud.rz(1.0)) * (k + 1) / n
                        quad(side, (hud.roadY0 - hud.roadYh) / (yA - hud.roadYh), (hud.roadY0 - hud.roadYh) / (yB - hud.roadYh))
                    }
                }
            }
        }

        readonly property var barEdges: [0, 12, 25, 40, 60]
        function leadZ(d) {
            for (var i = 0; i < 4; ++i)
                if (d < barEdges[i + 1]) return barZ(i + (d - barEdges[i]) / (barEdges[i + 1] - barEdges[i]))
            return barZ(4)
        }
        Item {
            id: leadCar
            property real zAnim: root.rawLead ? hud.leadZ(root.distance) : hud.barZ(root.gapBars > 0 ? root.gapBars - 1 : 4)
            Behavior on zAnim { SmoothedAnimation { duration: 200; velocity: -1 } }
            readonly property real s: 1.3 * root.u / zAnim
            visible: root.bulbCheck ? root.bootLit(6) : root.hasLead
            width: Math.round(48 * s); height: Math.round(30 * s)
            x: Math.round((hud.roadX + hud.bx(zAnim)) * root.u - width / 2)
            y: Math.round(hud.rz(zAnim) * root.u - 7 * s - height)
            readonly property color c: root.bulbCheck ? (root.bootLit(6) ? root.colPrimary : root.colIdle) : !root.hasLead ? root.colIdle
                                       : root.readoutColor
            Canvas {
                anchors.fill: parent
                antialiasing: true
                renderStrategy: Canvas.Immediate
                readonly property color c: leadCar.c
                onCChanged: requestPaint()
                onWidthChanged: requestPaint()
                onHeightChanged: requestPaint()
                onPaint: {
                    var ctx = getContext("2d")
                    ctx.reset()
                    ctx.clearRect(0, 0, width, height)
                    ctx.scale(width / 48, height / 30)
                    ctx.fillStyle = c
                    var shapes = [
                        [[4, 11], [44, 11], [48, 15], [48, 21], [44, 25], [4, 25], [0, 21], [0, 15]],
                        [[13, 1], [35, 1], [37, 3], [41, 12], [7, 12], [11, 3]],
                        [[4, 24], [14, 24], [14, 29], [13, 30], [5, 30], [4, 29]],
                        [[34, 24], [44, 24], [44, 29], [43, 30], [35, 30], [34, 29]]
                    ]
                    for (var k = 0; k < shapes.length; ++k) {
                        var p = shapes[k]
                        ctx.beginPath(); ctx.moveTo(p[0][0], p[0][1])
                        for (var j = 1; j < p.length; ++j) ctx.lineTo(p[j][0], p[j][1])
                        ctx.closePath(); ctx.fill()
                    }
                }
            }
        }

        SignIcon {
            id: auxSignIcon
            x: 171 * root.u; y: 56 * root.u; width: 120 * root.u; height: 120 * root.u
            code: root.bootLit(3) ? 7 : root.bulbCheck ? -1 : root.signDemo ? root.demoSign : root.auxSign
            value: root.signDemo ? 60 : root.lastLimit
            ghost: root.colOff
        }

        PixRect {
            anchors.fill: parent
            borderColor: root.colDanger
            borderWidth: 5 * root.u
            radius: 6 * root.u
            visible: root.collision && root.blink
            z: 100
        }
    }

    Item {
        anchors.fill: parent
        focus: root.debugKeys
        Keys.onPressed: {
            if (!root.debugKeys) return
            if (event.key === Qt.Key_R) { root.displayRotation = root.displayRotation === 0 ? 180 : 0; event.accepted = true }
            else if (event.key === Qt.Key_A || event.key === Qt.Key_Left)  { root.dbgTurnLeft = !root.dbgTurnLeft; event.accepted = true }
            else if (event.key === Qt.Key_D || event.key === Qt.Key_Right) { root.dbgTurnRight = !root.dbgTurnRight; event.accepted = true }
        }
    }
}
