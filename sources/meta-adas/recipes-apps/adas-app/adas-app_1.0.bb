SUMMARY = "ADAS HMI - Qt5 QML kiosk voi NanoDet-Plus + nhan dien bien bao"
LICENSE = "CLOSED"
PR = "r6"

SRC_URI = " \
    file://adas-app \
    file://adas.service \
    file://adas-launch.sh \
    file://adas.env \
    file://adas-obd-pair.sh \
    file://adas-obd-pair.service \
    file://adas-splash.service \
    file://adas-splash.rgb565 \
"

S = "${WORKDIR}/adas-app"

inherit cmake_qt5 systemd pkgconfig

DEPENDS = " \
    qtbase \
    qtdeclarative \
    qtquickcontrols2 \
    qtgraphicaleffects \
    opencv \
    onnxruntime \
    gstreamer1.0 \
    gstreamer1.0-plugins-base \
"

RDEPENDS:${PN} = " \
    adas-base-config \
    onnxruntime \
    opencv \
    v4l-utils \
    bluez5 \
    qtbase-plugins \
    qtdeclarative-qmlplugins \
    qtquickcontrols2-qmlplugins \
"

TARGET_CXXFLAGS:append = " -march=armv8-a+simd -mtune=cortex-a72 -O3 -ffast-math"

EXTRA_OECMAKE = " \
    -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_DIR=${STAGING_DIR_TARGET}/usr \
"

do_install:append() {
    install -d ${D}/opt/adas/models
    if [ -d "${S}/models" ]; then
        cp -r ${S}/models/* ${D}/opt/adas/models/ 2>/dev/null || true
    fi

    install -d ${D}/opt/adas/calib
    if [ -d "${S}/calib" ]; then
        cp -r ${S}/calib/* ${D}/opt/adas/calib/ 2>/dev/null || true
    fi

    install -d ${D}/opt/adas/qml
    if [ -d "${S}/qml" ]; then
        install -m 0644 ${S}/qml/*.qml ${D}/opt/adas/qml/
    fi

    install -d ${D}${systemd_system_unitdir}
    install -m 0644 ${WORKDIR}/adas.service \
        ${D}${systemd_system_unitdir}/adas.service

    install -d ${D}${libexecdir}/adas
    install -m 0755 ${WORKDIR}/adas-launch.sh \
        ${D}${libexecdir}/adas/adas-launch.sh

    install -m 0755 ${WORKDIR}/adas-obd-pair.sh ${D}${libexecdir}/adas/adas-obd-pair.sh
    install -m 0644 ${WORKDIR}/adas-obd-pair.service ${D}${systemd_system_unitdir}/adas-obd-pair.service

    install -m 0644 ${WORKDIR}/adas-splash.rgb565 ${D}/opt/adas/splash.rgb565
    install -m 0644 ${WORKDIR}/adas-splash.service ${D}${systemd_system_unitdir}/adas-splash.service

    install -d ${D}${bindir}
    if [ -f ${B}/adas_pi4 ]; then
        install -m 0755 ${B}/adas_pi4 ${D}${bindir}/adas_pi4
    fi

    install -d ${D}${sysconfdir}/adas
    install -m 0644 ${WORKDIR}/adas.env \
        ${D}${sysconfdir}/adas/adas.env
}

SYSTEMD_SERVICE:${PN}      = "adas.service adas-obd-pair.service adas-splash.service"
SYSTEMD_AUTO_ENABLE:${PN}  = "enable"

FILES:${PN} += " \
    ${bindir}/adas_pi4 \
    /opt/adas/ \
    ${libexecdir}/adas/ \
    ${sysconfdir}/adas/ \
    ${systemd_system_unitdir}/adas.service \
    ${systemd_system_unitdir}/adas-obd-pair.service \
    ${systemd_system_unitdir}/adas-splash.service \
"
