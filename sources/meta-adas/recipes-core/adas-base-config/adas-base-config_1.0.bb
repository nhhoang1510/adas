SUMMARY = "Base configuration files for ADAS system"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
    file://90-adas.rules \
    file://20-eth0.network \
    file://25-wlan0.network \
    file://hostapd.conf \
    file://10-watchdog.conf \
    file://99-adas-grayscale.conf \
"

S = "${WORKDIR}"

do_install() {
    install -d ${D}${sysconfdir}/udev/rules.d
    install -m 0644 ${WORKDIR}/90-adas.rules ${D}${sysconfdir}/udev/rules.d/90-adas.rules

    install -d ${D}${sysconfdir}/systemd/network
    install -m 0644 ${WORKDIR}/20-eth0.network ${D}${sysconfdir}/systemd/network/20-eth0.network
    install -m 0644 ${WORKDIR}/25-wlan0.network ${D}${sysconfdir}/systemd/network/25-wlan0.network

    install -m 0644 ${WORKDIR}/hostapd.conf ${D}${sysconfdir}/hostapd.conf

    install -d ${D}${sysconfdir}/systemd/system.conf.d
    install -m 0644 ${WORKDIR}/10-watchdog.conf ${D}${sysconfdir}/systemd/system.conf.d/10-watchdog.conf

    install -d ${D}${sysconfdir}/fonts/conf.d
    install -m 0644 ${WORKDIR}/99-adas-grayscale.conf ${D}${sysconfdir}/fonts/conf.d/99-adas-grayscale.conf
}

FILES:${PN} = " \
    ${sysconfdir}/udev/rules.d/ \
    ${sysconfdir}/systemd/network/ \
    ${sysconfdir}/hostapd.conf \
    ${sysconfdir}/systemd/system.conf.d/ \
    ${sysconfdir}/fonts/conf.d/ \
"
