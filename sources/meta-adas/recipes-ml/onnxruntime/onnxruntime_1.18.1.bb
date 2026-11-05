SUMMARY   = "ONNX Runtime prebuilt aarch64 - NEON SIMD optimized"
HOMEPAGE  = "https://onnxruntime.ai"
LICENSE   = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

PV = "1.18.1"
SRC_URI = "https://github.com/microsoft/onnxruntime/releases/download/v${PV}/onnxruntime-linux-aarch64-${PV}.tgz"
SRC_URI[sha256sum] = "c1dcd8ab29e8d227d886b6ee415c08aea893956acf98f0758a42a84f27c02851"

S = "${WORKDIR}/onnxruntime-linux-aarch64-${PV}"
COMPATIBLE_HOST = "aarch64.*-linux"

do_configure[noexec] = "1"
do_compile[noexec]   = "1"

do_install() {
    install -d ${D}${libdir}
    install -d ${D}${includedir}/onnxruntime

    cp -P ${S}/lib/libonnxruntime.so*    ${D}${libdir}/
    cp -a ${S}/include/.                  ${D}${includedir}/onnxruntime/
}

INHIBIT_PACKAGE_STRIP       = "1"
INHIBIT_PACKAGE_DEBUG_SPLIT = "1"
INSANE_SKIP:${PN}           += "ldflags already-stripped"

FILES:${PN}      = "${libdir}/libonnxruntime.so.*"
FILES:${PN}-dev  = "${libdir}/libonnxruntime.so ${includedir}/onnxruntime/"
