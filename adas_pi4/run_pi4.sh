#!/bin/bash
# ==============================================================================
# Script khoi dong ADAS tren Raspberry Pi 4 voi man hinh SPI ILI9486 + XPT2046
# ==============================================================================

# Cho cac thiet bi va framebuffer on dinh
sleep 2

# Khoi dong fbcp mirror frame sang man hinh SPI neu co cai dat
if command -v fbcp-ili9341 &> /dev/null; then
    sudo killall fbcp-ili9341 2>/dev/null
    /usr/local/bin/fbcp-ili9341 &
    sleep 1
fi

# Chon nen tang render Qt
# Neu co fbcp: dung eglfs (muot nhat)
# Neu khong co fbcp: dung linuxfb truc tiep vao /dev/fb1
if [ -e /dev/fb1 ]; then
    export QT_QPA_PLATFORM=linuxfb:fb=/dev/fb1
else
    export QT_QPA_PLATFORM=eglfs
fi

# Cau hinh Touch XPT2046
if [ -e /dev/input/event0 ]; then
    export QT_QPA_EVDEV_TOUCHSCREEN_PARAMETERS=/dev/input/event0:rotate=90
    export TSLIB_TSDEVICE=/dev/input/event0
fi

export QT_FONT_DPI=96
export QSG_RENDER_LOOP=basic

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/build" || exit 1

./adas_pi4 "$@"
