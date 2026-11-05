#!/bin/sh

set -eu

FB_DEV="/dev/fb-lcd"
CAM_DEV="/dev/adas-cam"
TOUCH_DEV="/dev/input/touch-lcd"
WAIT_SEC=3
APP="/usr/bin/adas_pi4"
MODEL_DIR="/opt/adas/models"

log() { echo "[$(date '+%H:%M:%S')] adas-launch: $*"; logger -t adas-launch "$*" 2>/dev/null || true; }

wait_for_node() {
    local node="$1"
    local limit=$(( $2 * 10 ))
    local i=0
    while [ ! -e "$node" ]; do
        i=$(( i + 1 ))
        [ "$i" -ge "$limit" ] && return 1
        sleep 0.1
    done
    return 0
}

log "Tim kiem framebuffer cho LCD..."

find_lcd_fb() {
    for fb in /sys/class/graphics/fb*; do
        [ -d "$fb" ] || continue
        name="$(cat "$fb/name" 2>/dev/null || true)"
        case "$name" in
            *piscreen*|*ili9486*|*ili9341*|*fbtft*|*st7789*)
                echo "/dev/$(basename "$fb")"
                return 0
                ;;
        esac
    done

    if [ -e "/dev/fb-lcd" ]; then
        echo "/dev/fb-lcd"
        return 0
    fi

    if [ -e "/dev/fb1" ]; then
        echo "/dev/fb1"
        return 0
    fi

    if [ -e "/dev/fb0" ]; then
        echo "/dev/fb0"
        return 0
    fi

    return 1
}

FB_TARGET=""
for attempt in $(seq 1 50); do
    FB_TARGET="$(find_lcd_fb || true)"
    [ -n "$FB_TARGET" ] && break
    sleep 0.1
done

if [ -n "$FB_TARGET" ]; then
    FB_DEV="$FB_TARGET"
    ln -sf "$FB_DEV" /dev/fb-lcd 2>/dev/null || true
    fb_driver="$(cat /sys/class/graphics/$(basename "$FB_DEV")/name 2>/dev/null || echo 'unknown')"
    log "Framebuffer OK: $FB_DEV (driver: $fb_driver)"
else
    log "WARNING: Khong tim thay framebuffer dac thu, fallback ve /dev/fb0"
    FB_DEV="/dev/fb0"
    ln -sf /dev/fb0 /dev/fb-lcd 2>/dev/null || true
fi

export QT_QPA_PLATFORM="linuxfb:fb=${FB_DEV}:size=480x320:mmSize=73x49"

log "Tim kiem camera USB / CSI tren Pi 4..."

is_camera() {
    real="$(basename "$(readlink -f "$1" 2>/dev/null || echo "$1")")"
    [ -e "/dev/$real" ] || return 1
    case "$(cat "/sys/class/video4linux/$real/name" 2>/dev/null)" in
        bcm2835-codec*|bcm2835-isp*|rpivid*) return 1 ;;
    esac
    command -v v4l2-ctl >/dev/null 2>&1 || { [ "$(cat "/sys/class/video4linux/$real/index" 2>/dev/null)" = 0 ]; return; }
    caps="$(v4l2-ctl --device="/dev/$real" --info 2>/dev/null | sed -n '/Device Caps/,$p')"
    echo "$caps" | grep -q "Video Capture" || return 1
    echo "$caps" | grep -qi "Memory-to-Memory" && return 1
    return 0
}

find_camera_node() {
    if [ -e "/dev/adas-cam" ] && is_camera /dev/adas-cam; then
        echo "/dev/adas-cam"
        return 0
    fi

    for dev in /sys/class/video4linux/video*; do
        [ -d "$dev" ] || continue
        if readlink -f "$dev/device" 2>/dev/null | grep -qi "usb"; then
            node="/dev/$(basename "$dev")"
            if is_camera "$node"; then
                echo "$node"
                return 0
            fi
        fi
    done

    for v in /dev/video*; do
        [ -e "$v" ] || continue
        if is_camera "$v"; then
            echo "$v"
            return 0
        fi
    done

    return 1
}

CAM_TARGET="$(find_camera_node || true)"

if [ -n "$CAM_TARGET" ]; then
    CAM_DEV="$CAM_TARGET"
    [ "$CAM_DEV" = /dev/adas-cam ] || ln -sf "$CAM_DEV" /dev/adas-cam 2>/dev/null || true
    real="$(basename "$(readlink -f "$CAM_DEV" 2>/dev/null || echo "$CAM_DEV")")"
    cam_name="$(cat "/sys/class/video4linux/$real/name" 2>/dev/null || echo 'USB/CSI Camera')"
    log "Camera OK: $CAM_DEV (ten: $cam_name)"
    export ADAS_CAMERA_DEVICE="$CAM_DEV"
else
    if [ -L /dev/adas-cam ] && ! is_camera /dev/adas-cam; then rm -f /dev/adas-cam; fi
    log "Khong thay camera - app tu do lai camera trong nen"
    export ADAS_CAMERA_DEVICE=""
fi

if [ -e "$TOUCH_DEV" ]; then
    log "Touch OK: $TOUCH_DEV"
    export QT_QPA_GENERIC_PLUGINS="evdevtouch:${TOUCH_DEV}"
else
    log "Touch $TOUCH_DEV chua co - evdevtouch tu do"
    export QT_QPA_GENERIC_PLUGINS="evdevtouch"
fi

echo 0 > /sys/class/graphics/fbcon/cursor_blink 2>/dev/null || true
printf '\033[?25l' > /dev/tty1 2>/dev/null || true
clear > /dev/tty1 2>/dev/null || true

for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -w "$g" ] && echo performance > "$g" 2>/dev/null || true
done
log "CPU governor: performance"

rfkill unblock wifi 2>/dev/null || true

NANODET_MODEL="${MODEL_DIR}/nanodet-plus-m_320.onnx"
if [ -f "$NANODET_MODEL" ]; then
    log "Model OK: NanoDet ($(du -sh "$NANODET_MODEL" | cut -f1))"
else
    log "WARNING: khong tim thay model NanoDet trong ${MODEL_DIR}"
fi

if [ ! -x "$APP" ]; then
    log "FATAL: Khong tim thay file thuc thi $APP"
    exit 1
fi
log "Khoi dong $APP tren $FB_DEV voi DualEngine"
EXTRA=""
[ -e /opt/adas/mock ] && { EXTRA="--mock"; log "MOCK: hien thi du lieu gia (xoa /opt/adas/mock de tat)"; }
exec "$APP" --model option_b $EXTRA
