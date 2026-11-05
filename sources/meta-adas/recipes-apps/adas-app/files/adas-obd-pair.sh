#!/bin/sh
F="${ADAS_OBD_BT_FILE:-/var/lib/adas/obd_bt}"
MIN_RSSI="${ADAS_OBD_PAIR_RSSI:--75}"
PINS="${ADAS_OBD_PINS:-1234 0000 6789}"
NAMES='OBD|ELM|V-LINK|VLINK|VGATE|CBT'
log() { echo "$*"; }
[ "${ADAS_OBD_AUTOPAIR:-1}" = "0" ] && { log "tat (ADAS_OBD_AUTOPAIR=0)"; exit 0; }
[ -n "${ADAS_OBD_BT:-}" ] && { log "ADAS_OBD_BT da dat san -> khong tu ghep"; exit 0; }
[ -n "${ADAS_OBD_DEV:-}" ] && { log "ADAS_OBD_DEV (USB) da dat -> khong tu ghep"; exit 0; }

btc() { bluetoothctl "$@" 2>/dev/null | tr -d '\033' | sed 's/\[[0-9;]*m//g'; }
paired() { btc info "$1" | grep -q "Paired: yes"; }

mkdir -p "$(dirname "$F")"
rfkill unblock bluetooth 2>/dev/null
while :; do
    for i in $(seq 20); do btc list | grep -q Controller && break; sleep 1; done
    if ! btc list | grep -q Controller; then log "chua thay Bluetooth (hci0)"; sleep 30; continue; fi
    btc power on >/dev/null

    if [ -s "$F" ]; then
        MAC=$(head -n 1 "$F")
        if paired "$MAC"; then log "da ghep doi $MAC"; exit 0; fi
        log "$MAC trong $F chua ghep doi -> ghep lai"
    else
        MAC=""
        { echo "scan on"; sleep 15; echo "scan off"; sleep 1; echo quit; } | bluetoothctl 2>/dev/null \
            | tr -d '\033' | sed 's/\[[0-9;]*m//g' > /tmp/obd_scan.txt
        BEST=""; BEST_RSSI=-999; NCAND=0; NKNOWN=0
        for M in $(btc devices | grep -aiE "$NAMES" | awk '{print $2}' | sort -u); do
            NCAND=$((NCAND + 1))
            R=$( { grep -a "$M" /tmp/obd_scan.txt; btc info "$M"; } | grep -a "RSSI" | grep -o -- '-[0-9][0-9]*' | tail -n 1)
            log "ung vien $M $(btc info "$M" | grep -a 'Name:' | head -n 1) RSSI ${R:-?}"
            if [ -n "$R" ]; then
                NKNOWN=$((NKNOWN + 1))
                if [ "$R" -ge "$MIN_RSSI" ] && [ "$R" -gt "$BEST_RSSI" ]; then BEST=$M; BEST_RSSI=$R; fi
            elif [ "$NCAND" -eq 1 ]; then BEST=$M
            fi
        done
        if [ "$NCAND" -gt 1 ] && [ "$NKNOWN" -eq 0 ]; then
            log "nhieu thiet bi OBD, khong biet RSSI -> khong tu ghep (dung pi_obd_pair.sh <MAC>)"; BEST=""
        fi
        MAC=$BEST
        [ -z "$MAC" ] && { [ "$NCAND" -eq 0 ] && log "chua thay ELM327 (da cam vao xe?)"; sleep 30; continue; }
    fi

    log "ghep doi $MAC..."
    for PIN in $PINS; do
        { echo "agent KeyboardOnly"; echo "default-agent"; echo "scan on"; sleep 6; echo "scan off";
          echo "pair $MAC"; sleep 4; echo "$PIN"; sleep 8; echo quit; } | bluetoothctl >/tmp/obd_pair.txt 2>&1
        paired "$MAC" && break
    done
    if paired "$MAC"; then
        btc trust "$MAC" >/dev/null
        echo "$MAC" > "$F"; sync
        log "DA GHEP DOI $MAC -> app se tu ket noi"
        exit 0
    fi
    log "ghep doi $MAC that bai: $(tr -d '\033' < /tmp/obd_pair.txt | grep -aiE 'fail|error' | tail -n 1)"
    sleep 30
done
