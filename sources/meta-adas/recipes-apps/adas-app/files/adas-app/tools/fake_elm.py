import os, sys, time, math, select, tty
link = sys.argv[1]; off_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0; unplug = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
t0 = time.time(); n = 0
def mk():
    m, s = os.openpty(); tty.setraw(s)
    try: os.unlink(link)
    except FileNotFoundError: pass
    os.symlink(os.ttyname(s), link); return m, s
m, s = mk(); echo = True; buf = b''
while True:
    t = time.time() - t0
    if t > unplug:
        os.close(m); os.close(s); os.unlink(link); print('unplugged', flush=True); time.sleep(3)
        m, s = mk(); echo = True; buf = b''; unplug = 1e9; print('replugged', flush=True)
    r, _, _ = select.select([m], [], [], 0.1)
    if not r: continue
    buf += os.read(m, 256)
    while b'\r' in buf:
        line, buf = buf.split(b'\r', 1); c = line.decode().strip().upper().replace(' ', '')
        out = (c + '\r') if echo else ''
        if c == 'ATZ': time.sleep(0.5); echo = True; out += '\r\rELM327 v1.5\r\r'
        elif c == 'ATE0': echo = False; out = 'OK\r\r'
        elif c == 'ATRV': out += '12.6V\r\r'
        elif c == 'ATDPN': out += 'A6\r\r'
        elif c == 'ATDP': out += 'AUTO,\r\r'
        elif c.startswith('AT'): out += 'OK\r\r'
        elif c == '0100':
            time.sleep(1.0)
            out += 'SEARCHING...\rUNABLE TO CONNECT\r\r' if t < off_s else 'SEARCHING...\r41 00 BE 3F A8 13\r\r'
        elif c in ('010D', '010D1'):
            time.sleep(0.03 if c == '010D1' else 0.08)
            v = int(45 + 40 * math.sin(t / 5)); n += 1
            out += '41 0D %02X\r\r' % v if t >= off_s else 'NO DATA\r\r'
        else: out += '?\r\r'
        os.write(m, (out + '>').encode())
