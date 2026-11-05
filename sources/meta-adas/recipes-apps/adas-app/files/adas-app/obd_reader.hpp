#pragma once
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <pthread.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <fstream>

class ObdReader {
public:
    enum State { OFF, NO_DEVICE, NO_ELM, NO_CAR, OK };

    ~ObdReader() { stop(); }

    void start(std::function<void(double)> onSpeed) {
        if (const char* e = getenv("ADAS_OBD"); e && atoi(e) == 0) { setState(OFF, "tắt (ADAS_OBD=0)"); return; }
        if (m_thr.joinable()) return;
        m_onSpeed = std::move(onSpeed);
        m_periodMs = envi("ADAS_OBD_PERIOD_MS", 100);
        m_staleS = envi("ADAS_OBD_STALE_MS", 1000) / 1000.0;
        m_run = true;
        m_thr = std::thread([this] { pthread_setname_np(pthread_self(), "adas-obd"); run(); });
    }

    void stop() {
        m_run = false;
        if (m_thr.joinable()) m_thr.join();
        closePort();
    }

    std::string json() const {
        std::lock_guard<std::mutex> lk(m_mtx);
        std::ostringstream ss;
        ss << "{\"state\":\"" << kStateNames[m_state] << "\",\"text\":\"" << m_text << "\",\"kph\":" << m_kph
           << ",\"hz\":" << m_hz << ",\"volt\":" << m_volt << ",\"dev\":\"" << m_dev << "\",\"elm\":\"" << m_elm
           << "\",\"proto\":\"" << m_proto << "\"}";
        return ss.str();
    }

    const char* stateName() const { std::lock_guard<std::mutex> lk(m_mtx); return kStateNames[m_state]; }

private:
    static constexpr const char* kStateNames[] = {"off", "no_device", "no_elm", "no_car", "ok"};
    using Clock = std::chrono::steady_clock;
    static int envi(const char* n, int d) { const char* e = getenv(n); return e && *e ? atoi(e) : d; }
    static std::string clean(std::string s) {
        s.erase(std::remove_if(s.begin(), s.end(), [](char c) { return c == '"' || c == '\\' || c < 32; }), s.end());
        return s.substr(0, 48);
    }
    static std::string protoName(const std::string& r) {
        static const char* names[] = {"", "SAE J1850 PWM", "SAE J1850 VPW", "ISO 9141-2", "ISO 14230-4 KWP (5 baud)",
                                      "ISO 14230-4 KWP (fast)", "ISO 15765-4 CAN (11 bit, 500k)",
                                      "ISO 15765-4 CAN (29 bit, 500k)", "ISO 15765-4 CAN (11 bit, 250k)",
                                      "ISO 15765-4 CAN (29 bit, 250k)", "SAE J1939 CAN"};
        std::string s = r;
        const bool autoSel = !s.empty() && s[0] == 'A';
        if (autoSel) s.erase(0, 1);
        if (s.size() != 1 || !isxdigit((unsigned char)s[0])) return "";
        const int n = (int)strtol(s.c_str(), nullptr, 16);
        if (n < 1 || n > 10) return "";
        return std::string(names[n]) + (autoSel ? " (tự dò)" : "");
    }
    static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

    void setState(State s, const std::string& text) {
        std::lock_guard<std::mutex> lk(m_mtx);
        m_state = s; m_text = text;
        if (s != OK) { m_kph = -1; m_hz = 0; }
    }

    void nap(int ms) {
        for (int i = 0; i < ms && m_run; i += 50) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    static speed_t baudConst(int b) {
        switch (b) { case 9600: return B9600; case 19200: return B19200; case 57600: return B57600;
                     case 115200: return B115200; case 230400: return B230400; default: return B38400; }
    }

    bool openPort(const std::string& dev, int baud) {
        closePort();
        m_fd = ::open(dev.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
        if (m_fd < 0) return false;
        termios tio{};
        if (tcgetattr(m_fd, &tio) == 0) {
            cfmakeraw(&tio);
            tio.c_cflag |= CLOCAL | CREAD;
            tio.c_cflag &= ~(CSTOPB | CRTSCTS);
            cfsetispeed(&tio, baudConst(baud)); cfsetospeed(&tio, baudConst(baud));
            tcsetattr(m_fd, TCSANOW, &tio);
        }
        tcflush(m_fd, TCIOFLUSH);
        return true;
    }

    void closePort() { if (m_fd >= 0) { ::close(m_fd); m_fd = -1; } }

    struct BtAddr { uint8_t b[6]; } __attribute__((packed));
    struct SockaddrRc { sa_family_t rc_family; BtAddr rc_bdaddr; uint8_t rc_channel; };
    static constexpr int kAfBluetooth = 31, kBtProtoRfcomm = 3;

    static std::string btMac() {
        std::string mac;
        if (const char* e = getenv("ADAS_OBD_BT"); e && *e) mac = e;
        else { std::ifstream f("/var/lib/adas/obd_bt"); f >> mac; }
        unsigned x[6];
        return sscanf(mac.c_str(), "%x:%x:%x:%x:%x:%x", &x[0], &x[1], &x[2], &x[3], &x[4], &x[5]) == 6 ? mac : "";
    }

    bool openBt(const std::string& mac, int channel) {
        closePort();
        unsigned x[6];
        sscanf(mac.c_str(), "%x:%x:%x:%x:%x:%x", &x[0], &x[1], &x[2], &x[3], &x[4], &x[5]);
        SockaddrRc a{};
        a.rc_family = kAfBluetooth; a.rc_channel = uint8_t(channel);
        for (int i = 0; i < 6; ++i) a.rc_bdaddr.b[5 - i] = uint8_t(x[i]);
        m_fd = ::socket(kAfBluetooth, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, kBtProtoRfcomm);
        if (m_fd < 0) { m_btErr = errno; return false; }
        if (::connect(m_fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0 && errno != EINPROGRESS) {
            m_btErr = errno; closePort(); return false;
        }
        for (int waited = 0; m_run && waited < 20000; waited += 100) {
            pollfd p{m_fd, POLLOUT, 0};
            if (::poll(&p, 1, 100) > 0) {
                int err = 0; socklen_t l = sizeof(err);
                getsockopt(m_fd, SOL_SOCKET, SO_ERROR, &err, &l);
                if (err == 0) return true;
                m_btErr = err; closePort(); return false;
            }
        }
        m_btErr = ETIMEDOUT; closePort(); return false;
    }

    void drain() {
        if (m_fd < 0) return;
        tcflush(m_fd, TCIFLUSH);
        char buf[256];
        while (::read(m_fd, buf, sizeof(buf)) > 0) {}
    }

    bool cmd(const std::string& c, std::string& resp, int timeoutMs) {
        resp.clear();
        if (m_fd < 0) return false;
        drain();
        const std::string out = c + "\r";
        if (::write(m_fd, out.data(), out.size()) != (ssize_t)out.size()) { m_ioError = true; return false; }
        std::string raw;
        const auto t0 = Clock::now();
        while (m_run) {
            const int left = timeoutMs - int(secs(t0, Clock::now()) * 1000);
            if (left <= 0) return false;
            pollfd p{m_fd, POLLIN, 0};
            const int r = ::poll(&p, 1, std::min(left, 100));
            if (r < 0) { if (errno == EINTR) continue; m_ioError = true; return false; }
            if (r == 0) continue;
            if (p.revents & (POLLERR | POLLHUP | POLLNVAL)) { m_ioError = true; return false; }
            char buf[256];
            const ssize_t n = ::read(m_fd, buf, sizeof(buf));
            if (n < 0) { if (errno == EAGAIN || errno == EINTR) continue; m_ioError = true; return false; }
            if (n == 0) { m_ioError = true; return false; }
            raw.append(buf, n);
            if (raw.find('>') != std::string::npos) break;
            if (raw.size() > 4096) raw.erase(0, 2048);
        }
        if (!m_run) return false;
        raw = raw.substr(0, raw.find('>'));
        for (const char* junk : {"SEARCHING...", "BUS INIT: ...", "BUS INIT:"}) {
            for (size_t p; (p = raw.find(junk)) != std::string::npos;) raw.erase(p, strlen(junk));
        }
        if (raw.compare(0, c.size(), c) == 0) raw.erase(0, c.size());
        for (char ch : raw) if (ch != ' ' && ch != '\r' && ch != '\n' && ch != '\0') resp += char(toupper((unsigned char)ch));
        return true;
    }

    static int parseSpeed(const std::string& r) {
        const size_t p = r.find("410D");
        if (p == std::string::npos || p + 6 > r.size()) return -1;
        char* end = nullptr;
        const std::string hx = r.substr(p + 4, 2);
        const long v = strtol(hx.c_str(), &end, 16);
        return (end && *end == '\0') ? int(v) : -1;
    }

    bool findAdapter() {
        if (const std::string mac = btMac(); !mac.empty()) {
            m_ioError = false;
            std::string r;
            bool elm = false;
            if (openBt(mac, envi("ADAS_OBD_BT_CH", 1)))
                for (int k = 0; k < 2 && !elm && !m_ioError && m_run; ++k)
                    elm = cmd("ATZ", r, 2500) && r.find("ELM") != std::string::npos;
            if (elm) {
                nap(200);
                std::lock_guard<std::mutex> lk(m_mtx);
                m_dev = "BT " + mac; m_elm = clean(r.substr(r.find("ELM")));
                return true;
            }
            closePort();
            const int e = m_btErr; m_btErr = 0;
            setState(NO_DEVICE, "không kết nối được ELM327 Bluetooth " + mac +
                     (e == EAFNOSUPPORT || e == EPROTONOSUPPORT ? " (Pi chưa có Bluetooth)"
                      : e == EHOSTDOWN || e == ETIMEDOUT || e == EHOSTUNREACH ? " (chưa cắm vào xe / ngoài tầm)"
                      : e == EACCES || e == ECONNREFUSED ? " (chưa ghép đôi? chạy pi_obd_pair.sh)"
                      : e ? std::string(" (") + strerror(e) + ")" : " (không trả lời ATZ)"));
            return false;
        }
        std::vector<std::string> devs;
        if (const char* e = getenv("ADAS_OBD_DEV"); e && *e) devs.push_back(e);
        else devs = {"/dev/adas-obd", "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2", "/dev/ttyUSB3",
                     "/dev/ttyACM0", "/dev/ttyACM1", "/dev/rfcomm0"};
        std::vector<int> bauds;
        if (int b = envi("ADAS_OBD_BAUD", 0); b > 0) bauds.push_back(b); else bauds = {38400, 9600, 115200};
        bool anyDev = false;
        for (const auto& d : devs) {
            struct stat st;
            if (::stat(d.c_str(), &st) != 0) continue;
            anyDev = true;
            for (int b : bauds) {
                if (!m_run || !openPort(d, b)) break;
                m_ioError = false;
                std::string r;
                bool elm = false;
                for (int k = 0; k < 2 && !elm && !m_ioError && m_run; ++k)
                    elm = cmd("ATZ", r, 2500) && r.find("ELM") != std::string::npos;
                if (elm) {
                    nap(200);
                    std::lock_guard<std::mutex> lk(m_mtx);
                    m_dev = d; m_elm = clean(r.substr(r.find("ELM")));
                    return true;
                }
                if (m_ioError) break;
            }
            closePort();
        }
        setState(anyDev ? NO_ELM : NO_DEVICE, anyDev ? "có cổng serial nhưng không phải ELM327" : "không thấy bộ ELM327");
        return false;
    }

    bool setupElm() {
        std::string r;
        for (const char* c : {"ATE0", "ATL0", "ATS0", "ATH0", "ATAT1", "ATSP0"}) {
            if (!cmd(c, r, 1500)) return false;
        }
        return true;
    }

    bool connectCar() {
        std::string r;
        if (!cmd("0100", r, 12000)) return false;
        if (r.find("4100") == std::string::npos) {
            setState(NO_CAR, r.find("UNABLETOCONNECT") != std::string::npos || r.find("NODATA") != std::string::npos
                                 ? "ELM327 OK, xe chưa trả lời (chưa bật khóa điện?)" : "ELM327 OK, xe chưa trả lời");
            return false;
        }
        std::string proto;
        if (cmd("ATDPN", r, 1000)) proto = protoName(r);
        if (proto.empty() && cmd("ATDP", r, 1000)) proto = clean(r);
        { std::lock_guard<std::mutex> lk(m_mtx); m_proto = proto; }
        m_speedCmd = "010D";
        if (cmd("010D1", r, 1000) && parseSpeed(r) >= 0) m_speedCmd = "010D1";
        return true;
    }

    int btRetryMs(Clock::time_point started) const {
        if (secs(started, Clock::now()) < 120) return 15000;
        return std::max(5, envi("ADAS_OBD_BT_RETRY_S", 45)) * 1000;
    }

    void run() {
        const auto started = Clock::now();
        while (m_run) {
            if (!findAdapter()) { nap(btMac().empty() ? 3000 : btRetryMs(started)); continue; }
            if (!setupElm()) { setState(NO_ELM, "ELM327 không trả lời lệnh cấu hình"); closePort(); nap(2000); continue; }
            setState(NO_CAR, "đang dò giao thức của xe...");
            while (m_run && !m_ioError && !connectCar()) nap(3000);
            if (!m_run) break;
            if (m_ioError) { setState(NO_DEVICE, "mất kết nối bộ ELM327"); closePort(); nap(2000); continue; }
            try {
                pollSpeed();
            } catch (const std::exception& e) {
                setState(NO_DEVICE, std::string("lỗi đọc OBD: ") + e.what());
            }
            if (m_onSpeed) m_onSpeed(-1);
            closePort();
            nap(1000);
        }
        if (m_onSpeed) m_onSpeed(-1);
    }

    void pollSpeed() {
        auto lastOk = Clock::now(), lastVolt = Clock::time_point(), rateT0 = Clock::now();
        bool stale = false;
        int answers = 0;
        std::string r;
        while (m_run) {
            const auto t = Clock::now();
            const bool ok = cmd(m_speedCmd, r, 1000);
            if (m_ioError) { setState(NO_DEVICE, "mất kết nối bộ ELM327"); return; }
            const int v = ok ? parseSpeed(r) : -1;
            const auto now = Clock::now();
            if (v >= 0 && v <= 250) {
                lastOk = now; stale = false; ++answers;
                if (m_onSpeed) m_onSpeed(v);
                std::lock_guard<std::mutex> lk(m_mtx);
                m_state = OK; m_kph = v; m_text = "đang đọc tốc độ";
            } else if (!stale && secs(lastOk, now) > m_staleS) {
                stale = true;
                if (m_onSpeed) m_onSpeed(-1);
                setState(NO_CAR, "xe không trả lời tốc độ");
            }
            if (secs(lastOk, now) > 5.0) return;
            if (secs(rateT0, now) >= 2.0) {
                std::lock_guard<std::mutex> lk(m_mtx);
                m_hz = answers / secs(rateT0, now); answers = 0; rateT0 = now;
            }
            if (secs(lastVolt, now) >= 2.0) {
                lastVolt = now;
                if (cmd("ATRV", r, 500)) {
                    const double volt = atof(r.c_str());
                    std::lock_guard<std::mutex> lk(m_mtx);
                    if (volt > 0) m_volt = volt;
                }
            }
            const int spent = int(secs(t, Clock::now()) * 1000);
            if (spent < m_periodMs) std::this_thread::sleep_for(std::chrono::milliseconds(m_periodMs - spent));
        }
    }

    std::thread m_thr;
    std::atomic<bool> m_run{false};
    std::function<void(double)> m_onSpeed;
    int m_fd = -1;
    bool m_ioError = false;
    int m_btErr = 0;
    int m_periodMs = 100;
    double m_staleS = 1.0;
    std::string m_speedCmd = "010D";

    mutable std::mutex m_mtx;
    State m_state = OFF;
    std::string m_text = "chưa chạy", m_dev, m_elm, m_proto;
    int m_kph = -1;
    double m_hz = 0, m_volt = 0;
};
