#ifndef MJPEGSERVER_HPP
#define MJPEGSERVER_HPP

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <sstream>
#include <fstream>
#include <functional>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/statvfs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/uio.h>
#include <sys/ioctl.h>
#include <linux/sockios.h>
#include <poll.h>
#include <sched.h>
#include <pthread.h>

#include <opencv2/opencv.hpp>

class MjpegServer {
public:
    using SwitchSourceFn = std::function<bool(const std::string&)>;
    using SetFrontOffsetFn = std::function<bool(double)>;
    using GetFrontOffsetFn = std::function<double()>;

    MjpegServer()
        : m_port(8080)
        , m_serverSock(-1)
        , m_running(false)
        , m_fps(0.0)
        , m_dist(-1.0)
        , m_speed(0)
        , m_speedLimit(60)
        , m_frameSeq(0)
        , m_lastClientActiveMs(0)
        , m_currentSource("camera")
        , m_lastFrameTime(std::chrono::steady_clock::now())
        , m_lastEncodeTime(std::chrono::steady_clock::now())
    {}

    ~MjpegServer() {
        stop();
    }

    void setSwitchSourceCallback(SwitchSourceFn cb) {
        m_onSwitchSource = cb;
    }

    void setFrontOffsetCallback(SetFrontOffsetFn cb) {
        m_onSetFrontOffset = cb;
    }

    void setGetFrontOffsetCallback(GetFrontOffsetFn cb) {
        m_onGetFrontOffset = cb;
    }

    void setCurrentSource(const std::string& src) {
        std::lock_guard<std::mutex> lock(m_statusMtx);
        m_currentSource = src;
    }

    std::string getCurrentSource() {
        std::lock_guard<std::mutex> lock(m_statusMtx);
        return m_currentSource;
    }

    void markClientActive() {
        auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        m_lastClientActiveMs.store(nowMs);
    }

    bool hasClients() const {
        if (m_clientCount.load() > 0) return true;
        auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        return (nowMs - m_lastClientActiveMs.load()) < 5000;
    }

    void updateStatusOnly(double fps, double dist, int speed, int speedLimit) {
        std::lock_guard<std::mutex> lock(m_statusMtx);
        m_fps = fps;
        m_dist = dist;
        m_speed = speed;
        m_speedLimit = speedLimit;
        m_lastFrameTime = std::chrono::steady_clock::now();
    }

    bool start(int port = 8080) {
        m_port = port;
        m_serverSock = socket(AF_INET, SOCK_STREAM, 0);
        if (m_serverSock < 0) {
            std::cerr << "[MjpegServer] Error creating socket\n";
            return false;
        }

        int opt = 1;
        setsockopt(m_serverSock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in serverAddr{};
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = INADDR_ANY;
        serverAddr.sin_port = htons(m_port);

        if (bind(m_serverSock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
            std::cerr << "[MjpegServer] Error binding to port " << m_port << "\n";
            close(m_serverSock);
            m_serverSock = -1;
            return false;
        }

        if (listen(m_serverSock, 15) < 0) {
            std::cerr << "[MjpegServer] Error listening on port " << m_port << "\n";
            close(m_serverSock);
            m_serverSock = -1;
            return false;
        }

        m_running = true;
        m_encThread = std::thread(&MjpegServer::encoderLoop, this);
        m_serverThread = std::thread(&MjpegServer::serverLoop, this);
        std::cout << "[MjpegServer] ADAS AI Studio & Streamer started at http://0.0.0.0:" << m_port << "\n";
        return true;
    }

    void stop() {
        bool wasRunning = m_running.exchange(false);
        if (wasRunning) {
            m_frameCv.notify_all();
            m_encCv.notify_all();
            if (m_serverSock >= 0) {
                shutdown(m_serverSock, SHUT_RDWR);
                close(m_serverSock);
                m_serverSock = -1;
            }
            if (m_serverThread.joinable()) {
                m_serverThread.join();
            }
            if (m_encThread.joinable()) {
                m_encThread.join();
            }
            {
                std::unique_lock<std::mutex> lk(m_handlersMtx);
                m_handlersCv.wait_for(lk, std::chrono::seconds(3), [&] {
                    return m_activeHandlers.load() == 0;
                });
            }
        }
    }

    void setFrame(const cv::Mat& camFrame, const std::string& telemJson, double fps = 0.0, double dist = -1.0, int speed = 0, int speedLimit = 60) {
        if (camFrame.empty()) return;
        {
            std::lock_guard<std::mutex> lock(m_statusMtx);
            m_fps = fps;
            m_dist = dist;
            m_speed = speed;
            m_speedLimit = speedLimit;
            m_latestTelemetryJson = telemJson;
            m_lastFrameTime = std::chrono::steady_clock::now();
        }

        if (!hasClients()) return;

        {
            std::lock_guard<std::mutex> lock(m_encMtx);
            m_encPendingCam = camFrame;
            m_encTelem = telemJson;
            m_encHasNew = true;
        }
        m_encCv.notify_one();
    }

    bool debugMask() const { return m_debugMask.load(std::memory_order_relaxed); }
    bool takeCalibReset() { return m_calibReset.exchange(false); }

private:
    void encoderLoop() {
        pthread_setname_np(pthread_self(), "adas-jpeg");
#if defined(__arm__) || defined(__aarch64__)
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(0, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
#endif
        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 45, cv::IMWRITE_JPEG_OPTIMIZE, 0 };
        while (m_running) {
            cv::Mat fCam;
            std::string telem;
            {
                std::unique_lock<std::mutex> lk(m_encMtx);
                m_encCv.wait_for(lk, std::chrono::milliseconds(200), [&] {
                    return m_encHasNew || !m_running;
                });
                if (!m_running) break;
                if (!m_encHasNew) continue;
                fCam = std::move(m_encPendingCam);
                telem = std::move(m_encTelem);
                m_encHasNew = false;
            }

            if (fCam.empty()) continue;

            cv::Mat encodeCam = fCam;
            cv::Mat resizedCam;
            if (fCam.cols > 640) {
                int target_w = 640;
                int target_h = static_cast<int>(std::round(fCam.rows * (640.0 / fCam.cols)));
                if (target_h % 2 != 0) target_h--;
                cv::resize(fCam, resizedCam, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);
                encodeCam = resizedCam;
            }

            std::vector<uchar> encodedCamBuf;
            bool okCam = false;
            try { okCam = cv::imencode(".jpg", encodeCam, encodedCamBuf, params); }
            catch (const cv::Exception& e) { fprintf(stderr, "[MjpegServer] encode error: %s\n", e.what()); }

            if (okCam) {
                {
                    std::lock_guard<std::mutex> lock(m_frameMtx);
                    m_latestJpeg = std::move(encodedCamBuf);
                    m_cachedTelemetryJson = std::move(telem);
                    m_frameSeq++;
                }
                m_frameCv.notify_all();
            }
        }
    }
    void serverLoop() {
        while (m_running) {
            sockaddr_in clientAddr{};
            socklen_t clientLen = sizeof(clientAddr);
            int clientSock = accept(m_serverSock, (struct sockaddr*)&clientAddr, &clientLen);
            if (clientSock < 0) {
                if (!m_running) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                continue;
            }

            struct timeval tv;
            tv.tv_sec = 3;
            tv.tv_usec = 0;
            setsockopt(clientSock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
            setsockopt(clientSock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));

            int nodelay = 1;
            setsockopt(clientSock, IPPROTO_TCP, TCP_NODELAY, (const char*)&nodelay, sizeof(nodelay));
            int sndBuf = 524288;
            setsockopt(clientSock, SOL_SOCKET, SO_SNDBUF, (const char*)&sndBuf, sizeof(sndBuf));

            if (m_activeHandlers.load() >= kMaxHandlers) {
                close(clientSock);
                continue;
            }
            m_activeHandlers++;

            std::thread([this, clientSock]() {
                pthread_setname_np(pthread_self(), "adas-http");
                try { handleClient(clientSock); }
                catch (const std::exception& e) { fprintf(stderr, "[MjpegServer] request error: %s\n", e.what()); }
                catch (...) { fprintf(stderr, "[MjpegServer] request error\n"); }
                close(clientSock);
                {
                    std::lock_guard<std::mutex> lk(m_handlersMtx);
                    m_activeHandlers--;
                }
                m_handlersCv.notify_all();
            }).detach();
        }
    }

    void handleClient(int clientSock) {
        std::vector<char> headerBuf(4096, 0);
        ssize_t bytesRead = recv(clientSock, headerBuf.data(), headerBuf.size() - 1, 0);
        if (bytesRead <= 0) return;

        std::string request(headerBuf.data(), bytesRead);

        if (request.find("GET /stream") != std::string::npos) {
            markClientActive();
            m_clientCount++;
            struct StreamClientGuard {
                std::atomic<int>& count;
                ~StreamClientGuard() { count--; }
            } clientGuard{m_clientCount};

            int nodelay = 1;
            setsockopt(clientSock, IPPROTO_TCP, TCP_NODELAY, (char*)&nodelay, sizeof(nodelay));

            int sndBuf = 262144;
            setsockopt(clientSock, SOL_SOCKET, SO_SNDBUF, (const char*)&sndBuf, sizeof(sndBuf));

            std::string header =
                "HTTP/1.1 200 OK\r\n"
                "Server: ADAS-Pi4-Streamer\r\n"
                "Connection: close\r\n"
                "Cache-Control: no-cache, no-store, must-revalidate, pre-check=0, post-check=0, max-age=0\r\n"
                "Pragma: no-cache\r\n"
                "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";

            if (send(clientSock, header.c_str(), header.length(), MSG_NOSIGNAL) <= 0) return;

            uint64_t lastSentSeq = 0;
            while (m_running) {
                std::vector<uchar> jpegData;
                uint64_t currentSeq = 0;
                {
                    std::unique_lock<std::mutex> lock(m_frameMtx);
                    m_frameCv.wait_for(lock, std::chrono::milliseconds(40), [&] {
                        return !m_running || m_frameSeq.load() != lastSentSeq;
                    });
                    if (!m_running) break;
                    if (m_frameSeq.load() == lastSentSeq) {
                        continue;
                    }
                    jpegData = m_latestJpeg;
                    currentSeq = m_frameSeq.load();
                }

                if (!jpegData.empty()) {
                    struct pollfd pfd{clientSock, POLLOUT, 0};
                    int pret = poll(&pfd, 1, 50);
                    if (pret < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) break;
                    if (pret == 0 || !(pfd.revents & POLLOUT)) {
                        lastSentSeq = currentSeq;
                        continue;
                    }

                    std::string partHeader = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                             std::to_string(jpegData.size()) + "\r\n\r\n";

                    if (!sendAll(clientSock, partHeader.data(), partHeader.size())) break;
                    if (!sendAll(clientSock, jpegData.data(), jpegData.size())) break;
                    if (!sendAll(clientSock, "\r\n", 2)) break;

                    lastSentSeq = currentSeq;
                }
            }
            return;
        }

        if (request.find("GET /api/live") != std::string::npos) {
            markClientActive();
            m_clientCount++;
            struct LiveClientGuard {
                std::atomic<int>& count;
                ~LiveClientGuard() { count--; }
            } clientGuard{m_clientCount};

            int nodelay = 1;
            setsockopt(clientSock, IPPROTO_TCP, TCP_NODELAY, (char*)&nodelay, sizeof(nodelay));
            std::string header =
                "HTTP/1.1 200 OK\r\n"
                "Connection: close\r\n"
                "Cache-Control: no-cache, no-store, must-revalidate\r\n"
                "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
            if (!sendAll(clientSock, header.data(), header.size())) return;

            uint64_t lastSentSeq = 0;
            while (m_running) {
                std::vector<uchar> jpegData;
                std::string telem;
                uint64_t seq = 0;
                {
                    std::unique_lock<std::mutex> lock(m_frameMtx);
                    m_frameCv.wait_for(lock, std::chrono::milliseconds(200), [&] {
                        return !m_running || m_frameSeq.load() != lastSentSeq;
                    });
                    if (!m_running) break;
                    seq = m_frameSeq.load();
                    if (seq == lastSentSeq || m_latestJpeg.empty()) continue;
                    jpegData = m_latestJpeg;
                    telem = m_cachedTelemetryJson;
                }
                lastSentSeq = seq;

                int queued = 0;
                if (ioctl(clientSock, SIOCOUTQ, &queued) == 0 && queued > 16384) continue;

                std::string part = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                   std::to_string(jpegData.size()) + "\r\nX-Frame-Seq: " + std::to_string(seq) +
                                   "\r\nX-Telemetry: " + telem + "\r\n\r\n";
                if (!sendAll(clientSock, part.data(), part.size())) break;
                if (!sendAll(clientSock, jpegData.data(), jpegData.size())) break;
                if (!sendAll(clientSock, "\r\n", 2)) break;
            }
            return;
        }

        if (request.find("GET /api/frame.jpg") != std::string::npos) {
            markClientActive();

            uint64_t waitSeq = 0;
            size_t wsPos = request.find("wait_seq=");
            if (wsPos != std::string::npos) {
                waitSeq = std::strtoull(request.c_str() + wsPos + 9, nullptr, 10);
            }

            std::vector<uchar> data;
            uint64_t seq = 0;
            std::string telem;
            double curFps = 0.0, curDist = -1.0;
            int curSpeed = 0, curLimit = 60;

            {
                std::unique_lock<std::mutex> lock(m_frameMtx);
                if (waitSeq > 0 && m_frameSeq.load() <= waitSeq && m_running) {
                    m_frameCv.wait_for(lock, std::chrono::milliseconds(50), [&] {
                        return !m_running || m_frameSeq.load() > waitSeq;
                    });
                }
                data = m_latestJpeg;
                seq = m_frameSeq.load();
                telem = m_cachedTelemetryJson;
            }

            {
                std::lock_guard<std::mutex> lock(m_statusMtx);
                curFps = m_fps;
                curDist = m_dist;
                curSpeed = m_speed;
                curLimit = m_speedLimit;
            }

            if (data.empty()) {
                sendJson(clientSock, 404, "{\"status\":\"no_frame\"}");
                return;
            }

            int nodelay = 1;
            setsockopt(clientSock, IPPROTO_TCP, TCP_NODELAY, (char*)&nodelay, sizeof(nodelay));

            std::ostringstream resp;
            resp << "HTTP/1.1 200 OK\r\n"
                 << "Content-Type: image/jpeg\r\n"
                 << "Cache-Control: no-cache, no-store, must-revalidate\r\n"
                 << "Access-Control-Allow-Origin: *\r\n"
                 << "Access-Control-Expose-Headers: X-Frame-Seq, X-Frame-Fps, X-Frame-Dist, X-Frame-Speed, X-Frame-Limit, X-Telemetry\r\n"
                 << "X-Frame-Seq: " << seq << "\r\n"
                 << "X-Frame-Fps: " << curFps << "\r\n"
                 << "X-Frame-Dist: " << curDist << "\r\n"
                 << "X-Frame-Speed: " << curSpeed << "\r\n"
                 << "X-Frame-Limit: " << curLimit << "\r\n";
            if (!telem.empty()) {
                resp << "X-Telemetry: " << telem << "\r\n";
            }
            resp << "Connection: close\r\n"
                 << "Content-Length: " << data.size() << "\r\n\r\n";
            std::string h = resp.str();
            sendAll(clientSock, h.c_str(), h.length());
            sendAll(clientSock, data.data(), data.size());
            return;
        }

        if (request.find("GET /api/telemetry") != std::string::npos) {
            markClientActive();
            std::string telem;
            {
                std::lock_guard<std::mutex> lock(m_statusMtx);
                telem = m_latestTelemetryJson.empty() ? "{}" : m_latestTelemetryJson;
            }
            sendJson(clientSock, 200, telem);
            return;
        }

        if (request.find("GET /api/debug") != std::string::npos) {
            markClientActive();
            const size_t lineEnd = request.find("\r\n");
            const size_t mPos = request.find("mask=");
            if (mPos != std::string::npos && mPos < lineEnd && mPos + 5 < request.size())
                m_debugMask.store(request[mPos + 5] == '1', std::memory_order_relaxed);
            sendJson(clientSock, 200, std::string("{\"status\":\"ok\",\"mask\":") + (debugMask() ? "true" : "false") + "}");
            return;
        }

        if (request.find("POST /api/calibration/reset") != std::string::npos) {
            markClientActive();
            m_calibReset.store(true);
            sendJson(clientSock, 200, "{\"status\":\"ok\"}");
            return;
        }

        if (request.find("GET /api/calibration") != std::string::npos) {
            markClientActive();
            double offset = m_onGetFrontOffset ? m_onGetFrontOffset() : 1.6;
            std::ostringstream ss;
            ss.imbue(std::locale::classic());
            ss << "{\"status\":\"ok\",\"camera_front_offset_m\":" << std::fixed << std::setprecision(2) << offset << "}";
            sendJson(clientSock, 200, ss.str());
            return;
        }

        if (request.find("POST /api/calibration/front_offset") != std::string::npos) {
            std::string fullReq = request;
            size_t clPos = fullReq.find("Content-Length: ");
            if (clPos == std::string::npos) clPos = fullReq.find("content-length: ");
            if (clPos != std::string::npos) {
                long long cl = std::atoll(fullReq.c_str() + clPos + 16);
                size_t headEnd = fullReq.find("\r\n\r\n");
                if (headEnd != std::string::npos) {
                    size_t bodyGot = fullReq.length() - (headEnd + 4);
                    while ((long long)bodyGot < cl) {
                        char buf[512];
                        ssize_t n = recv(clientSock, buf, std::min((long long)sizeof(buf), cl - (long long)bodyGot), 0);
                        if (n <= 0) break;
                        fullReq.append(buf, n);
                        bodyGot += n;
                    }
                }
            }
            double offset = 1.6;
            auto parseVal = [](const std::string& req, const std::string& key, double& val) {
                size_t pos = req.find(key);
                if (pos != std::string::npos) {
                    pos += key.length();
                    while (pos < req.length() && (req[pos] == ':' || req[pos] == '=' || req[pos] == ' ' || req[pos] == '"')) pos++;
                    double whole = 0.0, frac = 0.0, div = 1.0;
                    bool inFrac = false;
                    bool hasDigit = false;
                    while (pos < req.length() && (std::isdigit(req[pos]) || req[pos] == '.' || req[pos] == ',')) {
                        char c = req[pos++];
                        if (c == '.' || c == ',') { inFrac = true; }
                        else if (std::isdigit(c)) {
                            hasDigit = true;
                            if (!inFrac) whole = whole * 10.0 + (c - '0');
                            else { frac = frac * 10.0 + (c - '0'); div *= 10.0; }
                        }
                    }
                    if (hasDigit) {
                        val = whole + frac / div;
                        return true;
                    }
                }
                return false;
            };
            if (!parseVal(fullReq, "camera_front_offset_m", offset)) {
                parseVal(fullReq, "offset_m", offset);
            }
            offset = std::clamp(offset, 0.0, 3.0);
            bool ok = true;
            if (m_onSetFrontOffset) {
                ok = m_onSetFrontOffset(offset);
            }
            if (ok) {
                std::ostringstream ss;
                ss.imbue(std::locale::classic());
                ss << "{\"status\":\"ok\",\"camera_front_offset_m\":" << std::fixed << std::setprecision(2) << offset << "}";
                sendJson(clientSock, 200, ss.str());
            } else {
                sendJson(clientSock, 500, "{\"status\":\"error\",\"message\":\"Khong the luu thiet lap\"}");
            }
            return;
        }

        if (request.find("POST /api/source") != std::string::npos) {
            std::string fullReq = request;
            size_t clPos = fullReq.find("Content-Length: ");
            if (clPos == std::string::npos) clPos = fullReq.find("content-length: ");
            if (clPos != std::string::npos) {
                long long cl = std::atoll(fullReq.c_str() + clPos + 16);
                size_t headEnd = fullReq.find("\r\n\r\n");
                if (headEnd != std::string::npos) {
                    size_t bodyGot = fullReq.length() - (headEnd + 4);
                    while ((long long)bodyGot < cl) {
                        char buf[512];
                        ssize_t n = recv(clientSock, buf, std::min((long long)sizeof(buf), cl - (long long)bodyGot), 0);
                        if (n <= 0) break;
                        fullReq.append(buf, n);
                        bodyGot += n;
                    }
                }
            }

            std::string targetSource = "camera";
            if (fullReq.find("source=video") != std::string::npos) {
                targetSource = "/tmp/uploaded_test.mp4";
            }

            bool ok = true;
            if (m_onSwitchSource) {
                ok = m_onSwitchSource(targetSource);
            }
            if (ok) {
                setCurrentSource((targetSource == "camera") ? "camera" : "video");
                sendJson(clientSock, 200, "{\"status\":\"ok\",\"source\":\"" + getCurrentSource() + "\"}");
            } else {
                sendJson(clientSock, 409, "{\"status\":\"error\",\"message\":\"Khong the chuyen sang nguon nay\",\"source\":\"" + getCurrentSource() + "\"}");
            }
            return;
        }

        if (request.find("GET /api/status") != std::string::npos) {
            markClientActive();
            std::string curSrc = getCurrentSource();
            double curFps; int curSpeed, curLimit; double curDist;
            std::chrono::steady_clock::time_point lastTime;
            {
                std::lock_guard<std::mutex> lock(m_statusMtx);
                curFps = m_fps;
                curSpeed = m_speed;
                curDist = m_dist;
                curLimit = m_speedLimit;
                lastTime = m_lastFrameTime;
            }
            auto now = std::chrono::steady_clock::now();
            auto ageMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime).count();

            std::ostringstream json;
            json << "{\"source\":\"" << curSrc << "\","
                 << "\"fps\":" << curFps << ","
                 << "\"speed\":" << curSpeed << ","
                 << "\"dist\":" << curDist << ","
                 << "\"speedLimit\":" << curLimit << ","
                 << "\"frames\":" << m_frameSeq.load() << ","
                 << "\"frameAgeMs\":" << ageMs << "}";
            sendJson(clientSock, 200, json.str());
            return;
        }

        if (request.find("POST /upload") != std::string::npos) {
            handleVideoUpload(clientSock, request, headerBuf.data(), bytesRead);
            return;
        }

        markClientActive();
        serveDashboardHtml(clientSock);
    }

    void handleVideoUpload(int clientSock, const std::string& initRequest, const char* initBuf, ssize_t initLen) {
        size_t clPos = initRequest.find("Content-Length: ");
        if (clPos == std::string::npos) clPos = initRequest.find("content-length: ");
        if (clPos == std::string::npos) {
            sendJson(clientSock, 400, "{\"status\":\"error\",\"message\":\"Thieu Content-Length header\"}");
            return;
        }
        long long contentLength = std::atoll(initRequest.c_str() + clPos + 16);
        if (contentLength <= 0) {
            sendJson(clientSock, 400, "{\"status\":\"error\",\"message\":\"File video rong (0 bytes)\"}");
            return;
        }

        struct statvfs vfs;
        if (statvfs("/tmp", &vfs) == 0) {
            unsigned long long freeBytes = (unsigned long long)vfs.f_bavail * vfs.f_frsize;
            if ((unsigned long long)contentLength + 5000000ULL > freeBytes) {
                sendJson(clientSock, 507, "{\"status\":\"error\",\"message\":\"Bo nho /tmp khong du cho trong de luu video!\"}");
                return;
            }
        }

        size_t headerEnd = initRequest.find("\r\n\r\n");
        if (headerEnd == std::string::npos) {
            sendJson(clientSock, 400, "{\"status\":\"error\",\"message\":\"HTTP header khong hop le\"}");
            return;
        }
        size_t bodyStartInInit = headerEnd + 4;
        ssize_t bodyBytesInInit = initLen - bodyStartInInit;

        const std::string tempPath = "/tmp/upload_temp.mp4";
        const std::string finalPath = "/tmp/uploaded_test.mp4";

        FILE* fp = fopen(tempPath.c_str(), "wb");
        if (!fp) {
            sendJson(clientSock, 500, "{\"status\":\"error\",\"message\":\"Khong the tao file tren dia\"}");
            return;
        }

        if (bodyBytesInInit > 0) {
            size_t written = fwrite(initBuf + bodyStartInInit, 1, bodyBytesInInit, fp);
            if (written != (size_t)bodyBytesInInit) {
                fclose(fp);
                unlink(tempPath.c_str());
                sendJson(clientSock, 500, "{\"status\":\"error\",\"message\":\"Loi ghi dia ban dau\"}");
                return;
            }
        }

        struct timeval tvUpload;
        tvUpload.tv_sec = 20;
        tvUpload.tv_usec = 0;
        setsockopt(clientSock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tvUpload, sizeof(tvUpload));

        long long totalReceived = bodyBytesInInit;
        std::vector<char> chunk(65536);
        bool uploadSuccess = true;

        while (totalReceived < contentLength) {
            size_t toRead = std::min((long long)chunk.size(), contentLength - totalReceived);
            ssize_t n = recv(clientSock, chunk.data(), toRead, 0);
            if (n <= 0) {
                uploadSuccess = false;
                break;
            }
            size_t w = fwrite(chunk.data(), 1, n, fp);
            if (w != (size_t)n) {
                uploadSuccess = false;
                break;
            }
            totalReceived += n;
        }

        fflush(fp);
        fsync(fileno(fp));
        fclose(fp);

        if (!uploadSuccess || totalReceived < contentLength) {
            unlink(tempPath.c_str());
            std::cerr << "[MjpegServer] Upload bi gian doan: nhan " << totalReceived << "/" << contentLength << " bytes\n";
            sendJson(clientSock, 400, "{\"status\":\"error\",\"message\":\"Duong truyen Wi-Fi bi ngat giua chung. Vui long tai lai!\"}");
            return;
        }

        rename(tempPath.c_str(), finalPath.c_str());
        unlink((finalPath + ".calib.json").c_str());
        std::cout << "[MjpegServer] Da nhan tron ven video: " << finalPath << " (" << totalReceived << " bytes)\n";

        bool switchOk = false;
        if (m_onSwitchSource) {
            switchOk = m_onSwitchSource(finalPath);
        }

        if (switchOk) {
            setCurrentSource("video");
            sendJson(clientSock, 200, "{\"status\":\"ok\",\"message\":\"Upload thanh cong! AI dang chay video test.\",\"source\":\"video\"}");
        } else {
            sendJson(clientSock, 409, "{\"status\":\"error\",\"message\":\"Khong mo duoc video! Video phai la chuan MP4 (H.264/AVC). Hay chuyen doi bang ffmpeg!\",\"source\":\"camera\"}");
        }
    }

    void sendJson(int clientSock, int code, const std::string& body) {
        std::string statusText = (code == 200) ? "OK" : (code == 400) ? "Bad Request" : (code == 409) ? "Conflict" : (code == 507) ? "Insufficient Storage" : "Internal Error";
        std::ostringstream resp;
        resp << "HTTP/1.1 " << code << " " << statusText << "\r\n"
             << "Content-Type: application/json; charset=UTF-8\r\n"
             << "Access-Control-Allow-Origin: *\r\n"
             << "Connection: close\r\n"
             << "Content-Length: " << body.length() << "\r\n\r\n"
             << body;
        std::string s = resp.str();
        send(clientSock, s.c_str(), s.length(), MSG_NOSIGNAL);
    }

    static bool sendAll(int sock, const void* data, size_t len) {
        const char* p = static_cast<const char*>(data);
        while (len > 0) {
            ssize_t n = send(sock, p, len, MSG_NOSIGNAL);
            if (n > 0) { p += n; len -= static_cast<size_t>(n); continue; }
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        return true;
    }

    void serveDashboardHtml(int clientSock) {
        static const std::string html = R"raw_html(<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ADAS Perception Studio</title>
<style>
  :root {
    --bg: #0b1016; --panel: #121a23; --panel2: #18222d; --line: #243241; --text: #e6edf3; --muted: #8b9bab;
    --accent: #22d3ee; --ok: #22c55e; --warn: #f59e0b; --bad: #ef4444; --lane: #22d3ee; --veh: #a78bfa;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  header { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; padding: 10px 16px; border-bottom: 1px solid var(--line); background: var(--panel); position: sticky; top: 0; z-index: 5; }
  header h1 { font-size: 16px; margin: 0; letter-spacing: .3px; }
  header h1 span { color: var(--accent); }
  .pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; background: var(--panel2); border: 1px solid var(--line); color: var(--muted); font-size: 12px; white-space: nowrap; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
  .dot.ok { background: var(--ok); } .dot.warn { background: var(--warn); } .dot.bad { background: var(--bad); }
  .spacer { flex: 1; }
  button, .btn { font: inherit; color: var(--text); background: var(--panel2); border: 1px solid var(--line); border-radius: 8px; padding: 6px 12px; cursor: pointer; }
  button:hover { border-color: var(--accent); }
  button.on { background: #0e3a44; border-color: var(--accent); color: #c9f7ff; }
  main { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 14px; padding: 14px 16px; max-width: 1600px; margin: 0 auto; }
  @media (max-width: 1100px) { main { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
  .card h2 { margin: 0; padding: 9px 12px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .6px; color: var(--muted); border-bottom: 1px solid var(--line); display: flex; align-items: center; gap: 8px; }
  .card h2 .r { margin-left: auto; text-transform: none; letter-spacing: 0; font-weight: 400; }
  .views { display: grid; grid-template-columns: minmax(0, 1fr) 260px; gap: 14px; }
  @media (max-width: 760px) { .views { grid-template-columns: 1fr; } }
  .stage { position: relative; background: #000; aspect-ratio: 16 / 9; }
  .stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
  .stage .nosig { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: var(--muted); }
  #bev { display: block; width: 100%; height: auto; background: #071016; }
  .signRow { display: flex; align-items: center; gap: 10px; margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--line); }
  .signRow .k { font-size: 11px; color: var(--mut, #94a3b8); }
  .signRow .s { font-size: 11px; color: var(--mut, #94a3b8); }
  .signName { font-size: 16px; font-weight: 700; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; padding: 10px 12px; border-top: 1px solid var(--line); }
  .chips button { padding: 4px 10px; font-size: 12px; border-radius: 999px; }
  .side { display: flex; flex-direction: column; gap: 14px; }
  .alert { padding: 14px 12px; font-weight: 700; font-size: 18px; text-align: center; letter-spacing: .4px; background: #0f2a1a; color: #86efac; border-bottom: 1px solid var(--line); }
  .alert.l1 { background: #3a2a06; color: #fcd34d; } .alert.l2 { background: #4a0d0d; color: #fecaca; animation: blink .6s steps(2) infinite; }
  @keyframes blink { 50% { background: #7f1d1d; } }
  .stats { display: grid; grid-template-columns: 1fr 1fr; }
  .stat { padding: 10px 12px; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); }
  .stat:nth-child(2n) { border-right: 0; }
  .stat .k { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }
  .stat .v { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .stat .s { font-size: 12px; color: var(--muted); }
  table.kv { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
  table.kv td { padding: 5px 12px; border-bottom: 1px solid var(--line); }
  table.kv td:last-child { text-align: right; color: var(--text); }
  table.kv td:first-child { color: var(--muted); }
  table.kv tr:last-child td { border-bottom: 0; }
  .calrow { padding: 8px 12px 12px; border-top: 1px solid var(--line); }
  .calrow button { width: 100%; }
  .calrow button.arm { background: #3a1414; border-color: var(--bad); color: #fecaca; }
  .bar { height: 6px; background: var(--panel2); border-radius: 3px; overflow: hidden; margin: 8px 12px; }
  .bar > div { height: 100%; width: 0; background: var(--accent); transition: width .3s; }
  .calib-text { padding: 0 12px 10px; font-size: 13px; }
  .src { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 10px 12px; }
  .drop { margin: 0 12px 12px; padding: 16px 10px; border: 1px dashed var(--line); border-radius: 10px; text-align: center; color: var(--muted); cursor: pointer; font-size: 13px; }
  .drop:hover, .drop.over { border-color: var(--accent); color: var(--text); }
  #toast { position: fixed; left: 50%; bottom: 20px; transform: translateX(-50%); background: var(--panel2); border: 1px solid var(--line); padding: 10px 16px; border-radius: 10px; opacity: 0; transition: opacity .25s; pointer-events: none; z-index: 20; max-width: 90vw; }
  #toast.show { opacity: 1; } #toast.err { border-color: var(--bad); }
  .modal { position: fixed; inset: 0; background: rgba(0,0,0,.6); display: none; align-items: center; justify-content: center; z-index: 30; padding: 16px; }
  .modal.open { display: flex; }
  .modal .box { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 20px; width: min(460px, 100%); }
  .modal h3 { margin: 0 0 6px; }
  .modal p { color: var(--muted); margin: 0 0 14px; font-size: 13px; }
  .modal input[type=range] { width: 100%; }
  .modal .val { font-size: 28px; font-weight: 700; text-align: center; margin: 8px 0; }
  .modal .row { display: flex; gap: 8px; justify-content: flex-end; margin-top: 14px; }
</style>
</head>
<body>
<header>
  <h1>ADAS <span>Perception Studio</span></h1>
  <span class="pill" id="pSrc"><span class="dot" id="dSrc"></span><span id="tSrc">--</span></span>
  <span class="pill"><span class="dot" id="dAi"></span><span id="tAi">AI --</span></span>
  <span class="pill" id="pCam"><span class="dot bad" id="dCam"></span><span id="tCam">CAM --</span></span>
  <span class="pill" id="pObd"><span class="dot bad" id="dObd"></span><span id="tObdH">OBD --</span></span>
  <span class="pill" id="tFps">-- FPS</span>
  <span class="spacer"></span>
  <button id="bMask" title="Tô mask vạch làn (AI + kinh điển) do Pi vẽ lên ảnh">Mask làn</button>
  <button onclick="openModal()">Khoảng cách đầu xe</button>
  <a class="btn" href="/stream" target="_blank" title="Luồng MJPEG thô (không overlay)">MJPEG</a>
</header>

<main>
  <section>
    <div class="views">
      <div class="card">
        <h2>Camera <span class="r" id="tLat">--</span></h2>
        <div class="stage" id="stage">
          <canvas id="cam"></canvas>
          <div class="nosig" id="nosig">Đang chờ khung hình…</div>
        </div>
        <div class="chips" id="chips"></div>
      </div>
      <div class="card">
        <h2>Bird's-eye <span class="r">±8 m × 50 m</span></h2>
        <canvas id="bev" width="260" height="470"></canvas>
      </div>
    </div>
  </section>

  <aside class="side">
    <div class="card">
      <div class="alert" id="alert">AN TOÀN</div>
      <div class="stats">
        <div class="stat"><div class="k">Xe phía trước</div><div class="v" id="vLead">--</div><div class="s" id="sLead">không có</div></div>
        <div class="stat"><div class="k">Xe trong làn</div><div class="v" id="vInLane">--</div><div class="s">trên làn mình</div></div>
        <div class="stat"><div class="k">Lệch tâm làn</div><div class="v" id="vOff">--</div><div class="s" id="sLane">--</div></div>
        <div class="stat"><div class="k">Bề rộng làn</div><div class="v" id="vWidth">--</div><div class="s" id="sWidth">--</div></div>
      </div>
      <div class="signRow"><canvas id="signIco" width="44" height="44"></canvas>
        <div><div class="k">Biển báo</div><div class="signName" id="vSign">--</div><div class="s" id="sSign">--</div></div></div>
    </div>

    <div class="card">
      <h2>AI</h2>
      <table class="kv">
        <tr><td>NanoDet (xe)</td><td id="kDet">--</td></tr>
        <tr><td>Box giữ / phát hiện</td><td id="kFilt">--</td></tr>
        <tr><td>Xe đang bám</td><td id="kVeh">--</td></tr>
        <tr><td>Vòng xử lý / hiển thị</td><td id="kFps">--</td></tr>
      </table>
    </div>

    <div class="card">
      <h2>Hiệu chuẩn camera <span class="r" id="tCal">--</span></h2>
      <div class="bar"><div id="calBar"></div></div>
      <table class="kv">
        <tr><td>Chân trời / điểm tụ</td><td id="kHz">--</td></tr>
        <tr><td>Mép capo</td><td id="kHood">--</td></tr>
        <tr><td>Chiều cao / lệch ngang</td><td id="kCam">--</td></tr>
        <tr><td>Góc nhìn ngang</td><td id="kFov">--</td></tr>
      </table>
      <div class="calrow"><button id="bRecal" onclick="recalibrate()" title="Xóa hiệu chuẩn đã lưu và tự hiệu chuẩn lại từ đầu">Hiệu chuẩn lại</button></div>
    </div>

    <div class="card">
      <h2>Xe (OBD-II) <span class="r" id="tObd">--</span></h2>
      <table class="kv">
        <tr><td>Tốc độ</td><td id="kObdSpd">--</td></tr>
        <tr><td>Trạng thái</td><td id="kObdTxt">--</td></tr>
        <tr><td>Bộ đọc / giao thức</td><td id="kObdDev">--</td></tr>
        <tr><td>Ắc quy</td><td id="kObdV">--</td></tr>
      </table>
    </div>

    <div class="card">
      <h2>Nguồn video</h2>
      <div class="src">
        <button id="bCam" onclick="switchSource('camera')">Camera</button>
        <button id="bVid" onclick="switchSource('video')">Video test</button>
      </div>
      <div class="drop" id="drop">Thả file MP4 (H.264) vào đây hoặc bấm để chọn<br><small>video được lặp lại; tự hiệu chuẩn sau ~10–20 s</small>
        <div class="bar" id="upWrap" style="display:none"><div id="upBar"></div></div>
      </div>
      <input type="file" id="file" accept="video/mp4,video/*" hidden>
    </div>
  </aside>
</main>

<div class="modal" id="modal">
  <div class="box">
    <h3>Khoảng cách từ camera tới đầu xe</h3>
    <p>Khoảng cách đo từ ống kính; trừ đi số này để ra khoảng cách giữa hai đầu xe. Ước lượng bằng mắt là đủ.</p>
    <div class="val" id="offVal">1,6 m</div>
    <input type="range" id="offSlider" min="0" max="3" step="0.1" value="1.6" oninput="showOff(this.value)">
    <div class="row"><button onclick="closeModal()">Huỷ</button><button class="on" onclick="saveOff()">Lưu</button></div>
  </div>
</div>
<div id="toast"></div>

<script>
const $ = id => document.getElementById(id);
const cam = $('cam'), ctx = cam.getContext('2d');
const bev = $('bev'), bctx = bev.getContext('2d');
let T = null, lastSeq = 0, renderN = 0, renderT = performance.now(), renderFps = 0;

const LAYERS = [
  ['carpet', 'Thảm làn', true], ['lanes', 'Vạch làn', true], ['vehs', 'Xe', true], ['lead', 'Xe phía trước', true],
  ['raw', 'Box thô (detector)', false], ['geom', 'Chân trời & capo', true], ['signs', 'Biển báo', true]
];
const SIGNS = [['Giới hạn tốc độ','s'],['Hết hạn chế tốc độ','e'],['Hết mọi lệnh cấm','e'],['Khu đông dân cư','g'],
  ['Hết khu đông dân cư','g'],['Đường cao tốc','g'],['Hết đường cao tốc','g'],['Cấm đi ngược chiều','p'],['Dừng lại','p'],
  ['Giao với đường ưu tiên','d'],['Người đi bộ cắt ngang','d'],['Trẻ em','d'],['Giao với đường sắt','d'],
  ['Chỗ ngoặt nguy hiểm','d'],['Giao với đường không ưu tiên','d'],['Gờ giảm tốc phía trước','d'],['Biển báo','o']];
const signOf = c => SIGNS[c] || ['Biển báo', 'g'];
const signCol = k => k === 'p' || k === 's' ? '#ef4444' : k === 'd' ? '#facc15' : k === 'e' || k === 'o' ? '#cbd5e1' : '#3b82f6';
function signIcon(c2, k, x, y, r) {
  c2.save(); c2.lineWidth = Math.max(2, r * 0.22);
  if (k === 'd') {
    c2.beginPath(); c2.moveTo(x, y - r); c2.lineTo(x + r * 0.95, y + r * 0.75); c2.lineTo(x - r * 0.95, y + r * 0.75); c2.closePath();
    c2.fillStyle = '#fff'; c2.fill(); c2.strokeStyle = '#ef4444'; c2.stroke();
    c2.fillStyle = '#111'; c2.fillRect(x - r * 0.07, y - r * 0.35, r * 0.14, r * 0.55); c2.fillRect(x - r * 0.07, y + r * 0.3, r * 0.14, r * 0.14);
  } else if (k === 'g') {
    c2.fillStyle = '#2563eb'; c2.fillRect(x - r, y - r, 2 * r, 2 * r); c2.strokeStyle = '#fff'; c2.lineWidth = 2; c2.strokeRect(x - r + 3, y - r + 3, 2 * r - 6, 2 * r - 6);
  } else if (k === 'e') {
    c2.beginPath(); c2.arc(x, y, r * 0.9, 0, 2 * Math.PI); c2.fillStyle = '#fff'; c2.fill(); c2.strokeStyle = '#111'; c2.lineWidth = 1.5; c2.stroke();
    c2.clip(); c2.lineWidth = Math.max(1.5, r * 0.08);
    for (let i = -2; i <= 2; ++i) { const o = i * r * 0.3; c2.beginPath(); c2.moveTo(x - r + o, y + r + o); c2.lineTo(x + r + o, y - r + o); c2.stroke(); }
  } else if (k === 'o') {
    c2.beginPath(); c2.arc(x, y, r * 0.88, 0, 2 * Math.PI); c2.strokeStyle = '#cbd5e1'; c2.stroke();
  } else {
    c2.beginPath(); c2.arc(x, y, r * 0.88, 0, 2 * Math.PI); c2.fillStyle = '#fff'; c2.fill(); c2.strokeStyle = '#ef4444'; c2.stroke();
    if (k === 's') { c2.fillStyle = '#111'; c2.font = 'bold ' + Math.round(r * 0.75) + 'px system-ui'; c2.textAlign = 'center'; c2.textBaseline = 'middle'; c2.fillText('km/h', x, y); }
  }
  c2.restore();
}
const layers = {};
try { Object.assign(layers, JSON.parse(localStorage.getItem('adasLayers') || '{}')); } catch (e) {}
LAYERS.forEach(([k, label, def]) => {
  if (layers[k] === undefined) layers[k] = def;
  const b = document.createElement('button');
  b.textContent = label; b.className = layers[k] ? 'on' : '';
  b.onclick = () => { layers[k] = !layers[k]; b.className = layers[k] ? 'on' : ''; try { localStorage.setItem('adasLayers', JSON.stringify(layers)); } catch (e) {} draw(); };
  $('chips').appendChild(b);
});

function toast(msg, err) { const t = $('toast'); t.textContent = msg; t.className = 'show' + (err ? ' err' : ''); clearTimeout(t._h); t._h = setTimeout(() => t.className = '', 3500); }
const lvlCol = l => l === 2 ? '#ef4444' : l === 1 ? '#f59e0b' : '#22c55e';
const laneCol = l => l === 2 ? '#ef4444' : l === 1 ? '#facc15' : '#22c55e';
const laneFill = (l, a) => (l === 2 ? 'rgba(239,68,68,' : l === 1 ? 'rgba(250,204,21,' : 'rgba(34,197,94,') + a + ')';
const fmt = (v, d, u) => (v === undefined || v === null || v < 0) ? '--' : Number(v).toFixed(d) + (u || '');

let frameBmp = null;
function draw() {
  const W = cam.width, H = cam.height;
  if (!W || !H) return;
  ctx.clearRect(0, 0, W, H);
  if (frameBmp) ctx.drawImage(frameBmp, 0, 0, W, H);
  if (!T) return;
  const sx = W / (T.imgW || W), sy = H / (T.imgH || H);
  const P = p => [p[0] * sx, p[1] * sy];
  const col = laneCol(T.alert);

  if (layers.geom) {
    ctx.save(); ctx.setLineDash([10, 8]); ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(250,204,21,.7)'; ctx.beginPath(); ctx.moveTo(0, T.hzY * sy); ctx.lineTo(W, T.hzY * sy); ctx.stroke();
    if (T.hoodY) { ctx.strokeStyle = 'rgba(251,146,60,.7)'; ctx.beginPath(); ctx.moveTo(0, T.hoodY * sy); ctx.lineTo(W, T.hoodY * sy); ctx.stroke(); }
    ctx.setLineDash([]); ctx.strokeStyle = '#facc15'; ctx.beginPath();
    const vx = T.vpX * sx, vy = T.hzY * sy; ctx.moveTo(vx - 8, vy); ctx.lineTo(vx + 8, vy); ctx.moveTo(vx, vy - 8); ctx.lineTo(vx, vy + 8); ctx.stroke();
    ctx.restore();
  }
  if (layers.carpet && T.poly && T.poly.length >= 3) {
    ctx.save(); ctx.beginPath(); T.poly.forEach((p, i) => { const q = P(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); }); ctx.closePath();
    ctx.fillStyle = laneFill(T.alert, .26); ctx.fill(); ctx.restore();
  }
  if (layers.lanes) {
    const line = (pts, synth) => {
      if (!pts || pts.length < 2) return;
      ctx.save(); ctx.beginPath(); pts.forEach((p, i) => { const q = P(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
      ctx.lineWidth = synth ? 2 : 4; ctx.lineCap = 'round'; ctx.strokeStyle = synth ? 'rgba(148,163,184,.8)' : col; if (synth) ctx.setLineDash([8, 8]);
      ctx.stroke(); ctx.restore();
    };
    const b = T.bev || {};
    line(T.leftLane, b.L && b.L[4]); line(T.rightLane, b.R && b.R[4]);
    [T.pathL, T.pathR].forEach(pts => {
      if (!pts || pts.length < 2) return;
      ctx.save(); ctx.beginPath(); pts.forEach((p, i) => { const q = P(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
      ctx.lineWidth = 3; ctx.setLineDash([10, 6]); ctx.strokeStyle = '#22d3ee'; ctx.stroke(); ctx.restore();
    });
  }
  const eg = T.ego || {};
  if (layers.lanes && ((T.lane && T.lane !== 'NO_LANE') || eg.turn)) {
    const cx = W / 2, cy = H * 0.80;
    const lv = T.latVel || 0;
    const dir = T.alert > 0 ? (String(T.lane).indexOf('LEFT') >= 0 ? -1 : 1) : Math.abs(lv) > 0.25 ? Math.sign(lv) : 0;
    ctx.save(); ctx.textAlign = 'center';
    if (dir) {
      ctx.fillStyle = col; ctx.beginPath();
      ctx.moveTo(cx + dir * 46, cy); ctx.lineTo(cx + dir * 18, cy - 16); ctx.lineTo(cx + dir * 18, cy - 6);
      ctx.lineTo(cx - dir * 30, cy - 6); ctx.lineTo(cx - dir * 30, cy + 6); ctx.lineTo(cx + dir * 18, cy + 6); ctx.lineTo(cx + dir * 18, cy + 16);
      ctx.closePath(); ctx.fill();
      ctx.font = 'bold 15px system-ui'; ctx.fillText(dir < 0 ? 'lệch trái' : 'lệch phải', cx, cy - 24);
    }
    const txt = eg.turn ? 'xe đang rẽ ' + (eg.turn < 0 ? 'trái' : 'phải') + ' ' + Math.abs(eg.yaw).toFixed(0) + '°/s'
                          + (eg.curv ? ' (R≈' + Math.round(1 / Math.abs(eg.curv)) + ' m)' : '')
              : !T.roadShape ? 'phía trước: đường thẳng'
              : 'phía trước: cua ' + (T.roadShape < 0 ? 'trái' : 'phải') + (T.curveR ? ' (R≈' + Math.round(T.curveR) + ' m)' : '');
    ctx.font = '13px system-ui'; const tw = ctx.measureText(txt).width + 16;
    ctx.fillStyle = 'rgba(11,16,22,.75)'; ctx.fillRect(cx - tw / 2, cy + 24, tw, 22);
    ctx.fillStyle = eg.turn ? '#22d3ee' : T.roadShape ? '#facc15' : '#4ade80'; ctx.fillText(txt, cx, cy + 40);
    ctx.restore();
  }
  if (layers.raw && T.rawVehs) {
    ctx.save(); ctx.strokeStyle = 'rgba(255,255,255,.75)'; ctx.lineWidth = 1; ctx.font = '10px system-ui'; ctx.fillStyle = '#fff';
    T.rawVehs.forEach(v => { ctx.strokeRect(v.box[0] * sx, v.box[1] * sy, v.box[2] * sx, v.box[3] * sy); ctx.fillText(v.cls + ' ' + Math.round(v.conf * 100), v.box[0] * sx, v.box[1] * sy + v.box[3] * sy + 11); });
    ctx.restore();
  }
  if (layers.vehs && T.vehs) {
    T.vehs.forEach(v => {
      const x = v.box[0] * sx, y = v.box[1] * sy, w = v.box[2] * sx, h = v.box[3] * sy;
      const c = v.cutIn ? '#f59e0b' : v.pos === 'IN_LANE' ? '#22d3ee' : '#a78bfa';
      ctx.save(); ctx.strokeStyle = c; ctx.lineWidth = v.cutIn ? 2.5 : 1.5; ctx.strokeRect(x, y, w, h);
      label((v.cutIn ? 'TẠT ĐẦU ' : '') + '#' + v.id + ' ' + fmt(v.dist, 0, ' m'), x, y, c, '#0b1016'); ctx.restore();
    });
  }
  if (layers.signs && T.tsr && T.tsr.on && layers.geom && T.tsr.band) {
    ctx.save(); ctx.strokeStyle = 'rgba(96,165,250,.35)'; ctx.setLineDash([3, 6]); ctx.lineWidth = 1;
    ctx.strokeRect(1, T.tsr.band[0] * sy, W - 2, (T.tsr.band[1] - T.tsr.band[0]) * sy); ctx.restore();
  }
  if (layers.signs && T.signs) {
    T.signs.forEach(g => {
      const [x, y, w, h] = [g[0] * sx, g[1] * sy, g[2] * sx, g[3] * sy], s = signOf(g[4]), c = signCol(s[1]);
      ctx.save(); ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.strokeRect(x, y, w, h);
      const v = g[6] > 0 ? ' ' + g[6] + ' km/h' : g[6] < 0 ? ' (không phải)' : '';
      label(s[0] + v + ' ' + Math.round(g[5] * 100) + '%', x, y, c, s[1] === 'd' ? '#111' : '#fff'); ctx.restore();
    });
  }
  if (layers.lead && T.leadBox) {
    const [x, y, w, h] = [T.leadBox[0] * sx, T.leadBox[1] * sy, T.leadBox[2] * sx, T.leadBox[3] * sy];
    const c = lvlCol(T.threat), k = Math.max(8, Math.min(18, w / 4));
    ctx.save(); ctx.strokeStyle = c; ctx.lineWidth = 3; ctx.beginPath();
    [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]].forEach(([px, py, dx, dy]) => { ctx.moveTo(px + dx * k, py); ctx.lineTo(px, py); ctx.lineTo(px, py + dy * k); });
    ctx.stroke();
    label('#' + T.leadTrackId + '  ' + fmt(T.dist, 1, ' m'), x, y, c, '#fff', true);
    ctx.restore();
  }
}
function label(text, x, y, bg, fg, bold) {
  ctx.font = (bold ? 'bold 12px' : '11px') + ' system-ui';
  const w = ctx.measureText(text).width + 8, yy = Math.max(0, y - 17);
  ctx.fillStyle = bg; ctx.fillRect(x, yy, w, 16); ctx.fillStyle = fg; ctx.fillText(text, x + 4, yy + 12);
}

function drawBev() {
  const W = bev.width, H = bev.height, XR = 8, ZMAX = 50, ZMIN = -5;
  const px = x => W / 2 + x / XR * (W / 2), pz = z => H - (z - ZMIN) / (ZMAX - ZMIN) * H;
  bctx.fillStyle = '#071016'; bctx.fillRect(0, 0, W, H);
  bctx.strokeStyle = '#132532'; bctx.fillStyle = '#4b6475'; bctx.font = '10px system-ui'; bctx.lineWidth = 1;
  for (let z = 0; z <= ZMAX; z += 10) { bctx.beginPath(); bctx.moveTo(0, pz(z)); bctx.lineTo(W, pz(z)); bctx.stroke(); bctx.fillText(z + ' m', 4, pz(z) - 3); }
  for (let x = -6; x <= 6; x += 2) { bctx.beginPath(); bctx.moveTo(px(x), 0); bctx.lineTo(px(x), H); bctx.stroke(); }
  if (!T || !T.bev) return;
  const b = T.bev, z0 = Math.max(1, b.zNear || 3), z1 = 45;
  const xs = (m, z) => m[0] + m[1] * z + m[2] * z * z;
  const path = (m, flip) => { const p = []; for (let z = z0; z <= z1; z += 1) p.push([px(xs(m, z)), pz(z)]); return flip ? p.reverse() : p; };
  if (b.L && b.R && (b.L[3] || b.R[3])) {
    const pts = path(b.L).concat(path(b.R, true));
    bctx.beginPath(); pts.forEach((p, i) => i ? bctx.lineTo(p[0], p[1]) : bctx.moveTo(p[0], p[1])); bctx.closePath();
    bctx.fillStyle = laneFill(T.alert, .2); bctx.fill();
    [b.L, b.R].forEach(m => {
      if (!m[3]) return;
      bctx.save(); bctx.beginPath(); path(m).forEach((p, i) => i ? bctx.lineTo(p[0], p[1]) : bctx.moveTo(p[0], p[1]));
      bctx.lineWidth = m[4] ? 1.5 : 3; bctx.strokeStyle = m[4] ? '#64748b' : laneCol(T.alert); if (m[4]) bctx.setLineDash([6, 6]); bctx.stroke(); bctx.restore();
    });
  }
  (b.vehs || []).forEach(v => {
    const [x, z, w, id, lead, inPath] = v; if (z > ZMAX) return;
    const c = lead ? lvlCol(T.threat) : inPath ? '#22d3ee' : '#a78bfa';
    bctx.fillStyle = c; bctx.globalAlpha = .85; bctx.fillRect(px(x - w / 2), pz(z + 4.5), px(x + w / 2) - px(x - w / 2), pz(z) - pz(z + 4.5)); bctx.globalAlpha = 1;
    bctx.fillStyle = '#cbd5e1'; bctx.fillText(z.toFixed(0) + ' m', px(x + w / 2) + 3, pz(z + 2));
  });
  const cx = b.carX || 0, ego = '#e2e8f0';
  bctx.fillStyle = ego; bctx.fillRect(px(cx - 0.9), pz(0), px(cx + 0.9) - px(cx - 0.9), pz(-4.5) - pz(0));
  bctx.fillStyle = '#0b1016'; bctx.fillRect(px(cx - 0.7), pz(-0.6), px(cx + 0.7) - px(cx - 0.7), pz(-1.5) - pz(-0.6));
}

function panel() {
  if (!T) return;
  const a = $('alert'); a.textContent = T.warn || 'AN TOÀN'; const lv = Math.max(T.threat || 0, T.alert || 0); a.className = 'alert' + (lv === 2 ? ' l2' : lv === 1 ? ' l1' : '');
  $('vLead').textContent = T.hasLead ? fmt(T.dist, 1, ' m') : '--';
  $('sLead').textContent = T.hasLead ? ('#' + T.leadTrackId) : 'không có';
  $('vInLane').textContent = T.inLane != null ? T.inLane : '--';
  $('vOff').textContent = T.lane === 'NO_LANE' ? '--' : T.offset;
  $('sLane').textContent = T.laneText || T.lane;
  $('vWidth').textContent = T.lane === 'NO_LANE' ? '--' : fmt(T.laneWidth, 2, ' m');
  $('sWidth').textContent = 'điểm vạch ' + (T.lanePts ? T.lanePts.join(' / ') : '--');
  const ai = T.ai || {};
  $('kDet').textContent = fmt(ai.detHz, 1, ' Hz') + ' · ' + fmt(ai.detMs, 0, ' ms');
  const si = $('signIco'), sc2 = si.getContext('2d'); sc2.clearRect(0, 0, si.width, si.height);
  if (T.sign !== undefined && T.sign >= 0) {
    const s = signOf(T.sign); signIcon(sc2, s[1], 22, 22, 19);
    $('vSign').textContent = s[0] + (s[1] === 's' && T.speedValue > 0 && T.sign === T.speedSign ? ' ' + T.speedValue + ' km/h' : ''); $('vSign').style.color = signCol(s[1]);
  } else { $('vSign').textContent = 'không có'; $('vSign').style.color = ''; }
  $('sSign').textContent = !T.tsr ? '--' : !T.tsr.on ? 'tắt (thiếu model tsr_640x224.onnx)'
    : 'YOLO11n · ' + fmt(T.tsr.hz, 1, ' Hz') + ' · ' + fmt(T.tsr.ms, 0, ' ms');
  $('kFilt').textContent = T.detFilter ? T.detFilter[1] + ' / ' + T.detFilter[0] : '--';
  $('kVeh').textContent = (T.bev && T.bev.vehs ? T.bev.vehs.length : 0) + ' (trong làn ' + (T.inLane || 0) + ')';
  $('kFps').textContent = fmt(T.fps, 1) + ' / ' + renderFps + ' FPS';
  $('tFps').textContent = fmt(T.fps, 0, ' FPS');
  const aiOk = T.aiAge !== undefined && T.aiAge < 45;
  if (T.cvOnly) { $('dAi').className = 'dot ok'; $('tAi').textContent = 'CV cổ điển (không deep learning)'; }
  else { $('dAi').className = 'dot ' + (aiOk ? 'ok' : 'bad'); $('tAi').textContent = aiOk ? 'AI hoạt động' : 'AI không phản hồi'; }
  const cb = T.calib || {}, cc = T.calibCfg || {};
  $('calBar').style.width = Math.round((cb.progress || 0) * 100) + '%';
  $('tCal').textContent = cb.text || '--';
  $('kHz').textContent = fmt(cc.horizon, 3, 'H') + ' · ' + fmt(cc.vpX, 3, 'W');
  $('kHood').textContent = fmt(cc.hood, 3, 'H');
  $('kCam').textContent = fmt(cc.camH, 2, ' m') + ' · ' + (cc.latOff === undefined ? '--' : (cc.latOff >= 0 ? '+' : '') + Number(cc.latOff).toFixed(2) + ' m');
  $('kFov').textContent = fmt(cc.hfov, 0, '°');
  const ob = T.obd || {};
  const camOk = T.cam === 'live';
  $('dCam').className = 'dot ' + (camOk ? 'ok' : T.cam === 'video' ? 'warn' : 'bad');
  $('tCam').textContent = camOk ? 'CAM OK' : T.cam === 'video' ? 'CAM: video test' : 'CAM mất kết nối';
  $('dObd').className = 'dot ' + (ob.state === 'ok' ? 'ok' : 'bad');
  $('tObdH').textContent = ob.state === 'ok' ? 'OBD ' + ob.kph + ' km/h' : ob.state === 'off' ? 'OBD tắt' : 'OBD mất kết nối';
  $('pObd').title = ob.text || '';
  $('tObd').textContent = ob.state === 'ok' ? fmt(ob.hz, 1, ' Hz') : ob.state === 'off' ? 'tắt' : 'chưa có';
  $('kObdSpd').textContent = ob.state === 'ok' ? ob.kph + ' km/h' + (T.video ? ' (video: không dùng)' : '') : '--';
  $('kObdTxt').textContent = ob.text || '--';
  $('kObdDev').textContent = ob.dev ? ob.dev + ' · ' + (ob.elm || '?') + (ob.proto ? ' · ' + ob.proto : '') : '--';
  $('kObdV').textContent = ob.volt > 0 ? fmt(ob.volt, 1, ' V') : '--';
}

let abortCtl = null, lastFrameT = 0, sideT = 0;
async function showFrame(blob, seq, th) {
  lastSeq = seq;
  if (th) { try { T = JSON.parse(th); } catch (e) {} }
  const bmp = await createImageBitmap(blob);
  if (cam.width !== bmp.width || cam.height !== bmp.height) { cam.width = bmp.width; cam.height = bmp.height; }
  if (frameBmp) frameBmp.close();
  frameBmp = bmp; $('nosig').style.display = 'none';
  draw();
  const now = performance.now();
  if (now - sideT >= 120) { sideT = now; drawBev(); panel(); $('tLat').textContent = Math.round(now - lastFrameT) + ' ms/ảnh'; }
  lastFrameT = now;
  renderN++;
  if (now - renderT >= 1000) { renderFps = Math.round(renderN * 1000 / (now - renderT)); renderN = 0; renderT = now; }
}
let pend = null, painting = false;
function paint() {
  if (painting || !pend) return;
  const p = pend; pend = null; painting = true;
  requestAnimationFrame(() => showFrame(p.blob, p.seq, p.th).catch(() => {}).finally(() => { painting = false; paint(); }));
}
const dec = new TextDecoder();
function findCrlf2(b, from, to) {
  for (let i = from; i + 3 < to; i++) if (b[i] === 13 && b[i + 1] === 10 && b[i + 2] === 13 && b[i + 3] === 10) return i;
  return -1;
}
async function live() {
  abortCtl = new AbortController();
  const r = await fetch('/api/live', { signal: abortCtl.signal, cache: 'no-store' });
  if (!r.ok || !r.body) throw new Error('no live');
  const rd = r.body.getReader();
  let buf = new Uint8Array(256 * 1024), off = 0, end = 0;
  for (;;) {
    const { value, done } = await rd.read();
    if (done) throw new Error('live closed');
    if (end + value.length > buf.length) {
      if (off > 0) { buf.copyWithin(0, off, end); end -= off; off = 0; }
      if (end + value.length > buf.length) { const nb = new Uint8Array(Math.max(buf.length * 2, end + value.length)); nb.set(buf.subarray(0, end)); buf = nb; }
    }
    buf.set(value, end); end += value.length;
    for (;;) {
      const he = findCrlf2(buf, off, end);
      if (he < 0) break;
      const hdr = dec.decode(buf.subarray(off, he));
      const len = parseInt((hdr.match(/Content-Length:\s*(\d+)/i) || [])[1] || '-1', 10);
      if (len < 0) { off = he + 4; continue; }
      if (end < he + 4 + len) break;
      pend = { blob: new Blob([buf.subarray(he + 4, he + 4 + len)], { type: 'image/jpeg' }),
               seq: parseInt((hdr.match(/X-Frame-Seq:\s*(\d+)/i) || [])[1] || '0', 10),
               th: (hdr.match(/X-Telemetry:\s*(.*)/i) || [])[1] };
      off = he + 4 + len;
    }
    if (off === end) off = end = 0;
    paint();
  }
}
function piLost() {
  $('dCam').className = 'dot bad'; $('tCam').textContent = 'CAM mất kết nối';
  $('dObd').className = 'dot bad'; $('tObdH').textContent = 'OBD mất kết nối';
  $('dAi').className = 'dot bad'; $('tAi').textContent = 'AI --';
}
async function loop() {
  for (let fails = 0; fails < 3; ) {
    const s0 = lastSeq;
    try { await live(); } catch (e) { fails = lastSeq !== s0 ? 0 : fails + 1; $('nosig').style.display = 'flex'; $('nosig').textContent = 'Mất kết nối, đang thử lại…'; piLost(); await sleep(1000); }
  }
  for (;;) {
    const t0 = performance.now();
    try {
      abortCtl = new AbortController();
      const r = await fetch('/api/frame.jpg?wait_seq=' + lastSeq, { signal: abortCtl.signal, cache: 'no-store' });
      if (!r.ok) { await sleep(300); continue; }
      lastSeq = parseInt(r.headers.get('X-Frame-Seq') || '0', 10);
      const th = r.headers.get('X-Telemetry');
      if (th) { try { T = JSON.parse(th); } catch (e) {} }
      const bmp = await createImageBitmap(await r.blob());
      if (cam.width !== bmp.width || cam.height !== bmp.height) { cam.width = bmp.width; cam.height = bmp.height; }
      if (frameBmp) frameBmp.close();
      frameBmp = bmp; $('nosig').style.display = 'none';
      draw(); drawBev(); panel();
      $('tLat').textContent = Math.round(performance.now() - t0) + ' ms';
      renderN++; const now = performance.now();
      if (now - renderT >= 1000) { renderFps = Math.round(renderN * 1000 / (now - renderT)); renderN = 0; renderT = now; }
    } catch (e) { $('nosig').style.display = 'flex'; $('nosig').textContent = 'Mất kết nối, đang thử lại…'; piLost(); await sleep(1000); }
  }
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

function setSourceUI(src) {
  const isCam = src === 'camera';
  $('bCam').className = isCam ? 'on' : ''; $('bVid').className = isCam ? '' : 'on';
  $('tSrc').textContent = isCam ? 'Camera trực tiếp' : 'Video test (lặp)'; $('dSrc').className = 'dot ' + (isCam ? '' : 'warn');
}
function recalibrate() {
  const b = $('bRecal');
  if (!b.classList.contains('arm')) {
    b.classList.add('arm'); b.textContent = 'Bấm lần nữa để xóa hiệu chuẩn và làm lại';
    clearTimeout(b._h); b._h = setTimeout(() => { b.classList.remove('arm'); b.textContent = 'Hiệu chuẩn lại'; }, 6000);
    return;
  }
  clearTimeout(b._h); b.classList.remove('arm'); b.textContent = 'Hiệu chuẩn lại';
  fetch('/api/calibration/reset', { method: 'POST' }).then(r => r.json()).then(d => {
    if (d.status === 'ok') toast(T && T.video ? 'Đang hiệu chuẩn lại theo video (~10-20 s)'
                                             : 'Đang hiệu chuẩn lại: chạy xe ≥30 km/h trên đường thẳng có vạch làn vài phút');
    else toast('Không hiệu chuẩn lại được', true);
  }).catch(() => toast('Lỗi kết nối tới thiết bị', true));
}
function switchSource(src) {
  toast('Đang chuyển nguồn…');
  fetch('/api/source', { method: 'POST', body: 'source=' + src }).then(r => r.json()).then(d => {
    if (d.status === 'ok') { setSourceUI(d.source); toast('Đã chuyển sang ' + (d.source === 'camera' ? 'camera' : 'video test')); }
    else toast(d.message || 'Không chuyển được nguồn', true);
  }).catch(() => toast('Lỗi kết nối tới thiết bị', true));
}
function upload(file) {
  if (!file) return;
  const wrap = $('upWrap'), bar = $('upBar'); wrap.style.display = 'block'; bar.style.width = '0%';
  toast('Đang tải video lên…');
  const x = new XMLHttpRequest(); x.open('POST', '/upload', true); x.setRequestHeader('Content-Type', 'application/octet-stream');
  x.upload.onprogress = e => { if (e.lengthComputable) bar.style.width = Math.round(e.loaded / e.total * 100) + '%'; };
  x.onload = () => { wrap.style.display = 'none'; let d = {}; try { d = JSON.parse(x.responseText); } catch (e) {}
    if (x.status === 200 && d.status === 'ok') { setSourceUI('video'); toast(d.message || 'Đã tải xong'); } else toast(d.message || ('Lỗi tải video (' + x.status + ')'), true); };
  x.onerror = () => { wrap.style.display = 'none'; toast('Mất kết nối khi tải video', true); };
  x.send(file);
}
const drop = $('drop'), file = $('file');
drop.onclick = () => file.click(); file.onchange = () => upload(file.files[0]);
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); if (e.dataTransfer.files.length) upload(e.dataTransfer.files[0]); };
let lastSrc = '';
setInterval(() => fetch('/api/status').then(r => r.json()).then(d => { if (d.source && d.source !== lastSrc) { lastSrc = d.source; setSourceUI(d.source); } }).catch(() => {}), 1500);

function setMaskUI(on) { $('bMask').className = on ? 'on' : ''; }
fetch('/api/debug').then(r => r.json()).then(d => setMaskUI(d.mask)).catch(() => {});
$('bMask').onclick = () => fetch('/api/debug?mask=' + ($('bMask').className === 'on' ? 0 : 1)).then(r => r.json()).then(d => setMaskUI(d.mask)).catch(() => {});

const showOff = v => $('offVal').textContent = Number(v).toFixed(1).replace('.', ',') + ' m';
function openModal() {
  fetch('/api/calibration').then(r => r.json()).then(d => { if (d.camera_front_offset_m !== undefined) { $('offSlider').value = d.camera_front_offset_m; showOff(d.camera_front_offset_m); } }).catch(() => {});
  $('modal').classList.add('open');
}
function closeModal() { $('modal').classList.remove('open'); }
function saveOff() {
  const v = parseFloat($('offSlider').value);
  fetch('/api/calibration/front_offset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ camera_front_offset_m: v }) })
    .then(r => r.json()).then(d => { if (d.status === 'ok') { toast('Đã lưu: ' + v.toFixed(1).replace('.', ',') + ' m'); closeModal(); } else toast(d.message || 'Lỗi khi lưu', true); })
    .catch(() => toast('Lỗi kết nối tới thiết bị', true));
}
if (location.hash === '#setup') setTimeout(openModal, 300);

drawBev();
loop();
</script>
</body>
</html>)raw_html";

        std::ostringstream response;
        response << "HTTP/1.1 200 OK\r\n"
                 << "Content-Type: text/html; charset=UTF-8\r\n"
                 << "Content-Length: " << html.length() << "\r\n"
                 << "Connection: close\r\n\r\n"
                 << html;

        std::string resStr = response.str();
        sendAll(clientSock, resStr.c_str(), resStr.length());
    }

    int m_port;
    int m_serverSock;
    std::atomic<bool> m_running;
    std::thread m_serverThread;

    std::mutex m_frameMtx;
    std::condition_variable m_frameCv;
    std::vector<uchar> m_latestJpeg;
    std::atomic<bool> m_debugMask{false};
    std::atomic<bool> m_calibReset{false};
    double m_fps;
    double m_dist;
    int m_speed;
    int m_speedLimit;
    std::atomic<uint64_t> m_frameSeq;
    std::chrono::steady_clock::time_point m_lastFrameTime;
    std::chrono::steady_clock::time_point m_lastEncodeTime;

    std::mutex m_statusMtx;
    std::string m_latestTelemetryJson;
    std::string m_cachedTelemetryJson;
    std::string m_currentSource;
    SwitchSourceFn m_onSwitchSource;
    SetFrontOffsetFn m_onSetFrontOffset;
    GetFrontOffsetFn m_onGetFrontOffset;
    std::atomic<int> m_clientCount{0};
    std::atomic<int64_t> m_lastClientActiveMs{0};

    std::atomic<int> m_activeHandlers{0};
    std::mutex m_handlersMtx;
    std::condition_variable m_handlersCv;
    static constexpr int kMaxHandlers = 8;

    std::thread m_encThread;
    std::mutex m_encMtx;
    std::condition_variable m_encCv;
    cv::Mat m_encPendingCam;
    std::string m_encTelem;
    bool m_encHasNew = false;
};

#endif
