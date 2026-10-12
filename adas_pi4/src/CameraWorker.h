#ifndef CAMERAWORKER_H
#define CAMERAWORKER_H

#include <QThread>
#include <QString>
#include <opencv2/opencv.hpp>
#include "AdasController.h"
#include "FrameProvider.h"

class CameraWorker : public QThread {
    Q_OBJECT

signals:
    void frame_ready(int frameId);

public:
    CameraWorker(AdasController* ctrl, FrameProvider* provider, const std::string& source = "0", QObject* parent = nullptr)
        : QThread(parent)
        , m_ctrl(ctrl)
        , m_provider(provider)
        , m_source(source)
        , m_running(true)
        , m_frameCounter(0)
    {}

    void stop() {
        m_running = false;
    }

protected:
    void run() override {
        cv::VideoCapture cap;

        if (m_source == "0" || m_source == "webcam") {
            // Thử GStreamer pipeline tối ưu cho Pi4 / V4L2 USB camera (Logitech C270)
            std::string gst_pipeline =
                "v4l2src device=/dev/video0 ! "
                "image/jpeg, width=640, height=480, framerate=30/1 ! "
                "jpegdec ! "
                "videoconvert ! "
                "video/x-raw, format=BGR ! "
                "appsink drop=1";

            cap.open(gst_pipeline, cv::CAP_GSTREAMER);
            if (!cap.isOpened()) {
                cap.open(0, cv::CAP_V4L2);
            }
            if (!cap.isOpened()) {
                cap.open(0);
            }
        } else {
            cap.open(m_source);
        }

        if (!cap.isOpened()) {
            qWarning("[CameraWorker] Could not open video source: %s", m_source.c_str());
            return;
        }

        cv::Mat frame;
        int64 t_prev = cv::getTickCount();
        int fps_frames = 0;
        double current_fps = 0.0;
        double sim_speed = 50.0;

        while (m_running) {
            if (!cap.read(frame) || frame.empty()) {
                if (m_source != "0" && m_source != "webcam") {
                    // Loop lại nếu là video file test
                    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                    continue;
                }
                msleep(15);
                continue;
            }

            m_frameCounter++;
            fps_frames++;

            int64 t_now = cv::getTickCount();
            double elapsed_sec = (t_now - t_prev) / cv::getTickFrequency();
            if (elapsed_sec >= 0.5) {
                current_fps = fps_frames / elapsed_sec;
                fps_frames = 0;
                t_prev = t_now;
            }

            // Giả lập crop xe dẫn đầu nếu có
            cv::Mat crop_sample;
            int lead_x = frame.cols / 2 - 40;
            int lead_y = frame.rows / 2 - 20;
            if (lead_x >= 0 && lead_y >= 0 && lead_x + 80 <= frame.cols && lead_y + 60 <= frame.rows) {
                crop_sample = frame(cv::Rect(lead_x, lead_y, 80, 60)).clone();
            }

            // Cập nhật frame vào ImageProvider
            if (m_provider) {
                m_provider->updateFrame(frame, crop_sample);
            }

            // Cập nhật thông số qua invokeMethod (đảm bảo thread-safe với Qt GUI)
            QMetaObject::invokeMethod(m_ctrl, "setFps", Q_ARG(double, current_fps));
            QMetaObject::invokeMethod(m_ctrl, "setSpeed", Q_ARG(int, static_cast<int>(sim_speed)));
            QMetaObject::invokeMethod(m_ctrl, "setLaneStatus", Q_ARG(QString, "NORMAL"));
            QMetaObject::invokeMethod(m_ctrl, "setLaneOffset", Q_ARG(QString, "+0.04m"));
            QMetaObject::invokeMethod(m_ctrl, "setLaneText", Q_ARG(QString, "GIU LAN TOT"));
            QMetaObject::invokeMethod(m_ctrl, "setThreatLevel", Q_ARG(int, 0));
            QMetaObject::invokeMethod(m_ctrl, "setWarningText", Q_ARG(QString, "AN TOAN"));
            QMetaObject::invokeMethod(m_ctrl, "setHasLead", Q_ARG(bool, true));
            QMetaObject::invokeMethod(m_ctrl, "setLeadDist", Q_ARG(double, 18.5));
            QMetaObject::invokeMethod(m_ctrl, "setLeadTtc", Q_ARG(QString, "3.4s"));
            QMetaObject::invokeMethod(m_ctrl, "setLeadConf", Q_ARG(double, 92.0));
            QMetaObject::invokeMethod(m_ctrl, "setInLaneVehicles", Q_ARG(int, 1));

            // Phát signal frame_ready
            emit frame_ready(m_frameCounter);

            msleep(20); // Điều tiết ~30-40 FPS
        }

        cap.release();
    }

private:
    AdasController* m_ctrl;
    FrameProvider*  m_provider;
    std::string     m_source;
    bool            m_running;
    int             m_frameCounter;
};

#endif // CAMERAWORKER_H
