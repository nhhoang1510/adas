#ifndef FRAMEPROVIDER_H
#define FRAMEPROVIDER_H

#include <QQuickImageProvider>
#include <QImage>
#include <QMutex>
#include <QMutexLocker>
#include <opencv2/opencv.hpp>

class FrameProvider : public QQuickImageProvider {
public:
    FrameProvider() : QQuickImageProvider(QQuickImageProvider::Image) {
        cv::Mat blank = cv::Mat::zeros(320, 480, CV_8UC3);
        cv::putText(blank, "SYSTEM INITIALIZING...", cv::Point(100, 160),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(248, 189, 56), 1, cv::LINE_AA);
        m_currentFrame = matToQImage(blank);

        cv::Mat cropBlank(48, 64, CV_8UC3, cv::Scalar(42, 23, 15));
        cv::putText(cropBlank, "NO VEH", cv::Point(8, 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(105, 85, 71), 1, cv::LINE_AA);
        m_currentCrop = matToQImage(cropBlank);
    }

    static QImage matToQImage(const cv::Mat& bgr) {
        if (bgr.empty()) return QImage();
        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        QImage img(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
        return img.copy();
    }

    void updateFrame(const cv::Mat& bgr, const cv::Mat& crop = cv::Mat()) {
        QMutexLocker locker(&m_mutex);
        m_currentFrame = matToQImage(bgr);
        if (!crop.empty()) {
            m_currentCrop = matToQImage(crop);
        }
    }

    QImage requestImage(const QString &id, QSize *size, const QSize &requestedSize) override {
        Q_UNUSED(requestedSize);
        QMutexLocker locker(&m_mutex);
        QImage res;
        if (id.contains("crop")) {
            res = m_currentCrop;
        } else {
            res = m_currentFrame;
        }
        if (size) *size = res.size();
        return res;
    }

private:
    QImage m_currentFrame;
    QImage m_currentCrop;
    QMutex m_mutex;
};

#endif // FRAMEPROVIDER_H
