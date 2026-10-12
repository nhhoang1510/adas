#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickWindow>
#include <iostream>
#include <string>

#include "AdasController.h"
#include "FrameProvider.h"
#include "CameraWorker.h"

int main(int argc, char *argv[]) {
    // Tối ưu render cho màn hình nhỏ SPI trên Raspberry Pi
    qputenv("QSG_RENDER_LOOP", "basic");

    QGuiApplication app(argc, argv);
    app.setApplicationName("ADAS HMI Pi4");

    std::string video_source = "0";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--src" && i + 1 < argc) {
            video_source = argv[++i];
        }
    }

    AdasController adasBackend;
    auto* frameProvider = new FrameProvider();

    QQmlApplicationEngine engine;
    engine.addImageProvider("camera", frameProvider);

    // Context properties kết nối trực tiếp với Main.qml
    engine.rootContext()->setContextProperty("adasBackend", &adasBackend);

    CameraWorker adasWorker(&adasBackend, frameProvider, video_source);
    engine.rootContext()->setContextProperty("adasWorker", &adasWorker);

    const QUrl url(QStringLiteral("qrc:/Main.qml"));
    // Nếu chạy trực tiếp từ thư mục chứa file
    engine.load(QUrl::fromLocalFile("Main.qml"));
    if (engine.rootObjects().isEmpty()) {
        engine.load(QUrl::fromLocalFile("qml/Main.qml"));
    }

    if (engine.rootObjects().isEmpty()) {
        std::cerr << "[ERROR] Khong the tai Main.qml!" << std::endl;
        return -1;
    }

    adasWorker.start();

    int exec_res = app.exec();

    adasWorker.stop();
    adasWorker.wait();

    return exec_res;
}
