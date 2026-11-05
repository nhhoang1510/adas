#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickWindow>
#include <QFile>
#include <iostream>
#include <string>
#include <sched.h>
#include <pthread.h>
#include <clocale>

#include "AdasController.h"
#include "FrameProvider.h"
#include "CameraWorker.h"

#include <QSurfaceFormat>
#include <QFont>
#include <QTimer>
#include <QProcess>
#include <QElapsedTimer>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>

static void sdNotify(const char* msg) {
    const char* path = getenv("NOTIFY_SOCKET");
    if (!path || !*path) return;
    sockaddr_un sa{};
    sa.sun_family = AF_UNIX;
    const size_t n = strlen(path);
    if (n >= sizeof(sa.sun_path)) return;
    memcpy(sa.sun_path, path, n);
    if (sa.sun_path[0] == '@') sa.sun_path[0] = '\0';
    const int fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return;
    sendto(fd, msg, strlen(msg), MSG_NOSIGNAL, reinterpret_cast<sockaddr*>(&sa),
           socklen_t(offsetof(sockaddr_un, sun_path) + n));
    close(fd);
}

int main(int argc, char *argv[]) {
    adas_pin_to_core(0);
    std::cout << "[ADAS] Qt Main Thread pinned to Core 0" << std::endl;

    qputenv("QSG_RENDER_LOOP", "basic");

    QGuiApplication app(argc, argv);
    setlocale(LC_NUMERIC, "C");
    app.setQuitOnLastWindowClosed(false);
    app.setApplicationName("ADAS HMI Pi4");

    QFont appFont = app.font();
    appFont.setStyleStrategy(QFont::StyleStrategy(QFont::NoAntialias | QFont::NoSubpixelAntialias));
    appFont.setHintingPreference(QFont::PreferFullHinting);
    app.setFont(appFont);
    QQuickWindow::setTextRenderType(QQuickWindow::NativeTextRendering);

    std::string video_source = "0";
    std::string model_path = "";
    int onnx_cadence = 5;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--src" && i + 1 < argc) {
            video_source = argv[++i];
        } else if (std::string(argv[i]) == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::string(argv[i]) == "--cadence" && i + 1 < argc) {
            onnx_cadence = std::stoi(argv[++i]);
        }
    }

    if (model_path.empty()) {
        const char* env_model = getenv("ADAS_MODEL_PATH");
        if (env_model && strlen(env_model) > 0) {
            model_path = env_model;
        }
    }

    std::cout << "=== ADAS Hybrid Dual-Cadence Pipeline ===" << std::endl;
    std::cout << "  Source    : " << video_source << std::endl;
    std::cout << "  Model dir : " << (model_path.empty() ? "(auto-detect)" : model_path) << std::endl;
    std::cout << "  Cadence   : " << onnx_cadence << " frames" << std::endl;
    std::cout << "  Core Map  : Qt=0, FastPath=1, ONNX=2+3" << std::endl;
    std::cout << "==========================================" << std::endl;

    AdasController adasBackend;
    auto* frameProvider = new FrameProvider();

    QQmlApplicationEngine engine;
    engine.addImageProvider("camera", frameProvider);

    engine.rootContext()->setContextProperty("adasBackend", &adasBackend);

    static const char* kRadioScript =
        "w=0; i=$(iw dev wlan0 info 2>/dev/null); "
        "case \"$i\" in *'type AP'*) w=1;; *'type managed'*) iw dev wlan0 link 2>/dev/null | grep -q '^Connected' && w=1;; esac; "
        "b=0; h=$(hciconfig hci0 2>/dev/null); "
        "case \"$h\" in *'UP RUNNING'*) case \"$h\" in *PSCAN*|*ISCAN*) b=1;; esac; "
        "hcitool con 2>/dev/null | grep -q ACL && b=2;; esac; "
        "echo $w $b";
    auto* radioProc = new QProcess(&app);
    QObject::connect(radioProc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), [&adasBackend, radioProc]() {
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
        const QStringList v = QString::fromLatin1(radioProc->readAllStandardOutput()).split(' ', Qt::SkipEmptyParts);
#else
        const QStringList v = QString::fromLatin1(radioProc->readAllStandardOutput()).split(' ', QString::SkipEmptyParts);
#endif
        if (v.size() >= 2) { adasBackend.setWifiOn(v[0].trimmed() == "1"); const int b = v[1].trimmed().toInt(); adasBackend.setBtOn(b >= 1); adasBackend.setBtConnected(b == 2); }
    });
    auto pollRadio = [radioProc]() {
        if (radioProc->state() == QProcess::NotRunning) radioProc->start("/bin/sh", {"-c", kRadioScript});
    };
    auto* radioTimer = new QTimer(&app);
    QObject::connect(radioTimer, &QTimer::timeout, pollRadio);
    radioTimer->start(3000);
    QTimer::singleShot(500, radioProc, pollRadio);

    CameraWorker adasWorker(&adasBackend, frameProvider, video_source, model_path, onnx_cadence);
    engine.rootContext()->setContextProperty("adasWorker", &adasWorker);

    const QUrl qrcUrl(QStringLiteral("qrc:/qml/Main.qml"));
    engine.load(qrcUrl);

    if (engine.rootObjects().isEmpty()) {
        engine.load(QUrl(QStringLiteral("qrc:/Main.qml")));
    }

    if (engine.rootObjects().isEmpty()) {
        std::cerr << "[ADAS] QRC load failed, trying filesystem fallback..." << std::endl;
        QString appDir = QCoreApplication::applicationDirPath();
        QStringList searchPaths = {
            appDir + "/qml/Main.qml",
            appDir + "/Main.qml",
            "/opt/adas/qml/Main.qml",
            "qml/Main.qml",
            "Main.qml"
        };
        for (const QString& path : searchPaths) {
            if (QFile::exists(path)) {
                engine.load(QUrl::fromLocalFile(path));
                if (!engine.rootObjects().isEmpty()) {
                    std::cout << "[ADAS] Loaded QML from: " << path.toStdString() << std::endl;
                    break;
                }
            }
        }
    }

    if (engine.rootObjects().isEmpty()) {
        std::cerr << "[ERROR] Khong the tai Main.qml tu QRC hoac bat ky duong dan nao!" << std::endl;
        return -1;
    }

    adasWorker.start();

    sdNotify("READY=1");
    QElapsedTimer sinceStart;
    sinceStart.start();
    QTimer wdTimer;
    QObject::connect(&wdTimer, &QTimer::timeout, [&]() {
        if (adasWorker.isHealthy() || sinceStart.elapsed() < 20000) sdNotify("WATCHDOG=1");
        else qWarning("[ADAS] pipeline stalled, withholding watchdog ping");
    });
    wdTimer.start(1000);

    int exec_res = app.exec();
    sdNotify("STOPPING=1");

    adasWorker.requestStop();
    if (!adasWorker.wait(3000)) {
        std::cerr << "[ADAS] Worker stuck on shutdown, hard exit." << std::endl;
        std::_Exit(1);
    }

    return exec_res;
}
