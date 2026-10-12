#ifndef ADASCONTROLLER_H
#define ADASCONTROLLER_H

#include <QObject>
#include <QString>

/**
 * AdasController - C++ QObject mapping exact properties required by Main.qml & HudDevice.qml:
 *   adasBackend.fps
 *   adasBackend.speed
 *   adasBackend.laneOffset
 *   adasBackend.laneStatus ("NORMAL", "DRIFT_LEFT", "DRIFT_RIGHT", "DEPARTURE_LEFT", "DEPARTURE_RIGHT")
 *   adasBackend.laneText
 *   adasBackend.threatLevel (0, 1, 2)
 *   adasBackend.warningText
 *   adasBackend.hasLead
 *   adasBackend.leadConf
 *   adasBackend.leadDist
 *   adasBackend.leadTtc
 *   adasBackend.inLaneVehicles
 *   adasBackend.hasSign
 *   adasBackend.signCode
 *   adasBackend.signName
 *   adasBackend.signType
 *   adasBackend.soundMuted
 */
class AdasController : public QObject {
    Q_OBJECT

    Q_PROPERTY(double  fps              READ fps              WRITE setFps              NOTIFY fpsChanged)
    Q_PROPERTY(int     speed            READ speed            WRITE setSpeed            NOTIFY speedChanged)
    Q_PROPERTY(QString laneOffset       READ laneOffset       WRITE setLaneOffset       NOTIFY laneOffsetChanged)
    Q_PROPERTY(QString laneStatus       READ laneStatus       WRITE setLaneStatus       NOTIFY laneStatusChanged)
    Q_PROPERTY(QString laneText         READ laneText         WRITE setLaneText         NOTIFY laneTextChanged)
    Q_PROPERTY(int     threatLevel      READ threatLevel      WRITE setThreatLevel      NOTIFY threatLevelChanged)
    Q_PROPERTY(QString warningText      READ warningText      WRITE setWarningText      NOTIFY warningTextChanged)
    Q_PROPERTY(bool    hasLead          READ hasLead          WRITE setHasLead          NOTIFY hasLeadChanged)
    Q_PROPERTY(double  leadConf         READ leadConf         WRITE setLeadConf         NOTIFY leadConfChanged)
    Q_PROPERTY(double  leadDist         READ leadDist         WRITE setLeadDist         NOTIFY leadDistChanged)
    Q_PROPERTY(QString leadTtc          READ leadTtc          WRITE setLeadTtc          NOTIFY leadTtcChanged)
    Q_PROPERTY(int     inLaneVehicles   READ inLaneVehicles   WRITE setInLaneVehicles   NOTIFY inLaneVehiclesChanged)
    Q_PROPERTY(bool    hasSign          READ hasSign          WRITE setHasSign          NOTIFY hasSignChanged)
    Q_PROPERTY(QString signCode         READ signCode         WRITE setSignCode         NOTIFY signCodeChanged)
    Q_PROPERTY(QString signName         READ signName         WRITE setSignName         NOTIFY signNameChanged)
    Q_PROPERTY(QString signType         READ signType         WRITE setSignType         NOTIFY signTypeChanged)
    Q_PROPERTY(bool    soundMuted       READ soundMuted       WRITE setSoundMuted       NOTIFY soundMutedChanged)

public:
    explicit AdasController(QObject *parent = nullptr)
        : QObject(parent)
        , m_fps(0.0)
        , m_speed(0)
        , m_laneOffset("+0.00m")
        , m_laneStatus("NORMAL")
        , m_laneText("GIU LAN DUONG")
        , m_threatLevel(0)
        , m_warningText("HE THONG SAN SANG")
        , m_hasLead(false)
        , m_leadConf(0.0)
        , m_leadDist(-1.0)
        , m_leadTtc("--")
        , m_inLaneVehicles(0)
        , m_hasSign(false)
        , m_signCode("")
        , m_signName("")
        , m_signType("guide")
        , m_soundMuted(false)
    {}

    // Getters
    double  fps()            const { return m_fps; }
    int     speed()          const { return m_speed; }
    QString laneOffset()     const { return m_laneOffset; }
    QString laneStatus()     const { return m_laneStatus; }
    QString laneText()       const { return m_laneText; }
    int     threatLevel()    const { return m_threatLevel; }
    QString warningText()    const { return m_warningText; }
    bool    hasLead()        const { return m_hasLead; }
    double  leadConf()       const { return m_leadConf; }
    double  leadDist()       const { return m_leadDist; }
    QString leadTtc()        const { return m_leadTtc; }
    int     inLaneVehicles() const { return m_inLaneVehicles; }
    bool    hasSign()        const { return m_hasSign; }
    QString signCode()       const { return m_signCode; }
    QString signName()       const { return m_signName; }
    QString signType()       const { return m_signType; }
    bool    soundMuted()     const { return m_soundMuted; }

public slots:
    void setFps(double v) {
        if (m_fps != v) { m_fps = v; emit fpsChanged(); }
    }
    void setSpeed(int v) {
        if (m_speed != v) { m_speed = v; emit speedChanged(); }
    }
    void setLaneOffset(const QString& v) {
        if (m_laneOffset != v) { m_laneOffset = v; emit laneOffsetChanged(); }
    }
    void setLaneStatus(const QString& v) {
        if (m_laneStatus != v) { m_laneStatus = v; emit laneStatusChanged(); }
    }
    void setLaneText(const QString& v) {
        if (m_laneText != v) { m_laneText = v; emit laneTextChanged(); }
    }
    void setThreatLevel(int v) {
        if (m_threatLevel != v) { m_threatLevel = v; emit threatLevelChanged(); }
    }
    void setWarningText(const QString& v) {
        if (m_warningText != v) { m_warningText = v; emit warningTextChanged(); }
    }
    void setHasLead(bool v) {
        if (m_hasLead != v) { m_hasLead = v; emit hasLeadChanged(); }
    }
    void setLeadConf(double v) {
        if (m_leadConf != v) { m_leadConf = v; emit leadConfChanged(); }
    }
    void setLeadDist(double v) {
        if (m_leadDist != v) { m_leadDist = v; emit leadDistChanged(); }
    }
    void setLeadTtc(const QString& v) {
        if (m_leadTtc != v) { m_leadTtc = v; emit leadTtcChanged(); }
    }
    void setInLaneVehicles(int v) {
        if (m_inLaneVehicles != v) { m_inLaneVehicles = v; emit inLaneVehiclesChanged(); }
    }
    void setHasSign(bool v) {
        if (m_hasSign != v) { m_hasSign = v; emit hasSignChanged(); }
    }
    void setSignCode(const QString& v) {
        if (m_signCode != v) { m_signCode = v; emit signCodeChanged(); }
    }
    void setSignName(const QString& v) {
        if (m_signName != v) { m_signName = v; emit signNameChanged(); }
    }
    void setSignType(const QString& v) {
        if (m_signType != v) { m_signType = v; emit signTypeChanged(); }
    }
    void setSoundMuted(bool v) {
        if (m_soundMuted != v) { m_soundMuted = v; emit soundMutedChanged(); }
    }
    Q_INVOKABLE void toggleSoundMute() {
        setSoundMuted(!m_soundMuted);
    }

signals:
    void fpsChanged();
    void speedChanged();
    void laneOffsetChanged();
    void laneStatusChanged();
    void laneTextChanged();
    void threatLevelChanged();
    void warningTextChanged();
    void hasLeadChanged();
    void leadConfChanged();
    void leadDistChanged();
    void leadTtcChanged();
    void inLaneVehiclesChanged();
    void hasSignChanged();
    void signCodeChanged();
    void signNameChanged();
    void signTypeChanged();
    void soundMutedChanged();

private:
    double  m_fps;
    int     m_speed;
    QString m_laneOffset;
    QString m_laneStatus;
    QString m_laneText;
    int     m_threatLevel;
    QString m_warningText;
    bool    m_hasLead;
    double  m_leadConf;
    double  m_leadDist;
    QString m_leadTtc;
    int     m_inLaneVehicles;
    bool    m_hasSign;
    QString m_signCode;
    QString m_signName;
    QString m_signType;
    bool    m_soundMuted;
};

#endif // ADASCONTROLLER_H
