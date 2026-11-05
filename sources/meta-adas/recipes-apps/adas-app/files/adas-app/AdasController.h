#ifndef ADASCONTROLLER_H
#define ADASCONTROLLER_H

#include <QObject>
#include <QString>
#include <cmath>

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
    Q_PROPERTY(int     inLaneVehicles   READ inLaneVehicles   WRITE setInLaneVehicles   NOTIFY inLaneVehiclesChanged)
    Q_PROPERTY(bool    hasSign          READ hasSign          WRITE setHasSign          NOTIFY hasSignChanged)
    Q_PROPERTY(QString signCode         READ signCode         WRITE setSignCode         NOTIFY signCodeChanged)
    Q_PROPERTY(QString signName         READ signName         WRITE setSignName         NOTIFY signNameChanged)
    Q_PROPERTY(QString signType         READ signType         WRITE setSignType         NOTIFY signTypeChanged)
    Q_PROPERTY(int     speedSign        READ speedSign        WRITE setSpeedSign        NOTIFY speedSignChanged)
    Q_PROPERTY(int     auxSign          READ auxSign          WRITE setAuxSign          NOTIFY auxSignChanged)
    Q_PROPERTY(int     speedValue       READ speedValue       WRITE setSpeedValue       NOTIFY speedValueChanged)
    Q_PROPERTY(bool    soundMuted       READ soundMuted       WRITE setSoundMuted       NOTIFY soundMutedChanged)
    Q_PROPERTY(double  horizonY         READ horizonY         WRITE setHorizonY         NOTIFY horizonYChanged)
    Q_PROPERTY(double  pitchDeg         READ pitchDeg         WRITE setPitchDeg         NOTIFY pitchDegChanged)
    Q_PROPERTY(bool    turnLeft         READ turnLeft         WRITE setTurnLeft         NOTIFY turnLeftChanged)
    Q_PROPERTY(bool    turnRight        READ turnRight        WRITE setTurnRight        NOTIFY turnRightChanged)
    Q_PROPERTY(QString sourceState      READ sourceState      WRITE setSourceState      NOTIFY sourceStateChanged)
    Q_PROPERTY(bool    aiHealthy        READ aiHealthy        WRITE setAiHealthy        NOTIFY aiHealthyChanged)
    Q_PROPERTY(QString obdState         READ obdState         WRITE setObdState         NOTIFY obdStateChanged)
    Q_PROPERTY(int     calibProgress    READ calibProgress    WRITE setCalibProgress    NOTIFY calibProgressChanged)
    Q_PROPERTY(double  roadBend         READ roadBend         WRITE setRoadBend         NOTIFY roadBendChanged)
    Q_PROPERTY(bool    wifiOn           READ wifiOn           WRITE setWifiOn           NOTIFY wifiOnChanged)
    Q_PROPERTY(bool    btOn             READ btOn             WRITE setBtOn             NOTIFY btOnChanged)
    Q_PROPERTY(bool    btConnected      READ btConnected      WRITE setBtConnected      NOTIFY btConnectedChanged)

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
        , m_inLaneVehicles(0)
        , m_hasSign(false)
        , m_signCode("")
        , m_signName("")
        , m_signType("guide")
        , m_speedSign(-1)
        , m_auxSign(-1)
        , m_soundMuted(false)
        , m_horizonY(240.0)
        , m_pitchDeg(0.0)
        , m_turnLeft(false)
        , m_turnRight(false)
    {}

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
    int     inLaneVehicles() const { return m_inLaneVehicles; }
    bool    hasSign()        const { return m_hasSign; }
    QString signCode()       const { return m_signCode; }
    QString signName()       const { return m_signName; }
    QString signType()       const { return m_signType; }
    int     speedSign()      const { return m_speedSign; }
    int     auxSign()        const { return m_auxSign; }
    int     speedValue()     const { return m_speedValue; }
    bool    soundMuted()     const { return m_soundMuted; }
    double  horizonY()       const { return m_horizonY; }
    double  pitchDeg()       const { return m_pitchDeg; }
    bool    turnLeft()       const { return m_turnLeft; }
    bool    turnRight()      const { return m_turnRight; }
    QString sourceState()    const { return m_sourceState; }
    bool    aiHealthy()      const { return m_aiHealthy; }
    QString obdState()       const { return m_obdState; }
    bool    wifiOn()         const { return m_wifiOn; }
    bool    btOn()           const { return m_btOn; }
    bool    btConnected()    const { return m_btConnected; }
    int     calibProgress()  const { return m_calibProgress; }
    double  roadBend()       const { return m_roadBend; }

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
    void setSpeedSign(int v) {
        if (m_speedSign != v) { m_speedSign = v; emit speedSignChanged(); }
    }
    void setAuxSign(int v) {
        if (m_auxSign != v) { m_auxSign = v; emit auxSignChanged(); }
    }
    void setSpeedValue(int v) {
        if (m_speedValue != v) { m_speedValue = v; emit speedValueChanged(); }
    }
    void setSoundMuted(bool v) {
        if (m_soundMuted != v) { m_soundMuted = v; emit soundMutedChanged(); }
    }
    void setHorizonY(double v) {
        if (std::abs(m_horizonY - v) > 0.1) { m_horizonY = v; emit horizonYChanged(); }
    }
    void setPitchDeg(double v) {
        if (std::abs(m_pitchDeg - v) > 0.05) { m_pitchDeg = v; emit pitchDegChanged(); }
    }
    void setTurnLeft(bool v) {
        if (m_turnLeft != v) { m_turnLeft = v; emit turnLeftChanged(); }
    }
    void setTurnRight(bool v) {
        if (m_turnRight != v) { m_turnRight = v; emit turnRightChanged(); }
    }
    void setSourceState(const QString& v) {
        if (m_sourceState != v) { m_sourceState = v; emit sourceStateChanged(); }
    }
    void setAiHealthy(bool v) {
        if (m_aiHealthy != v) { m_aiHealthy = v; emit aiHealthyChanged(); }
    }
    void setObdState(const QString& v) {
        if (m_obdState != v) { m_obdState = v; emit obdStateChanged(); }
    }
    void setRoadBend(double v) {
        if (std::abs(m_roadBend - v) > 2e-4) { m_roadBend = v; emit roadBendChanged(); }
    }
    void setCalibProgress(int v) {
        if (m_calibProgress != v) { m_calibProgress = v; emit calibProgressChanged(); }
    }
    void setWifiOn(bool v) {
        if (m_wifiOn != v) { m_wifiOn = v; emit wifiOnChanged(); }
    }
    void setBtOn(bool v) {
        if (m_btOn != v) { m_btOn = v; emit btOnChanged(); }
    }
    void setBtConnected(bool v) {
        if (m_btConnected != v) { m_btConnected = v; emit btConnectedChanged(); }
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
    void inLaneVehiclesChanged();
    void hasSignChanged();
    void signCodeChanged();
    void signNameChanged();
    void signTypeChanged();
    void speedSignChanged();
    void auxSignChanged();
    void speedValueChanged();
    void soundMutedChanged();
    void horizonYChanged();
    void pitchDegChanged();
    void turnLeftChanged();
    void turnRightChanged();
    void sourceStateChanged();
    void aiHealthyChanged();
    void obdStateChanged();
    void calibProgressChanged();
    void roadBendChanged();
    void wifiOnChanged();
    void btOnChanged();
    void btConnectedChanged();

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
    int     m_inLaneVehicles;
    bool    m_hasSign;
    QString m_signCode;
    QString m_signName;
    QString m_signType;
    int     m_speedSign;
    int     m_auxSign;
    int     m_speedValue{0};
    bool    m_soundMuted;
    double  m_horizonY;
    double  m_pitchDeg;
    bool    m_turnLeft;
    bool    m_turnRight;
    QString m_sourceState{"LIVE"};
    bool    m_aiHealthy{true};
    QString m_obdState{"off"};
    int     m_calibProgress{-1};
    double  m_roadBend{0.0};
    bool    m_wifiOn{false};
    bool    m_btOn{false};
    bool    m_btConnected{false};
};

#endif
