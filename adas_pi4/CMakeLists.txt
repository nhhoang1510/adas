cmake_minimum_required(VERSION 3.16)
project(adas_pi4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler optimization cho ARM Cortex-A72 / Raspberry Pi 4
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    add_compile_options(
        -O3
        -march=armv8-a+simd
        -ffast-math
        -funroll-loops
        -fopenmp
    )
    message(STATUS "Target: ARM64 / Raspberry Pi 4 - NEON enabled")
else()
    add_compile_options(-O3 -march=native -ffast-math -fopenmp)
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 COMPONENTS Core Gui Qml Quick QuickControls2 REQUIRED)
message(STATUS "Qt5 version: ${Qt5_VERSION}")

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

add_executable(adas_pi4
    src/main_qt.cpp
    src/AdasController.h
    src/FrameProvider.h
    src/CameraWorker.h
)

target_include_directories(adas_pi4 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(adas_pi4 PRIVATE
    Qt5::Core
    Qt5::Gui
    Qt5::Qml
    Qt5::Quick
    Qt5::QuickControls2
    ${OpenCV_LIBS}
    pthread
)

# Copy QML files sang thu muc build
file(GLOB QML_FILES "${CMAKE_SOURCE_DIR}/qml/*.qml")
foreach(_qml ${QML_FILES})
    get_filename_component(_name ${_qml} NAME)
    configure_file(${_qml} ${CMAKE_BINARY_DIR}/${_name} COPYONLY)
endforeach()
