cmake_minimum_required(VERSION 3.16)
project(adas_perception CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio video)
find_package(Threads REQUIRED)

set(ONNXRUNTIME_DIR "" CACHE PATH "ONNX Runtime root (include/, lib/)")
find_path(ORT_INCLUDE onnxruntime_cxx_api.h HINTS ${ONNXRUNTIME_DIR}/include /usr/include/onnxruntime /usr/local/include/onnxruntime)
find_library(ORT_LIB onnxruntime HINTS ${ONNXRUNTIME_DIR}/lib /usr/lib /usr/local/lib)
if(NOT ORT_INCLUDE OR NOT ORT_LIB)
  message(FATAL_ERROR "ONNX Runtime not found: pass -DONNXRUNTIME_DIR=...")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(ADAS_ARCH_FLAGS -O3 -mcpu=cortex-a72 -ftree-vectorize)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
  set(ADAS_ARCH_FLAGS -O3 -mcpu=cortex-a72 -mfpu=neon-fp-armv8 -mfloat-abi=hard -ftree-vectorize)
else()
  set(ADAS_ARCH_FLAGS -O3)
endif()

add_library(adas_core INTERFACE)
target_include_directories(adas_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${ORT_INCLUDE})
target_link_libraries(adas_core INTERFACE ${OpenCV_LIBS} ${ORT_LIB} Threads::Threads)
target_compile_options(adas_core INTERFACE ${ADAS_ARCH_FLAGS} -Wall -Wextra -Wno-unused-parameter)

add_executable(adas_main app/adas_main.cpp)
target_link_libraries(adas_main adas_core)

add_executable(test_noise tools/cpp_test/test_noise.cpp)
target_link_libraries(test_noise adas_core)
add_executable(test_calib tools/cpp_test/test_calib.cpp)
target_link_libraries(test_calib adas_core)
add_executable(test_full tools/cpp_test/test_full.cpp)
target_link_libraries(test_full adas_core)
add_executable(test_pipeline tools/cpp_test/test_pipeline.cpp)
target_link_libraries(test_pipeline adas_core)
add_executable(test_flow tools/cpp_test/test_flow.cpp)
target_link_libraries(test_flow adas_core)

find_package(Qt5 QUIET COMPONENTS Core Gui Qml Quick QuickControls2 Network)
if(Qt5_FOUND)
  set(CMAKE_AUTOMOC ON)
  set(CMAKE_AUTORCC ON)
  add_executable(adas_pi4
    app/main_qt.cpp
    AdasController.h
    FrameProvider.h
    CameraWorker.h
    qml.qrc
  )
  target_link_libraries(adas_pi4
    adas_core
    Qt5::Core
    Qt5::Gui
    Qt5::Qml
    Qt5::Quick
    Qt5::QuickControls2
    Qt5::Network
  )
endif()
