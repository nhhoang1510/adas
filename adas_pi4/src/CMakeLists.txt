cmake_minimum_required(VERSION 3.20)
project(yolopv2_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===========================================================
# Build type mặc định Release
# ===========================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Tối ưu compiler flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast /openmp)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    add_compile_options(-O3 -march=native -ffast-math -fopenmp)
endif()

# ===========================================================
# Tìm OpenCV (cài qua vcpkg hoặc OpenCV installer)
# ===========================================================
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# ===========================================================
# ONNX Runtime - Tải SDK hoặc dùng vcpkg
# Đặt ONNXRUNTIME_DIR tới thư mục chứa include/ và lib/
# ===========================================================
if(NOT DEFINED ONNXRUNTIME_DIR)
    # Thử tự tìm trong thư mục phổ biến Windows
    set(_ORT_CANDIDATES
        "C:/onnxruntime"
        "C:/onnxruntime-win-x64-gpu-1.20.0"
        "$ENV{ONNXRUNTIME_DIR}"
    )
    foreach(_cand ${_ORT_CANDIDATES})
        if(EXISTS "${_cand}/include/onnxruntime_cxx_api.h")
            set(ONNXRUNTIME_DIR "${_cand}")
            break()
        endif()
    endforeach()
endif()

if(NOT DEFINED ONNXRUNTIME_DIR OR NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime SDK không tìm thấy!\n"
        "Tải tại: https://github.com/microsoft/onnxruntime/releases\n"
        "Sau đó: cmake -DONNXRUNTIME_DIR=C:/path/to/onnxruntime-win-x64-gpu-X.Y.Z ...\n"
    )
endif()

message(STATUS "ONNX Runtime dir: ${ONNXRUNTIME_DIR}")

# ===========================================================
# Tìm ONNX Runtime lib (GPU version ưu tiên)
# ===========================================================
find_library(ORT_LIB
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_DIR}/lib"
    NO_DEFAULT_PATH
)
if(NOT ORT_LIB)
    message(FATAL_ERROR "Không tìm thấy onnxruntime.lib trong ${ONNXRUNTIME_DIR}/lib")
endif()

# Kiểm tra nếu có CUDA EP
file(GLOB _ORT_CUDA_DLLS "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_cuda*")
if(_ORT_CUDA_DLLS)
    add_compile_definitions(ORT_HAS_CUDA)
    message(STATUS "ONNX Runtime CUDA EP: ON")
else()
    message(STATUS "ONNX Runtime CUDA EP: OFF (dùng CPU)")
endif()

# Kiểm tra DirectML EP (Windows only)
file(GLOB _ORT_DML_DLLS "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_dml*")
if(_ORT_DML_DLLS AND WIN32)
    add_compile_definitions(ORT_HAS_DML)
    message(STATUS "ONNX Runtime DirectML EP: ON")
endif()

# ===========================================================
# Target executable
# ===========================================================
add_executable(yolopv2_cpp
    main.cpp
)

target_include_directories(yolopv2_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${ONNXRUNTIME_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolopv2_cpp PRIVATE
    ${ORT_LIB}
    ${OpenCV_LIBS}
)

# ===========================================================
# Copy DLLs về thư mục output sau build (Windows)
# ===========================================================
if(WIN32)
    # Copy ONNX Runtime DLLs
    file(GLOB _ORT_DLLS "${ONNXRUNTIME_DIR}/lib/*.dll")
    foreach(_dll ${_ORT_DLLS})
        add_custom_command(TARGET yolopv2_cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_dll}" "$<TARGET_FILE_DIR:yolopv2_cpp>"
            COMMENT "Copying ${_dll}"
        )
    endforeach()

    # Copy OpenCV DLLs
    # Nếu dùng vcpkg, đặt VCPKG_TARGET_TRIPLET=x64-windows
    foreach(_lib ${OpenCV_LIBS})
        get_target_property(_dll_path ${_lib} IMPORTED_LOCATION_RELEASE)
        if(_dll_path AND EXISTS "${_dll_path}")
            string(REPLACE ".lib" ".dll" _dll_file "${_dll_path}")
            if(EXISTS "${_dll_file}")
                add_custom_command(TARGET yolopv2_cpp POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_dll_file}" "$<TARGET_FILE_DIR:yolopv2_cpp>"
                )
            endif()
        endif()
    endforeach()
endif()

# ===========================================================
# Install
# ===========================================================
install(TARGETS yolopv2_cpp RUNTIME DESTINATION bin)

message(STATUS "")
message(STATUS "=== Build Summary ===")
message(STATUS "  Executable  : yolopv2_cpp")
message(STATUS "  Build type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV      : ${OpenCV_VERSION}")
message(STATUS "  ORT         : ${ONNXRUNTIME_DIR}")
message(STATUS "=====================")
