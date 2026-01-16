cmake_minimum_required(VERSION 3.10)
project(detect-speech)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
add_compile_options(-O3)
add_definitions(-DWHISPER_FFMPEG)

# Include directories
include_directories(include src)

# Find FFmpeg dependencies using PkgConfig
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_check_modules(SWRESAMPLE REQUIRED libswresample)

# Find whisper and ggml libraries
find_library(WHISPER_LIB whisper REQUIRED)
find_library(GGML_LIB ggml REQUIRED)

# Threads support
find_package(Threads REQUIRED)

# Sources
set(SOURCES
    detect-speech.cpp
    src/common.cpp
    src/common-whisper.cpp
    src/ffmpeg-transcode.cpp
)

# Executable
add_executable(detect-speech ${SOURCES})

# Include directories for dependencies
target_include_directories(detect-speech PRIVATE 
    ${AVCODEC_INCLUDE_DIRS}
    ${AVFORMAT_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIRS}
    ${SWRESAMPLE_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(detect-speech PRIVATE
    ${WHISPER_LIB}
    ${GGML_LIB}
    ${AVCODEC_LIBRARIES}
    ${AVFORMAT_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    ${SWRESAMPLE_LIBRARIES}
    Threads::Threads
    ${CMAKE_DL_LIBS}
    m
)
