cmake_minimum_required(VERSION 3.16)
project(fcitx5-asr-addon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Fcitx5Core 5.0.0 REQUIRED)

add_library(fcitx5-asr MODULE asr.cpp)
target_link_libraries(fcitx5-asr PRIVATE Fcitx5::Core Fcitx5::Utils)
set_target_properties(fcitx5-asr PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "fcitx5-asr"
)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "Install prefix" FORCE)
endif()

set(ADDON_INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}/lib/fcitx5")
set(ADDON_INSTALL_DATADIR "${CMAKE_INSTALL_PREFIX}/share/fcitx5/addon")

set(FCITX5_ASR_LIB_PATH "${ADDON_INSTALL_LIBDIR}/libfcitx5-asr")

configure_file(fcitx5-asr.conf.in fcitx5-asr.conf @ONLY)

install(TARGETS fcitx5-asr DESTINATION "${ADDON_INSTALL_LIBDIR}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/fcitx5-asr.conf" DESTINATION "${ADDON_INSTALL_DATADIR}")
