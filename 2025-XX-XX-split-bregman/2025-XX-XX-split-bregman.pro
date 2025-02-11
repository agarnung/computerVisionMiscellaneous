TEMPLATE = app

SOURCES += \
        main.cpp

CONFIG += console c++17

CONFIG += link_pkgconfig
PKGCONFIG += opencv4

QMAKE_CXXFLAGS += -O3 -march=native
