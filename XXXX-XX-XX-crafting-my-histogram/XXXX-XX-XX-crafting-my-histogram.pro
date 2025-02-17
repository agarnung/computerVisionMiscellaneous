QT += core widgets gui

CONFIG += c++20

TEMPLATE = app

SOURCES += \
        main.cpp \

CONFIG += link_pkgconfig
PKGCONFIG += opencv4

QMAKE_CXXFLAGS += -O3 -march=native

