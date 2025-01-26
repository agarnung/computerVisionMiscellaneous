TEMPLATE = app

SOURCES += \
        main.cpp

CONFIG += c++20

CONFIG += link_pkgconfig
PKGCONFIG += opencv4

QMAKE_CXXFLAGS += -O3 -march=native -funroll-loops -fomit-frame-pointer -finline-functions -ftree-vectorize -mavx -mavx2 -msse4.2
