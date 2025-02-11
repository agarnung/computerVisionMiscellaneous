TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp

CONFIG += link_pkgconfig
PKGCONFIG += opencv

INCLUDEPATH += /usr/local/include/ImageMagick-7
QMAKE_CXXFLAGS += $(shell Magick++-config --cxxflags)
LIBS += $(shell /usr/local/bin/Magick++-config --libs)
DEFINES += "MAGICKCORE_QUANTUM_DEPTH=16"
DEFINES += "MAGICKCORE_HDRI_ENABLE=1"
