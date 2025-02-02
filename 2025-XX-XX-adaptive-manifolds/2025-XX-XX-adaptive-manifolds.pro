TEMPLATE = app

CONFIG += c++20

SOURCES += \
    main.cpp \

HEADERS += \

CONFIG += link_pkgconfig
PKGCONFIG += opencv4

QMAKE_CXXFLAGS += -O3 -march=native
