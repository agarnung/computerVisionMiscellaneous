TEMPLATE = app

CONFIG += c++20

SOURCES += \
    main.cpp \

PKGCONFIG += eigen3

QMAKE_CXXFLAGS += -O3 -march=native
