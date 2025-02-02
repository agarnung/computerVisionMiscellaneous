TEMPLATE = app

CONFIG += c++20

SOURCES += \
    main.cpp \

HEADERS += \

CONFIG += link_pkgconfig
PKGCONFIG += opencv4

INCLUDEPATH += /opt/matplotlib-cpp

PYTHON_VER = 3.10
PYTHON_PATH = /usr/include/python$${PYTHON_VER}
INCLUDEPATH += $$PYTHON_PATH
LIBS += -L/usr/lib/python$${PYTHON_VER}/config-$(shell python3-config --abiflags) -lpython$${PYTHON_VER}

QMAKE_CXXFLAGS += -O3 -march=native
