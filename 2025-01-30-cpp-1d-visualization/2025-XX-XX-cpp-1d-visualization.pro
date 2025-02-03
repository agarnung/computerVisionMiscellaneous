QT += core widgets gui

CONFIG += c++20

TEMPLATE = app

SOURCES += \
        main.cpp \

CONFIG += link_pkgconfig
PKGCONFIG += opencv4

LIBS += -L/usr/lib/x86_64-linux-gnu \
        -L/usr/lib \

MOC_DIR = moc # Esto debería forzar la generación de archivos moc en el directorio moc/

INCLUDEPATH += /opt/matplotlib-cpp
INCLUDEPATH += /usr/include/python3.10

LIBS += -lpython3.10

QMAKE_CXXFLAGS += -O3 -march=native # Flags de compilador para mejorar eficiencia de código máquina vectorizando operaciones


