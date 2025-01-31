TEMPLATE = app

SOURCES += \
        main.cpp

CONFIG -= debug
CONFIG += console c++17
CONFIG += link_pkgconfig

PKGCONFIG += opencv4
PKGCONFIG += eigen3

LIBS += -L/usr/lib/x86_64-linux-gnu \
        -L/usr/lib \

# Opciones de compilacion
QMAKE_CXXFLAGS += -fopenmp # Necesario para el compilador reconozca y procese las directivas de OpenMP
QMAKE_CXXFLAGS += -O0
#QMAKE_CXXFLAGS += -O3 -march=native
#QMAKE_CXXFLAGS += -march=native
#QMAKE_CXXFLAGS += -ftree-vectorize
#QMAKE_CXXFLAGS += -mfma
#QMAKE_CXXFLAGS += -mavx
#QMAKE_CXXFLAGS += -mavx2
#QMAKE_CXXFLAGS += -msse4.2
#QMAKE_CXXFLAGS += -funroll-loops -fomit-frame-pointer
#QMAKE_CXXFLAGS += -finline-functions # Forzar al compilador a usar inline en tus funciones pequeñas para mejorar la velocidad de ejecución

# Opciones de enlazado
QMAKE_LFLAGS += -fopenmp
#QMAKE_LFLAGS += -Wl,--strip-all # Optimización en el proceso de enlace.


