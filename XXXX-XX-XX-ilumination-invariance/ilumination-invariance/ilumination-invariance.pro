TEMPLATE = app

SOURCES += \
        main.cpp

CONFIG -= debug
CONFIG += console c++17
CONFIG += link_pkgconfig

PKGCONFIG += opencv
PKGCONFIG += eigen3
PKGCONFIG += fftw3

QMAKE_CXXFLAGS += -fopenmp
QMAKE_CXXFLAGS += -O3 -march=native
QMAKE_CXXFLAGS += -march=native
QMAKE_CXXFLAGS += -ftree-vectorize
QMAKE_CXXFLAGS += -mfma
QMAKE_CXXFLAGS += -mavx
QMAKE_CXXFLAGS += -mavx2
QMAKE_CXXFLAGS += -msse4.2
QMAKE_CXXFLAGS += -funroll-loops -fomit-frame-pointer
QMAKE_CXXFLAGS += -finline-functions

QMAKE_LFLAGS += -fopenmp
QMAKE_LFLAGS += -Wl,--strip-all

USE_QWT=no
LIB_DEPENDENCIES += DsiXml DsiQtd DsiData DsiLog DsiMisc DsiQtUtils DsiNumericalRecipes lapack gmp mpfr DsiMisc DsiCv
include ($(DSILIBS_SRC_PATH)/dsilibs.pri)

HEADERS += \
    functions.h \
