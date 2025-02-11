QT += core widgets

CONFIG += c++20

TEMPLATE = app

SOURCES += \
        main.cpp \

CONFIG += link_pkgconfig
PKGCONFIG += opencv
PKGCONFIG += fftw3

LIBS += -lopencv_core \
        -lopencv_highgui \
        -lopencv_imgproc \

USE_QWT=no
LIB_DEPENDENCIES += DsiXml DsiQtd DsiData DsiLog DsiMisc DsiQtUtils DsiMisc DsiLog DsiNumericalRecipes DsiCv
include ($(DSILIBS_SRC_PATH)/dsilibs.pri)
