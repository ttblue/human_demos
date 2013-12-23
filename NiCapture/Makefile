include ../../../ThirdParty/PSCommon/BuildSystem/CommonDefs.mak

BIN_DIR = ../../../Bin

INC_DIRS = \
	../../../Include \
	../../../ThirdParty/PSCommon/XnLib/Include \
	../../../ThirdParty/GL/ \
	/home/ankush/sources/openCV/OpenCV-2.4.2/include/opencv \

SRC_FILES = simple_capture.cpp

ifeq ("$(OSTYPE)","Darwin")
	CFLAGS += -DMACOS
	LDFLAGS += -framework OpenGL -framework GLUT
else
	CFLAGS += -DUNIX -DGLX_GLXEXT_LEGACY
	USED_LIBS += glut GL
endif

LIB_DIRS  += ../../../ThirdParty/PSCommon/XnLib/Bin/$(PLATFORM)-$(CFG) 
USED_LIBS += OpenNI2 XnLib

EXE_NAME = simple_capture

CFLAGS += -Wall 
USED_LIBS   += opencv_core opencv_highgui opencv_imgproc opencv_legacy 

include ../../../ThirdParty/PSCommon/BuildSystem/CommonCppMakefile
