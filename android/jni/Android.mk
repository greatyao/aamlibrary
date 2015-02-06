LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

OPENCV_CAMERA_MODULES:=on
OPENCV_INSTALL_MODULES:=off
OPENCV_LIB_TYPE:=STATIC

include E:\adt-bundle-windows-x86-20131030\OpenCV-2.4.9-android-sdk/sdk/native/jni/OpenCV.mk

LOCAL_SRC_FILES  := ../../src/AAM_Basic.cpp \
					../../src/AAM_IC.cpp \
					../../src/AAM_Util.cpp \
					../../src/AAM_Shape.cpp \
					../../src/AAM_PDM.cpp \
					../../src/AAM_PAW.cpp \
					../../src/AAM_TDM.cpp \
					../../src/AAM_CAM.cpp \
					../../src/VJfacedetect.cpp \
					DemoFit.cpp
					
LOCAL_C_INCLUDES += ../../src 
LOCAL_LDLIBS     += -llog -ldl  

LOCAL_MODULE := aamlibrary

include $(BUILD_SHARED_LIBRARY)
