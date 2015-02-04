set(SOURCE_FILES
	AAM_PDM.cpp
	AAM_TDM.cpp
	AAM_Util.cpp
	AAM_Shape.cpp
	AAM_CAM.cpp
	VJfacedetect.cpp
	AAM_MovieAVI.cpp
	AAM_PAW.cpp
	AAM_Basic.cpp
	AAM_IC.cpp
)

add_library(aamlibrary SHARED ${SOURCE_FILES})
target_link_libraries(aamlibrary ${OpenCV_LIBS})
set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
