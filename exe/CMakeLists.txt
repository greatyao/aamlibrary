INCLUDE_DIRECTORIES(
	${PROJECT_SOURCE_DIR}/src
)

add_executable(build  train.cpp)
target_link_libraries(build aamlibrary ${OpenCV_LIBS})
#set_target_properties(build PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_executable(fit  fit.cpp)
target_link_libraries(fit aamlibrary ${OpenCV_LIBS})
#set_target_properties(fit PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)


