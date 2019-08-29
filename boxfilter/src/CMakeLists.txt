include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB SOURCE_LISTS ./*.cc ./*.cpp)
add_executable(test_boxfilter ${SOURCE_LISTS})
