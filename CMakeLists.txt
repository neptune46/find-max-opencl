cmake_minimum_required(VERSION 3.15)
project(ocl_find_max LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

add_executable(ocl_find_max
    src/main.cpp
)

target_include_directories(ocl_find_max PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(ocl_find_max PRIVATE ${OpenCL_LIBRARIES})

# Copy kernel next to the executable for runtime compilation
add_custom_command(TARGET ocl_find_max POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.cl
            $<TARGET_FILE_DIR:ocl_find_max>/kernels.cl)

message(STATUS "OpenCL include dirs: ${OpenCL_INCLUDE_DIRS}")
message(STATUS "OpenCL libraries: ${OpenCL_LIBRARIES}")

