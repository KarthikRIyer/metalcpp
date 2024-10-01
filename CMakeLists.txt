cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
project(metalcpp)

#option(METAL_CPP_BUILD_EXAMPLES "Build examples" ON)

add_subdirectory(metal-cmake)  # Library definition

set(SOURCES src/main.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${QUARTZ_CORE_FRAMEWORK}
        "-framework AppKit"
        "-framework MetalKit"
        )

target_link_libraries(${PROJECT_NAME} PRIVATE METAL_CPP)

#if(METAL_CPP_BUILD_EXAMPLES)
#    add_subdirectory(src)  # Add targets
#endif(METAL_CPP_BUILD_EXAMPLES)
