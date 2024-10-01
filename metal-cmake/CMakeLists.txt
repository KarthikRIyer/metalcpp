find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(QUARTZ_CORE_FRAMEWORK QuartzCore REQUIRED)

# Library definition
add_library(METAL_CPP
        ${CMAKE_CURRENT_SOURCE_DIR}/definition.cpp
        )

# Metal cpp headers
target_include_directories(METAL_CPP PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp"
        )

# Metal cpp library (linker)
target_link_libraries(METAL_CPP
        "-framework Metal"
#        "-framework MetalKit"
#        "-framework AppKit"
        "-framework Foundation"
        "-framework QuartzCore"
        )