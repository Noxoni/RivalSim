cmake_minimum_required(VERSION 3.26)

project(RocketSimDiagnostic LANGUAGES CXX)

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)

set(ROCKETSIM_BINDING_SOURCE_DIR "" CACHE PATH "Pinned RocketSimPython checkout")
if(NOT EXISTS "${ROCKETSIM_BINDING_SOURCE_DIR}/src/RocketSim.h")
    message(FATAL_ERROR "Set ROCKETSIM_BINDING_SOURCE_DIR to the pinned RocketSimPython checkout")
endif()

file(GLOB_RECURSE ROCKETSIM_BULLET3_SOURCES CONFIGURE_DEPENDS
    "${ROCKETSIM_BINDING_SOURCE_DIR}/libsrc/bullet3-3.24/*.cpp"
)
file(GLOB_RECURSE ROCKETSIM_SOURCES CONFIGURE_DEPENDS
    "${ROCKETSIM_BINDING_SOURCE_DIR}/src/*.cpp"
)

add_library(rocketsim_diagnostic_core STATIC
    ${ROCKETSIM_BULLET3_SOURCES}
    ${ROCKETSIM_SOURCES}
)
if(IPO_SUPPORTED)
    set_target_properties(rocketsim_diagnostic_core PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    )
endif()
target_compile_features(rocketsim_diagnostic_core PUBLIC cxx_std_20)
target_compile_definitions(rocketsim_diagnostic_core PRIVATE
    RS_DONT_LOG=1
    XXH_INLINE_ALL=1
    XXH_STATIC_LINKING_ONLY=1
)
target_include_directories(rocketsim_diagnostic_core PUBLIC
    "${ROCKETSIM_BINDING_SOURCE_DIR}/libsrc"
    "${ROCKETSIM_BINDING_SOURCE_DIR}/src"
)

add_executable(rocketsim_diagnostic trace.cpp)
if(IPO_SUPPORTED)
    set_target_properties(rocketsim_diagnostic PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    )
endif()
target_compile_features(rocketsim_diagnostic PRIVATE cxx_std_20)
target_link_libraries(rocketsim_diagnostic PRIVATE rocketsim_diagnostic_core)
