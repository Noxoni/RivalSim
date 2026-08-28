cmake_minimum_required(VERSION 3.24)

project(rival_demo_recorder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BAKKESMOD_SDK_ROOT "" CACHE PATH "Pinned BakkesModSDK checkout")
if(NOT BAKKESMOD_SDK_ROOT)
    message(FATAL_ERROR
        "Set BAKKESMOD_SDK_ROOT to BakkesModSDK commit "
        "479e8f571cf554b25f4eeb64d611dec4133edcaf")
endif()

find_package(Git REQUIRED)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${BAKKESMOD_SDK_ROOT}" rev-parse HEAD
    OUTPUT_VARIABLE BAKKESMOD_SDK_HEAD
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE BAKKESMOD_SDK_HEAD_RESULT
)
if(NOT BAKKESMOD_SDK_HEAD_RESULT EQUAL 0 OR
   NOT BAKKESMOD_SDK_HEAD STREQUAL "479e8f571cf554b25f4eeb64d611dec4133edcaf")
    message(FATAL_ERROR
        "BAKKESMOD_SDK_ROOT must be the pinned SDK commit; got '${BAKKESMOD_SDK_HEAD}'")
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}/../.." rev-parse HEAD
    OUTPUT_VARIABLE RIVALREC_GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE RIVALREC_GIT_RESULT
)
if(NOT RIVALREC_GIT_RESULT EQUAL 0)
    set(RIVALREC_GIT_SHA "unknown")
endif()

set(RIVALREC_FORMAT_SOURCES
    src/recording_format.cpp
)

add_library(rivalrec_format STATIC ${RIVALREC_FORMAT_SOURCES})
target_include_directories(rivalrec_format PUBLIC include)
target_compile_definitions(rivalrec_format PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(rivalrec_format PUBLIC bcrypt)

add_executable(rivalrec_format_fixture tests/format_fixture.cpp)
target_link_libraries(rivalrec_format_fixture PRIVATE rivalrec_format)

include(CTest)
add_executable(rivalrec_policy_tests tests/recorder_policy_tests.cpp)
target_include_directories(rivalrec_policy_tests PRIVATE include)
add_test(NAME rivalrec_policy_tests COMMAND rivalrec_policy_tests)

add_library(rival_demo_recorder SHARED
    src/rival_demo_recorder.cpp
    src/rival_demo_recorder.hpp
    include/rivalrec/recorder_policy.hpp
)
set_target_properties(rival_demo_recorder PROPERTIES
    PREFIX ""
    OUTPUT_NAME "rival_demo_recorder"
)
target_include_directories(rival_demo_recorder PRIVATE
    include
    "${BAKKESMOD_SDK_ROOT}/include"
)
target_compile_definitions(rival_demo_recorder PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
    RIVALREC_GIT_SHA="${RIVALREC_GIT_SHA}"
    RIVALREC_SDK_REVISION="479e8f571cf554b25f4eeb64d611dec4133edcaf"
    RIVALREC_PLUGIN_BUILD="1.0.1"
)
target_link_libraries(rival_demo_recorder PRIVATE
    rivalrec_format
    "${BAKKESMOD_SDK_ROOT}/lib/pluginsdk.lib"
)

if(MSVC)
    foreach(target rivalrec_format rivalrec_format_fixture rivalrec_policy_tests rival_demo_recorder)
        target_compile_options(${target} PRIVATE /W4 /permissive- /EHsc /utf-8)
    endforeach()
endif()
