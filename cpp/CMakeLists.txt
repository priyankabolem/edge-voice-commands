cmake_minimum_required(VERSION 3.28)
project(kws_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ExecuTorch C++ runtime. By default this is the prebuilt SDK shipped in the
# executorch pip wheel; pass -DEXECUTORCH_DIR=<prefix>/share/cmake to use another
# installation (e.g. a build from source).
if(NOT EXECUTORCH_DIR)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import executorch, os; print(os.path.join(list(executorch.__path__)[0], 'share', 'cmake'))"
    OUTPUT_VARIABLE EXECUTORCH_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  set(KWS_EXECUTORCH_SOURCE "prebuilt (executorch pip wheel)")
else()
  set(KWS_EXECUTORCH_SOURCE "custom (${EXECUTORCH_DIR})")
endif()
find_package(executorch CONFIG REQUIRED PATHS "${EXECUTORCH_DIR}" NO_DEFAULT_PATH)
message(STATUS "ExecuTorch ${EXECUTORCH_BUILD_VERSION} from ${EXECUTORCH_DIR}")

string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type_upper)
string(STRIP "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_build_type_upper}}" KWS_EFFECTIVE_CXX_FLAGS)
configure_file(include/kws/build_info.h.in "${CMAKE_CURRENT_BINARY_DIR}/generated/kws/build_info.h")

add_library(kws_features STATIC src/assets.cc src/fft.cc src/log_mel.cc src/wav_reader.cc)
target_include_directories(kws_features PUBLIC include "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_options(kws_features PRIVATE -Wall -Wextra -Wpedantic)

add_library(kws_model STATIC src/kws_model.cc)
target_include_directories(kws_model PUBLIC include)
target_link_libraries(kws_model PUBLIC executorch::runtime executorch::backend_xnnpack)
target_compile_options(kws_model PRIVATE -Wall -Wextra)

foreach(app classify_wav batch_infer alloc_check)
  add_executable(${app} apps/${app}.cc)
  target_link_libraries(${app} PRIVATE kws_features kws_model)
endforeach()

if(APPLE)
  add_library(alloc_counter SHARED tools/alloc_counter.cc)
  target_include_directories(alloc_counter PUBLIC tools)
  target_link_libraries(alloc_check PRIVATE alloc_counter)
endif()

add_executable(dump_features apps/dump_features.cc)
target_link_libraries(dump_features PRIVATE kws_features)

add_executable(build_info apps/build_info.cc)
target_link_libraries(build_info PRIVATE kws_features)
