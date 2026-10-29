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
# installation (e.g. third_party/executorch-install/lib/cmake/ExecuTorch from
# scripts/build_executorch.sh).
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

# The pip wheel exports shared libraries as executorch::*; a source install
# (scripts/build_executorch.sh) exports static libraries under plain names that
# already force-load the runtime and backend so their registrations are kept.
if(TARGET executorch::runtime)
  set(KWS_ET_RUNTIME executorch::runtime)
  set(KWS_ET_XNNPACK executorch::backend_xnnpack)
  set(KWS_ET_THREADPOOL executorch::threadpool)
  set(KWS_ET_ETDUMP executorch::etdump)
else()
  set(KWS_ET_RUNTIME executorch extension_data_loader)
  set(KWS_ET_XNNPACK xnnpack_backend)
  set(KWS_ET_THREADPOOL extension_threadpool)
  set(KWS_ET_ETDUMP etdump flatccrt)
  # The source install does not record its version; the build script writes it.
  if(EXISTS "${EXECUTORCH_DIR}/../../../VERSION")
    file(STRINGS "${EXECUTORCH_DIR}/../../../VERSION" EXECUTORCH_BUILD_VERSION LIMIT_COUNT 1)
  endif()
  set(KWS_EXECUTORCH_SOURCE "source build (${EXECUTORCH_DIR})")
endif()
message(STATUS "ExecuTorch ${EXECUTORCH_BUILD_VERSION} from ${EXECUTORCH_DIR}")

string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type_upper)
string(STRIP "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_build_type_upper}}" KWS_EFFECTIVE_CXX_FLAGS)
configure_file(include/kws/build_info.h.in "${CMAKE_CURRENT_BINARY_DIR}/generated/kws/build_info.h")

add_library(kws_features STATIC src/assets.cc src/fft.cc src/log_mel.cc src/posterior.cc
                                src/ring_buffer.cc src/wav_reader.cc)
target_include_directories(kws_features PUBLIC include "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_options(kws_features PRIVATE -Wall -Wextra -Wpedantic)

add_library(kws_model STATIC src/kws_model.cc src/threads.cc)
target_include_directories(kws_model PUBLIC include)
target_link_libraries(kws_model PUBLIC ${KWS_ET_RUNTIME} ${KWS_ET_XNNPACK} ${KWS_ET_THREADPOOL})
target_compile_options(kws_model PRIVATE -Wall -Wextra)

foreach(app classify_wav batch_infer alloc_check bench stream)
  add_executable(${app} apps/${app}.cc)
  target_link_libraries(${app} PRIVATE kws_features kws_model)
endforeach()

if(APPLE)
  add_library(alloc_counter SHARED tools/alloc_counter.cc)
  target_include_directories(alloc_counter PUBLIC tools)
  target_link_libraries(alloc_check PRIVATE alloc_counter)
  target_link_libraries(stream PRIVATE alloc_counter)
  target_compile_definitions(stream PRIVATE KWS_HAVE_ALLOC_COUNTER)
endif()

add_executable(dump_features apps/dump_features.cc)
target_link_libraries(dump_features PRIVATE kws_features)

add_executable(build_info apps/build_info.cc)
target_link_libraries(build_info PRIVATE kws_features)

# The ExecuTorch v1.5.1 source install ships libetdump but not its headers, so the
# per-operator profiler is only built against runtimes that provide them.
find_file(KWS_ETDUMP_HEADER executorch/devtools/etdump/etdump_flatcc.h
          PATHS ${EXECUTORCH_INCLUDE_DIRS} NO_DEFAULT_PATH)
if(KWS_ETDUMP_HEADER)
  add_executable(profile_ops apps/profile_ops.cc)
  target_link_libraries(profile_ops PRIVATE kws_model ${KWS_ET_ETDUMP})
else()
  message(STATUS "ETDump headers not installed with this ExecuTorch; skipping profile_ops")
endif()
