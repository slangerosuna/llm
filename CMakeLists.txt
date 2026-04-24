cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CXX_COMPILER)
  find_program(ICPX_EXECUTABLE icpx)
  if(ICPX_EXECUTABLE)
    set(CMAKE_CXX_COMPILER "${ICPX_EXECUTABLE}" CACHE FILEPATH "C++ compiler" FORCE)
  endif()
endif()

project(llm_sycl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enforce Boost lookup through FindBoost (module mode) because many distros
# ship Boost without BoostConfig.cmake files.
# if(POLICY CMP0167)
#   cmake_policy(SET CMP0167 OLD)
# endif()
# set(Boost_NO_BOOST_CMAKE ON)
# find_package(Boost REQUIRED COMPONENTS json)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fsycl" COMPILER_SUPPORTS_FSYCL)
if(NOT COMPILER_SUPPORTS_FSYCL)
  message(FATAL_ERROR "A SYCL-capable compiler is required. Re-run CMake with a SYCL compiler, e.g. -DCMAKE_CXX_COMPILER=icpx")
endif()

file(GLOB_RECURSE HOST_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

if(HOST_SOURCES STREQUAL "")
  message(FATAL_ERROR "No C++ host sources found under src/.")
endif()

add_executable(llm_sycl ${HOST_SOURCES})

target_include_directories(llm_sycl PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(llm_sycl PRIVATE -fsycl -mavx2 -mf16c)
target_link_options(llm_sycl PRIVATE -fsycl)
# target_link_libraries(llm_sycl PRIVATE Boost::json)

enable_testing()
add_subdirectory(tests)
