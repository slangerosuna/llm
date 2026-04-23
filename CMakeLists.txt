cmake_minimum_required(VERSION 3.24)

project(llm_sycl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CMake >= 3.30 removes FindBoost (CMP0167), so use config mode.
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
find_package(Boost CONFIG REQUIRED COMPONENTS json system)

file(GLOB_RECURSE HOST_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

if(HOST_SOURCES STREQUAL "")
  message(FATAL_ERROR "No C++ host sources found under src/.")
endif()

add_executable(llm_sycl ${HOST_SOURCES})

target_compile_options(llm_sycl PRIVATE -fsycl)
target_link_options(llm_sycl PRIVATE -fsycl)
target_link_libraries(llm_sycl PRIVATE Boost::json Boost::system)
