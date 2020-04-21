cmake_minimum_required (VERSION 3.10)

IF(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE "Release")
	message(STATUS "Setting build type to ${CMAKE_BUILD_TYPE}")
ENDIF()

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "")
endif()

project("lti")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

###############################################################################
## Output directories.
###############################################################################
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

###############################################################################
## Use C++ 17.
###############################################################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
## Find packages
###############################################################################
find_package(Eigen3 CONFIG REQUIRED)

###############################################################################
# #Build header only lib with Eigen3
###############################################################################
add_library(
    ${PROJECT_NAME}
    INTERFACE
)
target_link_libraries(
    ${PROJECT_NAME} 
    INTERFACE
    Eigen3::Eigen
)
target_include_directories(
    ${PROJECT_NAME}
    INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

###############################################################################
## Add tests
###############################################################################
enable_testing()
add_subdirectory(test)
