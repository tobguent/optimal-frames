cmake_minimum_required(VERSION 2.8)
 
PROJECT(ReferenceFrameFilter)

FIND_PACKAGE(VTK REQUIRED)
include(${VTK_USE_FILE})

FIND_PACKAGE(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(ReferenceFrameFilter MACOSX_BUNDLE ReferenceFrameFilter vtkReferenceFrameFilter.cxx vtkReferenceFrameFilter.h)
 
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  MESSAGE(STATUS "Using OpenMP parallelization")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ELSE()
  MESSAGE(STATUS "Not using OpenMP parallelization")
ENDIF()

if(VTK_LIBRARIES)
  target_link_libraries(ReferenceFrameFilter ${VTK_LIBRARIES})
else()
  target_link_libraries(ReferenceFrameFilter vtkHybrid vtkWidgets)
endif()