cmake_minimum_required(VERSION 3.5.0)
project(VPStest)
set(CMAKE_BUILD_TYPE Release)
# find_package(OpenCV REQUIRED )
find_package(OpenCV 3.2.0 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)

set(OPENCV_ENABLE_NONFREE)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


include_directories(
include  
"/usr/include/eigen3"
  ${OpenCV_INCLUDE_DIRS}
  # ${Pangolin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
)


message(STATUS "Compile With map save/load function")
find_library(BOOST_SERIALIZATION boost_serialization)
if (NOT BOOST_SERIALIZATION)
    message(FATAL_ERROR "Can't find libboost_serialization")
endif()

add_library(Lib SHARED
src/utils.cpp

src/DataBase.cpp
src/ORBextractor.cc
src/map_viewer.cpp
src/Converter.cpp
)

target_link_libraries(Lib

${OpenCV_LIBS}
${BOOST_SERIALIZATION}
  # ${Pangolin_LIBRARIES}
  # ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
  #   Ceres::ceres
  #   glog
  GL
  GLU 
  glut    
  )

# Euroc ... Undistortion Img
add_executable(Undistortion undistortion_euroc.cpp)
target_link_libraries(Undistortion Lib)


# Euroc ... CamGtPose
add_executable(Transform Transform.cpp)
target_link_libraries(Transform Lib)

# Euroc ... make 3d map DB
add_executable(main_3dmap main_3dmap.cpp)
target_link_libraries(main_3dmap Lib)


add_executable(BA BA.cpp)
target_link_libraries(BA Lib Ceres::ceres)








