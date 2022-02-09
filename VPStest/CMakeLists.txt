cmake_minimum_required(VERSION 3.5.0)
project(VPStest)
set(CMAKE_BUILD_TYPE Release)
# find_package(OpenCV REQUIRED )
find_package(OpenCV 3.2.0 REQUIRED)
# find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)


LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


include_directories(
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
VPStest.cpp
DataBase.cpp
ORBextractor.cc
map_viewer.cpp
VPStestResult.cpp )

target_link_libraries(Lib
  
  ${OpenCV_LIBS}
  # ${Pangolin_LIBRARIES}
  ${BOOST_SERIALIZATION}
  ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
  #   Ceres::ceres
  #   glog
  GL
  GLU
  glut    
  )




# Kitti
add_executable(VPStest_kitti
  main_kitti.cpp

)
target_link_libraries(VPStest_kitti Lib)

# Kitti_query_one_image
add_executable(main_kitti_query_one_image
main_kitti_query_one_image.cpp

)
target_link_libraries(main_kitti_query_one_image Lib)

# Euroc
add_executable(VPStest_euroc
  main_euroc.cpp

)
target_link_libraries(VPStest_euroc Lib)

# Euroc Undistortion Img
add_executable(Undistortion
  undistortion_euroc.cpp

)
target_link_libraries(Undistortion Lib)

# Evaluation
add_executable(Evaluation
Evaluation.cpp
)
target_link_libraries(Evaluation Lib)

add_executable(LoadResult
LoadResult.cpp

)

target_link_libraries(LoadResult Lib)

add_executable(Evaluation_euroc
Evaluation_euroc.cpp

)

target_link_libraries(Evaluation_euroc Lib)









