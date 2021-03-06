cmake_minimum_required(VERSION 3.5.0)
project(VPStest)
set(CMAKE_BUILD_TYPE Release)
# find_package(OpenCV REQUIRED )
find_package(OpenCV 3.2.0 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)


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
src/VPStestResult.cpp
src/VPStest.cpp
src/Converter.cpp
 )

target_link_libraries(Lib
  
  ${OpenCV_LIBS}
  # ${Pangolin_LIBRARIES}
  ${BOOST_SERIALIZATION}
  ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    Ceres::ceres
  #   glog
  GL
  GLU
  glut    
  )





# GT pose for Evaluation
# add_executable(GT_trajectory GT_trajectory.cpp)
# target_link_libraries(GT_trajectory Lib)

# Euroc VPStest main
add_executable(VPStest_euroc main_euroc.cpp )
target_link_libraries(VPStest_euroc Lib )

# add_executable(VPStest_euroc_Compressed main_euroc_compressed.cpp )
# target_link_libraries(VPStest_euroc_Compressed Lib)


# Evaluation
# add_executable(LoadResult LoadResult.cpp)
# target_link_libraries(LoadResult Lib)


# add_executable(Evaluation_euroc Evaluation_euroc.cpp)
# target_link_libraries(Evaluation_euroc Lib)

# for debug
# add_executable(Query_one_Image Query_one_Image.cpp )
# target_link_libraries(Query_one_Image Lib)








