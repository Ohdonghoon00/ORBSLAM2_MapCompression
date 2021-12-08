cmake_minimum_required(VERSION 3.5.0)
project(VPStest)
set(CMAKE_BUILD_TYPE Release)
find_package(OpenCV REQUIRED )
# find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)


LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


include_directories(
  "/usr/include/eigen3"
  ${OpenCV_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
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
ORBextractor.cc )

target_link_libraries(Lib
  
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
  ${BOOST_SERIALIZATION}
  ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
  #   Ceres::ceres
  #   glog
      
  )

  add_executable(VPStest
    main.cpp

)
target_link_libraries(VPStest Lib)

add_executable(Evaluation
Evaluation.cpp

)

target_link_libraries(Evaluation Lib)

add_executable(dbow2test
dbow2test.cpp

)

target_link_libraries(dbow2test Lib)











