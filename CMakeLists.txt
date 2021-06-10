
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(ir_photo_calib)

set (CMAKE_CXX_FLAGS "-march=native -msse4.2 -mfpmath=sse")
set (CMAKE_BUILD_TYPE "Debug")
set (CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package( Boost COMPONENTS system filesystem REQUIRED )

#set(CMAKE_FRAMEWORK_PATH /home/manash/packages/eigen-3.2.9/build/install/share/eigen3/cmake)
find_package(Ceres REQUIRED)

OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
ENDIF()

if(OPENMP_FOUND)
  MESSAGE("OpenMP found")
  if(UNIX)
    SET(CMAKE_C_FLAGS_RELEASE "-O3  -Wall -Wextra -Wunused-variable  -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra -Wunused-variable -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
  endif(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  else(OPENMP_FOUND)
    MESSAGE("OpenMP not found")
    if(UNIX)
      SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -Wall -std=c++0x -Wunused-variable -Wno-unknown-pragmas -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
      SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall -std=c++0x -Wno-unknown-pragmas -Wunused-variable -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
    endif(UNIX)
endif(OPENMP_FOUND)

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${CERES_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIR}
                    include)

add_executable (main main.cpp irPhotoCalib.cpp utils.h)
target_link_libraries (main ${OpenCV_LIBRARIES}
                                ${CERES_LIBRARIES}
                                ${OPENMP_LIBRARIES})

# add_executable (demo demo.cpp irPhotoCalib.cpp)
# target_link_libraries (demo ${OpenCV_LIBRARIES}
#                                 ${CERES_LIBRARIES}
#                                 ${OPENMP_LIBRARIES})
                                
# add_executable (demo_loop demo_loop.cpp irPhotoCalib.cpp)
# target_link_libraries (demo_loop ${OpenCV_LIBRARIES}
#                                 ${CERES_LIBRARIES}
#                                 ${OPENMP_LIBRARIES}
#                                 ${Boost_LIBRARIES})

add_executable (gray2jet gray2jet.cpp)
target_link_libraries (gray2jet ${OpenCV_LIBRARIES})
