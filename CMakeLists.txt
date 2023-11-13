 cmake_minimum_required(VERSION 3.10)

 project(cmeans C)

 set(CMAKE_C_STANDARD 17)

 find_package(GSL REQUIRED)

 include_directories(${GSL_INCLUDE_DIRS})
 link_directories(${GSL_LIBRARY_DIRS})

 set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GSL_CFLAGS} -O3 -ffast-math -Ofast -funroll-loops -fprefetch-loop-arrays")

 if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
 set(CMAKE_C_COMPILER gcc)
 endif()

 if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
 set(CMAKE_C_COMPILER clang)
 endif()

 if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
 set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -arch arm64")
 endif()

 add_executable(exe python/cmeans/cmeans.c)
 target_link_libraries(exe ${GSL_LIBRARIES})
