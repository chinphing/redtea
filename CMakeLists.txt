cmake_minimum_required(VERSION 3.5)

#project redtea

PROJECT(readtea)

SET(CMAKE_C_COMPILER g++)
add_compile_options(-std=c++11)
  
#head file path  
INCLUDE_DIRECTORIES(  
    ../eigen/
)  

#source directory  
AUX_SOURCE_DIRECTORY(./ DIR_SRCS)  

#add executable file  
ADD_EXECUTABLE(test-lsl test-lsl.cpp)  
ADD_EXECUTABLE(test-log test-log.cpp)

