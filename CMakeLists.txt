cmake_minimum_required(VERSION 2.8)

project(AnnotionTool)
#find_package( OpenCV REQUIRED )

###################################################################################################
# - variables -------------------------------------------------------------------------------------
set(CMAKE_VERBOSE_MAKEFILE ON)

set(INCLUDE_DIR_OPENCV "" CACHE PATH "Set OPENCV include path")
set(LIBRARY_DIR_OPENCV "" CACHE PATH "Set OPENCV library path")

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
ADD_DEFINITIONS(-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/${CMAKE_BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/${CMAKE_BUILD_TYPE})

###################################################################################################
# - Windows crt library option --------------------------------------------------------------------

foreach(flag_var
    CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
endforeach(flag_var)

foreach(flag_var
    CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE
    CMAKE_CUDA_FLAGS_MINSIZEREL CMAKE_CUDA_FLAGS_RELWITHDEBINFO)
    string(REGEX REPLACE "-MD" "-MT" ${flag_var} "${${flag_var}}")
endforeach(flag_var)

add_definitions(
    -DOPENCV
)

###################################################################################################
# - Windows prebuilt paths setting ----------------------------------------------------------------
include_directories(
    ${INCLUDE_DIR_OPENCV}
)
link_directories(
    ${LIBRARY_DIR_OPENCV}
)

###################################################################################################
# - Windows excutable ----------------------------------------------------------------
add_executable( annotation
    annotation/annotationtool.cpp
)
add_executable( verification
    verification/verificationtool.cpp
)

target_link_libraries( annotation opencv_world430  ) # ${OpenCV_LIBS} opencv_highgui -L/usr/lib64 -ldl
target_compile_options( annotation PUBLIC -std=c++11 -fpermissive -w -Wall )

target_link_libraries( verification opencv_world430  ) # ${OpenCV_LIBS} opencv_highgui -L/usr/lib64 -ldl
target_compile_options( verification PUBLIC -std=c++11 -fpermissive -w -Wall )
