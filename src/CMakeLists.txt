cmake_minimum_required (VERSION 3.13)
project (TF_KNN LANGUAGES CUDA CXX)

option(BUILD_GPU "Enable building the tensorflow operator"
       ON)
option(BUILD_TEST "Enable building the tensorflow operator"
       ON)
option(BUILD_TF_OP "Enable building the tensorflow operator"
       ON)

# The version number.
set (TF_KNN_VERSION_MAJOR 0)
set (TF_KNN_VERSION_MINOR 1)
set (CMAKE_CXX_STANDARD 17)

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

#Search for python
find_package(Python 3.1 COMPONENTS Interpreter Development REQUIRED)

#TODO: Windows...
#list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

set(COMMON_SOURCE_LIST
    "kdtree.cpp"
    "tf_nndistance_g.cu"
    "tf_kdtree_g.cu")

set(COMMON_HEADERS_LIST
    "cutils.cuh"
    "kdtree.hpp"
    "nndistance.hpp")

set(EXEC_SOURCE_LIST
	"${COMMON_SOURCE_LIST}"
    "test_kdtree.cpp")

set(EXEC_HEADERS_LIST
    "${COMMON_HEADERS_LIST}")
    
set(INTERFACE_SOURCE_LIST
    "${COMMON_SOURCE_LIST}"
    "tf_nndistance_g.cu"
    "tf_nndistance.cpp"
    "tf_kdtree.cpp")


include(ExternalProject)
set(EIGEN_INSTALL_DIR "${PROJECT_SOURCE_DIR}/eigenlib")
ExternalProject_Add( eigen
  URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
  SOURCE_DIR "${EIGEN_INSTALL_DIR}"
  INSTALL_COMMAND ""
  BUILD_COMMAND ""
  CONFIGURE_COMMAND ""
  )

#Old, see below
#INCLUDE_DIRECTORIES("${EIGEN_INSTALL_DIR}")


##############################################################
## Tensorflow includes and libraries
execute_process(COMMAND "python" "-c" "import tensorflow as tf; import sys; ret = sys.stdout.write(tf.sysconfig.get_lib() + '/')" 
                OUTPUT_VARIABLE TF_LIB_DIR 
                RESULT_VARIABLE TF_LIB_RET_VAL)
execute_process(COMMAND "python" "-c" "import tensorflow as tf; import sys; ret = sys.stdout.write(tf.sysconfig.get_include() + '/')" 
                OUTPUT_VARIABLE TF_INCLUDE_DIR
                RESULT_VARIABLE TF_INCLUDE_RET_VAL) 
#TODO: Error checking not properly working
#TODO: Calling python_executable not working

if(NOT TF_LIB_RET_VAL EQUAL 0 OR NOT TF_INCLUDE_RET_VAL EQUAL 0)
    message( FATAL_ERROR "Error: Importing tensorflow in the active python environment failed")
endif()

#https://stackoverflow.com/questions/28597351/how-do-i-add-a-library-path-in-cmake
link_directories(${TF_LIB_DIR})


##############################################################
## Create targets and add definitions, dependencies...

if (BUILD_TEST)
    # add the executable
    add_executable(test_kdtree ${EXEC_SOURCE_LIST}) # ${EXEC_HEADERS_LIST})
endif()

#PYTHON_ADD_MODULE(oum_export_release tsitsiklis.cpp OrderedUpwindMethod.cpp utility.cpp OUMNPInterface.cpp)

#set_target_properties(test_kdtree PROPERTIES
#                            CUDA_SEPARABLE_COMPILATION ON)

add_library(tf_nndistance SHARED ${INTERFACE_SOURCE_LIST})

target_compile_definitions(tf_nndistance PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(test_kdtree cudart)
target_link_libraries(tf_nndistance tensorflow_framework cudart)

target_include_directories(test_kdtree PRIVATE ${EIGEN_INSTALL_DIR})
target_include_directories(tf_nndistance PRIVATE ${EIGEN_INSTALL_DIR} ${TF_INCLUDE_DIR})

#-fPIC
#set_property(tf_nndistance PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(tf_nndistance PUBLIC $<$<CONFIG:RELEASE>:NDEBUG>)
target_compile_options(test_kdtree PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(tf_nndistance PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

##################################
# Post-build commands
add_custom_command(TARGET tf_nndistance 
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tf_nndistance> "${PROJECT_SOURCE_DIR}/..")

##################################
# Tests
enable_testing()
add_test (NAME python-tests
  COMMAND "python" #-m pytest 
  ${PROJECT_SOURCE_DIR}/../scripts/test_knn_unit.py
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/../"
)

#Dependencies
add_dependencies(tf_nndistance eigen)
add_dependencies(test_kdtree eigen)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS tf_nndistance)


##################################
# Debug: Print all variables
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()