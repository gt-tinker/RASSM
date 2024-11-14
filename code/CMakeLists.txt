cmake_minimum_required(VERSION 3.10)

# residue based sparse gemm
project(rassm)

IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
    message(STATUS "Building DEBUG mode")
else()
    set(CMAKE_BUILD_TYPE "Release")
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Boost COMPONENTS program_options iostreams serialization)

# set SOURCE using the src directory
file(GLOB_RECURSE SOURCE "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE ASPT_SOURCE "${CMAKE_SOURCE_DIR}/aspt/*.cc")

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/jstream")

# add jstream/model.cc to the source
list(APPEND SOURCE "${CMAKE_SOURCE_DIR}/jstream/model.cc")

# boost is required for this project to run
if(Boost_FOUND)
    add_executable(rassm ${SOURCE})
    add_executable(aspt ${ASPT_SOURCE})
    target_include_directories(rassm PUBLIC "${CMAKE_SOURCE_DIR}/include;${PAPI_INCLUDE_DIRS};${Boost_INCLUDE_DIRS}")
    target_include_directories(aspt PUBLIC "${CMAKE_SOURCE_DIR}/include;${PAPI_INCLUDE_DIRS};${Boost_INCLUDE_DIRS}")

    if(APPLE) # Apple machines
        target_link_directories(rassm PUBLIC "/opt/homebrew/opt/libomp/lib")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wno-write-strings -std=c++17 -g -O3 -fopenmp")

    else() # Intel Machines
        find_package(OpenMP REQUIRED)
        # find_package(PAPI REQUIRED)
        include_directories(${PAPI_INCLUDE_DIRS})

        # target_compile_definitions(rassm PRIVATE LINUX=1)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            message(STATUS "Using GCC")
            target_compile_definitions(rassm PRIVATE GCC_COMPILER=1)

            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wextra -Wall -Wno-write-strings -std=c++17 -g -O3 -fopenmp")

        # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wno-write-strings -fsanitize=address -std=c++17 -g -O2")

        # INTEL COMPILER -- REQUIRED FOR THESE EXPERIMENTS
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
            message(STATUS "Using Intel")
            target_compile_definitions(rassm PRIVATE LINUX=1 INTEL_COMPILER=1)

            # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wno-write-strings -std=c++17 -g -O0 -qopenmp -march=core-avx2 -restrict -qmkl")

            # Trial Flag -- ALL EXPERIMENTS
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wno-write-strings -std=c++17 -O3 -parallel -dynamic -qopenmp -Ofast -march=core-avx2 -restrict -simd -vec -qmkl -qopt-prefetch=5 -qopt-prefetch-distance=0,4 -qopt-report=0")


            target_link_libraries(rassm ${PAPI_LIBRARIES})
            target_link_libraries(rassm OpenMP::OpenMP_CXX)
        else()
            # Not using GNU or Intel
            message(STATUS "Compiler Type: ${CMAKE_CXX_COMPILER_ID}")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wno-write-strings -std=c++17 -O3 -fopenmp -Ofast -march=core-avx2")

            target_compile_definitions(rassm PRIVATE LINUX=1)
        endif()

        target_link_libraries(rassm OpenMP::OpenMP_CXX)

    endif() # APPLE

    target_link_libraries(rassm ${Boost_LIBRARIES})

endif() # Boost_FOUND
