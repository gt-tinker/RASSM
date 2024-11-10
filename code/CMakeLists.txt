cmake_minimum_required(VERSION 3.10)

# residue based sparse gemm
project(respgemm)

IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
    message(STATUS "Building DEBUG mode")
else()
    set(CMAKE_BUILD_TYPE "Release")
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(SOURCE
    "${CMAKE_SOURCE_DIR}/src/main.cpp"
    "${CMAKE_SOURCE_DIR}/src/util.cpp"
    "${CMAKE_SOURCE_DIR}/src/common.cpp"
    "${CMAKE_SOURCE_DIR}/src/global.cpp"
)

find_package(Boost COMPONENTS program_options iostreams serialization)

set(INCLUDE
    "${CMAKE_SOURCE_DIR}/include/Reader.h"
    "${CMAKE_SOURCE_DIR}/include/Bitset.h"
    "${CMAKE_SOURCE_DIR}/include/experiments.h"

    "${CMAKE_SOURCE_DIR}/include/matrices/CSR.h"
    "${CMAKE_SOURCE_DIR}/include/matrices/CSC.h"
    "${CMAKE_SOURCE_DIR}/include/matrices/STM.h"
    "${CMAKE_SOURCE_DIR}/include/matrices/Matrix.h"

    "${CMAKE_SOURCE_DIR}/include/spmm/jstream.h"
    "${CMAKE_SOURCE_DIR}/include/spmm/kstream.h"
    "${CMAKE_SOURCE_DIR}/include/spmm/simple.h"
    "${CMAKE_SOURCE_DIR}/include/spmm/hybrid.h"

    "${CMAKE_SOURCE_DIR}/include/sddmm/simple.h"

    "${CMAKE_SOURCE_DIR}/include/utils/util.h"
    "${CMAKE_SOURCE_DIR}/include/utils/Statistics.h"
)

message(STATUS "${PAPI_INCLUDE_DIRS}")

if(Boost_FOUND)
    add_executable(resgemm ${SOURCE})
    target_include_directories(resgemm PUBLIC "${CMAKE_SOURCE_DIR}/include;${PAPI_INCLUDE_DIRS};${Boost_INCLUDE_DIRS}")

    if(APPLE) # Apple machines
        target_link_directories(resgemm PUBLIC "/opt/homebrew/opt/libomp/lib")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wno-write-strings -std=c++17 -g -O3 -fopenmp")

    # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wno-write-strings -std=c++17 -g -O3 -fopenmp -fsanitize=address -fno-omit-frame-pointer")
    else() # Intel Machines
        find_package(OpenMP REQUIRED)
        # find_package(PAPI REQUIRED)
        include_directories(${PAPI_INCLUDE_DIRS})

        # target_compile_definitions(resgemm PRIVATE LINUX=1)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            message(STATUS "Using GCC")
            target_compile_definitions(resgemm PRIVATE GCC_COMPILER=1)

            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wno-write-strings -std=c++17 -g -O3 -fopenmp")

        # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wno-write-strings -fsanitize=address -std=c++17 -g -O2")

        # INTEL COMPILER -- REQUIRED FOR THESE EXPERIMENTS
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
            message(STATUS "Using Intel")
            target_compile_definitions(resgemm PRIVATE LINUX=1 INTEL_COMPILER=1)

            # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wno-write-strings -std=c++17 -g -O0 -qopenmp -march=core-avx2 -restrict -qmkl")

            # Trial Flag -- ALL EXPERIMENTS
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wno-write-strings -std=c++17 -O3 -parallel -dynamic -qopenmp -Ofast -march=core-avx2 -restrict -simd -vec -qmkl -qopt-prefetch=5 -qopt-prefetch-distance=0,4 -qopt-report=0")


            target_link_libraries(resgemm ${PAPI_LIBRARIES})
            target_link_libraries(resgemm OpenMP::OpenMP_CXX)
        else()
            # Not using GNU or Intel
            message(STATUS "Compiler Type: ${CMAKE_CXX_COMPILER_ID}")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wno-write-strings -std=c++17 -O3 -fopenmp -Ofast")

            target_compile_definitions(resgemm PRIVATE LINUX=1)
        endif()

        target_link_libraries(resgemm OpenMP::OpenMP_CXX)

    endif() # APPLE

    target_link_libraries(resgemm ${Boost_LIBRARIES})

endif() # Boost_FOUND
