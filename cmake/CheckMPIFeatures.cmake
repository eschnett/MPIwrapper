include(CheckCXXSourceCompiles)
function(CheckMPIFeatures)
  if (NOT DEFINED HAVE_MPI_EXT OR NOT DEFINED MPI_HAS_QUERY_CUDA_SUPPORT)
    list(JOIN MPI_COMPILE_FLAGS " " CMAKE_REQUIRED_FLAGS)
    set(CMAKE_REQUIRED_INCLUDES ${MPI_INCLUDE_PATH})
    set(CMAKE_REQUIRED_LIBRARIES ${MPI_LIBRARIES})

    # We cannot use check_include_file here as <mpi.h> needs to be
    # included before <mpi-ext.h>, and check_include_file doesn't
    # support this.
    check_cxx_source_compiles(
      "
        #include <mpi.h>
        #include <mpi-ext.h>
        int main() {
          return 0;
        }
      "
      HAVE_MPI_EXT)
    if(${HAVE_MPI_EXT})
      set(HAVE_MPI_EXT 1)
    else()
      set(HAVE_MPI_EXT 0)
    endif()
    message(STATUS "HAVE_MPI_EXT=${HAVE_MPI_EXT}")

    list(APPEND CMAKE_REQUIRED_DEFINITIONS -DHAVE_MPI_EXT=${HAVE_MPI_EXT})

    check_cxx_source_compiles(
      "
        #include <mpi.h>
        #if HAVE_MPI_EXT
        #include <mpi-ext.h>
        #endif
        int main() {
          int result = MPIX_Query_cuda_support();
          return 0;
        }
      "
      HAVE_MPIX_QUERY_CUDA_SUPPORT)
    if(${HAVE_MPIX_QUERY_CUDA_SUPPORT})
      set(HAVE_MPIX_QUERY_CUDA_SUPPORT 1)
    else()
      set(HAVE_MPIX_QUERY_CUDA_SUPPORT 0)
    endif()

    check_cxx_source_compiles(
      "
        #include <mpi.h>
        #if HAVE_MPI_EXT
        #include <mpi-ext.h>
        #endif
        int main() {
          int result = MPIX_Query_hip_support();
          return 0;
        }
      "
      HAVE_MPIX_QUERY_HIP_SUPPORT)
    if(${HAVE_MPIX_QUERY_HIP_SUPPORT})
      set(HAVE_MPIX_QUERY_HIP_SUPPORT 1)
    else()
      set(HAVE_MPIX_QUERY_HIP_SUPPORT 0)
    endif()

    check_cxx_source_compiles(
      "
        #include <mpi.h>
        #if HAVE_MPI_EXT
        #include <mpi-ext.h>
        #endif
        int main() {
          int result = MPIX_Query_ze_support();
          return 0;
        }
      "
      HAVE_MPIX_QUERY_ZE_SUPPORT)
    if(${HAVE_MPIX_QUERY_ZE_SUPPORT})
      set(HAVE_MPIX_QUERY_ZE_SUPPORT 1)
    else()
      set(HAVE_MPIX_QUERY_ZE_SUPPORT 0)
    endif()

    list(REMOVE_ITEM CMAKE_REQUIRED_DEFINITIONS -DHAVE_MPI_EXT=${HAVE_MPI_EXT})
  endif()
endfunction()

CheckMPIFeatures()
