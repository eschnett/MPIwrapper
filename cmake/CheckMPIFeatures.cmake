include(CheckCXXSourceCompiles)

function(CheckMPIFeatures)
  if (NOT DEFINED HAVE_MPI_EXT OR NOT DEFINED HAVE_MPIX_QUERY_CUDA_SUPPORT)
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

    if(HAVE_MPI_EXT)
      list(APPEND CMAKE_REQUIRED_DEFINITIONS -DMPIWRAPPER_HAVE_MPI_EXT=1)
    endif()

    check_cxx_source_compiles(
      "
        #include <mpi.h>
        #ifdef MPIWRAPPER_HAVE_MPI_EXT
        #include <mpi-ext.h>
        #endif
        int main() {
          int result = MPIX_Query_cuda_support();
          return 0;
        }
      "
      HAVE_MPIX_QUERY_CUDA_SUPPORT)

    check_cxx_source_compiles(
      "
        #include <mpi.h>
        #ifdef MPIWRAPPER_HAVE_MPI_EXT
        #include <mpi-ext.h>
        #endif
        int main() {
          int result = MPIX_Query_hip_support();
          return 0;
        }
      "
      HAVE_MPIX_QUERY_HIP_SUPPORT)

    check_cxx_source_compiles(
      "
        #include <mpi.h>
        #ifdef MPIWRAPPER_HAVE_MPI_EXT
        #include <mpi-ext.h>
        #endif
        int main() {
          int result = MPIX_Query_rocm_support();
          return 0;
        }
      "
      HAVE_MPIX_QUERY_ROCM_SUPPORT)

    check_cxx_source_compiles(
      "
        #include <mpi.h>
        #ifdef MPIWRAPPER_HAVE_MPI_EXT
        #include <mpi-ext.h>
        #endif
        int main() {
          int result = MPIX_Query_ze_support();
          return 0;
        }
      "
      HAVE_MPIX_QUERY_ZE_SUPPORT)

    list(REMOVE_ITEM CMAKE_REQUIRED_DEFINITIONS -DMPIWRAPPER_HAVE_MPI_EXT=1)
  endif()
endfunction()

CheckMPIFeatures()
