cmake_minimum_required(VERSION 3.12...3.20)
project(
  MPIwrapper
  VERSION 2.9.0
  DESCRIPTION "MPI wrapper"
  LANGUAGES C CXX Fortran
  )

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 11)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

find_package(MPI REQUIRED)
message(STATUS "Using MPIEXEC_EXECUTABLE: ${MPIEXEC_EXECUTABLE}")
if(${MPIEXEC_EXECUTABLE} STREQUAL "MPIEXEC_EXECUTABLE-NOTFOUND")
  message(FATAL_ERROR "MPIEXEC_EXECUTABLE not set. Set the cmake variable MPIEXEC_EXECUTABLE to point to mpiexec.")
endif()
include(CheckMPIFeatures)



configure_file(src/mpiwrapper_version.h.in src/mpiwrapper_version.h @ONLY)

include(CheckTypeSize)
check_type_size(ptrdiff_t MPITRAMPOLINE_SIZEOF_CPTRDIFF_T LANGUAGE C)
if(${MPITRAMPOLINE_SIZEOF_CPTRDIFF_T} STREQUAL "" OR
    ${MPITRAMPOLINE_SIZEOF_CPTRDIFF_T} STREQUAL "0")
  message(FATAL_ERROR "Cannot determine sizeof(ptrdiff_t).")
endif()

configure_file(mpiabi/mpiabif.h.in mpiabi/mpiabif.h @ONLY)

add_custom_command(
  OUTPUT
  src/mpiabi_decl_constants_c.h
  src/mpiabi_decl_functions_c.h
  src/mpiabi_decl_constants_fortran.h
  src/mpiabi_decl_functions_fortran.h
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/gen/gen_decl.py
  DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/gen/gen_decl.py
  ${CMAKE_CURRENT_SOURCE_DIR}/mpiabi/mpi_constants.py
  ${CMAKE_CURRENT_SOURCE_DIR}/mpiabi/mpi_functions.py
  ${CMAKE_CURRENT_SOURCE_DIR}/mpiabi/mpi_constants_fortran.py
  ${CMAKE_CURRENT_SOURCE_DIR}/mpiabi/mpi_functions_fortran.py
  )

add_custom_command(
  OUTPUT
  src/mpiabi_defn_constants_c.h
  src/mpiabi_defn_functions_c.h
  src/mpiabi_defn_constants_fortran.h
  src/mpiabi_defn_functions_fortran.h
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/gen/gen_defn.py
  DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/gen/gen_defn.py
  ${CMAKE_CURRENT_SOURCE_DIR}/mpiabi/mpi_constants.py
  ${CMAKE_CURRENT_SOURCE_DIR}/mpiabi/mpi_functions.py
  ${CMAKE_CURRENT_SOURCE_DIR}/mpiabi/mpi_constants_fortran.py
  ${CMAKE_CURRENT_SOURCE_DIR}/mpiabi/mpi_functions_fortran.py
  )

add_library(mpiwrapper MODULE
  # *** C ***
  # source file
  src/mpiwrapper.cxx
  # header files
  src/mpiwrapper.hxx
  mpiabi/mpiabi.h
  # configured header files
  src/mpiwrapper_version.h
  # generated source files
  src/mpiabi_decl_constants_c.h
  src/mpiabi_decl_functions_c.h
  src/mpiabi_decl_constants_fortran.h
  src/mpiabi_decl_functions_fortran.h
  src/mpiabi_defn_constants_c.h
  src/mpiabi_defn_functions_c.h
  src/mpiabi_defn_functions_fortran.h
  # *** Fortran ***
  # source file
  src/mpiwrapper.f
  # header files
  mpiabi/mpiabif.h
  # generated source files
  src/mpiabi_defn_constants_fortran.h
  )
# set_target_properties(mpiwrapper PROPERTIES
#   SOVERSION 2.0.0
# )
target_include_directories(
  mpiwrapper
  PRIVATE mpiabi ${CMAKE_CURRENT_BINARY_DIR}/mpiabi ${CMAKE_CURRENT_BINARY_DIR}/src)
target_link_libraries(mpiwrapper PRIVATE MPI::MPI_CXX MPI::MPI_Fortran)

if(APPLE)
  # On macOS, check that the plugin `libmpiwrapper.so` is built with a
  # two-level namespace. If not, it will use the MPI functions
  # provided by MPItrampoline instead of the "real" MPI, which will
  # recurse infinitely, leading to a stack overflow and segfault.
  add_custom_command(
    TARGET mpiwrapper POST_BUILD
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/check_twolevel.sh ${CMAKE_CURRENT_BINARY_DIR}/libmpiwrapper.so
    COMMENT "Checking whether libmpiwrapper.so plugin uses a two-level namespace..."
    VERBATIM
    )
endif()

install(
  TARGETS mpiwrapper
  LIBRARY
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

configure_file(bin/mpiwrapperexec.in bin/mpiwrapperexec @ONLY)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/bin/mpiwrapperexec"
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )
