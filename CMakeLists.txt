cmake_minimum_required(VERSION 3.12...3.20)
project(
  MPIwrapper
  VERSION 2.5.0
  DESCRIPTION "MPI wrapper"
  LANGUAGES CXX Fortran
  )

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 11)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

find_package(MPI REQUIRED)
message(STATUS "Using MPIEXEC_EXECUTABLE: ${MPIEXEC_EXECUTABLE}")
if (${MPIEXEC_EXECUTABLE} STREQUAL "MPIEXEC_EXECUTABLE-NOTFOUND")
  message(FATAL_ERROR "MPIEXEC_EXECUTABLE not set. Set the cmake variable MPIEXEC_EXECUTABLE to point to mpiexec.")
endif()
include(CheckMPIFeatures)

configure_file(mpiwrapper_version.h.in mpiwrapper_version.h @ONLY)

add_custom_command(
  OUTPUT mpiabi_declarations.h
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_mpiabi_declarations.py >mpiabi_declarations.h
  DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/generate_mpiabi_declarations.py
  ${CMAKE_CURRENT_SOURCE_DIR}/mpi_constants.py
  ${CMAKE_CURRENT_SOURCE_DIR}/mpi_functions.py
  ${CMAKE_CURRENT_SOURCE_DIR}/mpi_constants_fortran.py
  ${CMAKE_CURRENT_SOURCE_DIR}/mpi_functions_fortran.py
  )

add_custom_command(
  OUTPUT mpiwrapper_definitions.h
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_mpiwrapper_definitions.py >mpiwrapper_definitions.h
  DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/generate_mpiwrapper_definitions.py
  ${CMAKE_CURRENT_SOURCE_DIR}/mpi_constants.py
  ${CMAKE_CURRENT_SOURCE_DIR}/mpi_functions.py
  ${CMAKE_CURRENT_SOURCE_DIR}/mpi_functions_fortran.py
  )

add_custom_command(
  OUTPUT mpiwrapper_definitions_fortran.h
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_mpiwrapper_definitions_fortran.py >mpiwrapper_definitions_fortran.h
  DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/generate_mpiwrapper_definitions_fortran.py
  ${CMAKE_CURRENT_SOURCE_DIR}/mpi_constants_fortran.py
  )

add_library(mpiwrapper MODULE
  mpiabi.h mpiabi_declarations.h
  mpiwrapper_version.h
  mpiwrapper.cxx mpiwrapper.hxx mpiwrapper_definitions.h
  mpiwrapper.f mpiwrapper_definitions_fortran.h)
# set_target_properties(mpiwrapper PROPERTIES
#   SOVERSION 2.0.0
# )
target_include_directories(mpiwrapper PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
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

configure_file(mpiwrapperexec.in mpiwrapperexec @ONLY)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/mpiwrapperexec"
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )
