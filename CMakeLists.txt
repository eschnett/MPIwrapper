#[[
# rm -rf build-openmpi $HOME/src/c/MPIstuff/mpiwrapper-openmpi
# cmake -S . -B build-openmpi -G Ninja -DMPIEXEC_EXECUTABLE=$(which mpiexec-openmpi-gcc11) -DCMAKE_CXX_COMPILER=mpicxx-openmpi-gcc11 -DCMAKE_Fortran_COMPILER=mpifort-openmpi-gcc11 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$HOME/src/c/MPIstuff/mpiwrapper-openmpi
# cmake --build build-openmpi --verbose && cmake --install build-openmpi --verbose

rm -rf build-openmpi $HOME/src/c/MPIstuff/mpiwrapper-openmpi
# Avoid the flag `-flat_namespace` in OPENMPI's mpicxx
# cmake -S . -B build-openmpi -G Ninja -DMPIEXEC_EXECUTABLE=$(which mpiexec-openmpi-gcc10) -DCMAKE_CXX_COMPILER=mpicxx-openmpi-gcc10 -DCMAKE_Fortran_COMPILER=mpifort-openmpi-gcc10 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$HOME/src/c/MPIstuff/mpiwrapper-openmpi
cmake \
    -S . \
    -B build-openmpi \
    -G Ninja \
    -DCMAKE_C_COMPILER=gcc-mp-11 \
    -DCMAKE_CXX_COMPILER=g++-mp-11 \
    -DCMAKE_Fortran_COMPILER=gfortran-mp-11 \
    -DMPI_C_ADDITIONAL_INCLUDE_DIRS=/opt/local/include/openmpi-gcc11 \
    -DMPI_C_LIB_NAMES='mpi' \
    -DMPI_CXX_ADDITIONAL_INCLUDE_DIRS=/opt/local/include/openmpi-gcc11 \
    -DMPI_CXX_LIB_NAMES='mpi_cxx;mpi' \
    -DMPI_Fortran_ADDITIONAL_INCLUDE_DIRS='/opt/local/include/openmpi-gcc11;/opt/local/lib/openmpi-gcc11' \
    -DMPI_Fortran_LIB_NAMES='mpi_usempif08;mpi_usempi_ignore_tkr;mpi_mpifh;mpi' \
    -DMPI_mpi_LIBRARY=/opt/local/lib/openmpi-gcc11/libmpi.dylib \
    -DMPI_mpi_cxx_LIBRARY=/opt/local/lib/openmpi-gcc11/libmpi_cxx.dylib \
    -DMPI_mpi_mpifh_LIBRARY=/opt/local/lib/openmpi-gcc11/libmpi_mpifh.dylib \
    -DMPI_mpi_usempi_ignore_tkr_LIBRARY=/opt/local/lib/openmpi-gcc11/libmpi_usempi_ignore_tkr.dylib \
    -DMPI_mpi_usempif08_LIBRARY=/opt/local/lib/openmpi-gcc11/libmpi_usempif08.dylib \
    -DMPIEXEC_EXECUTABLE=/opt/local/bin/mpiexec-openmpi-gcc11 \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=$HOME/src/c/MPIstuff/mpiwrapper-openmpi
cmake --build build-openmpi --verbose && cmake --install build-openmpi --verbose

rm -rf build-mpich $HOME/src/c/MPIstuff/mpiwrapper-mpich
# Avoid the flag `-flat_namespace` in MPICH's mpicxx
# cmake -S . -B build-mpich -G Ninja -DMPIEXEC_EXECUTABLE=$(which mpiexec-mpich-gcc10) -DCMAKE_CXX_COMPILER=mpicxx-mpich-gcc10 -DCMAKE_Fortran_COMPILER=mpifort-mpich-gcc10 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$HOME/src/c/MPIstuff/mpiwrapper-mpich
cmake \
    -S . \
    -B build-mpich \
    -G Ninja \
    -DCMAKE_CXX_COMPILER=g++-mp-10 \
    -DCMAKE_Fortran_COMPILER=gfortran-mp-10 \
    -DMPI_CXX_ADDITIONAL_INCLUDE_DIRS=/opt/local/include/mpich-gcc10 \
    -DMPI_CXX_LIB_NAMES='mpicxx;mpi;pmpi' \
    -DMPI_Fortran_ADDITIONAL_INCLUDE_DIRS=/opt/local/include/mpich-gcc10 \
    -DMPI_Fortran_LIB_NAMES='mpifort;mpi;pmpi' \
    -DMPI_mpi_LIBRARY=/opt/local/lib/mpich-gcc10/libmpi.dylib \
    -DMPI_mpicxx_LIBRARY=/opt/local/lib/mpich-gcc10/libmpicxx.dylib \
    -DMPI_mpifort_LIBRARY=/opt/local/lib/mpich-gcc10/libmpifort.dylib \
    -DMPI_pmpi_LIBRARY=/opt/local/lib/mpich-gcc10/libpmpi.dylib \
    -DMPIEXEC_EXECUTABLE=/opt/local/bin/mpiexec-mpich-gcc10 \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=$HOME/src/c/MPIstuff/mpiwrapper-mpich
cmake --build build-mpich --verbose && cmake --install build-mpich --verbose

rm -rf build-mpich-twolevel $HOME/src/c/MPIstuff/mpiwrapper-mpich-twolevel
cmake -S . -B build-mpich-twolevel -G Ninja -DMPIEXEC_EXECUTABLE=$HOME/mpich-3.4.2/bin/mpiexec -DCMAKE_CXX_COMPILER=$HOME/mpich-3.4.2/bin/mpicxx -DCMAKE_Fortran_COMPILER=$HOME/mpich-3.4.2/bin/mpifort -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$HOME/src/c/MPIstuff/mpiwrapper-mpich-twolevel
cmake --build build-mpich-twolevel --verbose && cmake --install build-mpich-twolevel --verbose



module load cmake
rm -rf build-openmpi-ubuntu $HOME/src/c/MPIstuff/mpiwrapper-openmpi-ubuntu
cmake -S . -B build-openmpi-ubuntu -DCMAKE_CXX_COMPILER=/usr/bin/mpic++ -DCMAKE_Fortran_COMPILER=/usr/bin/mpifort -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$HOME/src/c/MPIstuff/mpiwrapper-openmpi-ubuntu
cmake --build build-openmpi-ubuntu && cmake --install build-openmpi-ubuntu

module load cmake
rm -rf build-openmpi $HOME/src/c/MPIstuff/mpiwrapper-openmpi
cmake -S . -B build-openmpi -DCMAKE_CXX_COMPILER=/cm/shared/apps/openmpi/gcc-9/64/4.1.0/bin/mpic++ -DCMAKE_Fortran_COMPILER=/cm/shared/apps/openmpi/gcc-9/64/4.1.0/bin/mpifort -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$HOME/src/c/MPIstuff/mpiwrapper-openmpi
cmake --build build-openmpi && cmake --install build-openmpi

module load cmake
rm -rf build-openmpi-spack $HOME/src/c/MPIstuff/mpiwrapper-openmpi-spack
cmake -S . -B build-openmpi-spack -DCMAKE_CXX_COMPILER=$HOME/src/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-11.2.0/openmpi-4.1.1-3b4drmye35bg6hok7gk462yvoj6d4oqq/bin/mpicxx -DCMAKE_Fortran_COMPILER=$HOME/src/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-11.2.0/openmpi-4.1.1-3b4drmye35bg6hok7gk462yvoj6d4oqq/bin/mpifort -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$HOME/src/c/MPIstuff/mpiwrapper-openmpi-spack
cmake --build build-openmpi-spack && cmake --install build-openmpi-spack

module load cmake
rm -rf build-mpich $HOME/src/c/MPIstuff/mpiwrapper-mpich
cmake -S . -B build-mpich -DCMAKE_CXX_COMPILER=/cm/shared/apps/mpich/gcc-9/3.3.2/bin/mpic++ -DCMAKE_Fortran_COMPILER=/cm/shared/apps/mpich/gcc-9/3.3.2/bin/mpifort -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$HOME/src/c/MPIstuff/mpiwrapper-mpich
cmake --build build-mpich && cmake --install build-mpich

module load cmake
module load mpi
rm -rf build-intel $HOME/src/c/MPIstuff/mpiwrapper-intel
cmake -S . -B build-intel -DCMAKE_CXX_COMPILER=/cm/shared/apps/intel/mpi/2021.1.1/bin/mpicxx -DCMAKE_Fortran_COMPILER=/cm/shared/apps/intel/mpi/2021.1.1/bin/mpifc -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$HOME/src/c/MPIstuff/mpiwrapper-intel
cmake --build build-intel && cmake --install build-intel
module unload mpi
]]



cmake_minimum_required(VERSION 3.12...3.20)
project(
  MPIwrapper
  VERSION 2.2.0
  DESCRIPTION "MPI wrapper"
  LANGUAGES CXX Fortran
  )

set(CMAKE_CXX_STANDARD 11)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

find_package(MPI REQUIRED)
message(STATUS "Using MPIEXEC_EXECUTABLE: ${MPIEXEC_EXECUTABLE}")
if (${MPIEXEC_EXECUTABLE} STREQUAL "MPIEXEC_EXECUTABLE-NOTFOUND")
  message(FATAL_ERROR "MPIEXEC_EXECUTABLE not set. Set the cmake variable MPIEXEC_EXECUTABLE to point to mpiexec.")
endif()

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
  DESTINATION lib
  )

configure_file(mpiwrapperexec.in mpiwrapperexec @ONLY)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/mpiwrapperexec"
  DESTINATION bin
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )
