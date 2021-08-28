#[[
rm -rf build-openmpi $HOME/src/c/MPIstuff/mpiwrapper-openmpi
cmake -S . -B build-openmpi -G Ninja -DMPIEXEC_EXECUTABLE=$(which mpiexec-openmpi-gcc11) -DCMAKE_CXX_COMPILER=mpicxx-openmpi-gcc11 -DCMAKE_Fortran_COMPILER=mpifort-openmpi-gcc11 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$HOME/src/c/MPIstuff/mpiwrapper-openmpi
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
    -DMPI_mpi_LIBRARY='-L/opt/local/lib/mpich-gcc10;-lmpi' \
    -DMPI_mpicxx_LIBRARY='-L/opt/local/lib/mpich-gcc10;-lmpicxx' \
    -DMPI_pmpi_LIBRARY='-L/opt/local/lib/mpich-gcc10;-lpmpi' \
    -DMPI_Fortran_ADDITIONAL_INCLUDE_DIRS=/opt/local/include/mpich-gcc10 \
    -DMPI_Fortran_LIB_NAMES=mpifort \
    -DMPI_mpifort_LIBRARY='-L/opt/local/lib/mpich-gcc10;-lmpifort;-lmpi;-lpmpi' \
    -DMPIEXEC_EXECUTABLE=/opt/local/bin/mpiexec-mpich-gcc10 \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=$HOME/src/c/MPIstuff/mpiwrapper-mpich
cmake --build build-mpich --verbose && cmake --install build-mpich --verbose



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
  MPIwrapper VERSION 1.1.1
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

configure_file(mpiwrapper-version.h.in mpiwrapper-version.h @ONLY)

add_custom_command(
  OUTPUT mpiwrapper.f
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/fortran-cpp.sh ${CMAKE_CURRENT_SOURCE_DIR}/mpiwrapper.f.in mpiwrapper.f
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/mpiwrapper.f.in ${CMAKE_CURRENT_SOURCE_DIR}/mpi-constants-f.inc
  VERBATIM)

add_library(mpiwrapper MODULE mpiwrapper.hxx mpiwrapper-version.h mpiwrapper.cxx mpiwrapper.f)
target_include_directories(mpiwrapper PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(mpiwrapper PRIVATE MPI::MPI_CXX MPI::MPI_Fortran)

if(APPLE)
  # On macOS, check that the plugin `libmpiwrapper.so` is built with a
  # two-level namespace. If not, it will use the MPI functions
  # provided by MPItrampoline instead of the "real" MPI, which will
  # recurse infinitely, leading to a stack overflow and segfault.
  execute_process(
    COMMAND otool -hV ${CMAKE_CURRENT_BINARY_DIR}/libmpiwrapper.so
    OUTPUT_VARIABLE twolevel
    )
  if(NOT ("${twolevel}" MATCHES " TWOLEVEL "))
    message(FATAL_ERROR
      "Error in libmpiwrapper.so plugin:
      The plugin libmpiwrapper.so was built without a two-level
      namespace. This means that it will end up using the MPI
      functions provided by MPItrampoline instead of those by the
      actual MPI library. This will recurse infinitely, leading to a
      stack overflow and segfault.")
  endif()
endif()

install(
  TARGETS mpiwrapper
  LIBRARY
  DESTINATION lib
  )

configure_file(mpiwrapper-mpiexec.in mpiwrapper-mpiexec @ONLY)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/mpiwrapper-mpiexec"
  DESTINATION bin
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )
