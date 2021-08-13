#[[
rm -rf build-openmpi $HOME/src/c/MPIstuff/mpiwrapper-openmpi
cmake -S . -B build-openmpi -G Ninja -DCMAKE_CXX_COMPILER=mpicxx-openmpi-gcc11 -DCMAKE_Fortran_COMPILER=mpifort-openmpi-gcc11 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$HOME/src/c/MPIstuff/mpiwrapper-openmpi
cmake --build build-openmpi && cmake --install build-openmpi

rm -rf build-mpich $HOME/src/c/MPIstuff/mpiwrapper-openmpi
cmake -S . -B build-mpich -G Ninja -DCMAKE_CXX_COMPILER=mpicxx-mpich-gcc10 -DCMAKE_Fortran_COMPILER=mpifort-mpich-gcc10 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$HOME/src/c/MPIstuff/mpiwrapper-mpich
cmake --build build-mpich && cmake --install build-mpich



export PATH=$HOME/src/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-11.2.0/cmake-3.21.1-ztbpwc6mr65mqagnikcuuiifn3ox6btv/bin:$PATH
rm -rf build-openmpi ../MPIstuff-openmpi
cmake -S . -B build-openmpi -G Ninja -DCMAKE_CXX_COMPILER=/cm/shared/apps/openmpi/gcc-9/64/4.1.0/bin/mpic++ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../MPIstuff-openmpi
cmake --build build-openmpi && cmake --install build-openmpi

export PATH=$HOME/src/spack/opt/spack/linux-ubuntu18.04-skylake_avx512/gcc-11.2.0/cmake-3.21.1-ztbpwc6mr65mqagnikcuuiifn3ox6btv/bin:$PATH
module load mpi
rm -rf build-intel ../MPIstuff-intel
cmake -S . -B build-intel -G Ninja -DCMAKE_CXX_COMPILER=/cm/shared/apps/intel/mpi/2021.1.1/bin/mpicxx -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../MPIstuff-intel
cmake --build build-intel && cmake --install build-intel
module unload mpi
]]



cmake_minimum_required(VERSION 3.12...3.20)
project(
  MPIwrapper VERSION 1.0
  DESCRIPTION "MPI wrapper"
  LANGUAGES CXX Fortran
  )

set(CMAKE_CXX_STANDARD 11)

find_package(MPI REQUIRED)

add_custom_command(
  OUTPUT mpiwrapper.f
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/fortran-cpp.sh ${CMAKE_CURRENT_SOURCE_DIR}/mpiwrapper.f.in mpiwrapper.f
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/mpiwrapper.f.in ${CMAKE_CURRENT_SOURCE_DIR}/mpi-constants-f.inc
  VERBATIM)

add_library(mpiwrapper SHARED mpiwrapper.h mpiwrapper.cxx mpiwrapper.f)
target_link_libraries(mpiwrapper PRIVATE MPI::MPI_CXX MPI::MPI_Fortran)

install(
  TARGETS mpiwrapper
  )

# configure_file(mpiexec1.in mpiexec1 @ONLY)
# install(
#   FILES "${CMAKE_CURRENT_BINARY_DIR}/mpiexec1"
#   DESTINATION bin
#   PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
#   )
