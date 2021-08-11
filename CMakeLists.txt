#[[
rm -rf build
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=mpicxx-openmpi-gcc11 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$HOME/src/c/MPIstuff
cmake --build build && cmake --install build
]]

cmake_minimum_required(VERSION 3.12...3.20)
project(
  MPIwrapper VERSION 1.0
  DESCRIPTION "MPI wrapper"
  LANGUAGES CXX
  )

find_package(MPI REQUIRED)

add_library(mpiwrapper SHARED mpiwrapper.cxx)
target_link_libraries(mpiwrapper PUBLIC MPI::MPI_CXX)

install(
  TARGETS mpiwrapper
  )

# configure_file(mpiexec1.in mpiexec1 @ONLY)
# install(
#   FILES "${CMAKE_CURRENT_BINARY_DIR}/mpiexec1"
#   DESTINATION bin
#   PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
#   )
