#include "mpiwrapper.h"

#include <cstdarg>
#include <vector>

////////////////////////////////////////////////////////////////////////////////

// Wrap constants

#define MT(TYPE) WPI_##TYPE
#define CONSTANT(TYPE, NAME)                                                   \
  extern "C" TYPE const WPI_##NAME;                                            \
  TYPE const WPI_##NAME = (TYPE)MPI_##NAME;
#include "mpi-constants.inc"
#undef CONSTANT
#undef MT

////////////////////////////////////////////////////////////////////////////////

// Wrap functions

// Define prototypes for all functions
#define MT(TYPE) WPI_##TYPE
#define MP(TYPE) MPI_##TYPE
#define FUNCTION(RTYPE, NAME, PTYPES, PNAMES)                                  \
  extern "C" RTYPE WPI_##NAME PTYPES;
#include "mpi-functions.inc"
#undef FUNCTION

// Define implementations for most functions
#define MT(TYPE) WPI_##TYPE
#define MP(TYPE) MPI_##TYPE
#define FUNCTION(RTYPE, NAME, PTYPES, PNAMES)                                  \
  extern "C" RTYPE WPI_##NAME PTYPES { return MPI_##NAME PNAMES; }
#define SKIP_MANUAL_FUNCTIONS
#include "mpi-functions.inc"
#undef SKIP_MANUAL_FUNCTIONS
#undef FUNCTION

// Handle the remaining functions manually

// 3.7 Nonblocking Communication

extern "C" int MT(Waitany)(int count, MT(Request) array_of_requests[],
                           int *indx, MT(StatusPtr) status) {
  if (sizeof(MP(Request)) == sizeof(MT(Request)))
    return MP(Waitany)(count, (MP(Request) *)array_of_requests, indx,
                       (MP(StatusPtr))status);
  std::vector<MP(Request)> reqs(count);
  for (int i = 0; i < count; ++i)
    reqs[i] = (MP(Request))array_of_requests[i];
  const int ierr = MP(Waitany)(count, reqs.data(), indx, (MP(StatusPtr))status);
  for (int i = 0; i < count; ++i)
    array_of_requests[i] = (MT(Request))reqs[i];
  return ierr;
}

extern "C" int MT(Testany)(int count, MT(Request) array_of_requests[],
                           int *indx, int *flag, MT(StatusPtr) status) {
  if (sizeof(MP(Request)) == sizeof(MT(Request)))
    return MP(Testany)(count, (MP(Request) *)array_of_requests, indx, flag,
                       (MP(StatusPtr))status);
  std::vector<MP(Request)> reqs(count);
  for (int i = 0; i < count; ++i)
    reqs[i] = (MP(Request))array_of_requests[i];
  const int ierr =
      MP(Testany)(count, reqs.data(), indx, flag, (MP(StatusPtr))status);
  for (int i = 0; i < count; ++i)
    array_of_requests[i] = (MT(Request))reqs[i];
  return ierr;
}

extern "C" int MT(Waitall)(int count, MT(Request) array_of_requests[],
                           MT(StatusPtr) status) {
  if (sizeof(MP(Request)) == sizeof(MT(Request)))
    return MP(Waitall)(count, (MP(Request) *)array_of_requests,
                       (MP(StatusPtr))status);
  std::vector<MP(Request)> reqs(count);
  for (int i = 0; i < count; ++i)
    reqs[i] = (MP(Request))array_of_requests[i];
  const int ierr = MP(Waitall)(count, reqs.data(), (MP(StatusPtr))status);
  for (int i = 0; i < count; ++i)
    array_of_requests[i] = (MT(Request))reqs[i];
  return ierr;
}

extern "C" int MT(Testall)(int count, MT(Request) array_of_requests[],
                           int *flag, MT(StatusPtr) status) {
  if (sizeof(MP(Request)) == sizeof(MT(Request)))
    return MP(Testall)(count, (MP(Request) *)array_of_requests, flag,
                       (MP(StatusPtr))status);
  std::vector<MP(Request)> reqs(count);
  for (int i = 0; i < count; ++i)
    reqs[i] = (MP(Request))array_of_requests[i];
  const int ierr = MP(Testall)(count, reqs.data(), flag, (MP(StatusPtr))status);
  for (int i = 0; i < count; ++i)
    array_of_requests[i] = (MT(Request))reqs[i];
  return ierr;
}

extern "C" int MT(Waitsome)(int incount, MT(Request) array_of_requests[],
                            int *outcount, int array_of_indices[],
                            MT(Status) array_of_statuses[]) {
  if (sizeof(MP(Request)) == sizeof(MT(Request)) &&
      (MP(Status) *)array_of_statuses == MPI_STATUSES_IGNORE)
    return MP(Waitsome)(incount, (MP(Request) *)array_of_requests, outcount,
                        array_of_indices, MPI_STATUSES_IGNORE);
  std::vector<MP(Request)> reqs(incount);
  for (int i = 0; i < incount; ++i)
    reqs[i] = (MP(Request))array_of_requests[i];
  std::vector<MP(Status)> stats(incount);
  for (int i = 0; i < incount; ++i)
    stats[i] = (MP(Status))array_of_statuses[i];
  const int ierr = MP(Waitsome)(incount, reqs.data(), outcount,
                                array_of_indices, stats.data());
  for (int i = 0; i < incount; ++i)
    array_of_requests[i] = (MT(Request))reqs[i];
  for (int i = 0; i < incount; ++i)
    array_of_statuses[i] = (MT(Status))stats[i];
  return ierr;
}

extern "C" int MT(Testsome)(int incount, MT(Request) array_of_requests[],
                            int *outcount, int array_of_indices[],
                            MT(Status) array_of_statuses[]) {
  if (sizeof(MP(Request)) == sizeof(MT(Request)) &&
      (MP(Status) *)array_of_statuses == MPI_STATUSES_IGNORE)
    return MP(Waitsome)(incount, (MP(Request) *)array_of_requests, outcount,
                        array_of_indices, MPI_STATUSES_IGNORE);
  std::vector<MP(Request)> reqs(incount);
  for (int i = 0; i < incount; ++i)
    reqs[i] = (MP(Request))array_of_requests[i];
  std::vector<MP(Status)> stats(incount);
  for (int i = 0; i < incount; ++i)
    stats[i] = (MP(Status))array_of_statuses[i];
  const int ierr = MP(Testsome)(incount, reqs.data(), outcount,
                                array_of_indices, stats.data());
  for (int i = 0; i < incount; ++i)
    array_of_requests[i] = (MT(Request))reqs[i];
  for (int i = 0; i < incount; ++i)
    array_of_statuses[i] = (MT(Status))stats[i];
  return ierr;
}

// 3.9 Persistent Communication Requests

extern "C" int MT(Startall)(int count, MT(Request) array_of_requests[]) {
  if (sizeof(MP(Request)) == sizeof(MT(Request)))
    return MP(Startall)(count, (MP(Request) *)array_of_requests);
  std::vector<MP(Request)> reqs(count);
  for (int i = 0; i < count; ++i)
    reqs[i] = (MP(Request))array_of_requests[i];
  const int ierr = MP(Startall)(count, reqs.data());
  for (int i = 0; i < count; ++i)
    array_of_requests[i] = (MT(Request))reqs[i];
  return ierr;
}

// 4.1 Derived Datatypes

extern "C" int MT(Type_create_struct)(int count,
                                      const int array_of_blocklengths[],
                                      const MT(Aint) array_of_displacements[],
                                      const MT(Datatype) array_of_types[],
                                      MT(Datatype) * newtype) {
  if (sizeof(MP(Datatype)) == sizeof(MT(Datatype)))
    return MP(Type_create_struct)(
        count, array_of_blocklengths, (const MP(Aint) *)array_of_displacements,
        (const MP(Datatype) *)array_of_types, (MP(Datatype) *)newtype);
  std::vector<MP(Datatype)> types(count);
  for (int i = 0; i < count; ++i)
    types[i] = (MP(Datatype))array_of_types[i];
  const int ierr = MP(Type_create_struct)(
      count, array_of_blocklengths, (const MP(Aint) *)array_of_displacements,
      types.data(), (MP(Datatype) *)newtype);
  return ierr;
}

extern "C" int MT(Type_get_contents)(MT(Datatype) datatype, int max_integers,
                                     int max_addresses, int max_datatypes,
                                     int array_of_integers[],
                                     MT(Aint) array_of_addresses[],
                                     MT(Datatype) array_of_datatypes[]) {
  if (sizeof(MP(Datatype)) == sizeof(MT(Datatype)))
    return MP(Type_get_contents)(
        (MP(Datatype))datatype, max_integers, max_addresses, max_datatypes,
        array_of_integers, (MP(Aint) *)array_of_addresses,
        (MP(Datatype) *)array_of_datatypes);
  std::vector<MP(Datatype)> types(max_datatypes);
  const int ierr = MP(Type_get_contents)(
      (MP(Datatype))datatype, max_integers, max_addresses, max_datatypes,
      array_of_integers, (MP(Aint) *)array_of_addresses,
      (MP(Datatype) *)array_of_datatypes);
  for (int i = 0; i < max_datatypes; ++i)
    array_of_datatypes[i] = (MT(Datatype))types[i];
  return ierr;
}

// 5.8 All-to-All Scatter/Gather

extern "C" int MT(Alltoallw)(const void *sendbuf, const int sendcounts[],
                             const int sdispls[],
                             const MT(Datatype) sendtypes[], void *recvbuf,
                             const int recvcounts[], const int rdispls[],
                             const MT(Datatype) recvtypes[], MT(Comm) comm) {
  if (sizeof(MP(Datatype)) == sizeof(MT(Datatype)))
    return MP(Alltoallw)(
        sendbuf, sendcounts, sdispls, (MP(Datatype) *)sendtypes, recvbuf,
        recvcounts, rdispls, (const MP(Datatype) *)recvtypes, (MP(Comm))comm);
  int comm_size;
  MPI_Comm_size((MP(Comm))comm, &comm_size);
  std::vector<MP(Datatype)> stypes(comm_size);
  for (int i = 0; i < comm_size; ++i)
    stypes[i] = (MP(Datatype))sendtypes[i];
  std::vector<MP(Datatype)> rtypes(comm_size);
  for (int i = 0; i < comm_size; ++i)
    rtypes[i] = (MP(Datatype))recvtypes[i];
  const int ierr =
      MP(Alltoallw)(sendbuf, sendcounts, sdispls, stypes.data(), recvbuf,
                    recvcounts, rdispls, rtypes.data(), (MP(Comm))comm);
  return ierr;
}

// 5.12 Nonblocking Collective Operations

extern "C" int MT(Ialltoallw)(const void *sendbuf, const int sendcounts[],
                              const int sdispls[],
                              const MT(Datatype) sendtypes[], void *recvbuf,
                              const int recvcounts[], const int rdispls[],
                              const MT(Datatype) recvtypes[], MT(Comm) comm,
                              MT(Request) * request) {
  if (sizeof(MP(Datatype)) == sizeof(MT(Datatype)))
    return MP(Ialltoallw)(sendbuf, sendcounts, sdispls,
                          (MP(Datatype) *)sendtypes, recvbuf, recvcounts,
                          rdispls, (const MP(Datatype) *)recvtypes,
                          (MP(Comm))comm, (MP(Request) *)request);
  int comm_size;
  MPI_Comm_size((MP(Comm))comm, &comm_size);
  std::vector<MP(Datatype)> stypes(comm_size);
  for (int i = 0; i < comm_size; ++i)
    stypes[i] = (MP(Datatype))sendtypes[i];
  std::vector<MP(Datatype)> rtypes(comm_size);
  for (int i = 0; i < comm_size; ++i)
    rtypes[i] = (MP(Datatype))recvtypes[i];
  const int ierr = MP(Ialltoallw)(sendbuf, sendcounts, sdispls, stypes.data(),
                                  recvbuf, recvcounts, rdispls, rtypes.data(),
                                  (MP(Comm))comm, (MP(Request) *)request);
  return ierr;
}

// 7.6 Neighborhood Collective Communication on Procerss Topologies

extern "C" int MT(Neighbor_alltoallw)(
    const void *sendbuf, const int sendcounts[], const MT(Aint) sdispls[],
    const MT(Datatype) sendtypes[], void *recvbuf, const int recvcounts[],
    const MT(Aint) rdispls[], const MT(Datatype) recvtypes[], MT(Comm) comm) {
  if (sizeof(MP(Datatype)) == sizeof(MT(Datatype)))
    return MP(Neighbor_alltoallw)(
        sendbuf, sendcounts, (const MP(Aint) *)sdispls,
        (const MP(Datatype) *)sendtypes, recvbuf, recvcounts,
        (const MP(Aint) *)rdispls, (const MP(Datatype) *)recvtypes,
        (MP(Comm))comm);
  int comm_size;
  MPI_Comm_size((MP(Comm))comm, &comm_size);
  std::vector<MP(Datatype)> stypes(comm_size);
  for (int i = 0; i < comm_size; ++i)
    stypes[i] = (MP(Datatype))sendtypes[i];
  std::vector<MP(Datatype)> rtypes(comm_size);
  for (int i = 0; i < comm_size; ++i)
    rtypes[i] = (MP(Datatype))recvtypes[i];
  const int ierr = MP(Neighbor_alltoallw)(
      sendbuf, sendcounts, (const MP(Aint) *)sdispls, stypes.data(), recvbuf,
      recvcounts, (const MP(Aint) *)rdispls, rtypes.data(), (MP(Comm))comm);
  return ierr;
}

// 7.7 Nonblocking Neighborhood Communication on Process Topologies

extern "C" int MT(Ineighbor_alltoallw)(
    const void *sendbuf, const int sendcounts[], const MT(Aint) sdispls[],
    const MT(Datatype) sendtypes[], void *recvbuf, const int recvcounts[],
    const MT(Aint) rdispls[], const MT(Datatype) recvtypes[], MT(Comm) comm,
    MT(Request) * request) {
  if (sizeof(MP(Datatype)) == sizeof(MT(Datatype)))
    return MP(Ineighbor_alltoallw)(
        sendbuf, sendcounts, (const MP(Aint) *)sdispls,
        (const MP(Datatype) *)sendtypes, recvbuf, recvcounts,
        (const MP(Aint) *)rdispls, (const MP(Datatype) *)recvtypes,
        (MP(Comm))comm, (MP(Request) *)request);
  int comm_size;
  MPI_Comm_size((MP(Comm))comm, &comm_size);
  std::vector<MP(Datatype)> stypes(comm_size);
  for (int i = 0; i < comm_size; ++i)
    stypes[i] = (MP(Datatype))sendtypes[i];
  std::vector<MP(Datatype)> rtypes(comm_size);
  for (int i = 0; i < comm_size; ++i)
    rtypes[i] = (MP(Datatype))recvtypes[i];
  const int ierr = MP(Ineighbor_alltoallw)(
      sendbuf, sendcounts, (const MP(Aint) *)sdispls, stypes.data(), recvbuf,
      recvcounts, (const MP(Aint) *)rdispls, rtypes.data(), (MP(Comm))comm,
      (MP(Request) *)request);
  return ierr;
}

// 8.1 Implementation Information

extern "C" int MT(Get_version)(int *version, int *subversion) {
  // TODO: Synchronize with MPI_VERSION and MPI_SUBVERSION
  *version = 3;
  *subversion = 1;
  return MPI_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////

extern "C" int MT(Comm_spawn_multiple)(int count, char *array_of_commands[],
                                       char **array_of_argv[],
                                       const int array_of_maxprocs[],
                                       const MT(Info) array_of_info[], int root,
                                       MT(Comm) comm, MT(Comm) * intercomm,
                                       int array_of_errcodes[]) {
  if (sizeof(MP(Info)) == sizeof(MT(Info)))
    return MP(Comm_spawn_multiple)(
        count, array_of_commands, array_of_argv, array_of_maxprocs,
        (const MP(Info) *)array_of_info, root, (MP(Comm))comm,
        (MP(Comm) *)intercomm, array_of_errcodes);
  std::vector<MP(Info)> infos(count);
  for (int i = 0; i < count; ++i)
    infos[i] = (MP(Info))array_of_info[i];
  const int ierr = MP(Comm_spawn_multiple)(
      count, array_of_commands, array_of_argv, array_of_maxprocs, infos.data(),
      root, (MP(Comm))comm, (MP(Comm) *)intercomm, array_of_errcodes);
  return ierr;
}

// There is no way to forward varargs arguments. Should we return an error here?
extern "C" int MT(Pcontrol)(const int level, ...) { return MPI_SUCCESS; }

#undef MT
#undef MP

////////////////////////////////////////////////////////////////////////////////

// Wrap Fortran functions

// Define prototypes for the MPI Fortran functions
#define MT(TYPE) MPI_##TYPE
#define FUNCTION(RTYPE, NAME, PTYPES, PNAMES)                                  \
  extern "C" RTYPE mpi_##NAME##_ PTYPES;
#include "mpi-functions-f.inc"
#undef FUNCTION
#undef MT

// Define prototypes for our Fortran wrapper functions
#define MT(TYPE) WPI_##TYPE
#define FUNCTION(RTYPE, NAME, PTYPES, PNAMES)                                  \
  extern "C" RTYPE wpi_##NAME##_ PTYPES;
#include "mpi-functions-f.inc"
#undef FUNCTION
#undef MT

// Implement most Fortran wrapper functions
#define MT(TYPE) WPI_##TYPE
#define MP(TYPE) MPI_##TYPE
#define FUNCTION(RTYPE, NAME, PTYPES, PNAMES)                                  \
  extern "C" RTYPE wpi_##NAME##_ PTYPES { return mpi_##NAME##_ PNAMES; }
#define SKIP_MANUAL_FUNCTIONS
#include "mpi-functions-f.inc"
#undef SKIP_MANUAL_FUNCTIONS
#undef FUNCTION
#undef MT
#undef MP
