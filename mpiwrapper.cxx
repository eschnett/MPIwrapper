#include "mpiwrapper.hxx"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <type_traits>
#include <vector>

////////////////////////////////////////////////////////////////////////////////

extern "C" {
extern const char *const mpiwrapper_version;
extern const int mpiwrapper_version_major;
extern const int mpiwrapper_version_minor;
extern const int mpiwrapper_version_patch;

const char *const mpiwrapper_version = MPIWRAPPER_VERSION;
const int mpiwrapper_version_major = MPIWRAPPER_VERSION_MAJOR;
const int mpiwrapper_version_minor = MPIWRAPPER_VERSION_MINOR;
const int mpiwrapper_version_patch = MPIWRAPPER_VERSION_PATCH;

// Provided MPI ABI version (we use SemVer)
extern const int mpiabi_version_major;
extern const int mpiabi_version_minor;
extern const int mpiabi_version_patch;

const int mpiabi_version_major = MPIABI_VERSION_MAJOR;
const int mpiabi_version_minor = MPIABI_VERSION_MINOR;
const int mpiabi_version_patch = MPIABI_VERSION_PATCH;
}

////////////////////////////////////////////////////////////////////////////////

namespace {

struct WPI_Op_tuple {
  MPI_Op mpi_op;                  // created by MPI
  MPI_User_function *mpi_user_fn; // called by MPI
  WPI_User_function *wpi_user_fn; // registered by the application

  // mpi_user_fn is explicitly initialized by `init_op_map`
  WPI_Op_tuple() : mpi_op(MPI_OP_NULL), wpi_user_fn(nullptr) {}

  friend std::ostream &operator<<(std::ostream &os,
                                  const WPI_Op_tuple &wpi_op_tuple) {
    return os << "WPI_Op_tuple{mpi_op=" << wpi_op_tuple.mpi_op
              << ",mpi_user_fn=" << (const void *)wpi_op_tuple.mpi_user_fn
              << ",wpi_user_fn=" << (const void *)wpi_op_tuple.wpi_user_fn
              << "}";
  }
};

constexpr int maxN = 100;
std::array<WPI_Op_tuple, (sizeof(MPI_Op) == sizeof(WPI_Op) ? 0 : maxN)> op_map;

template <int N>
void mpi_op_wrapper(void *invec, void *inoutvec, int *len,
                    MPI_Datatype *mpi_datatype) {
  WPI_Datatype wpi_datatype(*mpi_datatype);
  op_map[N].wpi_user_fn(invec, inoutvec, len, &wpi_datatype);
}

namespace {
template <int Nstart, int Nend>
typename std::enable_if<(Nend < Nstart + 1), void>::type init_op_tuple() {}
template <int Nstart, int Nend>
typename std::enable_if<(Nend == Nstart + 1), void>::type init_op_tuple() {
  static_assert(0 <= Nstart &&
                Nstart < std::tuple_size<decltype(op_map)>::value);
  op_map[Nstart].mpi_user_fn = mpi_op_wrapper<Nstart>;
}
template <int Nstart, int Nend>
typename std::enable_if<(Nend > Nstart + 1), void>::type init_op_tuple() {
  constexpr int Nmid = (Nstart + Nend) / 2;
  static_assert(Nstart < Nmid && Nmid < Nend, "");
  init_op_tuple<Nstart, Nmid>();
  init_op_tuple<Nmid, Nend>();
}
} // namespace
__attribute__((__constructor__)) void init_op_map() {
  init_op_tuple<0, std::tuple_size<decltype(op_map)>::value>();
}

int Op_map_create(WPI_User_function *const wpi_user_fn_) {
  assert(wpi_user_fn_);
  static std::mutex m;
  const std::lock_guard<std::mutex> lock(m);
  for (int n = 0; n < int(op_map.size()); ++n) {
    if (!op_map[n].wpi_user_fn) {
      op_map[n].wpi_user_fn = wpi_user_fn_;
      return n;
    }
  }
  std::fprintf(stderr, "Too many MPI_Op created\n");
  std::exit(1);
}

void Op_map_free(const MPI_Op mpi_op_) {
  static std::mutex m;
  const std::lock_guard<std::mutex> lock(m);
  for (int n = 0; n < int(op_map.size()); ++n) {
    if (op_map[n].mpi_op == mpi_op_) {
      op_map[n].mpi_op = MPI_OP_NULL;
      op_map[n].wpi_user_fn = nullptr;
      return;
    }
  }
  std::fprintf(stderr, "Tried to free non-existing MPI_Op\n");
  std::exit(1);
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

// Wrap most constants functions automatically

extern "C" {
#include "mpiwrapper_definitions.h"
}

// Handle the remaining functions manually

// 3.7 Nonblocking Communication

extern "C" int MPIABI_Waitany(int count, MPIABI_Request array_of_requests[],
                              int *indx, MPIABI_StatusPtr status) {
  if (sizeof(MPI_Request) == sizeof(MPIABI_Request))
    return MPI_Waitany(count, (MPI_Request *)array_of_requests, indx,
                       (WPI_StatusPtr)status);
  std::vector<MPI_Request> reqs(count);
  for (int i = 0; i < count; ++i)
    reqs[i] = (WPI_Request)array_of_requests[i];
  const int ierr = MPI_Waitany(count, reqs.data(), indx, (WPI_StatusPtr)status);
  for (int i = 0; i < count; ++i)
    array_of_requests[i] = (WPI_Request)reqs[i];
  return ierr;
}

extern "C" int MPIABI_Testany(int count, MPIABI_Request array_of_requests[],
                              int *indx, int *flag, MPIABI_StatusPtr status) {
  if (sizeof(MPI_Request) == sizeof(MPIABI_Request))
    return MPI_Testany(count, (MPI_Request *)array_of_requests, indx, flag,
                       (WPI_StatusPtr)status);
  std::vector<MPI_Request> reqs(count);
  for (int i = 0; i < count; ++i)
    reqs[i] = (WPI_Request)array_of_requests[i];
  const int ierr =
      MPI_Testany(count, reqs.data(), indx, flag, (WPI_StatusPtr)status);
  for (int i = 0; i < count; ++i)
    array_of_requests[i] = (WPI_Request)reqs[i];
  return ierr;
}

extern "C" int MPIABI_Waitall(int count, MPIABI_Request array_of_requests[],
                              MPIABI_Status array_of_statuses[]) {
  const bool ignore_statuses =
      (MPI_Status *)array_of_statuses == MPI_STATUSES_IGNORE;
  if (sizeof(MPI_Request) == sizeof(MPIABI_Request) && ignore_statuses)
    return MPI_Waitall(count, (MPI_Request *)array_of_requests,
                       MPI_STATUSES_IGNORE);
  std::vector<MPI_Request> reqs(count);
  for (int i = 0; i < count; ++i)
    reqs[i] = (WPI_Request)array_of_requests[i];
  std::vector<MPI_Status> stats;
  if (!ignore_statuses) {
    stats.resize(count);
    for (int i = 0; i < count; ++i)
      stats[i] = (WPI_Status)array_of_statuses[i];
  }
  const int ierr = MPI_Waitall(
      count, reqs.data(), ignore_statuses ? MPI_STATUSES_IGNORE : stats.data());
  for (int i = 0; i < count; ++i)
    array_of_requests[i] = (WPI_Request)reqs[i];
  if (!ignore_statuses)
    for (int i = 0; i < count; ++i)
      array_of_statuses[i] = (WPI_Status)stats[i];
  return ierr;
}

extern "C" int MPIABI_Testall(int count, MPIABI_Request array_of_requests[],
                              int *flag, MPIABI_Status array_of_statuses[]) {
  const bool ignore_statuses =
      (MPI_Status *)array_of_statuses == MPI_STATUSES_IGNORE;
  if (sizeof(MPI_Request) == sizeof(MPIABI_Request) && ignore_statuses)
    return MPI_Testall(count, (MPI_Request *)array_of_requests, flag,
                       MPI_STATUSES_IGNORE);
  std::vector<MPI_Request> reqs(count);
  for (int i = 0; i < count; ++i)
    reqs[i] = (WPI_Request)array_of_requests[i];
  std::vector<MPI_Status> stats;
  if (!ignore_statuses) {
    stats.resize(count);
    for (int i = 0; i < count; ++i)
      stats[i] = (WPI_Status)array_of_statuses[i];
  }
  const int ierr =
      MPI_Testall(count, reqs.data(), flag,
                  ignore_statuses ? MPI_STATUSES_IGNORE : stats.data());
  for (int i = 0; i < count; ++i)
    array_of_requests[i] = (WPI_Request)reqs[i];
  if (!ignore_statuses)
    for (int i = 0; i < count; ++i)
      array_of_statuses[i] = (WPI_Status)stats[i];
  return ierr;
}

extern "C" int MPIABI_Waitsome(int incount, MPIABI_Request array_of_requests[],
                               int *outcount, int array_of_indices[],
                               MPIABI_Status array_of_statuses[]) {
  const bool ignore_statuses =
      (MPI_Status *)array_of_statuses == MPI_STATUSES_IGNORE;
  if (sizeof(MPI_Request) == sizeof(MPIABI_Request) && ignore_statuses)
    return MPI_Waitsome(incount, (MPI_Request *)array_of_requests, outcount,
                        array_of_indices, MPI_STATUSES_IGNORE);
  std::vector<MPI_Request> reqs(incount);
  for (int i = 0; i < incount; ++i)
    reqs[i] = (WPI_Request)array_of_requests[i];
  std::vector<MPI_Status> stats;
  if (!ignore_statuses) {
    stats.resize(incount);
    for (int i = 0; i < incount; ++i)
      stats[i] = (WPI_Status)array_of_statuses[i];
  }
  const int ierr =
      MPI_Waitsome(incount, reqs.data(), outcount, array_of_indices,
                   ignore_statuses ? MPI_STATUSES_IGNORE : stats.data());
  for (int i = 0; i < incount; ++i)
    array_of_requests[i] = (WPI_Request)reqs[i];
  if (!ignore_statuses)
    for (int i = 0; i < incount; ++i)
      array_of_statuses[i] = (WPI_Status)stats[i];
  return ierr;
}

extern "C" int MPIABI_Testsome(int incount, MPIABI_Request array_of_requests[],
                               int *outcount, int array_of_indices[],
                               MPIABI_Status array_of_statuses[]) {
  const bool ignore_statuses =
      (MPI_Status *)array_of_statuses == MPI_STATUSES_IGNORE;
  if (sizeof(MPI_Request) == sizeof(MPIABI_Request) && ignore_statuses)
    return MPI_Waitsome(incount, (MPI_Request *)array_of_requests, outcount,
                        array_of_indices, MPI_STATUSES_IGNORE);
  std::vector<MPI_Request> reqs(incount);
  for (int i = 0; i < incount; ++i)
    reqs[i] = (WPI_Request)array_of_requests[i];
  std::vector<MPI_Status> stats;
  if (!ignore_statuses) {
    stats.resize(incount);
    for (int i = 0; i < incount; ++i)
      stats[i] = (WPI_Status)array_of_statuses[i];
  }
  const int ierr =
      MPI_Testsome(incount, reqs.data(), outcount, array_of_indices,
                   ignore_statuses ? MPI_STATUSES_IGNORE : stats.data());
  for (int i = 0; i < incount; ++i)
    array_of_requests[i] = (WPI_Request)reqs[i];
  if (!ignore_statuses)
    for (int i = 0; i < incount; ++i)
      array_of_statuses[i] = (WPI_Status)stats[i];
  return ierr;
}

// 3.9 Persistent Communication Requests

extern "C" int MPIABI_Startall(int count, MPIABI_Request array_of_requests[]) {
  if (sizeof(MPI_Request) == sizeof(MPIABI_Request))
    return MPI_Startall(count, (MPI_Request *)array_of_requests);
  std::vector<MPI_Request> reqs(count);
  for (int i = 0; i < count; ++i)
    reqs[i] = (WPI_Request)array_of_requests[i];
  const int ierr = MPI_Startall(count, reqs.data());
  for (int i = 0; i < count; ++i)
    array_of_requests[i] = (WPI_Request)reqs[i];
  return ierr;
}

// 4.1 Derived Datatypes

extern "C" int
MPIABI_Type_create_struct(int count, const int array_of_blocklengths[],
                          const MPIABI_Aint array_of_displacements[],
                          const MPIABI_Datatype array_of_types[],
                          MPIABI_Datatype *newtype) {
  if (sizeof(MPI_Datatype) == sizeof(MPIABI_Datatype))
    return MPI_Type_create_struct(
        count, array_of_blocklengths, (const MPI_Aint *)array_of_displacements,
        (const MPI_Datatype *)array_of_types, (MPI_Datatype *)newtype);
  std::vector<MPI_Datatype> datatypes(count);
  for (int i = 0; i < count; ++i)
    datatypes[i] = (WPI_Datatype)array_of_types[i];
  const int ierr = MPI_Type_create_struct(
      count, array_of_blocklengths, (const MPI_Aint *)array_of_displacements,
      datatypes.data(), (WPI_DatatypePtr)newtype);
  return ierr;
}

extern "C" int MPIABI_Type_get_contents(MPIABI_Datatype datatype,
                                        int max_integers, int max_addresses,
                                        int max_datatypes,
                                        int array_of_integers[],
                                        MPIABI_Aint array_of_addresses[],
                                        MPIABI_Datatype array_of_datatypes[]) {
  if (sizeof(MPI_Datatype) == sizeof(MPIABI_Datatype))
    return MPI_Type_get_contents(
        (MPI_Datatype)datatype, max_integers, max_addresses, max_datatypes,
        array_of_integers, (MPI_Aint *)array_of_addresses,
        (MPI_Datatype *)array_of_datatypes);
  std::vector<MPI_Datatype> datatypes(max_datatypes);
  const int ierr = MPI_Type_get_contents(
      (WPI_Datatype)datatype, max_integers, max_addresses, max_datatypes,
      array_of_integers, (MPI_Aint *)array_of_addresses, datatypes.data());
  for (int i = 0; i < max_datatypes; ++i)
    array_of_datatypes[i] = (WPI_Datatype)datatypes[i];
  return ierr;
}

// 5.8 All-to-All Scatter/Gather

extern "C" int
MPIABI_Alltoallw(const void *sendbuf, const int sendcounts[],
                 const int sdispls[], const MPIABI_Datatype sendtypes[],
                 void *recvbuf, const int recvcounts[], const int rdispls[],
                 const MPIABI_Datatype recvtypes[], MPIABI_Comm comm) {
  if (sizeof(MPI_Datatype) == sizeof(MPIABI_Datatype))
    return MPI_Alltoallw(
        sendbuf, sendcounts, sdispls, (MPI_Datatype *)sendtypes, recvbuf,
        recvcounts, rdispls, (const MPI_Datatype *)recvtypes, (WPI_Comm)comm);
  int comm_size;
  MPI_Comm_size((WPI_Comm)comm, &comm_size);
  std::vector<MPI_Datatype> stypes(comm_size);
  for (int i = 0; i < comm_size; ++i)
    stypes[i] = (WPI_Datatype)sendtypes[i];
  std::vector<MPI_Datatype> rtypes(comm_size);
  for (int i = 0; i < comm_size; ++i)
    rtypes[i] = (WPI_Datatype)recvtypes[i];
  const int ierr =
      MPI_Alltoallw(sendbuf, sendcounts, sdispls, stypes.data(), recvbuf,
                    recvcounts, rdispls, rtypes.data(), (WPI_Comm)comm);
  return ierr;
}

// 5.9 Global Reduction Operations

extern "C" int MPIABI_Op_create(MPIABI_User_function *user_fn, int commute,
                                MPIABI_Op *op) {
  if (sizeof(MPI_Op) == sizeof(MPIABI_Op))
    return MPI_Op_create((MPI_User_function *)user_fn, commute, (MPI_Op *)op);
  const int n = Op_map_create((WPI_User_function *)user_fn);
  MPI_User_function *const mpi_user_fn = op_map[n].mpi_user_fn;
  const int ierr =
      MPI_Op_create((MPI_User_function *)mpi_user_fn, commute, (WPI_OpPtr)op);
  op_map[n].mpi_op = *(const MPI_Op *)(WPI_const_OpPtr)op;
  return ierr;
}

extern "C" int MPIABI_Op_free(MPIABI_Op *op) {
  if (sizeof(MPI_Op) == sizeof(MPIABI_Op))
    return MPI_Op_free((MPI_Op *)op);
  const MPI_Op old_op = *(const MPI_Op *)(WPI_const_OpPtr)op;
  const int ierr = MPI_Op_free((WPI_OpPtr)op);
  Op_map_free(old_op);
  return ierr;
}

// 5.12 Nonblocking Collective Operations

extern "C" int MPIABI_Ialltoallw(const void *sendbuf, const int sendcounts[],
                                 const int sdispls[],
                                 const MPIABI_Datatype sendtypes[],
                                 void *recvbuf, const int recvcounts[],
                                 const int rdispls[],
                                 const MPIABI_Datatype recvtypes[],
                                 MPIABI_Comm comm, MPIABI_Request *request) {
  if (sizeof(MPI_Datatype) == sizeof(MPIABI_Datatype))
    return MPI_Ialltoallw(sendbuf, sendcounts, sdispls,
                          (MPI_Datatype *)sendtypes, recvbuf, recvcounts,
                          rdispls, (const MPI_Datatype *)recvtypes,
                          (MPI_Comm)comm, (MPI_Request *)request);
  int comm_size;
  MPI_Comm_size((WPI_Comm)comm, &comm_size);
  std::vector<MPI_Datatype> stypes(comm_size);
  for (int i = 0; i < comm_size; ++i)
    stypes[i] = (WPI_Datatype)sendtypes[i];
  std::vector<MPI_Datatype> rtypes(comm_size);
  for (int i = 0; i < comm_size; ++i)
    rtypes[i] = (WPI_Datatype)recvtypes[i];
  const int ierr = MPI_Ialltoallw(sendbuf, sendcounts, sdispls, stypes.data(),
                                  recvbuf, recvcounts, rdispls, rtypes.data(),
                                  (WPI_Comm)comm, (WPI_RequestPtr)request);
  return ierr;
}

// 7.6 Neighborhood Collective Communication on Procerss Topologies

extern "C" int
MPIABI_Neighbor_alltoallw(const void *sendbuf, const int sendcounts[],
                          const MPIABI_Aint sdispls[],
                          const MPIABI_Datatype sendtypes[], void *recvbuf,
                          const int recvcounts[], const MPIABI_Aint rdispls[],
                          const MPIABI_Datatype recvtypes[], MPIABI_Comm comm) {
  if (sizeof(MPI_Datatype) == sizeof(MPIABI_Datatype))
    return MPI_Neighbor_alltoallw(
        sendbuf, sendcounts, (const MPI_Aint *)sdispls,
        (const MPI_Datatype *)sendtypes, recvbuf, recvcounts,
        (const MPI_Aint *)rdispls, (const MPI_Datatype *)recvtypes,
        (MPI_Comm)comm);
  int comm_size;
  MPI_Comm_size((WPI_Comm)comm, &comm_size);
  std::vector<MPI_Datatype> stypes(comm_size);
  for (int i = 0; i < comm_size; ++i)
    stypes[i] = (WPI_Datatype)sendtypes[i];
  std::vector<MPI_Datatype> rtypes(comm_size);
  for (int i = 0; i < comm_size; ++i)
    rtypes[i] = (WPI_Datatype)recvtypes[i];
  const int ierr = MPI_Neighbor_alltoallw(
      sendbuf, sendcounts, (const MPI_Aint *)sdispls, stypes.data(), recvbuf,
      recvcounts, (const MPI_Aint *)rdispls, rtypes.data(), (WPI_Comm)comm);
  return ierr;
}

// 7.7 Nonblocking Neighborhood Communication on Process Topologies

extern "C" int MPIABI_Ineighbor_alltoallw(
    const void *sendbuf, const int sendcounts[], const MPIABI_Aint sdispls[],
    const MPIABI_Datatype sendtypes[], void *recvbuf, const int recvcounts[],
    const MPIABI_Aint rdispls[], const MPIABI_Datatype recvtypes[],
    MPIABI_Comm comm, MPIABI_Request *request) {
  if (sizeof(MPI_Datatype) == sizeof(MPIABI_Datatype))
    return MPI_Ineighbor_alltoallw(
        sendbuf, sendcounts, (const MPI_Aint *)sdispls,
        (const MPI_Datatype *)sendtypes, recvbuf, recvcounts,
        (const MPI_Aint *)rdispls, (const MPI_Datatype *)recvtypes,
        (MPI_Comm)comm, (MPI_Request *)request);
  int comm_size;
  MPI_Comm_size((WPI_Comm)comm, &comm_size);
  std::vector<MPI_Datatype> stypes(comm_size);
  for (int i = 0; i < comm_size; ++i)
    stypes[i] = (WPI_Datatype)sendtypes[i];
  std::vector<MPI_Datatype> rtypes(comm_size);
  for (int i = 0; i < comm_size; ++i)
    rtypes[i] = (WPI_Datatype)recvtypes[i];
  const int ierr = MPI_Ineighbor_alltoallw(
      sendbuf, sendcounts, (const MPI_Aint *)sdispls, stypes.data(), recvbuf,
      recvcounts, (const MPI_Aint *)rdispls, rtypes.data(), (WPI_Comm)comm,
      (WPI_RequestPtr)request);
  return ierr;
}

// 8.1 Implementation Information

extern "C" int MPIABI_Get_library_version(char *version, int *resultlen) {
  char wrapped_version[MPI_MAX_LIBRARY_VERSION_STRING];
  int wrapped_resultlen;
  MPI_Get_library_version(wrapped_version, &wrapped_resultlen);

  // TODO: Add MPItrampoline version number as well
  *resultlen =
      snprintf(version, MPIABI_MAX_LIBRARY_VERSION_STRING,
               "MPIwrapper %s, using MPIABI %d.%d.%d, wrapping:\n%s",
               MPIWRAPPER_VERSION, MPIABI_VERSION_MAJOR, MPIABI_VERSION_MINOR,
               MPIABI_VERSION_PATCH, wrapped_version);
  return MPI_SUCCESS;
}

// 10.3 Process Manager Interface

extern "C" int MPIABI_Comm_spawn_multiple(
    int count, char *array_of_commands[], char **array_of_argv[],
    const int array_of_maxprocs[], const MPIABI_Info array_of_info[], int root,
    MPIABI_Comm comm, MPIABI_Comm *intercomm, int array_of_errcodes[]) {
  if (sizeof(MPI_Info) == sizeof(MPIABI_Info))
    return MPI_Comm_spawn_multiple(
        count, array_of_commands, array_of_argv, array_of_maxprocs,
        (const MPI_Info *)array_of_info, root, (MPI_Comm)comm,
        (MPI_Comm *)intercomm, array_of_errcodes);
  std::vector<MPI_Info> infos(count);
  for (int i = 0; i < count; ++i)
    infos[i] = (WPI_Info)array_of_info[i];
  const int ierr = MPI_Comm_spawn_multiple(
      count, array_of_commands, array_of_argv, array_of_maxprocs, infos.data(),
      root, (WPI_Comm)comm, (WPI_CommPtr)intercomm, array_of_errcodes);
  return ierr;
}
