#include <mpi.h>

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <array>
#include <type_traits>
#include <utility>
#include <vector>

// We implement this file in C++ so that we can automatically convert between
// the `MPI` and `MPIwrapper` types

////////////////////////////////////////////////////////////////////////////////

// Simple types

typedef uintptr_t MPIwrapper_Aint;
static_assert(sizeof(MPI_Aint) == sizeof(MPIwrapper_Aint));

typedef int64_t MPIwrapper_Count;
static_assert(sizeof(MPI_Count) == sizeof(MPIwrapper_Count));

typedef int MPIwrapper_Fint;
static_assert(sizeof(MPI_Fint) == sizeof(MPIwrapper_Fint));

typedef int64_t MPIwrapper_Offset;
static_assert(sizeof(MPI_Offset) == sizeof(MPIwrapper_Offset));

////////////////////////////////////////////////////////////////////////////////

// Handles

union MPIwrapper_Comm {
  MPI_Comm comm;
  uintptr_t padding;

  MPIwrapper_Comm() = default;
  MPIwrapper_Comm(MPI_Comm comm_) : comm(comm_) {}
  operator MPI_Comm() const { return comm; }
};
static_assert(sizeof(MPI_Comm) <= sizeof(MPIwrapper_Comm));

union MPIwrapper_Datatype {
  MPI_Datatype datatype;
  uintptr_t padding;

  MPIwrapper_Datatype() = default;
  MPIwrapper_Datatype(MPI_Datatype datatype_) : datatype(datatype_) {}
  operator MPI_Datatype() const { return datatype; }
};
static_assert(sizeof(MPI_Datatype) <= sizeof(MPIwrapper_Datatype));

union MPIwrapper_Errhandler {
  MPI_Errhandler errhandler;
  uintptr_t padding;

  MPIwrapper_Errhandler() = default;
  MPIwrapper_Errhandler(MPI_Errhandler errhandler_) : errhandler(errhandler_) {}
  operator MPI_Errhandler() const { return errhandler; }
};
static_assert(sizeof(MPI_Errhandler) <= sizeof(MPIwrapper_Errhandler));

union MPIwrapper_File {
  MPI_File file;
  uintptr_t padding;

  MPIwrapper_File() = default;
  MPIwrapper_File(MPI_File file_) : file(file_) {}
  operator MPI_File() const { return file; }
};
static_assert(sizeof(MPI_File) <= sizeof(MPIwrapper_File));

union MPIwrapper_Group {
  MPI_Group group;
  uintptr_t padding;

  MPIwrapper_Group() = default;
  MPIwrapper_Group(MPI_Group group_) : group(group_) {}
  operator MPI_Group() const { return group; }
};
static_assert(sizeof(MPI_Group) <= sizeof(MPIwrapper_Group));

union MPIwrapper_Info {
  MPI_Info info;
  uintptr_t padding;

  MPIwrapper_Info() = default;
  MPIwrapper_Info(MPI_Info info_) : info(info_) {}
  operator MPI_Info() const { return info; }
};
static_assert(sizeof(MPI_Info) <= sizeof(MPIwrapper_Info));

union MPIwrapper_Message {
  MPI_Message message;
  uintptr_t padding;

  MPIwrapper_Message() = default;
  MPIwrapper_Message(MPI_Message message_) : message(message_) {}
  operator MPI_Message() const { return message; }
};
static_assert(sizeof(MPI_Message) <= sizeof(MPIwrapper_Message));

union MPIwrapper_Op {
  MPI_Op op;
  uintptr_t padding;

  MPIwrapper_Op() = default;
  MPIwrapper_Op(MPI_Op op_) : op(op_) {}
  operator MPI_Op() const { return op; }
};
static_assert(sizeof(MPI_Op) <= sizeof(MPIwrapper_Op));

union MPIwrapper_Request {
  MPI_Request request;
  uintptr_t padding;

  MPIwrapper_Request() = default;
  MPIwrapper_Request(MPI_Request request_) : request(request_) {}
  operator MPI_Request() const { return request; }
};
static_assert(sizeof(MPI_Request) <= sizeof(MPIwrapper_Request));

union MPIwrapper_Win {
  MPI_Win win;
  uintptr_t padding;

  MPIwrapper_Win() = default;
  MPIwrapper_Win(MPI_Win win_) : win(win_) {}
  operator MPI_Win() const { return win; }
};
static_assert(sizeof(MPI_Win) <= sizeof(MPIwrapper_Win));

////////////////////////////////////////////////////////////////////////////////

// MPI_Status
// This is a difficult type since it has user-accessible fields.

// We put the MPI_Status struct in the beginning. This way, we can cast between
// MPI_Status* and MPIwrapper_Status*.
typedef struct MPIwrapper_Status {
  mutable union {
    MPI_Status status;
    struct {
      int f0;
      int f1;
      int f2;
      int f3;
      size_t f4;
    } OpenMPI; // also Spectrum
    struct {
      int f0;
      int f1;
      int f2;
      int f3;
      int f4;
    } MPICH; // also Intel
  } wrapped;
  mutable int MPI_SOURCE;
  mutable int MPI_TAG;
  mutable int MPI_ERROR;

  MPIwrapper_Status() = default;
  MPIwrapper_Status(const MPI_Status &status_)
      : MPI_SOURCE(status_.MPI_SOURCE), MPI_TAG(status_.MPI_TAG),
        MPI_ERROR(status_.MPI_ERROR) {
    wrapped.status = status_;
  }
  operator MPI_Status() const {
    wrapped.status.MPI_SOURCE = MPI_SOURCE;
    wrapped.status.MPI_TAG = MPI_TAG;
    wrapped.status.MPI_ERROR = MPI_ERROR;
    return wrapped.status;
  }
} MPIwrapper_Status;

static_assert(std::is_pod<MPIwrapper_Status>::value);
static_assert(sizeof MPIwrapper_Status::wrapped >= sizeof(MPI_Status));

namespace {

// TODO: Don't define this
typedef MPI_Status *MPI_StatusPtr;
typedef const MPI_Status *MPI_const_StatusPtr;

struct MPIwrapper_StatusPtr {
  MPIwrapper_Status *status;

  // We assume that the `MPI_Status` is actually an `MPIwrapper_Status`
  MPIwrapper_StatusPtr(MPI_Status *status_)
      : status((MPIwrapper_Status *)status_) {
    if (status_ != MPI_STATUS_IGNORE && status_ != MPI_STATUSES_IGNORE) {
      status->MPI_SOURCE = status->wrapped.status.MPI_SOURCE;
      status->MPI_TAG = status->wrapped.status.MPI_TAG;
      status->MPI_ERROR = status->wrapped.status.MPI_ERROR;
    }
  }
  operator MPI_StatusPtr() {
    assert(status);
    return &status->wrapped.status;
  }
};

struct MPIwrapper_const_StatusPtr {
  const MPIwrapper_Status *status;

  // We assume that the `MPI_Status` is actually an `MPIwrapper_Status`
  MPIwrapper_const_StatusPtr(const MPI_Status *status_)
      : status((const MPIwrapper_Status *)status_) {
    if (status_ != MPI_STATUS_IGNORE && status_ != MPI_STATUSES_IGNORE) {
      status->MPI_SOURCE = status->wrapped.status.MPI_SOURCE;
      status->MPI_TAG = status->wrapped.status.MPI_TAG;
      status->MPI_ERROR = status->wrapped.status.MPI_ERROR;
    }
  }
  operator MPI_const_StatusPtr() {
    if (&status->wrapped.status != MPI_STATUS_IGNORE &&
        &status->wrapped.status != MPI_STATUSES_IGNORE) {
      status->wrapped.status.MPI_SOURCE = status->MPI_SOURCE;
      status->wrapped.status.MPI_TAG = status->MPI_TAG;
      status->wrapped.status.MPI_ERROR = status->MPI_ERROR;
    }
    return &status->wrapped.status;
  }
};

////////////////////////////////////////////////////////////////////////////////

// Call-back function types

typedef int(MPIwrapper_Comm_copy_attr_function)(MPIwrapper_Comm, int, void *,
                                                void *, void *, int *);
typedef int(MPIwrapper_Comm_delete_attr_function)(MPIwrapper_Comm, int, void *,
                                                  void *);
typedef void(MPIwrapper_Comm_errhandler_function)(MPIwrapper_Comm *, int *,
                                                  ...);
typedef int(MPIwrapper_Copy_function)(MPIwrapper_Comm, int, void *, void *,
                                      void *, int *);
typedef int(MPIwrapper_Delete_function)(MPIwrapper_Comm, int, void *, void *);
typedef void(MPIwrapper_File_errhandler_function)(MPIwrapper_File *, int *,
                                                  ...);
typedef int(MPIwrapper_Type_copy_attr_function)(MPIwrapper_Datatype, int,
                                                void *, void *, void *, int *);
typedef int(MPIwrapper_Type_delete_attr_function)(MPIwrapper_Datatype, int,
                                                  void *, void *);
typedef void(MPIwrapper_User_function)(void *a, void *b, int *len,
                                       MPIwrapper_Datatype *datatype);
typedef int(MPIwrapper_Win_copy_attr_function)(MPIwrapper_Win, int, void *,
                                               void *, void *, int *);
typedef int(MPIwrapper_Win_delete_attr_function)(MPIwrapper_Win, int, void *,
                                                 void *);
typedef void(MPIwrapper_Win_errhandler_function)(MPIwrapper_Win *, int *, ...);

////////////////////////////////////////////////////////////////////////////////

// Translate user functions for MPI_Op

struct op_translation_t {
  MPIwrapper_User_function *user_function;
  MPI_Op op;
  op_translation_t() : user_function(nullptr) {}
};
const int num_op_translations = 10;
std::array<op_translation_t, num_op_translations> op_translations;

int insert_op_translation(MPIwrapper_User_function *const user_fn) {
  int n = 0;
  while (n < int(op_translations.size()) && op_translations[n].user_function)
    ++n;
  if (n == int(op_translations.size())) {
    fprintf(stderr, "Too many operators defined\n");
    exit(1);
  }
  op_translations[n].user_function = user_fn;
  op_translations[n].op = MPI_OP_NULL;
  return n;
}
void free_op_translation(const MPI_Op op) {
  int n = 0;
  while (n < int(op_translations.size()) && op != op_translations[n].op)
    ++n;
  if (n == int(op_translations.size())) {
    fprintf(stderr, "Could not find operator\n");
    exit(1);
  }
  op_translations[n] = op_translation_t();
}

template <int N>
void wrapper_function(void *a, void *b, int *len, MPI_Datatype *datatype) {
  op_translations[N].user_function(a, b, len, (MPIwrapper_Datatype *)datatype);
}
const std::array<MPI_User_function *, num_op_translations> wrapper_functions{
    wrapper_function<0>, wrapper_function<1>, wrapper_function<2>,
    wrapper_function<3>, wrapper_function<4>, wrapper_function<5>,
    wrapper_function<6>, wrapper_function<7>, wrapper_function<8>,
    wrapper_function<9>,
};
static_assert(op_translations.size() == wrapper_functions.size());

} // namespace

////////////////////////////////////////////////////////////////////////////////

// Wrap constants

#define MT(TYPE) MPIwrapper_##TYPE
#define CONSTANT(TYPE, NAME)                                                   \
  extern "C" const TYPE MPIWRAPPER_##NAME;                                     \
  const TYPE MPIWRAPPER_##NAME = (TYPE)MPI_##NAME;
#include "mpi-constants.inc"
#undef CONSTANT
#undef MT

////////////////////////////////////////////////////////////////////////////////

// Wrap functions

#define SKIP_MANUAL_FUNCTIONS

#define MT(TYPE) MPIwrapper_##TYPE
#define MP(TYPE) MPI_##TYPE
#define FUNCTION(RTYPE, NAME, PTYPES, PNAMES)                                  \
  extern "C" RTYPE MPIwrapper_##NAME PTYPES { return MPI_##NAME PNAMES; }
#include "mpi-functions.inc"
#undef FUNCTION

// Handle array types manually

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

// 5.9 Global Reduction Operations

extern "C" int MT(Op_create)(MT(User_function) * user_fn, int commute,
                             MT(Op) * op) {
  const int n = insert_op_translation(user_fn);
  const int ierr = MP(Op_create)(wrapper_functions[n], commute, (MP(Op) *)op);
  op_translations[n].op = *op;
  return ierr;
}

extern "C" int MT(Op_free)(MT(Op) * op) {
  free_op_translation(*op);
  const int ierr = MP(Op_free)((MP(Op) *)op);
  return ierr;
}

#undef MT
#undef MP

#undef SKIP_MANUAL_FUNCTIONS
