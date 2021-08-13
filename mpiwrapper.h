#ifndef MPIWRAPPER_H
#define MPIWRAPPER_H

#include <mpi.h>

#include <stdint.h>

#include <type_traits>

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
// This is a difficult type to wrap since it has user-accessible fields.

#define MPIWRAPPER_STATUS_SIZE 10

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
    } padding_OpenMPI; // also Spectrum MPI
    struct {
      int f0;
      int f1;
      int f2;
      int f3;
      int f4;
    } padding_MPICH; // also Intel MPI
  } wrapped;
  mutable int MPI_SOURCE;
  mutable int MPI_TAG;
  mutable int MPI_ERROR;

  MPIwrapper_Status() = default;
  MPIwrapper_Status(const MPI_Status &status)
      : MPI_SOURCE(status.MPI_SOURCE), MPI_TAG(status.MPI_TAG),
        MPI_ERROR(status.MPI_ERROR) {
    wrapped.status = status;
  }
  operator MPI_Status() const {
    wrapped.status.MPI_SOURCE = MPI_SOURCE;
    wrapped.status.MPI_TAG = MPI_TAG;
    wrapped.status.MPI_ERROR = MPI_ERROR;
    return wrapped.status;
  }

  void to_wrapper() const {
    if (&wrapped.status != MPI_STATUS_IGNORE &&
        &wrapped.status != MPI_STATUSES_IGNORE) {
      MPI_SOURCE = wrapped.status.MPI_SOURCE;
      MPI_TAG = wrapped.status.MPI_TAG;
      MPI_ERROR = wrapped.status.MPI_ERROR;
    }
  }
  void from_wrapper() const {
    if (&wrapped.status != MPI_STATUS_IGNORE &&
        &wrapped.status != MPI_STATUSES_IGNORE) {
      wrapped.status.MPI_SOURCE = MPI_SOURCE;
      wrapped.status.MPI_TAG = MPI_TAG;
      wrapped.status.MPI_ERROR = MPI_ERROR;
    }
  }
} MPIwrapper_Status;

static_assert(std::is_pod<MPIwrapper_Status>::value);
static_assert(sizeof MPIwrapper_Status::wrapped >= sizeof(MPI_Status));
static_assert(MPIWRAPPER_STATUS_SIZE * sizeof(MPIwrapper_Fint) ==
              sizeof(MPIwrapper_Status));

typedef MPIwrapper_Status *MPIwrapper_StatusPtr;
typedef const MPIwrapper_Status *MPIwrapper_const_StatusPtr;

struct MPI_StatusPtr {
  MPIwrapper_Status *wrapper_status;

  MPI_StatusPtr(MPIwrapper_Status *wrapper_status_)
      : wrapper_status(wrapper_status_) {
    wrapper_status->from_wrapper();
  }
  ~MPI_StatusPtr() { wrapper_status->to_wrapper(); }
  operator MPI_Status *() const { return &wrapper_status->wrapped.status; }
};

struct MPI_const_StatusPtr {
  const MPIwrapper_Status *wrapper_status;

  MPI_const_StatusPtr(const MPIwrapper_Status *wrapper_status_)
      : wrapper_status(wrapper_status_) {
    wrapper_status->from_wrapper();
  }
  ~MPI_const_StatusPtr() { wrapper_status->to_wrapper(); }
  operator const MPI_Status *() const {
    return &wrapper_status->wrapped.status;
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

#endif // #ifndef MPIWRAPPER_H
