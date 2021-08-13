#ifndef MPIWRAPPER_H
#define MPIWRAPPER_H

#include <mpi.h>

#include <stdint.h>

#include <type_traits>

// We implement this file in C++ so that we can automatically convert between
// the `MPI` and `MPIwrapper` types

////////////////////////////////////////////////////////////////////////////////

// Simple types

typedef uintptr_t WPI_Aint;
static_assert(sizeof(MPI_Aint) == sizeof(WPI_Aint));

typedef int64_t WPI_Count;
static_assert(sizeof(MPI_Count) == sizeof(WPI_Count));

typedef int WPI_Fint;
static_assert(sizeof(MPI_Fint) == sizeof(WPI_Fint));

typedef int64_t WPI_Offset;
static_assert(sizeof(MPI_Offset) == sizeof(WPI_Offset));

////////////////////////////////////////////////////////////////////////////////

// Handles

union WPI_Comm {
  MPI_Comm comm;
  uintptr_t padding;

  WPI_Comm() = default;
  WPI_Comm(MPI_Comm comm_) : comm(comm_) {}
  operator MPI_Comm() const { return comm; }
};
static_assert(sizeof(MPI_Comm) <= sizeof(WPI_Comm));

union WPI_Datatype {
  MPI_Datatype datatype;
  uintptr_t padding;

  WPI_Datatype() = default;
  WPI_Datatype(MPI_Datatype datatype_) : datatype(datatype_) {}
  operator MPI_Datatype() const { return datatype; }
};
static_assert(sizeof(MPI_Datatype) <= sizeof(WPI_Datatype));

union WPI_Errhandler {
  MPI_Errhandler errhandler;
  uintptr_t padding;

  WPI_Errhandler() = default;
  WPI_Errhandler(MPI_Errhandler errhandler_) : errhandler(errhandler_) {}
  operator MPI_Errhandler() const { return errhandler; }
};
static_assert(sizeof(MPI_Errhandler) <= sizeof(WPI_Errhandler));

union WPI_File {
  MPI_File file;
  uintptr_t padding;

  WPI_File() = default;
  WPI_File(MPI_File file_) : file(file_) {}
  operator MPI_File() const { return file; }
};
static_assert(sizeof(MPI_File) <= sizeof(WPI_File));

union WPI_Group {
  MPI_Group group;
  uintptr_t padding;

  WPI_Group() = default;
  WPI_Group(MPI_Group group_) : group(group_) {}
  operator MPI_Group() const { return group; }
};
static_assert(sizeof(MPI_Group) <= sizeof(WPI_Group));

union WPI_Info {
  MPI_Info info;
  uintptr_t padding;

  WPI_Info() = default;
  WPI_Info(MPI_Info info_) : info(info_) {}
  operator MPI_Info() const { return info; }
};
static_assert(sizeof(MPI_Info) <= sizeof(WPI_Info));

union WPI_Message {
  MPI_Message message;
  uintptr_t padding;

  WPI_Message() = default;
  WPI_Message(MPI_Message message_) : message(message_) {}
  operator MPI_Message() const { return message; }
};
static_assert(sizeof(MPI_Message) <= sizeof(WPI_Message));

union WPI_Op {
  MPI_Op op;
  uintptr_t padding;

  WPI_Op() = default;
  WPI_Op(MPI_Op op_) : op(op_) {}
  operator MPI_Op() const { return op; }
};
static_assert(sizeof(MPI_Op) <= sizeof(WPI_Op));

union WPI_Request {
  MPI_Request request;
  uintptr_t padding;

  WPI_Request() = default;
  WPI_Request(MPI_Request request_) : request(request_) {}
  operator MPI_Request() const { return request; }
};
static_assert(sizeof(MPI_Request) <= sizeof(WPI_Request));

union WPI_Win {
  MPI_Win win;
  uintptr_t padding;

  WPI_Win() = default;
  WPI_Win(MPI_Win win_) : win(win_) {}
  operator MPI_Win() const { return win; }
};
static_assert(sizeof(MPI_Win) <= sizeof(WPI_Win));

////////////////////////////////////////////////////////////////////////////////

// MPI_Status
// This is a difficult type to wrap since it has user-accessible fields.

#define WPI_STATUS_SIZE 10

// We put the MPI_Status struct in the beginning. This way, we can cast between
// MPI_Status* and WPI_Status*.
typedef struct WPI_Status {
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

  WPI_Status() = default;
  WPI_Status(const MPI_Status &status)
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
} WPI_Status;

static_assert(std::is_pod<WPI_Status>::value);
static_assert(sizeof WPI_Status::wrapped >= sizeof(MPI_Status));
static_assert(WPI_STATUS_SIZE * sizeof(WPI_Fint) == sizeof(WPI_Status));

typedef WPI_Status *WPI_StatusPtr;
typedef const WPI_Status *WPI_const_StatusPtr;

struct MPI_StatusPtr {
  WPI_Status *wrapper_status;

  MPI_StatusPtr(WPI_Status *wrapper_status_) : wrapper_status(wrapper_status_) {
    wrapper_status->from_wrapper();
  }
  ~MPI_StatusPtr() { wrapper_status->to_wrapper(); }
  operator MPI_Status *() const { return &wrapper_status->wrapped.status; }
};

struct MPI_const_StatusPtr {
  const WPI_Status *wrapper_status;

  MPI_const_StatusPtr(const WPI_Status *wrapper_status_)
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

typedef int(WPI_Comm_copy_attr_function)(WPI_Comm, int, void *, void *, void *,
                                         int *);
typedef int(WPI_Comm_delete_attr_function)(WPI_Comm, int, void *, void *);
typedef void(WPI_Comm_errhandler_function)(WPI_Comm *, int *, ...);
typedef int(WPI_Copy_function)(WPI_Comm, int, void *, void *, void *, int *);
typedef int(WPI_Delete_function)(WPI_Comm, int, void *, void *);
typedef void(WPI_File_errhandler_function)(WPI_File *, int *, ...);
typedef int(WPI_Type_copy_attr_function)(WPI_Datatype, int, void *, void *,
                                         void *, int *);
typedef int(WPI_Type_delete_attr_function)(WPI_Datatype, int, void *, void *);
typedef void(WPI_User_function)(void *a, void *b, int *len,
                                WPI_Datatype *datatype);
typedef int(WPI_Win_copy_attr_function)(WPI_Win, int, void *, void *, void *,
                                        int *);
typedef int(WPI_Win_delete_attr_function)(WPI_Win, int, void *, void *);
typedef void(WPI_Win_errhandler_function)(WPI_Win *, int *, ...);

#endif // #ifndef MPIWRAPPER_H
