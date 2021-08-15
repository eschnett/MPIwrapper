#ifndef MPIWRAPPER_H
#define MPIWRAPPER_H

#include <mpi.h>

#include <stdint.h>
#include <string.h>

#include <type_traits>

// We implement this file in C++ so that we can automatically convert between
// the `MPI` and `MPIwrapper` types

////////////////////////////////////////////////////////////////////////////////

// Simple types

typedef intptr_t WPI_Aint;
static_assert(sizeof(MPI_Aint) == sizeof(WPI_Aint));

typedef int64_t WPI_Count;
static_assert(sizeof(MPI_Count) == sizeof(WPI_Count));

typedef int WPI_Fint;
static_assert(sizeof(MPI_Fint) == sizeof(WPI_Fint));

typedef int64_t WPI_Offset;
static_assert(sizeof(MPI_Offset) == sizeof(WPI_Offset));

////////////////////////////////////////////////////////////////////////////////

// Handles

template <typename MPI_T> struct alignas(alignof(uintptr_t)) WPI_Handle {
  MPI_T handle;
  uint8_t padding[sizeof(uintptr_t) - sizeof(MPI_T)];
  void pad() { memset(padding, 0, sizeof padding); }

  WPI_Handle() = default;
  WPI_Handle(MPI_T handle_) : handle(handle_) { pad(); }
  operator MPI_T() const { return handle; }
};

template <typename MPI_T> struct WPI_HandlePtr {
  WPI_Handle<MPI_T> *whandleptr;

  WPI_HandlePtr() = default;
  WPI_HandlePtr(MPI_T *handleptr) : whandleptr((WPI_Handle<MPI_T> *)handleptr) {
    whandleptr->pad();
  }
  operator MPI_T *() const { return &whandleptr->handle; }

  // These checks should logically be in `WPI_Handle`. However, C++ doesn't
  // allow them there because the type `WPI_Handle` isn't complete yet, so we
  // put them here.
  static_assert(sizeof(MPI_T) <= sizeof(WPI_Handle<MPI_T>), "");
  static_assert(alignof(MPI_T) <= alignof(WPI_Handle<MPI_T>), "");
  static_assert(sizeof(WPI_Handle<MPI_T>) == sizeof(uintptr_t), "");
  static_assert(alignof(WPI_Handle<MPI_T>) == alignof(uintptr_t), "");
  static_assert(std::is_pod<WPI_Handle<MPI_T>>::value, "");
};

typedef MPI_Comm *MPI_CommPtr;
typedef WPI_Handle<MPI_Comm> WPI_Comm;
typedef WPI_HandlePtr<MPI_Comm> WPI_CommPtr;

typedef MPI_Datatype *MPI_DatatypePtr;
typedef WPI_Handle<MPI_Datatype> WPI_Datatype;
typedef WPI_HandlePtr<MPI_Datatype> WPI_DatatypePtr;

typedef MPI_Errhandler *MPI_ErrhandlerPtr;
typedef WPI_Handle<MPI_Errhandler> WPI_Errhandler;
typedef WPI_HandlePtr<MPI_Errhandler> WPI_ErrhandlerPtr;

typedef MPI_File *MPI_FilePtr;
typedef WPI_Handle<MPI_File> WPI_File;
typedef WPI_HandlePtr<MPI_File> WPI_FilePtr;

typedef MPI_Group *MPI_GroupPtr;
typedef WPI_Handle<MPI_Group> WPI_Group;
typedef WPI_HandlePtr<MPI_Group> WPI_GroupPtr;

typedef MPI_Info *MPI_InfoPtr;
typedef WPI_Handle<MPI_Info> WPI_Info;
typedef WPI_HandlePtr<MPI_Info> WPI_InfoPtr;

typedef MPI_Message *MPI_MessagePtr;
typedef WPI_Handle<MPI_Message> WPI_Message;
typedef WPI_HandlePtr<MPI_Message> WPI_MessagePtr;

typedef MPI_Op *MPI_OpPtr;
typedef WPI_Handle<MPI_Op> WPI_Op;
typedef WPI_HandlePtr<MPI_Op> WPI_OpPtr;

typedef MPI_Request *MPI_RequestPtr;
typedef WPI_Handle<MPI_Request> WPI_Request;
typedef WPI_HandlePtr<MPI_Request> WPI_RequestPtr;

typedef MPI_Win *MPI_WinPtr;
typedef WPI_Handle<MPI_Win> WPI_Win;
typedef WPI_HandlePtr<MPI_Win> WPI_WinPtr;

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
