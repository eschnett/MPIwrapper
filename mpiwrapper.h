#ifndef MPIWRAPPER_H
#define MPIWRAPPER_H

#include <mpi.h>

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <type_traits>

// We implement this file in C++ so that we can automatically convert between
// the `MPI` and `MPIwrapper` types

////////////////////////////////////////////////////////////////////////////////

// Simple types

typedef intptr_t WPI_Aint;
static_assert(sizeof(MPI_Aint) == sizeof(WPI_Aint), "");
static_assert(alignof(MPI_Aint) == alignof(WPI_Aint), "");

typedef int64_t WPI_Count;
static_assert(sizeof(MPI_Count) == sizeof(WPI_Count), "");
static_assert(alignof(MPI_Count) == alignof(WPI_Count), "");

typedef int WPI_Fint;
static_assert(sizeof(MPI_Fint) == sizeof(WPI_Fint), "");
static_assert(alignof(MPI_Fint) == alignof(WPI_Fint), "");

typedef int64_t WPI_Offset;
static_assert(sizeof(MPI_Offset) == sizeof(WPI_Offset), "");
static_assert(alignof(MPI_Offset) == alignof(WPI_Offset), "");

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
  WPI_Handle<MPI_T> *wrapped_handleptr;

  WPI_HandlePtr() = default;
  WPI_HandlePtr(MPI_T *handleptr)
      : wrapped_handleptr((WPI_Handle<MPI_T> *)handleptr) {
    wrapped_handleptr->pad();
  }
  operator MPI_T *() const { return &wrapped_handleptr->handle; }

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
    } padding_OpenMPI; // also IBM Spectrum MPI
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
  mutable bool is_ignore;

  WPI_Status() = default;
  WPI_Status(const MPI_Status &status) {
    assert((const MPI_Status *)this != MPI_STATUS_IGNORE &&
           (const MPI_Status *)this != MPI_STATUSES_IGNORE);
    wrapped.status = status;
    MPI_SOURCE = wrapped.status.MPI_SOURCE;
    MPI_TAG = wrapped.status.MPI_TAG;
    MPI_ERROR = wrapped.status.MPI_ERROR;
  }
  operator MPI_Status() const {
    assert((const MPI_Status *)this != MPI_STATUS_IGNORE &&
           (const MPI_Status *)this != MPI_STATUSES_IGNORE);
    wrapped.status.MPI_SOURCE = MPI_SOURCE;
    wrapped.status.MPI_TAG = MPI_TAG;
    wrapped.status.MPI_ERROR = MPI_ERROR;
    return wrapped.status;
  }
} WPI_Status;

static_assert(sizeof WPI_Status::wrapped >= sizeof(MPI_Status), "");
static_assert(alignof WPI_Status::wrapped >= alignof(MPI_Status), "");
static_assert(sizeof(WPI_Status) >= sizeof(MPI_Status), "");
static_assert(alignof(WPI_Status) >= alignof(MPI_Status), "");
static_assert(WPI_STATUS_SIZE * sizeof(WPI_Fint) == sizeof(WPI_Status), "");
static_assert(std::is_pod<WPI_Status>::value, "");

typedef WPI_Status *WPI_StatusPtr;
typedef const WPI_Status *WPI_const_StatusPtr;

struct MPI_StatusPtr {
  WPI_Status *statusptr;

  MPI_StatusPtr(WPI_Status *statusptr_) : statusptr(statusptr_) {}
  ~MPI_StatusPtr() {
    if ((MPI_Status *)statusptr != MPI_STATUS_IGNORE &&
        (MPI_Status *)statusptr != MPI_STATUSES_IGNORE) {
      statusptr->MPI_SOURCE = statusptr->wrapped.status.MPI_SOURCE;
      statusptr->MPI_TAG = statusptr->wrapped.status.MPI_TAG;
      statusptr->MPI_ERROR = statusptr->wrapped.status.MPI_ERROR;
    }
  }
  operator MPI_Status *() const {
    if ((MPI_Status *)statusptr == MPI_STATUS_IGNORE ||
        (MPI_Status *)statusptr == MPI_STATUSES_IGNORE)
      return MPI_STATUS_IGNORE;
    statusptr->wrapped.status.MPI_SOURCE = statusptr->MPI_SOURCE;
    statusptr->wrapped.status.MPI_TAG = statusptr->MPI_TAG;
    statusptr->wrapped.status.MPI_ERROR = statusptr->MPI_ERROR;
    return &statusptr->wrapped.status;
  }
};

struct MPI_const_StatusPtr {
  const WPI_Status *statusptr;

  MPI_const_StatusPtr(const WPI_Status *statusptr_) : statusptr(statusptr_) {}
  ~MPI_const_StatusPtr() {
    if ((const MPI_Status *)statusptr != MPI_STATUS_IGNORE &&
        (const MPI_Status *)statusptr != MPI_STATUSES_IGNORE) {
      statusptr->MPI_SOURCE = statusptr->wrapped.status.MPI_SOURCE;
      statusptr->MPI_TAG = statusptr->wrapped.status.MPI_TAG;
      statusptr->MPI_ERROR = statusptr->wrapped.status.MPI_ERROR;
    }
  }
  operator const MPI_Status *() const {
    if ((const MPI_Status *)statusptr == MPI_STATUS_IGNORE ||
        (const MPI_Status *)statusptr == MPI_STATUSES_IGNORE)
      return MPI_STATUS_IGNORE;
    statusptr->wrapped.status.MPI_SOURCE = statusptr->MPI_SOURCE;
    statusptr->wrapped.status.MPI_TAG = statusptr->MPI_TAG;
    statusptr->wrapped.status.MPI_ERROR = statusptr->MPI_ERROR;
    return &statusptr->wrapped.status;
  }
};

////////////////////////////////////////////////////////////////////////////////

// Call-back function types

typedef int WPI_Comm_copy_attr_function(WPI_Comm oldcomm, int comm_keyval,
                                        void *extra_state,
                                        void *attribute_val_in,
                                        void *attribute_val, int *flag);
typedef int WPI_Comm_delete_attr_function(WPI_Comm comm, int comm_keyval,
                                          void *attribute_val,
                                          void *extra_state);
typedef void WPI_Comm_errhandler_function(WPI_Comm *, int *, ...);
typedef WPI_Comm_copy_attr_function WPI_Copy_function;
#if 0
// TODO: Handle conversions
typedef int WPI_Datarep_conversion_function(void *userbuf,
                                            WPI_Datatype datatype, int count,
                                            void *filebuf, WPI_Offset position,
                                            void *extra_state);
typedef int WPI_Datarep_extent_function(WPI_Datatype datatype,
                                        WPI_Aint *file_extent,
                                        void *extra_state);
#endif
typedef WPI_Comm_delete_attr_function WPI_Delete_function;
typedef void WPI_File_errhandler_function(WPI_File *, int *, ...);
typedef int WPI_Grequest_cancel_function(void *extra_state, int complete);
typedef int WPI_Grequest_free_function(void *extra_state);
#if 0
// TODO: Handle status correctly
typedef int WPI_Grequest_query_function(void *extra_state, WPI_Status *status);
#endif
typedef int WPI_Type_copy_attr_function(WPI_Datatype oldtype, int type_keyval,
                                        void *extra_state,
                                        void *attribute_val_in,
                                        void *attribute_val_out, int *flag);
typedef int WPI_Type_delete_attr_function(WPI_Datatype datatype,
                                          int type_keyval, void *attribute_val,
                                          void *extra_state);
typedef void WPI_User_function(void *invec, void *inoutvec, int *len,
                               WPI_Datatype *datatype);
typedef int WPI_Win_copy_attr_function(WPI_Win oldwin, int win_keyval,
                                       void *extra_state,
                                       void *attribute_val_in,
                                       void *attribute_val_out, int *flag);
typedef int WPI_Win_delete_attr_function(WPI_Win win, int win_keyval,
                                         void *attribute_val,
                                         void *extra_state);
typedef void WPI_Win_errhandler_function(WPI_Win *, int *, ...);

#endif // #ifndef MPIWRAPPER_H
