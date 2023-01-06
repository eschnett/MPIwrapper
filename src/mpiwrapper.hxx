#ifndef MPIWRAPPER_HXX
#define MPIWRAPPER_HXX

#include "mpiwrapper_version.h"

#include "mpiabi.h"

#include <mpi.h>
#if HAVE_MPI_EXT
#include <mpi-ext.h>
#endif

#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>

// We implement this file in C++ so that we can automatically convert
// between the `MPI` and `MPIABI` types. We also use it for thread
// safety.

////////////////////////////////////////////////////////////////////////////////

// Compile-time constants

// Ensure that the MPI standard version is supported.
// We allow the major version to be larger because the MPI standard is
// very conservative in making non-backward-compatible changes.
static_assert((MPI_VERSION == MPIABI_MPI_VERSION &&
               MPI_SUBVERSION >= MPIABI_MPI_SUBVERSION) ||
                  MPI_VERSION >= MPIABI_MPI_VERSION,
              "");

static_assert(MPI_MAX_DATAREP_STRING <= MPIABI_MAX_DATAREP_STRING, "");
static_assert(MPI_MAX_ERROR_STRING <= MPIABI_MAX_ERROR_STRING, "");
static_assert(MPI_MAX_INFO_KEY <= MPIABI_MAX_INFO_KEY, "");
static_assert(MPI_MAX_INFO_VAL <= MPIABI_MAX_INFO_VAL, "");
static_assert(MPI_MAX_LIBRARY_VERSION_STRING <=
                  MPIABI_MAX_LIBRARY_VERSION_STRING,
              "");
static_assert(MPI_MAX_OBJECT_NAME <= MPIABI_MAX_OBJECT_NAME, "");
static_assert(MPI_MAX_PORT_NAME <= MPIABI_MAX_PORT_NAME, "");
static_assert(MPI_MAX_PROCESSOR_NAME <= MPIABI_MAX_PROCESSOR_NAME, "");

// Simple types

typedef MPIABI_Aint WPI_Aint;
static_assert(sizeof(MPI_Aint) == sizeof(WPI_Aint), "");
static_assert(alignof(MPI_Aint) == alignof(WPI_Aint), "");

// TODO:
// - MPICH uses 8 bytes for `MPI_Count` and `MPI_Offset`
// - OpenMPI uses at most `ssize_t` for `MPI_Count` and `MPI_Offset`,
//   which has only 32 bits on 32-bit architectures.
// - We follow the MPICH conventions. This means that MPIwrapper won't
//   build with OpenMPI on 32-bit architectures.
// - We could correct this, and this would require adding code to
//   convert between these integer types both for the C and Fortran
//   functions.
typedef MPIABI_Count WPI_Count;
static_assert(sizeof(MPI_Count) == sizeof(WPI_Count), "");
static_assert(alignof(MPI_Count) == alignof(WPI_Count), "");

typedef MPIABI_Fint WPI_Fint;
static_assert(sizeof(MPI_Fint) == sizeof(WPI_Fint), "");
static_assert(alignof(MPI_Fint) == alignof(WPI_Fint), "");

typedef MPIABI_Offset WPI_Offset;
static_assert(sizeof(MPI_Offset) == sizeof(WPI_Offset), "");
static_assert(alignof(MPI_Offset) == alignof(WPI_Offset), "");

////////////////////////////////////////////////////////////////////////////////

// Handles

template <typename MPI_Handle> struct WPI_Handle {
  uintptr_t wpi_handle;

  // We need to initialize the padding to ensure that handles can be compared
  // for equality
  void pad() { wpi_handle = (uintptr_t)(MPI_Handle)wpi_handle; }

  WPI_Handle() = default;

  // Convert from and to MPI: Handle padding
  WPI_Handle(MPI_Handle mpi_handle) : wpi_handle((uintptr_t)mpi_handle) {}
  operator MPI_Handle() const { return (MPI_Handle)wpi_handle; }

  WPI_Handle(uintptr_t wrapped) : wpi_handle(wrapped) {}
  operator uintptr_t() const { return wpi_handle; }
};

template <typename MPI_Handle> struct WPI_HandlePtr {
  WPI_Handle<MPI_Handle> *wrapped_handle_ptr;

  WPI_HandlePtr() = default;

  // Pad the wrapped handle. This is necessary if the handle was not initialized
  // by the application, the MPI library did set the handle, and the MPI wrapper
  // is now returning to the application.
  ~WPI_HandlePtr() { wrapped_handle_ptr->pad(); }

  // Convert from and to MPI: Cast pointer and handle padding. We assume that
  // the pointed-to handle is actually a wrapped handle, not just an MPI handle.
  WPI_HandlePtr(MPI_Handle *mpi_handle_ptr)
      : wrapped_handle_ptr((WPI_Handle<MPI_Handle> *)mpi_handle_ptr) {
    wrapped_handle_ptr->pad();
  }
  operator MPI_Handle *() const { return (MPI_Handle *)wrapped_handle_ptr; }

  // Convert from and to MPIABI: Cast pointer
  WPI_HandlePtr(uintptr_t *abi_handle_ptr)
      : wrapped_handle_ptr((WPI_Handle<MPI_Handle> *)abi_handle_ptr) {}
  operator uintptr_t *() const { return (uintptr_t *)wrapped_handle_ptr; }

  // These checks should logically be in `WPI_Handle`. However, C++ doesn't
  // allow them there because the type `WPI_Handle` isn't complete there, so we
  // put them here.
  static_assert(sizeof(MPI_Handle) <= sizeof(WPI_Handle<MPI_Handle>), "");
  static_assert(alignof(MPI_Handle) <= alignof(WPI_Handle<MPI_Handle>), "");
  static_assert(sizeof(WPI_Handle<MPI_Handle>) == sizeof(uintptr_t), "");
  static_assert(alignof(WPI_Handle<MPI_Handle>) == alignof(uintptr_t), "");
  static_assert(std::is_pod<WPI_Handle<MPI_Handle>>::value, "");
};

template <typename MPI_Handle> struct WPI_const_HandlePtr {
  const WPI_Handle<MPI_Handle> *wrapped_handle_ptr;

  WPI_const_HandlePtr() = default;

  // Convert from and to MPI: Cast pointer.
  WPI_const_HandlePtr(const MPI_Handle *mpi_handle_ptr)
      : wrapped_handle_ptr((const WPI_Handle<MPI_Handle> *)mpi_handle_ptr) {}
  operator const MPI_Handle *() const {
    return (const MPI_Handle *)wrapped_handle_ptr;
  }

  // Convert from and to MPIABI: Cast pointer
  WPI_const_HandlePtr(const uintptr_t *abi_handle_ptr)
      : wrapped_handle_ptr((const WPI_Handle<MPI_Handle> *)abi_handle_ptr) {}
  operator const uintptr_t *() const {
    return (const uintptr_t *)wrapped_handle_ptr;
  }
};

typedef MPI_Comm *MPI_CommPtr;
typedef WPI_Handle<MPI_Comm> WPI_Comm;
typedef WPI_HandlePtr<MPI_Comm> WPI_CommPtr;
typedef WPI_const_HandlePtr<MPI_Comm> WPI_const_CommPtr;

typedef MPI_Datatype *MPI_DatatypePtr;
typedef WPI_Handle<MPI_Datatype> WPI_Datatype;
typedef WPI_HandlePtr<MPI_Datatype> WPI_DatatypePtr;
typedef WPI_const_HandlePtr<MPI_Datatype> WPI_const_DatatypePtr;

typedef MPI_Errhandler *MPI_ErrhandlerPtr;
typedef WPI_Handle<MPI_Errhandler> WPI_Errhandler;
typedef WPI_HandlePtr<MPI_Errhandler> WPI_ErrhandlerPtr;
typedef WPI_const_HandlePtr<MPI_Errhandler> WPI_const_ErrhandlerPtr;

typedef MPI_File *MPI_FilePtr;
typedef WPI_Handle<MPI_File> WPI_File;
typedef WPI_HandlePtr<MPI_File> WPI_FilePtr;
typedef WPI_const_HandlePtr<MPI_File> WPI_const_FilePtr;

typedef MPI_Group *MPI_GroupPtr;
typedef WPI_Handle<MPI_Group> WPI_Group;
typedef WPI_HandlePtr<MPI_Group> WPI_GroupPtr;
typedef WPI_const_HandlePtr<MPI_Group> WPI_const_GroupPtr;

typedef MPI_Info *MPI_InfoPtr;
typedef WPI_Handle<MPI_Info> WPI_Info;
typedef WPI_HandlePtr<MPI_Info> WPI_InfoPtr;
typedef WPI_const_HandlePtr<MPI_Info> WPI_const_InfoPtr;

typedef MPI_Message *MPI_MessagePtr;
typedef WPI_Handle<MPI_Message> WPI_Message;
typedef WPI_HandlePtr<MPI_Message> WPI_MessagePtr;
typedef WPI_const_HandlePtr<MPI_Message> WPI_const_MessagePtr;

typedef MPI_Op *MPI_OpPtr;
typedef WPI_Handle<MPI_Op> WPI_Op;
typedef WPI_HandlePtr<MPI_Op> WPI_OpPtr;
typedef WPI_const_HandlePtr<MPI_Op> WPI_const_OpPtr;

typedef MPI_Request *MPI_RequestPtr;
typedef WPI_Handle<MPI_Request> WPI_Request;
typedef WPI_HandlePtr<MPI_Request> WPI_RequestPtr;
typedef WPI_const_HandlePtr<MPI_Request> WPI_const_RequestPtr;

typedef MPI_Win *MPI_WinPtr;
typedef WPI_Handle<MPI_Win> WPI_Win;
typedef WPI_HandlePtr<MPI_Win> WPI_WinPtr;
typedef WPI_const_HandlePtr<MPI_Win> WPI_const_WinPtr;

////////////////////////////////////////////////////////////////////////////////

// MPI_Status

#define WPI_STATUS_SIZE MPIABI_STATUS_SIZE

// The memory layout of WPI_Status must be identical to MPIABI_Status
struct WPI_Status : MPIABI_Status {
  WPI_Status() = default;

  WPI_Status(const MPIABI_Status &abi_status) : MPIABI_Status(abi_status) {}

  // Convert from and to MPI: Handle fields
  WPI_Status(const MPI_Status &mpi_status) {
    assert(&mpi_status != MPI_STATUS_IGNORE &&
           &mpi_status != MPI_STATUSES_IGNORE);
    memcpy(this, &mpi_status, sizeof mpi_status);
    // Set MPIABI fields from MPI status
    MPI_SOURCE = mpi_status.MPI_SOURCE;
    MPI_TAG = mpi_status.MPI_TAG;
    MPI_ERROR = mpi_status.MPI_ERROR;
  }
  operator MPI_Status() const {
    // Set MPI status from MPIABI fields
    MPI_Status mpi_status;
    memcpy(&mpi_status, this, sizeof mpi_status);
    mpi_status.MPI_SOURCE = MPI_SOURCE;
    mpi_status.MPI_TAG = MPI_TAG;
    mpi_status.MPI_ERROR = MPI_ERROR;
    return mpi_status;
  }
};

static_assert(sizeof(WPI_Status) == sizeof(MPIABI_Status), "");
static_assert(alignof(WPI_Status) == alignof(MPIABI_Status), "");
static_assert(sizeof WPI_Status::mpi_status >= sizeof(MPI_Status), "");
namespace {
using mpi_status_type = decltype(WPI_Status::mpi_status);
static_assert(std::is_union<mpi_status_type>::value, "");
static_assert(alignof(mpi_status_type) >= alignof(MPI_Status), "");
} // namespace
static_assert(sizeof(WPI_Status) >= sizeof(MPI_Status), "");
static_assert(alignof(WPI_Status) >= alignof(MPI_Status), "");
static_assert(WPI_STATUS_SIZE * sizeof(WPI_Fint) == sizeof(WPI_Status), "");
static_assert(std::is_pod<WPI_Status>::value, "");

struct WPI_StatusPtr {
  WPI_Status *wrapped_status_ptr;

  WPI_StatusPtr() = default;

  // Set the wrapped fields. This is necessary if the status was not initialized
  // by the application, the MPI library did set the status, and the MPI wrapper
  // is now returning to the application.
  ~WPI_StatusPtr() {
    MPI_Status *mpi_status_ptr = (MPI_Status *)wrapped_status_ptr;
    if (mpi_status_ptr != MPI_STATUS_IGNORE &&
        mpi_status_ptr != MPI_STATUSES_IGNORE) {
      wrapped_status_ptr->MPI_SOURCE = mpi_status_ptr->MPI_SOURCE;
      wrapped_status_ptr->MPI_TAG = mpi_status_ptr->MPI_TAG;
      wrapped_status_ptr->MPI_ERROR = mpi_status_ptr->MPI_ERROR;
    }
  }

  // Convert from and to MPI: Cast pointer and handle fields. We assume the
  // pointed-to status is a wrapped status, not just an MPI status.
  WPI_StatusPtr(MPI_Status *mpi_status_ptr)
      : wrapped_status_ptr((WPI_Status *)mpi_status_ptr) {
    if (mpi_status_ptr != MPI_STATUS_IGNORE &&
        mpi_status_ptr != MPI_STATUSES_IGNORE) {
      wrapped_status_ptr->MPI_SOURCE = mpi_status_ptr->MPI_SOURCE;
      wrapped_status_ptr->MPI_TAG = mpi_status_ptr->MPI_TAG;
      wrapped_status_ptr->MPI_ERROR = mpi_status_ptr->MPI_ERROR;
    }
  }
  operator MPI_Status *() const { return (MPI_Status *)wrapped_status_ptr; }

  // Convert from and to MPIABI: Cast pointer
  WPI_StatusPtr(MPIABI_Status *abi_status_ptr)
      : wrapped_status_ptr((WPI_Status *)abi_status_ptr) {}
  operator MPIABI_Status *() const {
    return (MPIABI_Status *)wrapped_status_ptr;
  }
};

struct WPI_const_StatusPtr {
  const WPI_Status *wrapped_status_ptr;

  WPI_const_StatusPtr() = default;

  // Set the wrapped fields. This is necessary if the status was not initialized
  // by the application, the MPI library did set the status, and the MPI wrapper
  // is now returning to the application.
  ~WPI_const_StatusPtr() {
    const MPI_Status *mpi_status_ptr = (const MPI_Status *)wrapped_status_ptr;
    if (mpi_status_ptr != MPI_STATUS_IGNORE &&
        mpi_status_ptr != MPI_STATUSES_IGNORE) {
      const_cast<WPI_Status *>(wrapped_status_ptr)->MPI_SOURCE =
          mpi_status_ptr->MPI_SOURCE;
      const_cast<WPI_Status *>(wrapped_status_ptr)->MPI_TAG =
          mpi_status_ptr->MPI_TAG;
      const_cast<WPI_Status *>(wrapped_status_ptr)->MPI_ERROR =
          mpi_status_ptr->MPI_ERROR;
    }
  }

  // Convert from and to MPI: Cast pointer and handle fields. We assume the
  // pointed-to status is a wrapped status, not just an MPI status.
  WPI_const_StatusPtr(const MPI_Status *mpi_status_ptr)
      : wrapped_status_ptr((const WPI_Status *)mpi_status_ptr) {
    if (mpi_status_ptr != MPI_STATUS_IGNORE &&
        mpi_status_ptr != MPI_STATUSES_IGNORE) {
      const_cast<WPI_Status *>(wrapped_status_ptr)->MPI_SOURCE =
          mpi_status_ptr->MPI_SOURCE;
      const_cast<WPI_Status *>(wrapped_status_ptr)->MPI_TAG =
          mpi_status_ptr->MPI_TAG;
      const_cast<WPI_Status *>(wrapped_status_ptr)->MPI_ERROR =
          mpi_status_ptr->MPI_ERROR;
    }
  }
  operator const MPI_Status *() const {
    return (const MPI_Status *)wrapped_status_ptr;
  }

  // Convert from and to MPIABI: Cast pointer
  WPI_const_StatusPtr(const MPIABI_Status *abi_status_ptr)
      : wrapped_status_ptr((const WPI_Status *)abi_status_ptr) {}
  operator const MPIABI_Status *() const {
    return (const MPIABI_Status *)wrapped_status_ptr;
  }
};

////////////////////////////////////////////////////////////////////////////////

// Call-back function types

// TODO: Handle conversions

typedef int WPI_Comm_copy_attr_function(WPI_Comm oldcomm, int comm_keyval,
                                        void *extra_state,
                                        void *attribute_val_in,
                                        void *attribute_val, int *flag);
typedef int WPI_Comm_delete_attr_function(WPI_Comm comm, int comm_keyval,
                                          void *attribute_val,
                                          void *extra_state);
typedef void WPI_Comm_errhandler_function(WPI_Comm *, int *, ...);
typedef WPI_Comm_errhandler_function WPI_Comm_errhandler_fn;
typedef WPI_Comm_copy_attr_function WPI_Copy_function;
typedef int WPI_Datarep_conversion_function(void *userbuf,
                                            WPI_Datatype datatype, int count,
                                            void *filebuf, WPI_Offset position,
                                            void *extra_state);
typedef int WPI_Datarep_extent_function(WPI_Datatype datatype,
                                        WPI_Aint *file_extent,
                                        void *extra_state);
typedef WPI_Comm_delete_attr_function WPI_Delete_function;
typedef void WPI_File_errhandler_function(WPI_File *, int *, ...);
typedef WPI_File_errhandler_function WPI_File_errhandler_fn;
typedef int WPI_Grequest_cancel_function(void *extra_state, int complete);
typedef int WPI_Grequest_free_function(void *extra_state);
typedef int WPI_Grequest_query_function(void *extra_state, WPI_Status *status);
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
typedef WPI_Win_errhandler_function WPI_Win_errhandler_fn;

// Declarations

#ifdef __cplusplus
extern "C" {
#endif

#include "mpiabi_decl_constants_c.h"
#include "mpiabi_decl_constants_fortran.h"
#include "mpiabi_decl_functions_c.h"
#include "mpiabi_decl_functions_fortran.h"

#ifdef __cplusplus
}
#endif

#endif // #ifndef MPIWRAPPER_HXX
