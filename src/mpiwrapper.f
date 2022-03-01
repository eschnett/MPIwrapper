      subroutine mpiwrapper_export_fortran_constants
      implicit none

      include "mpif.h"
      include "mpiabif.h"

!     Assert that MPI_STATUS_SIZE <= MPIABI_STATUS_SIZE
      integer(1 - max(0, MPI_STATUS_SIZE - MPIABI_STATUS_SIZE)) check_mpi_status_size

!     integer(1 - abs(MPI_ADDRESS_KIND - 4)) check_mpi_address_kind
      integer(1 - abs(MPI_COUNT_KIND - 8)) check_mpi_count_kind
      integer(1 - abs(MPI_INTEGER_KIND - 4)) check_mpi_integer_kind
      integer(1 - abs(MPI_OFFSET_KIND - 8)) check_mpi_offset_kind

      include "mpiabi_defn_constants_fortran.h"

      call mpiwrapper_store_sentinels(
     &     MPI_STATUS_IGNORE,
     &     MPI_STATUSES_IGNORE)

      end
