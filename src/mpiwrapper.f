      subroutine mpiwrapper_export_fortran_constants
      implicit none
      include "mpif.h"

!     Assert that MPI_STATUS_SIZE <= 6
      integer cond
      parameter (cond = max(0, MPI_STATUS_SIZE - 6))
      integer(1 - abs(cond)) check

      include "mpiabi_defn_constants_fortran.h"

      call mpiwrapper_store_sentinels(
     &     MPI_STATUS_IGNORE,
     &     MPI_STATUSES_IGNORE)

      end