      subroutine mpiwrapper_export_fortran_constants
      implicit none
      include "mpif.h"

!     Assert that MPI_STATUS_SIZE = 6
      integer cond
      parameter (cond = MPI_STATUS_SIZE - 6)
      integer(1 - abs(cond)) check

      include "mpiwrapper_definitions_fortran.h"

      end
