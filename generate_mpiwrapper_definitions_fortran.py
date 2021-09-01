#!/usr/bin/env python3

import re
from string import Template

from mpi_constants_fortran import constants_fortran

def wrap(line):
    lines = []
    while len(line) > 72:
        lines.append(line[0:72] + "&")
        line = "     &" + line[72:]
    lines.append(line)
    return "\n".join(lines)

print()
print("!     Fortran constants")

# Declarations
for (tp, nm) in constants_fortran:
    subs = {'mpi_nm': nm,
            'abi_nm': re.sub(r"MPI_", "MPIABI_", nm)}
    tmpl = []
    tmpl.append("      integer $abi_nm")
    tmpl.append("      common /$abi_nm/ $abi_nm")
    print("\n".join(map(lambda line: wrap(Template(line).substitute(subs)), tmpl)))

# Definitions
for (tp, nm) in constants_fortran:
    subs = {'mpi_nm': nm,
            'abi_nm': re.sub(r"MPI_", "MPIABI_", nm)}
    tmpl = []
    tmpl.append("      $abi_nm = $mpi_nm")
    print("\n".join(map(lambda line: wrap(Template(line).substitute(subs)), tmpl)))
