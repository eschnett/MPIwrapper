#!/usr/bin/env python3

import os
import re
from string import Template
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "mpiabi"))

from mpi_constants import constants
from mpi_functions import functions
from mpi_constants_fortran import constants_fortran
from mpi_functions_fortran import functions_fortran

with open("src/mpiabi_decl_constants_c.h", "w") as file:
    file.write("// Declare C MPI constants\n")
    file.write("\n")
    for (tp, nm) in constants:
        subs = {'abi_tp': re.sub(r"MPI(X?)_", r"MPI\1ABI_", tp),
                'abi_nm': re.sub(r"MPI(X?)_", r"MPI\1ABI_", nm)}
        file.write(Template("extern $abi_tp const $abi_nm;\n").substitute(subs))

with open("src/mpiabi_decl_functions_c.h", "w") as file:
    file.write("// Declare C MPI functions\n")
    file.write("\n")
    for (tp, nm, args, flags) in functions:
        subs = {'abi_tp': re.sub(r"MPI(X?)_", r"MPI\1ABI_", tp),
                'abi_nm': re.sub(r"MPI(X?)_", r"MPI\1ABI_", nm)}
        for (i, (atp, anm)) in enumerate(args):
            subs['abi_atp{0}'.format(i)] = re.sub(r"MPI(X?)_", r"MPI\1ABI_", atp)
            subs['anm{0}'.format(i)] = anm
        tmpl = ["$abi_tp $abi_nm("]
        for (i, (atp, anm)) in enumerate(args):
            tmpl.append("  $abi_atp{0} $anm{0},".format(i))
        tmpl[-1] = re.sub(r",?$", "", tmpl[-1])  # remove trailing comma of last argument
        tmpl.append(");")
        file.write(Template("\n".join(tmpl)).substitute(subs))
        file.write("\n")
        file.write("\n")

with open("src/mpiabi_decl_constants_fortran.h", "w") as file:
    file.write("// Declare Fortran MPI constants\n")
    file.write("\n")
    for (tp, nm) in constants_fortran:
        subs = {'abi_tp': re.sub(r"MPI(X?)_\w+", r"MPI\1ABI_Fint", tp),
                'abi_nm': re.sub(r"MPI(X?)_", r"MPI\1ABI_", nm).lower() + "_"}
        file.write(Template("extern $abi_tp const $abi_nm;\n").substitute(subs))

with open("src/mpiabi_decl_functions_fortran.h", "w") as file:
    file.write("// Declare Fortran MPI functions\n")
    file.write("\n")
    num_ierror_args = 0
    for (tp, nm, args) in functions_fortran:
        subs = {'abi_tp': re.sub(r"MPI(X?)_\w+", r"MPI\1ABI_Fint", tp),
                'abi_nm': re.sub(r"MPI(X?)_", r"MPI\1ABI_", nm).lower() + "_"}
        for (i, (atp, anm)) in enumerate(args):
            num_ierror_args += anm == "ierror"
            subs['abi_atp{0}'.format(i)] = re.sub(r"MPI(X?)_\w+", r"MPI\1ABI_Fint", atp)
            subs['anm{0}'.format(i)] = anm
        tmpl = ["$abi_tp $abi_nm("]
        for (i, (atp, anm)) in enumerate(args):
            tmpl.append("  $abi_atp{0} $anm{0},".format(i))
        tmpl[-1] = re.sub(r",?$", "", tmpl[-1])  # remove trailing comma of last argument
        tmpl.append(");")
        # tmpl = ["extern $abi_tp (* const $abi_nm)("]
        # for (i, (atp, anm)) in enumerate(args):
        #     tmpl.append("  $abi_atp{0} $anm{0},".format(i))
        # tmpl[-1] = re.sub(r",?$", "", tmpl[-1])  # remove trailing comma of last argument
        # tmpl.append(");")
        file.write(Template("\n".join(tmpl)).substitute(subs))
        file.write("\n")
        file.write("\n")
    assert num_ierror_args == 0 if nm in ["MPI_Wtime", "MPI_Wtick"] else 1
