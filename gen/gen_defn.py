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

def wpi_type(tp):
    tp = re.sub(r"MPI(X?)_", r"WPI\1_", tp)
    tp = re.sub(r"WPI(X?)_(Comm|Datatype|Errhandler|File|Group|Info|Message|Op|Request|Status|Win) \*", r"WPI\1_\2Ptr", tp)
    tp = re.sub(r"const WPI(X?)_(Comm|Datatype|Errhandler|File|Group|Info|Message|Op|Request|Status|Win)", r"WPI\1_const_\2", tp)
    return tp

def wrap(line):
    lines = []
    while len(line) > 72:
        lines.append(line[0:72] + "&")
        line = "     &" + line[72:]
    lines.append(line)
    return "\n".join(lines)

with open("src/mpiabi_defn_constants_c.h", "w") as file:
    file.write("// Define C MPI constants\n")
    file.write("\n")
    for (tp, nm) in constants:
        subs = {'mpi_tp': tp,
                'abi_tp': re.sub(r"MPI(X?)_", r"MPI\1ABI_", tp),
                'mpi_nm': nm,
                'abi_nm': re.sub(r"MPI(X?)_", r"MPI\1ABI_", nm)}
        rcast = "($abi_tp)" if re.search(r"MPI(X?)_", tp) else ""
        file.write(Template("$abi_tp const $abi_nm = "+rcast+"$mpi_nm;\n").substitute(subs))

with open("src/mpiabi_defn_functions_c.h", "w") as file:
    file.write("// Define C MPI functions\n")
    file.write("\n")
    for (tp, nm, args, flags) in functions:
        assert flags is None or flags == "manual"
        if flags == "manual":
            continue
        subs = {'mpi_tp': tp,
                'wpi_tp': re.sub(r"MPI(X?)_", r"WPI\1_", tp),
                'abi_tp': re.sub(r"MPI(X?)_", r"MPI\1ABI_", tp),
                'mpi_nm': nm,
                'abi_nm': re.sub(r"MPI(X?)_", r"MPI\1ABI_", nm)}
        for (i, (atp, anm)) in enumerate(args):
            subs['mpi_atp{0}'.format(i)] = atp
            subs['wpi_atp{0}'.format(i)] = wpi_type(atp)
            subs['abi_atp{0}'.format(i)] = re.sub(r"MPI(X?)_", r"MPI\1ABI_", atp)
            subs['anm{0}'.format(i)] = anm
        tmpl = ["$abi_tp $abi_nm("]
        for (i, (atp, anm)) in enumerate(args):
            tmpl.append("  $abi_atp{0} $anm{0},".format(i))
        tmpl[-1] = re.sub(r",?$", "", tmpl[-1])  # remove trailing comma of last argument
        tmpl.append(") {");
        rcast = "($abi_tp)($wpi_tp)" if re.search(r"MPI(X?)_", tp) else ""
        tmpl.append("  return "+rcast+"$mpi_nm(");
        for (i, (atp, anm)) in enumerate(args):
            acast = "($mpi_atp{0})($wpi_atp{0})".format(i) if re.search(r"MPI(X?)_", atp) else ""
            tmpl.append("    "+acast+"$anm{0},".format(i))
        tmpl[-1] = re.sub(r",?$", "", tmpl[-1])  # remove trailing comma of last argument
        tmpl.append("  );");
        tmpl.append("}");
        file.write(Template("\n".join(tmpl)).substitute(subs))
        file.write("\n")

with open("src/mpiabi_defn_constants_fortran.h", "w") as file:
    file.write("!     Define Fortran MPI constants\n")
    file.write("\n")

    # Declarations
    for (tp, nm) in constants_fortran:
        subs = {'mpi_nm': nm,
                'abi_nm': re.sub(r"MPI(X?)_", r"MPI\1ABI_", nm)}
        tmpl = []
        tmpl.append("      integer $abi_nm")
        tmpl.append("      common /$abi_nm/ $abi_nm")
        file.write("\n".join(map(lambda line: wrap(Template(line).substitute(subs)), tmpl)))
        file.write("\n")

    # Definitions
    for (tp, nm) in constants_fortran:
        subs = {'mpi_nm': nm,
                'abi_nm': re.sub(r"MPI(X?)_", r"MPI\1ABI_", nm)}
        tmpl = []
        tmpl.append("      $abi_nm = $mpi_nm")
        file.write("\n".join(map(lambda line: wrap(Template(line).substitute(subs)), tmpl)))
        file.write("\n")

with open("src/mpiabi_defn_functions_fortran.h", "w") as file:
    file.write("// Define Fortran MPI functions\n")
    file.write("\n")
    for (tp, nm, args) in functions_fortran:
        subs = {'mpi_tp': tp,
                'abi_tp': re.sub(r"MPI(X?)_\w+", r"MPIABI_Fint", tp),
                'mpi_nm': nm.lower() + "_",
                'abi_nm': re.sub(r"MPI(X?)_", r"MPI\1ABI_", nm).lower() + "_"}
        for (i, (atp, anm)) in enumerate(args):
            subs['mpi_atp{0}'.format(i)] = atp
            subs['abi_atp{0}'.format(i)] = re.sub(r"MPI(X?)_\w+", r"MPIABI_Fint", atp)
            subs['anm{0}'.format(i)] = anm
        tmpl = ["extern $abi_tp $mpi_nm("]
        for (i, (atp, anm)) in enumerate(args):
            tmpl.append("  $abi_atp{0} $anm{0},".format(i))
        tmpl[-1] = re.sub(r",?$", "", tmpl[-1])  # replace trailing comma of last argument
        tmpl.append(");");
        tmpl.append("$abi_tp $abi_nm(")
        for (i, (atp, anm)) in enumerate(args):
            tmpl.append("  $abi_atp{0} $anm{0},".format(i))
        tmpl[-1] = re.sub(r",?$", "", tmpl[-1])  # replace trailing comma of last argument
        tmpl.append(") {");
        tmpl.append("  return $mpi_nm(");
        for (i, (atp, anm)) in enumerate(args):
            tmpl.append("    $anm{0},".format(i))
        tmpl[-1] = re.sub(r",?$", "", tmpl[-1])  # replace trailing comma of last argument
        tmpl.append("  );")
        tmpl.append("}");
        file.write(Template("\n".join(tmpl)).substitute(subs))
        file.write("\n")
