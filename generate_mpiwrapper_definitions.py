#!/usr/bin/env python3

import re
from string import Template

from mpi_constants import constants
from mpi_functions import functions
from mpi_functions_fortran import functions_fortran

def wpi_type(tp):
    tp = re.sub(r"MPI_", "WPI_", tp)
    tp = re.sub(r"WPI_(Comm|Datatype|Errhandler|File|Group|Info|Message|Op|Request|Status|Win) \*", r"WPI_\1Ptr", tp)
    tp = re.sub(r"const WPI_(Comm|Datatype|Errhandler|File|Group|Info|Message|Op|Request|Status|Win)", r"WPI_const_\1", tp)
    return tp

print()
print("// C constants")
for (tp, nm) in constants:
    subs = {'mpi_tp': tp,
            'abi_tp': re.sub(r"MPI_", "MPIABI_", tp),
            'mpi_nm': nm,
            'abi_nm': re.sub(r"MPI_", "MPIABI_", nm)}
    rcast = "($abi_tp)" if re.search(r"MPI_", tp) else ""
    print(Template("$abi_tp const $abi_nm = "+rcast+"$mpi_nm;").substitute(subs))

print()
print("// C functions")
for (tp, nm, args, flags) in functions:
    assert flags is None or flags == "manual"
    if flags == "manual":
        continue
    subs = {'mpi_tp': tp,
            'wpi_tp': re.sub(r"MPI_", "WPI_", tp),
            'abi_tp': re.sub(r"MPI_", "MPIABI_", tp),
            'mpi_nm': nm,
            'abi_nm': re.sub(r"MPI_", "MPIABI_", nm)}
    for (i, (atp, anm)) in enumerate(args):
        subs['mpi_atp{0}'.format(i)] = atp
        subs['wpi_atp{0}'.format(i)] = wpi_type(atp)
        subs['abi_atp{0}'.format(i)] = re.sub(r"MPI_", "MPIABI_", atp)
        subs['anm{0}'.format(i)] = anm
    tmpl = ["$abi_tp $abi_nm("]
    for (i, (atp, anm)) in enumerate(args):
        tmpl.append("  $abi_atp{0} $anm{0},".format(i))
    tmpl[-1] = re.sub(r",?$", "", tmpl[-1])  # remove trailing comma of last argument
    tmpl.append(") {");
    tmpl.append("  fprintf(stderr, \"$mpi_nm.0\\n\");")
    rcast = "($abi_tp)($wpi_tp)" if re.search(r"MPI_", tp) else ""
    # tmpl.append("  return "+rcast+"$mpi_nm(");
    tmpl.append("  $abi_tp res = "+rcast+"$mpi_nm(");
    for (i, (atp, anm)) in enumerate(args):
        acast = "($mpi_atp{0})($wpi_atp{0})".format(i) if re.search(r"MPI_", atp) else ""
        tmpl.append("    "+acast+"$anm{0},".format(i))
    tmpl[-1] = re.sub(r",?$", "", tmpl[-1])  # remove trailing comma of last argument
    tmpl.append("  );");
    tmpl.append("  fprintf(stderr, \"$mpi_nm.9\\n\");")
    tmpl.append("  return res;")
    tmpl.append("}");
    print(Template("\n".join(tmpl)).substitute(subs))

# Fortran constants are defined via Fortran code

print()
print("// Fortran functions")
for (tp, nm, args) in functions_fortran:
    subs = {'mpi_tp': tp,
            'abi_tp': re.sub(r"MPI_\w+", "MPIABI_Fint", tp),
            'mpi_nm': nm.lower() + "_",
            'abi_nm': re.sub(r"MPI_", "MPIABI_", nm).lower() + "_"}
    for (i, (atp, anm)) in enumerate(args):
        subs['mpi_atp{0}'.format(i)] = atp
        subs['abi_atp{0}'.format(i)] = re.sub(r"MPI_\w+", "MPIABI_Fint", atp)
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
    # tmpl = ["extern $abi_tp $mpi_nm("]
    # for (i, (atp, anm)) in enumerate(args):
    #     tmpl.append("  $abi_atp{0} $anm{0},".format(i))
    # tmpl[-1] = re.sub(r",?$", "", tmpl[-1])  # replace trailing comma of last argument
    # tmpl.append(");");
    # tmpl.append("$abi_tp (* const $abi_nm)(")
    # for (i, (atp, anm)) in enumerate(args):
    #     tmpl.append("  $abi_atp{0} $anm{0},".format(i))
    # tmpl[-1] = re.sub(r",?$", "", tmpl[-1])  # remove trailing comma of last argument
    # tmpl.append(") = $mpi_nm;")
    print(Template("\n".join(tmpl)).substitute(subs))
