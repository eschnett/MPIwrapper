#!/bin/sh
cpp $1 | sed -e 's/&/\n     /g' >$2
