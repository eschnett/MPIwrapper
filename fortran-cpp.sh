#!/bin/sh
cpp $1 |
    sed -e 's/\[n\]/\n      /g' |
    sed -e 's/\[c\]/                                                                        \&\n     \&/g' |
    sed -e 's/\(^.\{72\}\) *\&$/\1\&/g' >$2
