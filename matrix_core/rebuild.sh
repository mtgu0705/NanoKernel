#!/bin/sh
ARCH=gfx936

rm -rf build && mkdir build && cd build 

/opt/rocm/bin/hipcc -x hip ../matrix_core.cc  --offload-arch=$ARCH  -O3 -Wall -save-temps -o matrix_core.exe
