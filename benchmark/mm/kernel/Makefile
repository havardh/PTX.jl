
LIBCLC=../../../lib/libclc


all: Float32.cu  Float64.cu  Int32.cu  Int64.cu
	nvcc -m64 -arch=sm_20 -O3 -ptx *.cu


clang: Float64.cu
	clang -ccc-host-triple ptx64 -Xclang -target-feature -Xclang -ptx64 -Xclang -target-feature -Xclang +sm20 -I${LIBCLC}/generic/include -include clc/clc.h -Dcl_clang_storage_class_specifiers -O3 Float64.cl -S -o Float64_clang.ptx
