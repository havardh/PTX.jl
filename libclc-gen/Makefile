LLVM_HOME=/Volumes/Untitled/projects/llvm-3.4
INCLUDE=-I${LLVM_HOME}/include
LLVM_CONFIG=`llvm-config --cxxflags`

all: libclc-gen
	./libclc-gen ../lib/nvptx64-nvidia-cuda.ll 2> .opencl.jl
	grep -v 'Not supported' .opencl.jl >../src/OpenCL.jl


libclc-gen: gen.o emit-julia-declarations-pass.o
	clang++ -g -rdynamic gen.o emit-julia-declarations-pass.o `llvm-config --cxxflags --libs all` `llvm-config --ldflags` -o libclc-gen

gen.o : gen.cpp
	clang++ -g -c gen.cpp ${LLVM_CONFIG}

emit-julia-declarations-pass.o : emit-julia-declarations-pass.cpp
	clang++ -g -c emit-julia-declarations-pass.cpp ${LLVM_CONFIG}

