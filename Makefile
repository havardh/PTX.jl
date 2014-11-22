LLVM_HOME=/Volumes/Untitled/projects/llvm-3.4
INCLUDE=-I${LLVM_HOME}/include
LLVM_CONFIG=`llvm-config --cxxflags`

#clang++ -cc1 -fno-builtin -emit-llvm-bc -triple nvptx64-nvidia-cuda lowered-julia-array.cl


all: julia2ptx

julia2ptx: main.o linker.o lower-array-pass.o
	clang++ -g -rdynamic main.o linker.o lower-array-pass.o `llvm-config --cxxflags --libs all` `llvm-config --ldflags` -o julia2ptx

main.o : main.cpp
	clang++ -g -c main.cpp ${LLVM_CONFIG}

linker.o: linker.cpp
	clang++ -g -c linker.cpp ${LLVM_CONFIG}

lower-array-pass.o: lower-array-pass.cpp
	clang++ -g -c lower-array-pass.cpp ${LLVM_CONFIG}
