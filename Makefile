LLVM_HOME=/Volumes/Untitled/projects/llvm-3.4
INCLUDE=-I${LLVM_HOME}/include




all: main.cpp
	clang++ -rdynamic main.cpp linker.cpp lower-array-pass.cpp `llvm-config --cxxflags --libs all` `llvm-config --ldflags` -o julia2ptx

