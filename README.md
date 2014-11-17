
Small adhoc compiler step to produce PTX code using the LLVM NVPTX backend for NVIDIA GPUs.

Relies on the __@code_llvm__ function for extrating a LLVM function declaration.

Needs the patch found on https://github.com/havardh/julia/tree/noinline to produce suitable functions.

Supports a subset of Julia based on vectors. 
