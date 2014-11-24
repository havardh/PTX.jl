module CudaMM

using BenchmarkLite

import CUDA

type Benchmark{T} <: Proc end

Base.string{T}(::Benchmark{T}) = "Cuda GPU(" * string(T) * ")"
Base.length{T}(p::Benchmark{T}, n) = n
Base.isvalid{T}(p::Benchmark{T}, n) = n > 0

function Base.start{T}(p::Benchmark{T}, n)

  dev = CUDA.CuDevice(0)
  ctx = CUDA.create_context(dev)

  md = CUDA.CuModule(filename="./kernel/"*string(T)*".ptx")
  mm = CUDA.CuFunction(md, "MatrixMultiply")

  A = rand(T, n*n);
  B = rand(T, n*n);
  
  A_gpu = CUDA.CuArray(A)
  B_gpu = CUDA.CuArray(B)
  C_gpu = CUDA.CuArray(T, n*n)

  (ctx, mm, (A_gpu, B_gpu, C_gpu))

end

function Base.run{T}(p::Benchmark{T}, n, state)

  (ctx, mm, (A_gpu, B_gpu, C_gpu)) = state

  gs = n > 32 ? div(n, 32) : 1
  bs = n > 32 ? 32 : n

  CUDA.launch(mm, (gs, gs), (bs, bs), (A_gpu, B_gpu, C_gpu, n, n, n))

end

function Base.done{T}(p::Benchmark{T}, n, state)

  (ctx, mm, (A_gpu, B_gpu, C_gpu)) = state

  result = CUDA.to_host(C_gpu)

  CUDA.free(A_gpu)
  CUDA.free(B_gpu)
  CUDA.free(C_gpu)

  CUDA.destroy(ctx)

end

export Benchmark

end
