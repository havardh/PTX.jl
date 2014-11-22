module JuliaMM

using BenchmarkLite

include("../gpu.jl")

function MatrixMultiply(A, B, C, n, m, k, v)

  col = get_global_id(0);
  row = get_global_id(1);

  i = 0;
  while i < m

    v = v + (A[i + row*m] * B[col + i*k]);
    i += 1;
  end 

  C[col + row*k] = v;

  return
end

type Benchmark{T} <: Proc end

Base.string{T}(::Benchmark{T}) = "Julia GPU(" * string(T) * ")"
Base.length{T}(p::Benchmark{T}, n) = n
Base.isvalid{T}(p::Benchmark{T}, n) = n > 0

function Base.start{T}(p::Benchmark{T}, n)
  dev, ctx = create()
  ptx = code_ptx(MatrixMultiply, 
    (GPUArray{T},
    GPUArray{T},
    GPUArray{T},
    Int64, Int64, Int64, T)
  )

  md = CUDA.CuModule(source=ptx)
  mm = CUDA.CuFunction(md, "MatrixMultiply")

  A = GPUArray{T}(rand(T, n*n));
  B = GPUArray{T}(rand(T, n*n));
  
  A_gpu = CUDA.CuArray(A.data)
  B_gpu = CUDA.CuArray(B.data)
  C_gpu = CUDA.CuArray(T, n*n)

  (ctx, mm, (A_gpu, B_gpu, C_gpu))

end

function Base.run{T}(p::Benchmark{T}, n, state)

  (ctx, mm, (A_gpu, B_gpu, C_gpu)) = state

  gs = n > 32 ? div(n, 32) : 1
  bs = n > 32 ? 32 : n

  CUDA.launch(mm, (gs, gs), (bs, bs), (A_gpu, B_gpu, C_gpu, n, n, n, zero(T)))

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
