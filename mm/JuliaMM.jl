module JuliaMM

using BenchmarkLite

include("../gpu.jl")

function MatrixMultiply(A, B, C, n, m, k)

  col = get_global_id(0);
  row = get_global_id(1);

  i = 0;
  v = 0;
  while i < m

    v = v + (A[i + row*m] * B[col + i*k]);
    i += 1;
  end 

  C[col + row*k] = v;

end

type Benchmark <: Proc end

Base.string(::Benchmark) = "Julia GPU"
Base.length(p::Benchmark, n) = n
Base.isvalid(p::Benchmark, n) = n > 0

function Base.start(p::Benchmark, n)
  dev, ctx = create()
  ptx = @code_ptx MatrixMultiply(
    GPUArray{Int64}([0]),
    GPUArray{Int64}([0]),
    GPUArray{Int64}([0]),
    0, 20, 0
  )

  md = CUDA.CuModule(source=ptx)
  mm = CUDA.CuFunction(md, "MatrixMultiply1")

  A = GPUArray{Int64}(rand(Int64, n*n));
  B = GPUArray{Int64}(rand(Int64, n*n));
  
  A_gpu = CUDA.CuArray(A.data)
  B_gpu = CUDA.CuArray(B.data)
  C_gpu = CUDA.CuArray(Int64, n*n)

  (ctx, mm, (A_gpu, B_gpu, C_gpu))

end

function Base.run(p::Benchmark, n, state)

  (ctx, mm, (A_gpu, B_gpu, C_gpu)) = state

  gs = n > 32 ? div(n, 32) : 1
  bs = n > 32 ? 32 : n

  CUDA.launch(mm, (gs, gs), (bs, bs), (A_gpu, B_gpu, C_gpu, n, n, n))

end

function Base.done(p::Benchmark, n, state)

  (ctx, mm, (A_gpu, B_gpu, C_gpu)) = state

  result = CUDA.to_host(C_gpu)

  CUDA.free(A_gpu)
  CUDA.free(B_gpu)
  CUDA.free(C_gpu)

  CUDA.destroy(ctx)

end

export Benchmark

end
