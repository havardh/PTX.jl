module JuliaBlur
using BenchmarkLite
include("../gpu.jl")

@inline IDX(i,j) = i+j*4
@inline ON_BORDER(i,j) = (i==0 || j==0 || i==3 || j==3)
@inline ON_BORDER_HOST(i, j) = ON_BORDER(i-1,j-1)

# Kernel
function blur(IN::GPUArray{Int64}, OUT::GPUArray{Int64})

  idx = get_global_id(0)
  idy = get_global_id(1)

  v = 0

  if !ON_BORDER(idx, idy)
    for i=-1:1, j=-1:1
      v = v + IN[IDX(idx+i, idy+j)]
    end
    v = div(v, 9)
  else
    v = IN[IDX(idx, idy)]
  end

  OUT[IDX(idx, idy)] = v

  return
end


type Benchmark <: Proc end

Base.string(::Benchmark) = "JuliaBlur"
Base.length(p::Benchmark, n) = n
Base.isvalid(p::Benchmark, n) = n & (n-1) == 0
function Base.start(p::Benchmark, n)

  dev, ctx = create()
  ptx = @code_ptx blur(GPUArray{Int64}([1]), GPUArray{Int64}([1]))
  md = CUDA.CuModule(source=ptx)
  kernel = CUDA.CuFunction(md, "blur1")

  img = GPUArray{Int64}(rand(Int64, n*n))

  img_in = CUDA.CuArray(img.data)
  img_out = CUDA.CuArray(Int64, n * n)

  (dev, ctx, md, kernel, img, img_in, img_out)
end

function Base.run(p::Benchmark, n, state)

  (dev, ctx, md, kernel, img, img_in, img_out) = state;

  gs = n > 32 ? div(n, 32) : 1
  bs = n > 32 ? 32 : n

  CUDA.launch(kernel, (gs, gs), (bs, bs), (img_in, img_out))

end
function Base.done(p::Benchmark, n, state)

  (dev, ctx, md, kernel, img, img_in, img_out) = state;

  result = CUDA.to_host(img_out)

  CUDA.free(img_in)
  CUDA.free(img_out)

  CUDA.destroy(ctx)

end

export Benchmark

end

