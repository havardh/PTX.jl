module CUDABlur

using BenchmarkLite
import CUDA


type Benchmark <: Proc end

Base.string(::Benchmark) = "CUDABlur"
Base.length(p::Benchmark, n) = n
Base.isvalid(p::Benchmark, n) = n & (n-1) == 0
function Base.start(p::Benchmark, n)

  dev = CUDA.CuDevice(0)
  ctx = CUDA.create_context(dev)
  md = CUDA.CuModule(filename="blur.ptx")
  blur = CUDA.CuFunction(md, "blur")

  img = rand(Int64, n*n)

  img_in = CUDA.CuArray(img)
  img_out = CUDA.CuArray(Int64, n * n)

  (dev, ctx, md, blur, img, img_in, img_out)
end

function Base.run(p::Benchmark, n, state)

  (dev, ctx, md, blur, img, img_in, img_out) = state;

  gs = n > 32 ? div(n, 32) : 1
  bs = n > 32 ? 32 : n

  CUDA.launch(blur, (gs, gs), (bs, bs), (img_in, img_out))

end
function Base.done(p::Benchmark, n, state)

  (dev, ctx, md, blur, img, img_in, img_out) = state;

  result = CUDA.to_host(img_out)

  CUDA.free(img_in)
  CUDA.free(img_out)

  CUDA.destroy(ctx)

end

export Benchmark

end
