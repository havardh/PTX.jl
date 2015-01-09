import CUDA
using PTX

function cuda_run(image)
  n = 256
  dev = CUDA.CuDevice(0)
  ctx = CUDA.create_context(dev)

  md = CUDA.CuModule(filename="blur.ptx")
  kernel = CUDA.CuFunction(md, "blur")

  img = GPUArray{Float32}(image)

  img_in = CUDA.CuArray(img.data)
  img_out = CUDA.CuArray(Float32, n * n)

  gs = 16
  bs = 16

  CUDA.launch(kernel, (gs, gs), (bs, bs), (img_in, img_out))

  result = CUDA.to_host(img_out)

  CUDA.free(img_in)
  CUDA.free(img_out)

  CUDA.destroy(ctx)

  return result
end
